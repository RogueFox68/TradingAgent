import json
import time
import datetime
import pandas as pd
import numpy as np
import os
import sys
import requests
import config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# yfinance is imported lazily. It is still the PRIMARY bar source (see
# fetch_daily_bars) because MIN_VOLUME below is calibrated against consolidated
# volume, which is what Yahoo reports. But it broke outright in 2026-09, and an
# import-time failure of it used to take the whole scan down before it could
# reach the Alpaca fallback.
yf = None
_yf_unavailable = False


def _load_yfinance():
    """Import yfinance on first use; None (logged once) if it won't import."""
    global yf, _yf_unavailable
    if yf is not None or _yf_unavailable:
        return yf
    try:
        import yfinance as _mod
    except Exception as e:
        _yf_unavailable = True
        print(f"[!] yfinance unavailable ({e}) — falling back to Alpaca bars.")
        return None
    yf = _mod
    return yf


# --- CONFIGURATION ---
MIN_VOLUME = 2_000_000   # Increased liquidity floor (CONSOLIDATED volume)
MIN_PRICE = 15.00        
MAX_PRICE = 1000.00
OUTPUT_FILE = "dragnet_candidates.json"
BENCHMARK_TICKER = "SPY"

# Alpaca's free feed is IEX, which is a single venue — roughly a few percent of
# consolidated volume. MIN_VOLUME cannot be applied to it directly, and guessing
# a scale factor to make it fit would be exactly the mistake of calibrating a
# proxy by eye. So when bars come from Alpaca, liquidity is filtered by RANK
# instead: keep the most-liquid names in the scanned universe by dollar volume.
# That is scale-invariant, so it needs no calibration, and it preserves the
# filter's actual intent ("trade the liquid end of the market").
ALPACA_LIQUIDITY_RANK = 400

# A scan that finds nothing is a FAILED scan, not an empty market. Publishing it
# would overwrite the Beelink's good targets with nothing, and the fleet's
# staleness check would not catch it — the file would be fresh, just empty.
MIN_VIABLE_CANDIDATES = 5

# Institutional Seed List: Ensures we analyze the "Leaders" first
# In a Bull Trend, these are the assets most likely to have a "High Confidence" LLM score.
SEED_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "AMD", "NFLX", "INTC", "TMO", "LIN", "WMT", "SBUX", "INTU",
    "QCOM", "AMGN", "ISRG", "TXN", "V", "MA", "UNH", "JNJ", "PG", "PM",
    "PLTR", "COIN", "MSTR", "SHOP", "SNOW", "SPOT", "NET", "DDOG"
]

def _alert(message):
    """Best-effort Discord ping to the overseer webhook. A scan that dies
    quietly is the failure mode this whole file is being hardened against, so
    a failure to alert is itself printed."""
    hook = getattr(config, "WEBHOOK_OVERSEER", "")
    if not hook or "YOUR" in hook:
        return
    try:
        requests.post(hook, json={"content": message, "username": "Market Scanner"},
                      timeout=10)
    except Exception as e:
        print(f"[!] Overseer alert failed: {e}")


# --- GATEKEEPER (CST EDITION) ---
def is_mission_time():
    now = datetime.datetime.now()
    if now.weekday() > 4: 
        print(f"[Gatekeeper] Today is {now.strftime('%A')}. Market Closed.")
        return False
    current_minutes = now.hour * 60 + now.minute
    market_open = 8 * 60 + 30 
    market_close = 15 * 60     
    if market_open <= current_minutes <= market_close:
        return True
    print(f"[Gatekeeper] Time is {now.strftime('%H:%M')}. Market Closed.")
    return False

# --- AUTHENTICATION ---
def get_alpaca_client():
    try:
        return TradingClient(config.API_KEY, config.SECRET_KEY, paper=config.PAPER)
    except Exception as e:
        print(f"[!] Error loading Alpaca client from config: {e}")
        return None

# --- NATIVE MATH ENGINE ---
class TechnicalMath:
    @staticmethod
    def get_sma(series, window):
        return series.rolling(window=window).mean()

    @staticmethod
    def get_ema(series, window):
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def get_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_adx(high, low, close, window=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([
            (high - low), 
            (high - close.shift(1)).abs(), 
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        tr = tr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/window, adjust=False).mean() / tr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/window, adjust=False).mean()

    @staticmethod
    def get_relative_strength(ticker_series, benchmark_series):
        """
        Calculates the RS Line (Ticker / Benchmark).
        Returns True if the RS is trending up (current ratio > 14d average).
        """
        rs_line = ticker_series / benchmark_series
        rs_sma = rs_line.rolling(window=14).mean()
        return rs_line.iloc[-1] > rs_sma.iloc[-1]

# --- BAR SOURCES ---------------------------------------------------------
# Two providers, one shape. Every consumer below takes {symbol: DataFrame} with
# Open/High/Low/Close/Volume columns, so which provider answered never leaks
# into the strategy logic — except for volume scale, which fetch_daily_bars
# reports back so the caller can filter appropriately.

_data_client = None


def _alpaca_data_client():
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
    return _data_client


def _bars_from_yfinance(symbols, days):
    """{symbol: OHLCV DataFrame} from Yahoo, or {} if it cannot answer."""
    mod = _load_yfinance()
    if mod is None:
        return {}
    period = f"{days}d" if days < 365 else f"{max(2, round(days / 365))}y"
    raw = mod.download(symbols, period=period, interval="1d",
                       group_by="ticker", progress=False)
    if raw is None or raw.empty:
        return {}

    out = {}
    for sym in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym not in raw.columns.levels[0]:
                    continue
                df = raw[sym]
            elif len(symbols) == 1:
                df = raw
            else:
                # Exactly one ticker survived a multi-ticker request, so
                # yfinance flattened the MultiIndex away. There is no way to
                # tell WHICH one it is, and guessing would assign one stock's
                # bars to every symbol in the chunk. Drop the chunk.
                continue
            df = df.dropna()
            if not df.empty:
                out[sym] = df
        except Exception as e:
            print(f"   [!] yfinance parse error for {sym}: {e}")
    return out


def _bars_from_alpaca(symbols, days):
    """{symbol: OHLCV DataFrame} from Alpaca, or {} if it cannot answer.

    Chunked: the request goes in the URL, so a 4,800-symbol universe in one
    call is rejected on length alone."""
    client = _alpaca_data_client()
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    out = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        try:
            raw = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start)).df
        except Exception as e:
            print(f"   [!] Alpaca bars error ({chunk[0]}..): {e}")
            continue
        if raw is None or raw.empty:
            continue
        # BarSet.df is a (symbol, timestamp) MultiIndex.
        if isinstance(raw.index, pd.MultiIndex):
            for sym in raw.index.get_level_values("symbol").unique():
                df = raw.xs(sym, level="symbol")
                out[sym] = pd.DataFrame({
                    "Open": df["open"], "High": df["high"], "Low": df["low"],
                    "Close": df["close"], "Volume": df["volume"],
                }).dropna()
    return out


def fetch_daily_bars(symbols, days):
    """Daily OHLCV for `symbols`, from Yahoo if it answers and Alpaca if not.

    Returns (bars, source). `source` is "yfinance" or "alpaca" and matters for
    ONE thing: Alpaca's free feed is IEX-only, so its Volume column is a single
    venue's share and cannot be compared against MIN_VOLUME. Prices/indicators
    are unaffected — a daily OHLC bar is a daily OHLC bar."""
    if not symbols:
        return {}, "none"
    try:
        bars = _bars_from_yfinance(symbols, days)
    except Exception as e:
        print(f"   [!] yfinance download failed: {e}")
        bars = {}
    if bars:
        return bars, "yfinance"

    print("   -> yfinance returned nothing; falling back to Alpaca bars.")
    return _bars_from_alpaca(symbols, days), "alpaca"


# --- LOGIC ---

def get_market_universe(client):
    print("--- 🕸️ DEPLOYING PRECISION DRAGNET ---")
    # Preserve order of institutional seed list using dict keys
    universe = list(dict.fromkeys(SEED_LIST)) 
    
    try:
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = client.get_all_assets(req)
        alpaca_list = [
            a.symbol for a in assets 
            if a.tradable and a.marginable and a.shortable and "." not in a.symbol
        ]
        # Combine seed list with Alpaca universe, keeping seeds at the front
        universe.extend([s for s in alpaca_list if s not in universe])
        print(f"   -> Universe expanded: {len(universe)} assets (Seed priority enabled).")
        return universe
    except Exception as e:
        print(f"[!] Alpaca Error: {e}. Falling back to Seed List.")
        return universe

def filter_by_volume(tickers):
    print(f"2. Filtering for Institutional Liquidity (Vol > {MIN_VOLUME/1_000_000:.1f}M)...")
    liquid = []
    ranked = []          # (dollar_volume, symbol) — only used on the Alpaca path
    fell_back = False
    chunk_size = 100

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        bars, source = fetch_daily_bars(chunk, days=10)
        if source == "alpaca":
            fell_back = True
        for sym, df in bars.items():
            try:
                if df.empty:
                    continue
                avg_vol = float(df["Volume"].tail(3).mean())
                curr_price = float(df["Close"].iloc[-1])
                if not (MIN_PRICE < curr_price < MAX_PRICE):
                    continue
                if source == "alpaca":
                    # IEX-only volume: rank instead of threshold (see the
                    # ALPACA_LIQUIDITY_RANK note at the top of this file).
                    ranked.append((avg_vol * curr_price, sym))
                elif avg_vol > MIN_VOLUME:
                    liquid.append(sym)
            except Exception as e:
                print(f"   Warning skipping liquid {sym}: {e}")

    if fell_back:
        # Keep the seed list regardless of rank: those are the institutional
        # leaders the scan is built around, and they are liquid by definition.
        ranked.sort(reverse=True)
        top = [sym for _, sym in ranked[:ALPACA_LIQUIDITY_RANK]]
        seeds = [s for s in SEED_LIST if s in {sym for _, sym in ranked}]
        liquid.extend(s for s in seeds if s not in top)
        liquid.extend(top)
        liquid = list(dict.fromkeys(liquid))
        print(f"   -> Alpaca (IEX) volume is single-venue; ranked to the top "
              f"{ALPACA_LIQUIDITY_RANK} by dollar volume instead of a "
              f"{MIN_VOLUME/1_000_000:.1f}M threshold.")

    print(f"   -> Final Liquid List: {len(liquid)} stocks.")
    return liquid

def analyze_technicals(tickers):
    print(f"3. Analyzing High-Conviction Technicals for {len(tickers)} stocks...")
    candidates = []
    if not tickers:
        return []

    # Benchmark (SPY) for Relative Strength. Price-only, so either source works.
    spy_bars, _ = fetch_daily_bars([BENCHMARK_TICKER], days=730)
    spy_data = None
    if BENCHMARK_TICKER in spy_bars:
        spy_data = spy_bars[BENCHMARK_TICKER]["Close"]
        if isinstance(spy_data, pd.DataFrame):
            spy_data = spy_data.squeeze()
    else:
        print("[!] Benchmark unavailable. RS Filtering disabled.")

    bars, source = fetch_daily_bars(tickers, days=730)
    if not bars:
        # LOUD: the caller refuses to publish an empty scan, but say why here.
        print("   [!] No bars from ANY source — technical analysis impossible.")
        return []
    print(f"   -> {len(bars)}/{len(tickers)} symbols returned bars (via {source}).")

    for sym, df in bars.items():
        try:
            if len(df) < 205:
                continue
            df = df.dropna()

            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            rsi = TechnicalMath.get_rsi(close)
            adx = TechnicalMath.get_adx(df['High'], df['Low'], close)
            sma200 = TechnicalMath.get_sma(close, 200)
            ema20 = TechnicalMath.get_ema(close, 20)
            ema50 = TechnicalMath.get_ema(close, 50)

            curr_price = float(close.iloc[-1])
            curr_rsi = float(rsi.iloc[-1])
            curr_adx = float(adx.iloc[-1])

            # Relative Strength Check
            is_outperforming = True
            if spy_data is not None:
                # Align SPY data length with Ticker data length
                spy_sliced = spy_data.tail(len(close))
                is_outperforming = TechnicalMath.get_relative_strength(close, spy_sliced)

            # --- STRATEGY SEGREGATION ---

            # 1. TREND TARGETS (The "Institutional Leader" Setup)
            if (curr_price > ema20.iloc[-1] > ema50.iloc[-1] > sma200.iloc[-1] and
                    is_outperforming and
                    curr_adx > 25 and
                    55 < curr_rsi < 75):
                candidates.append({"symbol": sym, "type": "trend_targets", "score": curr_adx})

            # 2. SURVIVOR TARGETS (High Quality Dip)
            elif curr_price > sma200.iloc[-1] and curr_rsi < 40:
                candidates.append({"symbol": sym, "type": "survivor_targets", "score": (50 - curr_rsi)})

            # 3. WHEEL TARGETS (The Stable Income Setup)
            elif curr_price > sma200.iloc[-1] and 40 <= curr_rsi <= 55 and curr_adx < 25:
                candidates.append({"symbol": sym, "type": "wheel_targets", "score": (50 - curr_rsi)})

            # 4. SHORT TARGETS (The Breakdown)
            elif curr_price < sma200.iloc[-1]:
                candidates.append({"symbol": sym, "type": "short_targets", "score": curr_adx})

        except Exception as e:
            # Was a bare `except: continue`, which hid the difference between
            # "this ticker has odd data" and "the maths is broken for every
            # ticker" — the second one looks like a quiet market.
            print(f"   Warning skipping technicals for {sym}: "
                  f"{type(e).__name__}: {e}")

    return candidates

def get_earnings_date(ticker_symbol):
    """True = safe to trade (no earnings inside 2 days). Alpaca has no earnings
    calendar, so this stays on yfinance; with yfinance down the guard is simply
    unavailable and every ticker is allowed through, as it always was on a
    lookup failure."""
    mod = _load_yfinance()
    if mod is None:
        return True
    try:
        ticker = mod.Ticker(ticker_symbol)
        
        # Suppress 404 noise from ETFs or missing calendars
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            calendar = ticker.calendar
            
        if calendar is None or (hasattr(calendar, 'empty') and calendar.empty):
            return True
        
        # Extract date from various yfinance formats
        next_earnings = None
        if hasattr(calendar, 'iloc'):
            next_earnings = calendar.iloc[0][0]
        elif isinstance(calendar, dict):
            next_earnings = calendar.get('Earnings Date', [None])[0]
            
        if next_earnings and isinstance(next_earnings, (datetime.date, datetime.datetime, pd.Timestamp)):
            if hasattr(next_earnings, 'to_pydatetime'):
                next_earnings = next_earnings.to_pydatetime()
            if next_earnings.tzinfo:
                next_earnings = next_earnings.replace(tzinfo=None)
            
            days_until = (next_earnings - datetime.datetime.now()).days
            if 0 <= days_until <= 2:
                return False
    except Exception:
        # Missing/odd calendars are routine (ETFs have none) and this is a
        # best-effort guard, so a failure allows the ticker through — as before.
        pass
    return True

def run_dragnet():
    if not is_mission_time():
        sys.exit(0)

    client = get_alpaca_client()
    if not client: return
    
    all_tickers = get_market_universe(client)
    liquid_tickers = filter_by_volume(all_tickers)
    results = analyze_technicals(liquid_tickers)

    # A scan that found (almost) nothing is a FAILED scan, not a quiet market.
    # Writing it would hand sector_scout_3 an empty candidate list, which
    # produces an empty active_targets.json, which is SCP'd over the Beelink's
    # good targets — and the fleet's 24h staleness check would NOT catch it,
    # because the file it receives is perfectly fresh. It is just empty.
    # Leaving the previous file in place and failing loudly is strictly better:
    # stale targets are a known, alerted state; empty ones are silent.
    if len(results) < MIN_VIABLE_CANDIDATES:
        msg = (f"Dragnet found only {len(results)} candidates from "
               f"{len(liquid_tickers)} liquid / {len(all_tickers)} scanned "
               f"(minimum {MIN_VIABLE_CANDIDATES}). Refusing to overwrite "
               f"{OUTPUT_FILE} — the previous scan stands.")
        print(f"\n[!] SCAN ABORTED: {msg}")
        _alert(f"🚨 **DRAGNET SCAN ABORTED**\n{msg}\n"
               f"Usually the bar source is down — check yfinance/Alpaca "
               f"reachability on this host.")
        sys.exit(1)

    
    final_output = {
        "trend_targets": [], "survivor_targets": [],
        "wheel_targets": [], "short_targets": []
    }
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n4. Applying Earnings Safety Guard...")
    for item in results:
        if get_earnings_date(item['symbol']):
            cat = item['type']
            if len(final_output[cat]) < 10:
                final_output[cat].append({
                    "symbol": item['symbol'],
                    "tech_score": round(item['score'], 2)
                })
            
    print("\n--- 🎯 PRECISION DRAGNET COMPLETE ---")
    print(json.dumps(final_output, indent=4))
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=4)
    print(f"   ✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_dragnet()
