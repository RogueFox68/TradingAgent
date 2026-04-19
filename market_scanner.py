import json
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import config
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# --- CONFIGURATION ---
MIN_VOLUME = 2_000_000   # Increased liquidity floor
MIN_PRICE = 15.00        
MAX_PRICE = 1000.00
OUTPUT_FILE = "dragnet_candidates.json"
BENCHMARK_TICKER = "SPY"

# Institutional Seed List: Ensures we analyze the "Leaders" first
# In a Bull Trend, these are the assets most likely to have a "High Confidence" LLM score.
SEED_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "PEP", "COST",
    "ADBE", "CSCO", "AMD", "NFLX", "INTC", "TMO", "LIN", "WMT", "SBUX", "INTU",
    "QCOM", "AMGN", "ISRG", "TXN", "V", "MA", "UNH", "JNJ", "PGE", "PM",
    "SND", "PLTR", "COIN", "MSTR", "SHOP", "SNOW", "SOTP", "NET", "DDOG"
]

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
    liquid_tickers = []
    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            df = yf.download(chunk, period="5d", interval="1d", group_by='ticker', progress=False)
            if df.empty: continue
            
            for sym in chunk:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if sym not in df.columns.levels[0]: continue
                        data = df[sym]
                    elif len(chunk) == 1:
                        data = df
                    else:
                        # Edge case: If exactly one stock succeeds out of multi-ticker chunk, 
                        # yfinance flattens the multi-index. Skip this single survivor to avoid 
                        # assigning its data falsely to other tickers in the loop.
                        continue
                        
                    if data.empty: continue
                    avg_vol = data['Volume'].tail(3).mean()
                    curr_price = float(data['Close'].iloc[-1])
                    if avg_vol > MIN_VOLUME and MIN_PRICE < curr_price < MAX_PRICE:
                        liquid_tickers.append(sym)
                except: continue
        except: pass
    print(f"   -> Final Liquid List: {len(liquid_tickers)} stocks.")
    return liquid_tickers

def analyze_technicals(tickers):
    print(f"3. Analyzing High-Conviction Technicals for {len(tickers)} stocks...")
    candidates = []
    if not tickers: return []
    
    # Download Benchmark (SPY) for Relative Strength calculations
    try:
        spy_data = yf.download(BENCHMARK_TICKER, period="2y", interval="1d", progress=False)['Close']
    except Exception as e:
        print(f"[!] Benchmark Error: {e}. RS Filtering disabled.")
        spy_data = None

    try:
        # Download all tickers in one batch
        data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
        
        for sym in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.levels[0]: continue
                    df = data[sym].copy()
                elif len(tickers) == 1:
                    df = data.copy()
                else:
                    continue
                    
                if len(df) < 205: continue 
                df = df.dropna()
                
                # Core Math
                close = df['Close']
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
                
                # 4. CONDOR TARGETS (The Chop/Sideways Setup)
                elif curr_adx < 20 and 40 < curr_rsi < 60:
                    candidates.append({"symbol": sym, "type": "condor_targets", "score": (20 - curr_adx)})
                
                # 5. SHORT TARGETS (The Breakdown)
                elif curr_price < sma200.iloc[-1] and curr_adx > 25:
                    candidates.append({"symbol": sym, "type": "short_targets", "score": curr_adx})

            except: continue
    except Exception as e:
        print(f"   [!] Tech Analysis Error: {e}")
    return candidates

def get_earnings_date(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
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
    except: pass
    return True

def run_dragnet():
    if not is_mission_time():
        sys.exit(0)

    client = get_alpaca_client()
    if not client: return
    
    all_tickers = get_market_universe(client)
    liquid_tickers = filter_by_volume(all_tickers)
    results = analyze_technicals(liquid_tickers)
    
    final_output = {
        "trend_targets": [], "survivor_targets": [],
        "wheel_targets": [], "condor_targets": [], "short_targets": []
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
