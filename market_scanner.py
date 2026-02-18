import json
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# --- CONFIGURATION ---
MIN_VOLUME = 1_500_000   # Liquidity Check
MIN_PRICE = 15.00        # No Penny Stocks
MAX_PRICE = 500.00
OUTPUT_FILE = "dragnet_candidates.json"
KEYS_FILE = "keys.json"

# --- GATEKEEPER (CST EDITION) ---
def is_mission_time():
    """
    Returns True ONLY if it is:
    1. A Weekday (Mon-Fri)
    2. AND (08:00 AM CST) OR (Market Hours 08:30 - 15:00 CST)
    """
    now = datetime.datetime.now()
    
    # 1. Block Weekends (5=Sat, 6=Sun)
    if now.weekday() > 4: 
        print(f"[Gatekeeper] Today is {now.strftime('%A')}. Market Closed. Aborting.")
        return False

    # 2. STRICT 08:30 CST START
    # We want to catch the market right at open or slightly after, 
    # but scanning at 8:00 AM (pre-market) gives misleading volume/price data.
    current_minutes = now.hour * 60 + now.minute
    market_open = 8 * 60 + 30  # 08:30 CST
    market_close = 15 * 60     # 15:00 CST (3:00 PM)

    if market_open <= current_minutes <= market_close:
        return True
        
    print(f"[Gatekeeper] Time is {now.strftime('%H:%M')}. Market Closed. Aborting.")
    return False

# --- AUTHENTICATION ---
def get_alpaca_client():
    try:
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
        return TradingClient(keys['APCA_API_KEY_ID'], keys['APCA_API_SECRET_KEY'], paper=True)
    except Exception as e:
        print(f"[!] Error loading keys.json: {e}")
        return None

# --- NATIVE MATH ENGINE ---
class TechnicalMath:
    @staticmethod
    def get_sma(series, window):
        return series.rolling(window=window).mean()

    @staticmethod
    def get_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_atr(high, low, close, window=14):
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        return tr.ewm(alpha=1/window, adjust=False).mean()

    @staticmethod
    def get_adx(high, low, close, window=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = TechnicalMath.get_atr(high, low, close, window)
        tr = tr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/window, adjust=False).mean() / tr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/window, adjust=False).mean()

# --- LOGIC ---

def get_market_universe(client):
    print("--- 🕸️ DEPLOYING DRAGNET (Universe Scan) ---")
    print("1. Fetching asset list from Alpaca...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
            assets = client.get_all_assets(req)
            clean_list = [
                a.symbol for a in assets 
                if a.tradable and a.marginable and a.shortable and "." not in a.symbol
            ]
            print(f"   -> Found {len(clean_list)} tradeable assets.")
            return clean_list
        except Exception as e:
            print(f"   [!] Alpaca Error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))
    
    print("   [!] Failed to fetch assets after retries.")
    return []

def filter_by_volume(tickers, chunk_size=200):
    print(f"2. Filtering for Liquidity (Vol > {MIN_VOLUME/1_000_000:.1f}M)...")
    liquid_tickers = []
    total = len(tickers)
    for i in range(0, total, chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            df = yf.download(chunk, period="5d", interval="1d", group_by='ticker', progress=False, threads=True)
            if len(chunk) > 1:
                for sym in chunk:
                    try:
                        if sym not in df.columns.levels[0]: continue
                        data = df[sym]
                        if data.empty: continue
                        avg_vol = data['Volume'].tail(3).mean()
                        curr_price = data['Close'].iloc[-1]
                        if avg_vol > MIN_VOLUME and MIN_PRICE < curr_price < MAX_PRICE:
                            liquid_tickers.append(sym)
                    except: continue
            else:
                avg_vol = df['Volume'].tail(3).mean()
                curr_price = df['Close'].iloc[-1]
                if avg_vol > MIN_VOLUME and MIN_PRICE < curr_price < MAX_PRICE:
                    liquid_tickers.append(tickers[0])
            print(f"   -> Batch {i}/{total}: Found {len(liquid_tickers)} candidates...", end='\r')
        except: pass
    print(f"\n   -> Final Liquid List: {len(liquid_tickers)} stocks.")
    return liquid_tickers

def analyze_technicals(tickers):
    print(f"3. Analyzing Technicals (Segregated Lists) for {len(tickers)} stocks...")
    candidates = []
    if not tickers: return []
    try:
        data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False, threads=True)
        for sym in tickers:
            try:
                if len(tickers) > 1:
                    if sym not in data.columns.levels[0]: continue
                    df = data[sym].copy()
                else: df = data.copy()
                if len(df) < 205: continue 
                df = df.dropna()
                
                df['rsi'] = TechnicalMath.get_rsi(df['Close'])
                df['adx'] = TechnicalMath.get_adx(df['High'], df['Low'], df['Close'])
                df['sma200'] = TechnicalMath.get_sma(df['Close'], 200)
                
                current = df.iloc[-1]
                rsi = float(current['rsi'])
                adx = float(current['adx'])
                sma200 = float(current['sma200'])
                price = float(current['Close'])
                if pd.isna(rsi) or pd.isna(adx) or pd.isna(sma200): continue

                # --- EXCLUSIVE LIST LOGIC ---
                # We use if/elif to ensure a ticker only ends up in ONE list.

                # 1. TREND TARGETS (Momentum)
                # Buying Strength: Price > SMA200, Strong Trend (ADX > 25), RSI > 55 (But not insane)
                if price > sma200 and adx > 25 and 55 < rsi < 75:
                    candidates.append({"symbol": sym, "type": "trend_targets", "score": adx})
                
                # 2. SURVIVOR TARGETS (Dip Buyers)
                # Buying Weakness in Uptrend: Price > SMA200, RSI < 40 (Oversold)
                elif price > sma200 and rsi < 40:
                    candidates.append({"symbol": sym, "type": "survivor_targets", "score": (50-rsi)})

                # 3. WHEEL TARGETS (Stable/Neutral Bull)
                # Price > SMA200, RSI Healthy (40-55), lower ADX preferred for range
                elif price > sma200 and 40 <= rsi <= 55:
                    candidates.append({"symbol": sym, "type": "wheel_targets", "score": (50-rsi)})
                
                # 4. CONDOR TARGETS (Chop/Sideways)
                # No Trend (ADX < 20), RSI Middle
                elif adx < 20 and 40 < rsi < 60:
                    candidates.append({"symbol": sym, "type": "condor_targets", "score": (20-adx)})
                
                # 5. SHORT TARGETS (Bearish)
                # Price < SMA200 (Downtrend)
                elif price < sma200:
                    # Score based on how "bad" it is (Lower RSI = crash mode, Higher RSI = rip sell)
                    candidates.append({"symbol": sym, "type": "short_targets", "score": adx})

            except: continue
    except Exception as e:
        print(f"   [!] Tech Analysis Error: {e}")
    return candidates


# Known ETF suffixes and patterns to skip
ETF_KEYWORDS = ["SPY", "QQQ", "IWM", "ETF"]  # Specific known ones
ETF_PATTERNS = ["SCH", "VT", "IQ", "XL", "GLD", "SLV"]  # Common ETF prefixes

def is_etf(symbol):
    """
    Quick heuristic to detect ETFs before calling Yahoo.
    ETFs are typically 3-4 chars, no numbers, often end in specific patterns.
    """
    # Length check - most ETFs are 3-4 chars
    if len(symbol) <= 4 and symbol.isalpha():
        # Check Alpaca asset info (you already have this client!)
        # For dragnet speed, we just use heuristics + yahoo 404 suppression
        for p in ETF_PATTERNS:
            if symbol.startswith(p): return True
        return False
    return False

def get_earnings_date(ticker_symbol):
    """
    Checks if earnings are within the next 2 days.
    Returns: True if SAFE (No earnings soon), False if DANGER.
    Skips ETFs gracefully.
    """
    # Quick ETF Check
    if is_etf(ticker_symbol):
        return True

    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Suppress 404s - ETFs and some stocks simply don't have calendars
        import contextlib
        import io
        
        with contextlib.redirect_stderr(io.StringIO()):
            calendar = ticker.calendar
            
        # If calendar is None or empty, assume safe (ETF or no data)
        if calendar is None:
            return True
            
        # Handle different calendar formats yfinance returns
        if hasattr(calendar, 'empty') and calendar.empty:
            return True
            
        if isinstance(calendar, dict) and not calendar:
            return True

        # --- Earning Date Extraction ---
        if hasattr(calendar, 'iloc'):
            next_earnings = calendar.iloc[0][0]
        elif isinstance(calendar, dict):
            earnings_dates = calendar.get('Earnings Date', [])
            if not earnings_dates:
                return True
            next_earnings = earnings_dates[0]
        else:
            return True

        if isinstance(next_earnings, (datetime.date, datetime.datetime)):
            if isinstance(next_earnings, pd.Timestamp):
                next_earnings = next_earnings.to_pydatetime()
            
            if next_earnings.tzinfo:
                next_earnings = next_earnings.replace(tzinfo=None)

            now = datetime.datetime.now()
            days_until = (next_earnings - now).days
            
            if 0 <= days_until <= 2:
                print(f"   [!] {ticker_symbol} has earnings in {days_until} days. Skipping.")
                return False
                
    except Exception:
        # Silently pass - 404s, missing data, ETFs all land here
        # Default to SAFE so we don't block valid tickers
        pass
        
    return True

def run_dragnet():
    # --- GATEKEEPER CHECK ---
    if not is_mission_time():
        sys.exit(0) # Exit cleanly
    # ------------------------

    client = get_alpaca_client()
    if not client: return
    all_tickers = get_market_universe(client)
    liquid_tickers = filter_by_volume(all_tickers)
    results = analyze_technicals(liquid_tickers)
    
    final_output = {
        "trend_targets": [], 
        "survivor_targets": [],
        "wheel_targets": [], 
        "condor_targets": [],
        "short_targets": []
    }
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # [FILTER] Earnings Guard
    safe_results = []
    print("\n4. Checking Earnings Calendar (Safety Guard)...")
    for item in results:
        if get_earnings_date(item['symbol']):
            safe_results.append(item)
    
    for item in safe_results:
        category = item['type']
        if len(final_output[category]) < 10: 
            # [MODIFIED] Store Object instead of String
            # Old: final_output[category].append(item['symbol'])
            # New:
            final_output[category].append({
                "symbol": item['symbol'],
                "tech_score": round(item['score'], 2)
            })
            
    print("\n--- 🎯 DRAGNET RESULTS ---")
    print(json.dumps(final_output, indent=4))
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=4)
        print(f"   ✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_dragnet()
