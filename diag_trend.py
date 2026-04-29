import json
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from market_scanner import TechnicalMath

# Download Benchmark (SPY) for Relative Strength calculations
spy_data = yf.download("SPY", period="2y", interval="1d", progress=False)['Close']
data = yf.download(["NVDA", "AAPL", "MSFT", "MSTR", "META", "TSLA"], period="2y", interval="1d", group_by='ticker', progress=False)

for sym in ["NVDA", "AAPL", "MSFT", "MSTR", "META", "TSLA"]:
    print(f"\nEvaluating {sym}")
    try:
        df = data[sym].copy()
        if len(df) < 205: continue 
        df = df.dropna()
        
        # Core Math
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
            
        print(f"type(spy_data): {type(spy_data)}")
        if isinstance(spy_data, pd.DataFrame):
            spy_series = spy_data.squeeze()
        else:
            spy_series = spy_data
            
        rsi = TechnicalMath.get_rsi(close)
        adx = TechnicalMath.get_adx(df['High'], df['Low'], close)
        sma200 = TechnicalMath.get_sma(close, 200)
        ema20 = TechnicalMath.get_ema(close, 20)
        ema50 = TechnicalMath.get_ema(close, 50)
        
        curr_price = float(close.iloc[-1])
        curr_rsi = float(rsi.iloc[-1])
        curr_adx = float(adx.iloc[-1])
        
        print(f"Price: {curr_price:.2f}, EMA20: {ema20.iloc[-1]:.2f}, EMA50: {ema50.iloc[-1]:.2f}, SMA200: {sma200.iloc[-1]:.2f}")
        print(f"RSI: {curr_rsi:.2f}, ADX: {curr_adx:.2f}")

        # Relative Strength Check
        is_outperforming = True
        if spy_series is not None:
            # Align SPY data length with Ticker data length
            spy_sliced = spy_series.tail(len(close))
            is_outperforming = TechnicalMath.get_relative_strength(close, spy_sliced)
            
        print(f"Is outperforming: {is_outperforming}")

        if (curr_price > ema20.iloc[-1] > ema50.iloc[-1] > sma200.iloc[-1] and 
            is_outperforming and 
            curr_adx > 25 and 
            55 < curr_rsi < 75):
            print(f"--> {sym} is a TREND TARGET!")
        else:
            print("--> Failed conditions:")
            if not (curr_price > ema20.iloc[-1] > ema50.iloc[-1] > sma200.iloc[-1]):
                print("  - Price/EMA/SMA order incorrect")
            if not is_outperforming:
                print("  - Not outperforming")
            if not (curr_adx > 25):
                print(f"  - ADX <= 25 ({curr_adx:.2f})")
            if not (55 < curr_rsi < 75):
                print(f"  - RSI not between 55 and 75 ({curr_rsi:.2f})")
    except Exception as e:
        print(f"Error: {e}")
