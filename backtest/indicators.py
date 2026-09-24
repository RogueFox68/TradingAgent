"""EMA / RSI / ADX computed the way the fleet's `ta` library computes them.

The bots use `ta` (EMAIndicator, RSIIndicator, ADXIndicator). `ta` is
sdist-only and does not build on every toolchain, so it is not a dependency
here; these are vectorised equivalents, and test_backtest.IndicatorParityTest
checks them against `ta` wherever `ta` imports.

EMA and RSI are exact: `ta` is itself a pandas `ewm` call. ADX is Wilder's
recursion, which `ta` runs as a Python loop with a sum-of-first-N seed; this
uses `ewm(alpha=1/N)` with pandas' own seed. The two differ only by the seed,
whose weight decays as (13/14)^n - under 1e-4 after ~130 bars. The bots
compute on the newest 200-500 bars and never read the first ones, and the
simulator computes over the whole cached series, so the seed never reaches a
decision. Measured against ta 0.11.0 on a 500-bar series: EMA and RSI
identical, ADX within 2e-4 over the last 300 bars.
"""
import numpy as np
import pandas as pd


def ema(close, window):
    return close.ewm(span=window, min_periods=window, adjust=False).mean()


def rsi(close, window=14):
    diff = close.diff(1)
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)
    emaup = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    emadn = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = emaup / emadn
    out = pd.Series(np.where(emadn == 0, 100.0, 100 - (100 / (1 + rs))), index=close.index)
    return out.where(emaup.notna())


def adx(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - \
         pd.concat([low, prev_close], axis=1).min(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    pos = up.where((up > down) & (up > 0), 0.0)
    neg = down.where((down > up) & (down > 0), 0.0)
    a = 1 / window
    trs = tr.ewm(alpha=a, adjust=False).mean()
    dip = 100 * pos.ewm(alpha=a, adjust=False).mean() / trs.replace(0, np.nan)
    din = 100 * neg.ewm(alpha=a, adjust=False).mean() / trs.replace(0, np.nan)
    dx = (100 * (dip - din).abs() / (dip + din).replace(0, np.nan)).fillna(0.0)
    out = dx.ewm(alpha=a, adjust=False).mean()
    out.iloc[: 2 * window] = np.nan
    return out


def add_trend_indicators(df, fast, slow, adx_window=14):
    df = df.copy()
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    df["adx"] = adx(df["high"], df["low"], df["close"], adx_window)
    return df


def add_rsi(df, window):
    df = df.copy()
    df["rsi"] = rsi(df["close"], window)
    return df
