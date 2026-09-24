"""Option chains, option prices and VIX history for the wheel backtest.

What Alpaca can and cannot tell us, which shapes everything downstream:

  * Option history starts in February 2024. Nothing earlier exists.
  * There are NO historical option quotes - only bars built from trades. A
    contract that did not trade on a day has no price that day. The wheel
    backtest therefore prices every fill off that day's traded VWAP (with a
    spread haircut), and treats "no trade" the way wheel_bot treats a
    spread too wide to quote: the order does not happen.
  * Expired contracts are listed only under status=inactive, so every chain
    is the union of the active and inactive listings.

Everything is cached under the cache dir, so a rerun makes no requests.
"""
import io
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import data
from .schedule import ET

CHAIN_COLUMNS = ["symbol", "type", "strike", "expiry"]
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]


def _opt_client():
    if "options" not in data._clients:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        c = data._config()
        data._clients["options"] = OptionHistoricalDataClient(c.API_KEY, c.SECRET_KEY)
    return data._clients["options"]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def chain(underlying, exp_from, exp_to, strike_lo, strike_hi, cache_dir):
    """Every put and call on `underlying` expiring in [exp_from, exp_to] with
    a strike in [strike_lo, strike_hi], expired contracts included.

    Contracts whose root is not the underlying (adjusted deliverables after a
    corporate action, e.g. AAPL1) are dropped: their 100-share assumption
    does not hold."""
    p = Path(cache_dir) / "chains" / (
        f"{underlying}_{exp_from:%Y%m%d}_{exp_to:%Y%m%d}_{strike_lo:.2f}_{strike_hi:.2f}.pkl")
    if p.exists():
        return _load(p)
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetOptionContractsRequest
    rows = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        token = None
        while True:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying], status=status,
                expiration_date_gte=exp_from, expiration_date_lte=exp_to,
                strike_price_gte=f"{strike_lo:.2f}", strike_price_lte=f"{strike_hi:.2f}",
                limit=10000, page_token=token)
            resp = data._retry(lambda: data.trading_client().get_option_contracts(req),
                               f"option chain {underlying}")
            for c in resp.option_contracts or []:
                if (c.root_symbol or underlying) != underlying:
                    continue
                if c.size is not None and str(c.size) not in ("100", "100.0"):
                    continue
                t = getattr(c.type, "value", c.type)
                rows.append((c.symbol, "P" if str(t).lower() == "put" else "C",
                             float(c.strike_price), pd.Timestamp(c.expiration_date).date()))
            token = resp.next_page_token
            if not token:
                break
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS).drop_duplicates("symbol")
    _save(p, df)
    return df


def option_bars(symbols, start, end, cache_dir):
    """{symbol: daily bars indexed by ET date}. An empty frame means the
    contract never traded in the window - a real answer, cached as such."""
    out, missing = {}, []
    for s in symbols:
        p = Path(cache_dir) / "optbars" / f"{s}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
        if p.exists():
            out[s] = _load(p)
        else:
            missing.append(s)
    if not missing:
        return out
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    for k in range(0, len(missing), 100):
        part = missing[k:k + 100]
        req = OptionBarsRequest(symbol_or_symbols=part, timeframe=TimeFrame.Day,
                                start=start, end=end)
        resp = data._retry(lambda: _opt_client().get_option_bars(req), f"option bars {part[0]}..")
        df = resp.df if resp.data else pd.DataFrame()
        for s in part:
            if not df.empty and s in df.index.get_level_values(0):
                sdf = df.xs(s)
                idx = pd.DatetimeIndex(sdf.index)
                idx = idx.tz_convert(ET) if idx.tz is not None else idx
                sdf = sdf.reindex(columns=BAR_COLUMNS)
                sdf.index = pd.Index(idx.date)
                sdf = sdf[~sdf.index.duplicated(keep="last")].sort_index()
            else:
                sdf = pd.DataFrame(columns=BAR_COLUMNS)
            out[s] = sdf
            _save(Path(cache_dir) / "optbars" / f"{s}_{start:%Y%m%d}_{end:%Y%m%d}.pkl", sdf)
    return out


CBOE_VIX_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def parse_vix_csv(text):
    """CBOE's VIX_History.csv: DATE,OPEN,HIGH,LOW,CLOSE with MM/DD/YYYY dates."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(df["CLOSE"].astype(float).to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y").dt.date)
    return s[~s.index.duplicated(keep="last")].sort_index()


def vix_history(cache_dir, csv_path=None):
    """Daily VIX closes. The fleet's own primary VIX source is CBOE, and so
    is this: CBOE publishes the full daily history as a CSV. A local copy
    can be passed with csv_path when the Corsair cannot reach the CDN.

    There is no fallback to a proxy (VIXY, realised volatility): the wheel's
    gates are VIX-level thresholds, and a mis-scaled input would move them
    silently (trading-bot-fleet CLAUDE.md, "Data sources & resilience")."""
    if csv_path:
        return parse_vix_csv(Path(csv_path).read_text(encoding="utf-8"))
    p = Path(cache_dir) / "vix" / f"VIX_History_{date.today():%Y%m%d}.csv"
    if not p.exists():
        import requests
        r = data._retry(lambda: requests.get(CBOE_VIX_CSV, timeout=30), "CBOE VIX history")
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(r.text, encoding="utf-8")
    return parse_vix_csv(p.read_text(encoding="utf-8"))
