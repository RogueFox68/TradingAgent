"""Market data for the backtest: Alpaca bars, cached on disk.

Same feed and request shape as the bots: 15m bars with no `feed` argument
(the account default - IEX on the free plan - is what trend_bot and
survivor_bot compute on) and no session filter, so extended-hours bars feed
the indicators exactly as they do live.

Never `limit` alongside `start`: Alpaca returns the OLDEST bars in the window
and truncates, which is how three of the fleet's four fetch sites traded on
weeks-old data (trading-bot-fleet CLAUDE.md, "Market Data Correctness").
Requests here page the whole window.
"""
import os
import pickle
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .schedule import ET

_clients = {}

# A real US equity ticker: 1-5 capital letters. Alpaca's INACTIVE asset list
# is ~20k entries of mostly non-tradable placeholders - CUSIPs (0029900E0),
# contingent value rights (003CVR016), escrows (097ESC016) - and the bars
# endpoint rejects a whole request over any one of them.
_TICKER = re.compile(r"^[A-Z]{1,5}$")
_INVALID = re.compile(r"invalid symbol: ([^\s\"',]+)")


class InvalidSymbol(Exception):
    def __init__(self, symbol):
        super().__init__(f"invalid symbol: {symbol}")
        self.symbol = symbol


def _config():
    import config  # TradingAgent's gitignored config.py (API_KEY, SECRET_KEY, PAPER)
    return config


def data_client():
    if "data" not in _clients:
        from alpaca.data.historical import StockHistoricalDataClient
        c = _config()
        _clients["data"] = StockHistoricalDataClient(c.API_KEY, c.SECRET_KEY)
    return _clients["data"]


def trading_client():
    if "trading" not in _clients:
        from alpaca.trading.client import TradingClient
        c = _config()
        _clients["trading"] = TradingClient(c.API_KEY, c.SECRET_KEY, paper=getattr(c, "PAPER", True))
    return _clients["trading"]


def _retry(fn, what, tries=4):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # network / 429 / 5xx: back off, then give up loudly
            m = _INVALID.search(str(e))
            if m:
                # a permanent answer about one symbol, not a transient failure:
                # retrying the same request can only fail the same way
                raise InvalidSymbol(m.group(1)) from e
            if k == tries - 1:
                raise RuntimeError(f"{what} failed after {tries} attempts: {e}") from e
            wait = 2 ** (k + 1)
            print(f"   [data] {what}: {type(e).__name__}: {e} - retrying in {wait}s")
            time.sleep(wait)


def sessions(start, end):
    """[(open, close)] tz-aware ET Timestamps for each trading day in
    [start, end], from Alpaca's calendar - so holidays and early closes are
    the exchange's, not a guess."""
    from alpaca.trading.requests import GetCalendarRequest
    cal = _retry(lambda: trading_client().get_calendar(GetCalendarRequest(start=start, end=end)),
                 "calendar")
    out = []
    for d in cal:
        o, c = pd.Timestamp(d.open), pd.Timestamp(d.close)
        o = o.tz_localize(ET) if o.tzinfo is None else o.tz_convert(ET)
        c = c.tz_localize(ET) if c.tzinfo is None else c.tz_convert(ET)
        out.append((o, c))
    return out


def _cache_path(cache_dir, kind, sym, start, end):
    safe = sym.replace("/", "_")
    return Path(cache_dir) / kind / f"{safe}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"


def _like(t, tz):
    """t as a Timestamp comparable with an index in timezone `tz`."""
    t = pd.Timestamp(t)
    if tz is None:
        return t.tz_localize(None) if t.tzinfo is not None else t
    return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)


def _cache_index(cache_dir, kind):
    """{symbol: [(start, end, path)]} for every cached file of `kind`, from ONE
    directory listing.

    The first version globbed the directory once per symbol. With ~15,000
    symbols and ~15,000 cached files that is ~225 million directory entries
    read, silently - on the Corsair it looked like a hang after the
    "Universe and daily bars" line."""
    d = Path(cache_dir) / kind
    idx = {}
    if not d.is_dir():
        return idx
    with os.scandir(d) as it:
        for e in it:
            if not e.name.endswith(".pkl"):
                continue
            parts = e.name[:-4].rsplit("_", 2)
            if len(parts) == 3:
                idx.setdefault(parts[0], []).append((parts[1], parts[2], Path(e.path)))
    return idx


def _covering_cache(index, sym, start, end):
    """A cached frame for `sym` whose date range contains [start, end], if
    any - so a run over a sub-window (the wheel backtest over the research
    run's download) reuses bars instead of fetching them again. The frame is
    sliced to the requested window; ends are compared as dates, because the
    cache name carries dates."""
    s0, e0 = f"{start:%Y%m%d}", f"{end:%Y%m%d}"
    for s1, e1, p in index.get(sym.replace("/", "_"), ()):
        if s1 <= s0 and e1 >= e0:
            with open(p, "rb") as f:
                df = pickle.load(f)
            if df.empty:
                return df
            idx = pd.DatetimeIndex(df.index)
            return df[(idx >= _like(start, idx.tz)) & (idx <= _like(end, idx.tz))]
    return None


def _fetch(symbols, start, end, timeframe, kind, cache_dir, chunk, adjustment=None):
    """`adjustment`: None = the API default (raw), which is what the bots
    trade on and what the 3-month ablation uses. Multi-year research must
    pass "split" or "all": on raw bars a 10:1 split is a -90% bar, which fires
    every stop and books a fortune for every short."""
    from alpaca.data.requests import StockBarsRequest
    if adjustment:
        kind = f"{kind}_{adjustment}"
    out, missing = {}, []
    index = _cache_index(cache_dir, kind)
    print(f"   [data] {kind}: checking the cache for {len(symbols)} symbols "
          f"({sum(len(v) for v in index.values())} cached files)...")
    for n, s in enumerate(symbols, 1):
        p = _cache_path(cache_dir, kind, s, start, end)
        cov = None
        if p.exists():
            with open(p, "rb") as f:
                cov = pickle.load(f)
        else:
            cov = _covering_cache(index, s, start, end)
        if cov is not None:
            out[s] = cov
        else:
            missing.append(s)
        if n % 2000 == 0:
            print(f"   [data] {kind}: {n}/{len(symbols)} checked, {len(out)} from cache")
    if missing:
        print(f"   [data] fetching {kind} bars for {len(missing)} symbols "
              f"({len(out)} cached)...")
    for k in range(0, len(missing), chunk):
        part = missing[k:k + chunk]
        kw = {}
        if adjustment:
            from alpaca.data.enums import Adjustment
            kw["adjustment"] = Adjustment(adjustment)
        df = pd.DataFrame()
        todo = list(part)
        while todo:
            req = StockBarsRequest(symbol_or_symbols=todo, timeframe=timeframe,
                                   start=start, end=end, **kw)
            try:
                resp = _retry(lambda: data_client().get_stock_bars(req), f"{kind} bars {todo[0]}..")
            except InvalidSymbol as e:
                if e.symbol not in todo:
                    raise RuntimeError(f"Alpaca rejected {e.symbol!r}, which was not requested") from e
                # drop it and re-ask for the rest; it is cached below as
                # having no bars, so a rerun does not ask again
                print(f"   [data] skipping {e.symbol}: Alpaca says it is not a valid symbol")
                todo.remove(e.symbol)
                continue
            df = resp.df if resp.data else pd.DataFrame()
            break
        for s in part:
            if not df.empty and s in df.index.get_level_values(0):
                sdf = df.xs(s)[["open", "high", "low", "close", "volume"]].sort_index()
            else:
                sdf = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            out[s] = sdf
            p = _cache_path(cache_dir, kind, s, start, end)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump(sdf, f)
        print(f"   [data] {kind}: {min(k + chunk, len(missing))}/{len(missing)}")
    return out


def bars_15m(symbols, start, end, cache_dir, adjustment=None):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return _fetch(sorted(set(symbols)), start, end, TimeFrame(15, TimeFrameUnit.Minute),
                  "15m", cache_dir, chunk=25, adjustment=adjustment)


def bars_daily(symbols, start, end, cache_dir, adjustment=None):
    from alpaca.data.timeframe import TimeFrame
    return _fetch(sorted(set(symbols)), start, end, TimeFrame.Day, "1d", cache_dir,
                  chunk=200, adjustment=adjustment)


def equity_universe(include_inactive=False):
    """Every active, tradable US equity symbol Alpaca lists today.

    Survivorship caveat: a name delisted during the window is absent. For a
    top-100-by-dollar-volume list over three months that is a handful of names
    at most, and it biases arm A the same way it biased the live scanner,
    which also started from get_all_assets(ACTIVE).

    `include_inactive=True` adds delisted names, for multi-year research,
    where the bias is no longer small: a universe built from today's
    survivors has already removed every stock that went to zero, which
    flatters any strategy that buys losers. Alpaca still serves bars for many
    delisted symbols, not all, so this reduces the bias rather than removing
    it."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    assets = list(_retry(lambda: trading_client().get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)), "assets"))
    syms = {a.symbol for a in assets if a.tradable}
    if include_inactive:
        inactive = _retry(lambda: trading_client().get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.INACTIVE)),
            "inactive assets")
        syms |= {a.symbol for a in inactive}
    return sorted(s for s in syms if _TICKER.match(s))


def topn_lists(daily, session_dates, n=100, lookback=20, min_price=5.0, min_rows=15):
    """{date: [symbols]} - the point-in-time most active N names for each
    session, ranked by mean DOLLAR volume over the `lookback` sessions BEFORE
    that date. Nothing from the ranked day itself is used.

    Dollar volume, not share volume: share volume ranks a $3 stock above a
    $300 one for the same money traded. And the rank is scale-invariant, so
    IEX's partial volume ranks the same names consolidated volume would
    (market_scanner makes the same argument for its Alpaca fallback)."""
    dv, px = {}, {}
    for s, df in daily.items():
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
        d = pd.Series(df["close"].to_numpy() * df["volume"].to_numpy(), index=idx.date)
        dv[s] = d[~d.index.duplicated()]
        c = pd.Series(df["close"].to_numpy(), index=idx.date)
        px[s] = c[~c.index.duplicated()]
    if not dv:
        return {}
    dvw = pd.DataFrame(dv).sort_index()
    pxw = pd.DataFrame(px).sort_index()
    mean_dv = dvw.rolling(lookback, min_periods=min_rows).mean()
    out = {}
    for day in session_dates:
        prior = mean_dv.index[mean_dv.index < day]
        if len(prior) == 0:
            continue
        last = prior[-1]
        row = mean_dv.loc[last]
        ok = pxw.loc[last] >= min_price
        out[day] = list(row[ok].dropna().nlargest(n).index)
    return out
