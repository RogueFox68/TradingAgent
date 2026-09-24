"""Daily-bar portfolio strategies over the point-in-time top-N universe.

Timing, identical for every strategy:
  signal from data up to the CLOSE of day s -> enter at the OPEN of s+1 ->
  hold for H sessions -> leave at the open of s+H+1.
Returns are open-to-open, so a position never earns the close-to-open gap
of the day its signal was computed on.

Holding periods longer than the formation interval overlap, the standard
Jegadeesh-Titman construction: a cohort is formed every `step` sessions and
held for `hold` sessions, so hold/step cohorts are live at once, each with
weight step/hold. Costs are charged on turnover of the combined book
(sum of |weight change|), so a name that stays in consecutive cohorts is not
charged for leaving and re-entering.

Gross exposure is 1.0 for every strategy. A long-short book is half long,
half short, so its returns are directly comparable with a long-only one.
Short borrow cost is NOT modelled.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    open: pd.DataFrame        # dates x symbols, adjusted
    close: pd.DataFrame       # dates x symbols, adjusted
    members: dict             # date -> [symbols], point-in-time top-N (known at that open)
    spy_close: pd.Series

    @property
    def dates(self):
        return self.close.index


def build_panel(daily, members, spy="SPY"):
    """Wide adjusted open/close for every symbol that is ever a member, plus SPY."""
    syms = sorted({s for v in members.values() for s in v} | {spy})
    opens, closes = {}, {}
    for s in syms:
        df = daily.get(s)
        if df is None or df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
        d = pd.Index(idx.date)
        o = pd.Series(df["open"].to_numpy(dtype=float), index=d)
        c = pd.Series(df["close"].to_numpy(dtype=float), index=d)
        opens[s] = o[~o.index.duplicated()]
        closes[s] = c[~c.index.duplicated()]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    return Panel(o, c, members, c[spy] if spy in c else pd.Series(dtype=float))


def open_to_open(panel):
    """r[t] = open[t+1] / open[t] - 1. A missing next open (halt, delisting)
    counts as 0: the position is treated as flat, not as a total loss or a
    windfall - neither is knowable from bars alone."""
    o = panel.open
    r = o.shift(-1) / o - 1
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_cohorts(panel, formation_weights, hold, step, cost_bps):
    """Daily net returns of the overlapping-cohort book.

    formation_weights: DataFrame (dates x symbols), non-zero only on
    formation days, each row summing to gross 1.0 (or all zero = cash)."""
    W = formation_weights.reindex(index=panel.dates, columns=panel.open.columns).fillna(0.0)
    # position on day t = cohorts formed on s in [t-hold, t-1]
    P = W.rolling(hold, min_periods=1).sum().shift(1).fillna(0.0) * (step / hold)
    r = open_to_open(panel)[P.columns]
    gross = (P * r).sum(axis=1)
    turnover = P.diff().abs().sum(axis=1)
    turnover.iloc[0] = P.iloc[0].abs().sum()
    return gross - turnover * cost_bps / 1e4


def _members_at(panel, d):
    return [s for s in panel.members.get(d, []) if s in panel.close.columns]


def reversal_weights(panel, lookback, k, long_short, step=1):
    """Short-term reversal: buy the k members with the WORST `lookback`-day
    return; with long_short, also short the k best."""
    c = panel.close
    past = c.shift(lookback)
    ret = c / past - 1
    W = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for i, d in enumerate(c.index):
        if i % step:
            continue
        m = _members_at(panel, d)
        x = ret.loc[d, m].dropna() if m else pd.Series(dtype=float)
        if len(x) < (2 * k if long_short else k):
            continue
        x = x.sort_values()
        leg = 0.5 if long_short else 1.0
        W.loc[d, x.index[:k]] = leg / k
        if long_short:
            W.loc[d, x.index[-k:]] -= leg / k
    return W


def momentum_weights(panel, lookback, skip, k, spy_filter, step):
    """Cross-sectional momentum: buy the k members with the best return from
    `lookback` to `skip` sessions ago (skipping the most recent month, which
    reverses). With spy_filter, hold cash while SPY is below its 200-day
    average."""
    c = panel.close
    ret = c.shift(skip) / c.shift(lookback) - 1
    spy_ok = pd.Series(True, index=c.index)
    if spy_filter and len(panel.spy_close):
        s = panel.spy_close.reindex(c.index)
        spy_ok = s > s.rolling(200, min_periods=200).mean()
    W = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for i, d in enumerate(c.index):
        if i % step or not bool(spy_ok.loc[d]):
            continue
        m = _members_at(panel, d)
        x = ret.loc[d, m].dropna() if m else pd.Series(dtype=float)
        if len(x) < k:
            continue
        W.loc[d, x.sort_values().index[-k:]] = 1.0 / k
    return W


def equal_weight_weights(panel):
    """Benchmark: every current member, equally weighted, rebalanced daily."""
    c = panel.close
    W = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for d in c.index:
        m = [s for s in _members_at(panel, d) if not np.isnan(c.loc[d, s])]
        if m:
            W.loc[d, m] = 1.0 / len(m)
    return W


def spy_buy_and_hold(panel, cost_bps):
    W = pd.DataFrame(0.0, index=panel.dates, columns=panel.open.columns)
    if "SPY" in W:
        W["SPY"] = 1.0
    return run_cohorts(panel, W, hold=1, step=1, cost_bps=cost_bps)
