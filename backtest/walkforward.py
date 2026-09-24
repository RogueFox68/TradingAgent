"""Walk-forward selection, performance statistics, and the pass bar.

The rule that keeps a parameter search honest: settings are chosen using
ONLY data that precedes the period they are scored on. For each fold
(a calendar year for daily strategies, a quarter for the 15m bots), every
trial in a family is ranked by Sharpe over all EARLIER folds; the winner's
returns in that fold, and only those, go into the family's out-of-sample
(OOS) record. The OOS record is the result. The full-period table of every
trial is printed only for transparency, next to the Deflated Sharpe Ratio,
which shows how much of the best full-period number is selection luck.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_N = NormalDist()
EULER_GAMMA = 0.5772156649

# Pre-registered pass bar. Decided before any result was seen; changing it
# after a run is exactly the kind of search this tool exists to prevent.
PASS_BAR = dict(
    min_oos_sharpe=0.5,        # annualised, net of costs
    min_psr=0.95,              # P(true OOS Sharpe > 0)
    min_positive_folds=2 / 3,  # share of OOS folds with a positive return
    beat_spy_sharpe=True,      # OOS Sharpe >= SPY buy-and-hold over the same days
    min_oos_folds=4,           # a single good year or quarter is not a record
)


def fold_labels(index, freq):
    idx = pd.DatetimeIndex(pd.to_datetime(list(index)))
    if freq == "Y":
        return pd.Index([str(d.year) for d in idx], name="fold")
    if freq == "Q":
        return pd.Index([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in idx], name="fold")
    raise ValueError(freq)


def sharpe(r):
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd or np.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def psr(r, sr_star_ann=0.0):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): the probability
    that the true Sharpe exceeds sr_star, given the sample's length, skew and
    fat tails. Computed on per-period Sharpe."""
    r = pd.Series(r).dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 10 or not sd or np.isnan(sd):
        return float("nan")
    sr = r.mean() / sd
    star = sr_star_ann / math.sqrt(TRADING_DAYS)
    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0          # pandas reports EXCESS kurtosis
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - star) * math.sqrt(n - 1) / math.sqrt(denom))


def deflated_sharpe(best, trial_sharpes_ann):
    """PSR of the best trial against the Sharpe that the BEST of N
    zero-skill trials would be expected to reach by luck. Below 0.95: the
    best full-period result is not distinguishable from selection noise."""
    s = [x for x in trial_sharpes_ann if not np.isnan(x)]
    n = len(s)
    if n < 2:
        return float("nan"), float("nan")
    v = np.var(np.array(s) / math.sqrt(TRADING_DAYS), ddof=1)
    z1 = _N.inv_cdf(1 - 1 / n)
    z2 = _N.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    sr0_ann = sr0 * math.sqrt(TRADING_DAYS)
    return psr(best, sr0_ann), sr0_ann


def max_drawdown(r):
    eq = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def ann_return(r):
    r = pd.Series(r).dropna()
    if not len(r):
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def summarize(r, freq):
    r = pd.Series(r).dropna()
    folds = r.groupby(fold_labels(r.index, freq)).apply(lambda x: float((1 + x).prod() - 1))
    return dict(days=len(r), ann_return_pct=100 * ann_return(r),
                ann_vol_pct=100 * float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan"),
                sharpe=sharpe(r), psr=psr(r), max_dd_pct=100 * max_drawdown(r),
                positive_folds=float((folds > 0).mean()) if len(folds) else float("nan"),
                fold_returns={k: 100 * v for k, v in folds.items()})


def walk_forward(trials, freq, min_train_folds=1):
    """trials: {name: daily return Series}. Returns (oos Series, choices).

    choices: [(fold, chosen trial, its training Sharpe)]. The first
    `min_train_folds` folds are training-only and never scored."""
    df = pd.DataFrame(trials).sort_index()
    labels = fold_labels(df.index, freq)
    folds = list(dict.fromkeys(labels))
    pieces, choices = [], []
    for j, f in enumerate(folds):
        if j < min_train_folds:
            continue
        train = df[labels.isin(folds[:j])]
        test = df[labels == f]
        scores = {c: sharpe(train[c]) for c in df.columns}
        ranked = [c for c in df.columns if not np.isnan(scores[c])]
        if not ranked:
            continue
        best = max(ranked, key=lambda c: scores[c])   # first wins ties: grid order
        pieces.append(test[best])
        choices.append((f, best, scores[best]))
    oos = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return oos, choices


def verdict(stats, spy_sharpe, bar=PASS_BAR):
    """[(criterion, passed, detail)] for one family's OOS record."""
    n_folds = len(stats.get("fold_returns", {}))
    out = [
        ("at least %d OOS folds" % bar["min_oos_folds"], n_folds >= bar["min_oos_folds"], str(n_folds)),
        ("OOS Sharpe >= %.2f" % bar["min_oos_sharpe"],
         stats["sharpe"] >= bar["min_oos_sharpe"], "%.2f" % stats["sharpe"]),
        ("P(Sharpe > 0) >= %.2f" % bar["min_psr"],
         (stats["psr"] or 0) >= bar["min_psr"], "%.3f" % stats["psr"]),
        ("positive in >= %.0f%% of OOS folds" % (100 * bar["min_positive_folds"]),
         stats["positive_folds"] >= bar["min_positive_folds"] - 1e-9,
         "%.0f%%" % (100 * stats["positive_folds"])),
    ]
    if bar.get("beat_spy_sharpe"):
        out.append(("Sharpe >= SPY buy-and-hold over the same days",
                    stats["sharpe"] >= spy_sharpe, "%.2f vs %.2f" % (stats["sharpe"], spy_sharpe)))
    return [(c, bool(p) and not (isinstance(p, float) and np.isnan(p)), d) for c, p, d in out]
