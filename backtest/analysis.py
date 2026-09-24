"""Metrics, the random-selection null, and the LLM signal tests.

Three kinds of evidence, in increasing statistical power:

  1. Arm metrics (fleet simulations). Realistic but few trades per arm, so
     every comparison carries a block-bootstrap interval on the paired daily
     P&L difference.
  2. The random null. For "the LLM only helped because it traded less", the
     LLM arm is placed inside the distribution of arms that approve the same
     NUMBER of candidates per run at random. Inside it: selection added
     nothing that approving fewer names would not.
  3. Candidate-level forward returns. Every candidate the scout analysed -
     approved or rejected - with the return that followed. This uses all
     ~6,000 decisions instead of the few hundred trades the bots made, and
     does not depend on the simulator at all.
"""
import math
import random
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from .scout_log import EQUITY_BUCKETS
from .schedule import ET

DIRECTION = {"trend_targets": 1, "survivor_targets": 1, "short_targets": -1}


# --- 1. arm metrics -----------------------------------------------------------

def daily_pnl(result):
    eq = result.daily_equity.sort_index()
    prev = pd.concat([pd.Series([result.start_equity]), eq.iloc[:-1]]).to_numpy()
    return pd.Series(eq.to_numpy() - prev, index=eq.index)


def arm_metrics(result, spy_daily_ret=None):
    tr = result.trades
    closed = [t for t in tr if t.reason != "open_at_end"]
    pnl = sum(t.pnl for t in tr)
    cap_days = sum(t.capital_days for t in tr)
    eq = result.daily_equity.sort_index()
    dp = daily_pnl(result)
    dret = dp / pd.concat([pd.Series([result.start_equity]), eq.iloc[:-1]]).to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[result.start_equity], eq.to_numpy()]))
    dd = (np.concatenate([[result.start_equity], eq.to_numpy()]) - peak) / peak
    rets = [t.ret for t in closed]
    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [-t.pnl for t in closed if t.pnl < 0]
    m = dict(
        arm=result.arm,
        trades=len(tr),
        trades_per_day=len(tr) / max(len(eq), 1),
        total_pnl=pnl,
        return_pct=100 * pnl / result.start_equity,
        win_rate=100 * len(wins) / len(closed) if closed else float("nan"),
        avg_trade_ret_pct=100 * statistics.mean(rets) if rets else float("nan"),
        profit_factor=(sum(wins) / sum(losses)) if losses else float("nan"),
        max_drawdown_pct=100 * float(dd.min()) if len(dd) else 0.0,
        sharpe=float(dret.mean() / dret.std() * math.sqrt(252)) if len(dret) > 2 and dret.std() > 0 else float("nan"),
        capital_days=cap_days,
        # Window P&L over the capital the arm kept deployed ON AVERAGE: the
        # "did it trade better, or just less?" number. An arm that sits in
        # cash is scored on what it did deploy, not on the calendar. Not
        # annualised - annualising per dollar-day turns a 1% two-hour trade
        # into thousands of percent.
        return_on_deployed_pct=100 * pnl / (cap_days / _calendar_days(eq)) if cap_days else float("nan"),
        avg_exposure_pct=100 * cap_days / (result.start_equity * _calendar_days(eq)) if len(eq) else 0.0,
        worst5pct_trade_ret_pct=100 * float(np.mean(sorted(rets)[: max(1, len(rets) // 20)])) if rets else float("nan"),
        overnight_holds=sum(1 for t in tr if t.exit_time.astimezone(ET).date() != t.entry_time.astimezone(ET).date()),
    )
    if spy_daily_ret is not None and len(dret) > 5:
        j = pd.concat([dret.rename("a"), spy_daily_ret.rename("m")], axis=1, join="inner").dropna()
        if len(j) > 5 and j["m"].var() > 0:
            beta = float(np.cov(j["a"], j["m"])[0, 1] / j["m"].var())
            m["beta_vs_spy"] = beta
            m["alpha_ann_pct"] = 100 * 252 * float(j["a"].mean() - beta * j["m"].mean())
    return m


def _calendar_days(eq):
    if len(eq) < 2:
        return 1.0
    return max((pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days + 1, 1)


def by_bot(result):
    days = _calendar_days(result.daily_equity.sort_index())
    out = {}
    for bot in ("trend_bot", "survivor_bot"):
        tr = [t for t in result.trades if t.bot == bot]
        closed = [t for t in tr if t.reason != "open_at_end"]
        cap = sum(t.capital_days for t in tr)
        out[bot] = dict(trades=len(tr), pnl=sum(t.pnl for t in tr),
                        win_rate=100 * sum(t.pnl > 0 for t in closed) / len(closed) if closed else float("nan"),
                        return_on_deployed_pct=100 * sum(t.pnl for t in tr) / (cap / days) if cap else float("nan"))
    return out


def block_bootstrap_diff(a, b, block=5, draws=5000, seed=0):
    """Mean paired daily P&L difference (a - b) with a 95% moving-block
    bootstrap interval. Blocks keep the autocorrelation that overnight holds
    put into daily P&L."""
    j = pd.concat([daily_pnl(a).rename("a"), daily_pnl(b).rename("b")], axis=1, join="inner").fillna(0.0)
    d = (j["a"] - j["b"]).to_numpy()
    n = len(d)
    if n < block * 2:
        return dict(mean=float(d.mean()) if n else float("nan"), lo=float("nan"), hi=float("nan"),
                    days=n, total=float(d.sum()))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(math.ceil(n / block))
    means = np.empty(draws)
    for i in range(draws):
        s = rng.choice(starts, size=k)
        means[i] = np.concatenate([d[x:x + block] for x in s])[:n].mean()
    return dict(mean=float(d.mean()), lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)), days=n,
                total=float(d.sum()))


def null_percentile(value, null_values):
    """Share of random-selection draws the value beats (0-100)."""
    v = [x for x in null_values if not (isinstance(x, float) and math.isnan(x))]
    if not v or value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return 100.0 * sum(x < value for x in v) / len(v)


# --- decision statistics (no market data needed) ----------------------------

def decision_stats(runs, start, end, threshold=0.66):
    """What the LLM gate did, from the log alone."""
    rows = []
    for r in runs:
        if not (start <= r.started.date() <= end):
            continue
        for c in r.candidates:
            if c.bucket not in EQUITY_BUCKETS + ("wheel_targets",):
                continue
            no_news = (all(x is None for x in (c.t1, c.t2, c.t3, c.social))
                       and not c.failed)
            rows.append(dict(
                month=r.started.strftime("%Y-%m"), bucket=c.bucket, approved=c.approved,
                tech=c.tech, llm=c.llm_score(), no_news=no_news,
                needed_llm=(threshold - 0.3 * c.tech) / 0.7, ai_error=c.ai_error))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    per_bucket = df.groupby("bucket").agg(
        candidates=("approved", "size"), approval_pct=("approved", lambda s: 100 * s.mean()),
        mean_tech=("tech", "mean"), mean_llm=("llm", "mean"),
        llm_needed=("needed_llm", "mean"),
        no_news_pct=("no_news", lambda s: 100 * s.mean()))
    per_month = df.pivot_table(index="month", columns="bucket", values="approved",
                               aggfunc=lambda s: round(100 * s.mean(), 1))
    # How much of the gate is the LLM's opinion vs arithmetic? A candidate
    # whose tech score leaves it needing an LLM score above what the LLM gives
    # the AVERAGE candidate is rejected by the weighting, not by a judgment.
    typical_llm = float(df["llm"].median())
    df["needs_above_typical"] = df["needed_llm"] > typical_llm
    mech = df.groupby("bucket").apply(
        lambda g: pd.Series(dict(
            needs_above_typical_pct=100 * g["needs_above_typical"].mean(),
            approval_if_needs_above_pct=100 * g.loc[g["needs_above_typical"], "approved"].mean()
            if g["needs_above_typical"].any() else float("nan"),
            approval_if_not_pct=100 * g.loc[~g["needs_above_typical"], "approved"].mean()
            if (~g["needs_above_typical"]).any() else float("nan"))),
        include_groups=False)
    no_news_approved = int(df.loc[df["no_news"], "approved"].sum())
    return dict(per_bucket=per_bucket, per_month=per_month, mechanical=mech,
                typical_llm=typical_llm, no_news_total=int(df["no_news"].sum()),
                no_news_approved=no_news_approved, n=len(df),
                ai_error_pct=100 * float(df["ai_error"].mean()),
                outages=llm_outage_runs(runs, start, end),
                news_outages=news_outage_runs(runs, start, end))


def llm_outage_runs(runs, start, end, share=0.5):
    """Runs where most LLM calls FAILED. sector_scout_3.ask_llama scores an
    exception as 0.0 ("AI Failed"), not as missing, so an LM Studio outage
    rejects every candidate - and the run still publishes a status-"success"
    file with empty buckets, which the fleet reads as a deliberate stand-by.
    These runs are the LLM being down, not the LLM judging."""
    out = []
    for r in runs:
        if not (start <= r.started.date() <= end) or not r.candidates:
            continue
        err = sum(c.ai_error for c in r.candidates) / len(r.candidates)
        if err >= share:
            out.append(dict(run_start=r.started.strftime("%Y-%m-%d %H:%M"),
                            candidates=len(r.candidates),
                            ai_error_pct=round(100 * err, 1),
                            approved=sum(c.approved for c in r.candidates),
                            published=r.published))
    return out


def news_outage_runs(runs, start, end, share=0.5):
    """Runs where MORE than `share` of the candidates had no news in any tier.
    sector_scout_3.publish_abort_reason refuses to publish these
    (NO_NEWS_ABORT_SHARE, also 0.5 and also "more than"). For a run from
    before that guard, a listing here is what it would have stopped, and a
    run that looks normal is a false positive of the threshold."""
    out = []
    for r in runs:
        if not (start <= r.started.date() <= end) or not r.candidates:
            continue
        missing = sum(not c.has_news() for c in r.candidates) / len(r.candidates)
        if missing > share:
            out.append(dict(run_start=r.started.strftime("%Y-%m-%d %H:%M"),
                            candidates=len(r.candidates),
                            no_news_pct=round(100 * missing, 1),
                            approved=sum(c.approved for c in r.candidates),
                            published=r.published))
    return out


def latency_stats(runs, start, end):
    lat = [r.llm_latency_s / 60 for r in runs
           if r.published and r.llm_latency_s and start <= r.started.date() <= end]
    if not lat:
        return {}
    s = sorted(lat)
    return dict(runs=len(s), median_min=statistics.median(s), p10_min=s[len(s) // 10],
                p90_min=s[(9 * len(s)) // 10], max_min=s[-1])


def dead_time(schedule, session_list):
    """Share of regular-session 15m steps in which the fleet's targets file
    was empty or stale - the hours the bots could not enter anything at all."""
    total = empty = 0
    for o, c in session_list:
        t = o
        while t < c:
            total += 1
            tg = schedule.at(t.to_pydatetime())
            if not any(tg.get(b) for b in EQUITY_BUCKETS):
                empty += 1
            t += pd.Timedelta(minutes=15)
    return dict(steps=total, empty_steps=empty, empty_pct=100 * empty / total if total else float("nan"))


# --- 3. candidate forward returns / information coefficient -----------------

def _first_bar_at_or_after(a_t, t_ns):
    return int(np.searchsorted(a_t, t_ns, side="left"))


def candidate_forward_returns(runs, market, session_list, start, end, horizons=(1, 3, 5)):
    """One row per (symbol, bucket, session day) - its FIRST appearance that
    day, so a name the scout re-analyses three times a day counts once.

    Entry: open of the first regular-session bar at/after publication.
    Exit:  last regular-session close `h` sessions later.
    `lat_ret`: the move between the scanner finishing and publication - what
    the LLM's runtime cost (or saved) before anyone could act.
    Returns are signed by bucket direction (a short candidate that falls
    scores positive). Candidates whose LLM call failed (`ai_error`) carry a
    0.0 that is an exception, not a verdict; they are kept, flagged, and
    excluded by the signal tests."""
    closes_by_day = [c for _, c in session_list]
    opens_by_day = [o for o, _ in session_list]
    seen = set()
    rows = []
    for ri, r in enumerate(runs):
        if not r.published or not (start <= r.scout_done.date() <= end):
            continue
        for c in r.candidates:
            if c.bucket not in DIRECTION or not market.has(c.symbol):
                continue
            pub = pd.Timestamp(r.scout_done).tz_convert(ET)
            key = (c.symbol, c.bucket, pub.date())
            if key in seen:
                continue
            seen.add(key)
            a = market.sym[c.symbol]
            # first regular-session bar at/after publication
            di = next((i for i, (o, cl) in enumerate(session_list) if cl > pub), None)
            if di is None:
                continue
            entry_t = max(pub, opens_by_day[di])
            i = _first_bar_at_or_after(a["t"], entry_t.value)
            if i >= len(a["t"]) or a["t"][i] >= closes_by_day[di].value:
                continue
            entry = a["open"][i]
            sign = DIRECTION[c.bucket]
            row = dict(run=ri, date=pub.date(), symbol=c.symbol, bucket=c.bucket,
                       approved=c.approved, tech=c.tech, llm=c.llm_score(),
                       confidence=c.confidence, ai_error=c.ai_error)
            for h in horizons:
                dj = di + h
                if dj >= len(closes_by_day):
                    row[f"fwd{h}"] = np.nan
                    continue
                j = int(np.searchsorted(a["t"], closes_by_day[dj].value, side="left")) - 1
                row[f"fwd{h}"] = sign * (a["close"][j] / entry - 1) if j > i else np.nan
            if r.scanner_done is not None:
                sd = pd.Timestamp(r.scanner_done).tz_convert(ET)
                # newest bar COMPLETED when the scanner finished
                k = int(np.searchsorted(a["t"], sd.value - pd.Timedelta(minutes=15).value,
                                        side="right")) - 1
                # only meaningful when the scanner finished inside the same session
                if k >= 0 and sd >= opens_by_day[di] and a["t"][k] >= opens_by_day[di].value:
                    row["lat_ret"] = sign * (entry / a["close"][k] - 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap_mean_diff(df, col, group_col="approved", cluster="date", draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[col])
    days = d[cluster].unique()
    if len(days) < 5:
        return (float("nan"), float("nan"))
    by = {k: g for k, g in d.groupby(cluster)}
    out = []
    for _ in range(draws):
        pick = rng.choice(days, size=len(days))
        s = pd.concat([by[x] for x in pick])
        a, b = s.loc[s[group_col], col], s.loc[~s[group_col], col]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def _judged(fwd):
    return fwd[~fwd["ai_error"]] if "ai_error" in fwd else fwd


def approved_vs_rejected(fwd, horizons=(1, 3, 5)):
    fwd = _judged(fwd)
    rows = []
    for b in list(DIRECTION) + ["ALL"]:
        d = fwd if b == "ALL" else fwd[fwd["bucket"] == b]
        for h in horizons:
            col = f"fwd{h}"
            dd = d.dropna(subset=[col])
            if dd.empty:
                continue
            a, r = dd.loc[dd["approved"], col], dd.loc[~dd["approved"], col]
            lo, hi = _cluster_bootstrap_mean_diff(dd, col)
            rows.append(dict(bucket=b, horizon=h, n_approved=len(a), n_rejected=len(r),
                             approved_mean_pct=100 * a.mean() if len(a) else np.nan,
                             rejected_mean_pct=100 * r.mean() if len(r) else np.nan,
                             diff_pct=100 * (a.mean() - r.mean()) if len(a) and len(r) else np.nan,
                             diff_lo_pct=100 * lo, diff_hi_pct=100 * hi))
    return pd.DataFrame(rows)


def rank_ic(fwd, horizons=(1, 3, 5), min_names=5):
    """Fama-MacBeth rank IC per (run, bucket) cross-section.

    `llm_resid_ic` is the one that answers the question: the rank correlation
    between forward return and the part of the LLM score the tech score does
    not explain (LLM rank residualised on tech rank within the cross-section).
    Mean over cross-sections, t = mean / (sd / sqrt(n))."""
    fwd = _judged(fwd)
    rows = []
    for h in horizons:
        col = f"fwd{h}"
        acc = defaultdict(list)
        for (_, b), g in fwd.dropna(subset=[col]).groupby(["run", "bucket"]):
            if len(g) < min_names:
                continue
            rf = g[col].rank()
            rt, rl = g["tech"].rank(), g["llm"].rank()
            if rt.nunique() > 1:
                beta = np.polyfit(rt, rl, 1)
                resid = rl - np.polyval(beta, rt)
            else:
                resid = rl - rl.mean()
            for name, x in (("tech_ic", rt), ("llm_ic", rl), ("llm_resid_ic", pd.Series(resid, index=g.index)),
                            ("confidence_ic", g["confidence"].rank())):
                if x.nunique() > 1 and rf.nunique() > 1:
                    acc[(b, name)].append(float(np.corrcoef(x, rf)[0, 1]))
                    acc[("ALL", name)].append(acc[(b, name)][-1])
        for (b, name), v in acc.items():
            n = len(v)
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=1)) if n > 1 else float("nan")
            rows.append(dict(bucket=b, horizon=h, measure=name, cross_sections=n, mean_ic=mu,
                             t_stat=mu / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")))
    return pd.DataFrame(rows).sort_values(["bucket", "horizon", "measure"]) if rows else pd.DataFrame()


def latency_drift(fwd):
    if "lat_ret" not in fwd:
        return pd.DataFrame()
    rows = []
    for b in list(DIRECTION) + ["ALL"]:
        d = fwd if b == "ALL" else fwd[fwd["bucket"] == b]
        d = d.dropna(subset=["lat_ret"])
        for label, sel in (("approved", d["approved"]), ("all", pd.Series(True, index=d.index))):
            x = d.loc[sel, "lat_ret"]
            if len(x) > 1:
                rows.append(dict(bucket=b, set=label, n=len(x), mean_move_pct=100 * x.mean(),
                                 se_pct=100 * x.std(ddof=1) / math.sqrt(len(x))))
    return pd.DataFrame(rows)


def shadow_vote_returns(fwd, votes_path):
    """Forward return by shadow-specialist decision (approve/watch/reject),
    matched to the scout run whose `updated` stamp the snapshot carries."""
    import json
    rows = []
    with open(votes_path, encoding="utf-8") as f:
        for line in f:
            try:
                snap = json.loads(line)
            except ValueError:
                continue
            day = pd.Timestamp(snap.get("updated")).tz_convert(ET).date() if snap.get("updated") else None
            for v in snap.get("votes", []):
                if v.get("advisor_failed"):
                    continue
                rows.append(dict(date=day, symbol=v.get("symbol"), bucket=v.get("strategy_bucket"),
                                 decision=v.get("decision")))
    if not rows:
        return pd.DataFrame()
    votes = pd.DataFrame(rows).drop_duplicates(["date", "symbol", "bucket"])
    j = fwd.merge(votes, on=["date", "symbol", "bucket"], how="inner")
    out = []
    for dec, g in j.groupby("decision"):
        for h in (1, 3, 5):
            x = g[f"fwd{h}"].dropna()
            if len(x):
                out.append(dict(decision=dec, horizon=h, n=len(x), mean_pct=100 * x.mean()))
    return pd.DataFrame(out)
