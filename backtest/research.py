"""Strategy research: can ANY pre-declared rule set clear an out-of-sample bar?

    python -m backtest.research
    python -m backtest.research --start 2019-01-01 --intraday-start 2024-07-01

Run from the TradingAgent directory on the Corsair (config.py supplies the
Alpaca keys). See backtest/README.md, "Strategy research".

THE HYPOTHESIS LIST BELOW IS THE PRE-REGISTRATION. It was written before
any result existed, and the git history dates it. Adding a family or
widening a grid after seeing results is allowed only as a NEW, dated
entry, and every trial ever run counts toward the Deflated Sharpe - a
search that quietly grows until something passes is what this file exists
to prevent.
"""
import argparse
import itertools
import math
import sys
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from . import daily as D
from . import rules, schedule
from . import walkforward as W
from .run_backtest import _fmt, _table
from .schedule import ET

# --- pre-registered hypotheses (2026-09-24) --------------------------------

HYPOTHESES = {
    "reversal_daily": dict(
        kind="daily", freq="Y",
        why="The scout's momentum candidates REVERSED over Jun-Sep 2026 (trend picks fell, "
            "short picks rose). Short-term reversal over days is a long-documented effect. "
            "Buy the top-N's recent losers; optionally short the recent winners.",
        grid=dict(lookback=[3, 5, 10], hold=[3, 10], k=[10], long_short=[False, True]),
    ),
    "momentum_daily": dict(
        kind="daily", freq="Y",
        why="Cross-sectional momentum over 6-12 months, skipping the last month, is the "
            "most replicated anomaly in equities and works on weeks-to-months horizons, "
            "not on 15-minute bars. Optional SPY 200-day filter for crash risk.",
        grid=dict(lookback=[126, 252], skip=[21], k=[10, 20], rebalance=[21],
                  spy_filter=[False, True]),
    ),
    "trend_15m": dict(
        kind="intraday", freq="Q",
        why="trend_bot's own entries on the top-N universe with wider/narrower exits, and "
            "INVERTED (fade its crossovers), since its picks reversed.",
        grid=dict(trend_stop_tp=[(-0.05, 0.08), (-0.03, 0.06), (-0.08, 0.15)],
                  trend_invert=[False, True]),
    ),
    "survivor_15m": dict(
        kind="intraday", freq="Q",
        why="survivor_bot's RSI-dip entries on the top-N universe with alternative exits.",
        grid=dict(surv_stop_tp=[(-0.03, 0.05), (-0.05, 0.10), (-0.02, 0.03)]),
    ),
}


def grid_points(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def trial_name(family, p):
    def f(v):
        if isinstance(v, tuple):
            return "/".join(f"{x:+.0%}" for x in v)
        if isinstance(v, bool):
            return "y" if v else "n"
        return str(v)
    return family + "[" + ",".join(f"{k}={f(v)}" for k, v in p.items()) + "]"


# --- family runners -----------------------------------------------------------

def run_daily_family(name, panel, cost_bps):
    out = {}
    for p in grid_points(HYPOTHESES[name]["grid"]):
        if name == "reversal_daily":
            Wt = D.reversal_weights(panel, p["lookback"], p["k"], p["long_short"])
            r = D.run_cohorts(panel, Wt, hold=p["hold"], step=1, cost_bps=cost_bps)
        else:
            Wt = D.momentum_weights(panel, p["lookback"], p["skip"], p["k"],
                                    p["spy_filter"], step=p["rebalance"])
            r = D.run_cohorts(panel, Wt, hold=p["rebalance"], step=p["rebalance"], cost_bps=cost_bps)
        out[trial_name(name, p)] = r
        print(f"   {trial_name(name, p)}: Sharpe {W.sharpe(r):.2f}")
    return out


def run_intraday_family(name, market, sessions, sched, equity, slippage_bps, seed):
    from . import sim
    out = {}
    for p in grid_points(HYPOTHESES[name]["grid"]):
        v = {}
        if "trend_stop_tp" in p:
            v.update(bots=("trend_bot",), trend_stop=p["trend_stop_tp"][0],
                     trend_tp=p["trend_stop_tp"][1], trend_invert=p["trend_invert"])
        if "surv_stop_tp" in p:
            v.update(bots=("survivor_bot",), surv_stop=p["surv_stop_tp"][0],
                     surv_tp=p["surv_stop_tp"][1])
        res = sim.Simulator(market, sessions, equity, slippage_bps, seed, variant=v).run(sched)
        eq = res.daily_equity.sort_index()
        r = eq / pd.concat([pd.Series([res.start_equity]), eq.iloc[:-1]]).to_numpy() - 1
        r.index = pd.DatetimeIndex(pd.to_datetime(list(r.index)))
        out[trial_name(name, p)] = r
        print(f"   {trial_name(name, p)}: {len(res.trades)} trades, Sharpe {W.sharpe(r):.2f}")
    return out


# --- report -------------------------------------------------------------------

def _bench_over(bench, idx):
    return bench.reindex(idx).dropna()


def family_section(name, trials, spy_r, ew_r, min_train_folds):
    h = HYPOTHESES[name]
    freq = h["freq"]
    oos, choices = W.walk_forward(trials, freq, min_train_folds)
    idle = [n for n, r in trials.items() if not (pd.Series(r).fillna(0) != 0).any()]
    lines = [f"### {name}\n", f"_{h['why']}_\n",
             f"{len(trials)} trials. Folds: {'calendar years' if freq == 'Y' else 'quarters'}; "
             f"the first {min_train_folds} fold(s) train only.\n"]
    if idle:
        # A trial that never traded has no Sharpe and can never be chosen. It
        # usually means the universe is smaller than the portfolio (k > top-N)
        # or the data is missing, which should be loud, not a quiet "no record".
        lines.append(f"**Warning: {len(idle)} of {len(trials)} trials never traded** "
                     f"(e.g. `{idle[0]}`). Check --top-n against the portfolio size k, and the data.\n")
    if oos.empty:
        return lines + ["_Not enough folds for an out-of-sample record._\n"], None
    st = W.summarize(oos, freq)
    spy = _bench_over(spy_r, oos.index)
    ew = _bench_over(ew_r, oos.index) if ew_r is not None else None
    spy_st = W.summarize(spy, freq) if len(spy) else None
    rows = [dict(series=f"{name} (walk-forward OOS)", **{k: v for k, v in st.items() if k != "fold_returns"})]
    if spy_st:
        rows.append(dict(series="SPY buy-and-hold, same days",
                         **{k: v for k, v in spy_st.items() if k != "fold_returns"}))
    if ew is not None and len(ew):
        rows.append(dict(series="top-N equal weight, same days",
                         **{k: v for k, v in W.summarize(ew, freq).items() if k != "fold_returns"}))
    lines += ["**Out-of-sample record (the result):**\n",
              _table(rows, ["series", "days", "ann_return_pct", "ann_vol_pct", "sharpe", "psr",
                            "max_dd_pct", "positive_folds"]), ""]
    fr = [dict(fold=f, chosen=c, train_sharpe=s,
               oos_return_pct=st["fold_returns"].get(f),
               spy_return_pct=(spy_st["fold_returns"].get(f) if spy_st else None))
          for f, c, s in choices]
    lines += ["**Fold by fold (what walk-forward picked, and how it did next):**\n",
              _table(fr, ["fold", "chosen", "train_sharpe", "oos_return_pct", "spy_return_pct"]), ""]
    v = W.verdict(st, spy_st["sharpe"] if spy_st else float("nan"))
    passed = all(p for _, p, _ in v)
    lines += ["**Pass bar:**\n",
              _table([dict(criterion=c, result="PASS" if p else "fail", value=d) for c, p, d in v],
                     ["criterion", "result", "value"]),
              f"\n**{'PASSES - candidate for paper trading beside the live bots' if passed else 'Does not pass.'}**\n"]
    return lines, dict(family=name, passed=passed, oos=oos, stats=st)


def transparency_section(all_trials):
    names = list(all_trials)
    sharpes = {n: W.sharpe(all_trials[n]) for n in names}
    best = max((n for n in names if not np.isnan(sharpes[n])), key=lambda n: sharpes[n], default=None)
    rows = [dict(trial=n, full_period_sharpe=sharpes[n],
                 ann_return_pct=100 * W.ann_return(all_trials[n]),
                 max_dd_pct=100 * W.max_drawdown(all_trials[n])) for n in names]
    rows.sort(key=lambda r: -(r["full_period_sharpe"] if not np.isnan(r["full_period_sharpe"]) else -99))
    lines = ["## Every trial, full period (transparency only - NOT the result)\n",
             f"{len(names)} trials in total. The best full-period Sharpe is what a naive search "
             "would report. The Deflated Sharpe Ratio asks whether it beats the best of "
             f"{len(names)} zero-skill trials; below 0.95, it does not.\n"]
    if best:
        dsr, sr0 = W.deflated_sharpe(all_trials[best], list(sharpes.values()))
        lines.append(f"- Best: `{best}`, Sharpe {_fmt(sharpes[best])}. Luck benchmark for "
                     f"{len(names)} trials: Sharpe {_fmt(sr0)}. **Deflated Sharpe: {_fmt(dsr, 3)}**\n")
        lines.append("  (Trials of different lengths are pooled here; intraday trials cover a "
                     "shorter window, so this is indicative.)\n")
    lines.append(_table(rows, ["trial", "full_period_sharpe", "ann_return_pct", "max_dd_pct"]))
    return lines


# --- main -----------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1),
                   help="first day of the daily-strategy record (data is fetched ~15 months earlier)")
    p.add_argument("--end", type=date.fromisoformat,
                   default=date.today() - timedelta(days=1))
    p.add_argument("--intraday-start", type=date.fromisoformat, default=date(2024, 7, 1),
                   help="first day of the 15m-bot record (15m bars are heavy; default ~2 years)")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--cost-bps", type=float, default=5.0, help="per side, daily strategies")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="per side, 15m bots")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--min-train-folds", type=int, default=1)
    p.add_argument("--families", nargs="*", default=list(HYPOTHESES), choices=list(HYPOTHESES))
    p.add_argument("--no-delisted", action="store_true",
                   help="rank only today's active assets (faster, survivorship-biased)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fleet-repo", default=str(Path(__file__).resolve().parents[2] / "trading-bot-fleet"))
    p.add_argument("--allow-unverified-rules", action="store_true")
    p.add_argument("--cache-dir", default="backtest_cache")
    p.add_argument("--out", default="backtest_research_out")
    return p.parse_args(argv)


def main(argv=None):
    from . import data
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fams = args.families
    intraday = [f for f in fams if HYPOTHESES[f]["kind"] == "intraday"]
    daily_f = [f for f in fams if HYPOTHESES[f]["kind"] == "daily"]

    problems = rules.verify_against_fleet(args.fleet_repo) if intraday else []
    if problems and not args.allow_unverified_rules:
        for p in problems:
            print("   -", p)
        sys.exit("15m families replay the fleet's rules, which could not be verified. Point "
                 "--fleet-repo at a trading-bot-fleet checkout, or pass --allow-unverified-rules.")

    data_start = min(args.start, args.intraday_start) - timedelta(days=460)
    t0 = datetime.combine(data_start, dtime(0), tzinfo=ET)
    t1 = min(datetime.combine(args.end, dtime(23, 59), tzinfo=ET), datetime.now(ET) - timedelta(minutes=20))

    print("Universe (active + delisted)..." if not args.no_delisted else "Universe (active only)...")
    universe = data.equity_universe(include_inactive=not args.no_delisted)
    daily_bars = data.bars_daily(universe + ["SPY"], t0, t1, args.cache_dir, adjustment="all")
    spy = daily_bars.get("SPY")
    if spy is None or spy.empty:
        sys.exit("No SPY daily bars - cannot build the calendar or the benchmark.")
    cal = sorted(pd.DatetimeIndex(spy.index).tz_convert(ET).date)
    print(f"Ranking point-in-time top-{args.top_n} over {len(cal)} sessions...")
    members = data.topn_lists(daily_bars, cal, n=args.top_n)
    panel = D.build_panel(daily_bars, members)
    del daily_bars

    spy_r = D.spy_buy_and_hold(panel, args.cost_bps)
    spy_r.index = pd.DatetimeIndex(pd.to_datetime(list(spy_r.index)))
    ew_r = D.run_cohorts(panel, D.equal_weight_weights(panel), 1, 1, args.cost_bps)
    ew_r.index = spy_r.index

    all_trials, sections, results = {}, [], []
    start_ts = pd.Timestamp(args.start)
    for f in daily_f:
        print(f"Family {f}...")
        tr = run_daily_family(f, panel, args.cost_bps)
        tr = {k: _trim(v, start_ts) for k, v in tr.items()}
        all_trials.update(tr)
        sec, res = family_section(f, tr, spy_r, _trim(ew_r, start_ts), args.min_train_folds)
        sections += sec
        results.append(res)

    if intraday:
        from . import sim
        sess = data.sessions(args.intraday_start, args.end)
        idates = [o.date() for o, _ in sess]
        sched = schedule.topn_schedule({d: members[d] for d in idates if d in members},
                                       flat_conf=0.68, name=f"top{args.top_n}")
        syms = sorted(sched.symbols())
        print(f"15m bars for {len(syms)} symbols ({args.intraday_start} .. {args.end})...")
        f0 = datetime.combine(args.intraday_start - timedelta(days=21), dtime(0), tzinfo=ET)
        bars = data.bars_15m(syms, f0, t1, args.cache_dir, adjustment="split")
        market = sim.Market(bars)
        del bars
        for f in intraday:
            print(f"Family {f}...")
            tr = run_intraday_family(f, market, sess, sched, args.equity, args.slippage_bps, args.seed)
            all_trials.update(tr)
            sec, res = family_section(f, tr, spy_r, None, args.min_train_folds)
            sections += sec
            results.append(res)

    passed = [r["family"] for r in results if r and r["passed"]]
    report = [
        "# Strategy research: walk-forward, out-of-sample\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Daily strategies {args.start} .. {args.end}; "
        f"15m bots {args.intraday_start} .. {args.end}. Universe: point-in-time top-{args.top_n} "
        f"by trailing 20-day dollar volume ({'active + delisted' if not args.no_delisted else 'active only'}). "
        f"Costs: {args.cost_bps} bps/side (daily), {args.slippage_bps} bps/side slippage (15m).\n",
        "## Verdict\n",
        (f"**Passed the pre-registered bar:** {', '.join(passed)}. Next step is paper trading beside "
         "the live bots, not live money." if passed else
         "**No family passed the pre-registered bar.** On this evidence, none of these rule sets "
         "has an edge worth trading after costs."),
        "",
        "Pass bar, fixed before any run: " + "; ".join(
            f"{k}={v}" for k, v in W.PASS_BAR.items()) + ".\n",
        f"Rules for the 15m families: {'verified against ' + args.fleet_repo if not problems else 'NOT VERIFIED'}.\n",
        "## Families\n", *sections, *transparency_section(all_trials),
        "\n## Limitations\n",
        "- Daily strategies trade at the next open with a flat cost per side; no market impact, "
        "no short borrow cost, and a missing next open counts as a flat day.",
        "- Delisted names are included where Alpaca still serves their bars, not everywhere.",
        "- 15m families: regime/VIX gating and CFO reallocation are not simulated; bars are "
        "split-adjusted, which the live bots' bars are not.",
        "- Walk-forward removes the selection bias of choosing settings, not the bias of having "
        "chosen these four families. That is why the list is pre-registered and dated.",
    ]
    (out / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    pd.DataFrame(all_trials).to_csv(out / "trial_daily_returns.csv")
    oos = {r["family"]: r["oos"] for r in results if r}
    if oos:
        pd.DataFrame(oos).to_csv(out / "oos_daily_returns.csv")
    print(f"Wrote {out / 'research_report.md'}")
    return 0


def _trim(r, start_ts):
    r = r.copy()
    r.index = pd.DatetimeIndex(pd.to_datetime(list(r.index)))
    return r[r.index >= start_ts]


if __name__ == "__main__":
    sys.exit(main())
