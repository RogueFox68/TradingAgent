"""LLM-gate ablation backtest - entry point.

    python -m backtest.run_backtest --scout-log scout_log.txt
    python -m backtest.run_backtest --scout-log scout_log.txt --decisions-only

Run from the TradingAgent directory on the Corsair (it reads config.py for the
Alpaca keys). --decisions-only needs no keys and no network: it reports what
the LLM gate did from the log alone. See backtest/README.md.
"""
import argparse
import csv
import math
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

import pandas as pd

from . import analysis, rules, schedule, scout_log
from .schedule import ET

# NYSE full-day closures, used ONLY by --decisions-only (no API access). The
# full run takes the calendar from Alpaca.
_NYSE_HOLIDAYS_2026 = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
                       date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
                       date(2026, 11, 26), date(2026, 12, 25)}


def approx_sessions(start, end):
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in _NYSE_HOLIDAYS_2026:
            out.append((pd.Timestamp(datetime.combine(d, dtime(9, 30)), tz=ET),
                        pd.Timestamp(datetime.combine(d, dtime(16, 0)), tz=ET)))
        d += timedelta(days=1)
    return out


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _df_table(df, floatfmt=2):
    if df is None or len(df) == 0:
        return "_(no data)_"
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    return _table(df.to_dict("records"), list(df.columns))


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scout-log", required=True, help="scout_log.txt from the Corsair")
    p.add_argument("--shadow-votes", help="shadow_advisor_votes.jsonl (optional)")
    p.add_argument("--start", type=date.fromisoformat,
                   help="window start (default: 3 months before --end)")
    p.add_argument("--end", type=date.fromisoformat,
                   help="window end (default: the day before the log's last run)")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--null-draws", type=int, default=500,
                   help="random-selection draws for arm D (each is a full simulation)")
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fleet-repo", default=str(Path(__file__).resolve().parents[2] / "trading-bot-fleet"),
                   help="trading-bot-fleet checkout, to verify the pinned strategy rules")
    p.add_argument("--allow-unverified-rules", action="store_true",
                   help="run even if the fleet source is missing or disagrees (stated in the report)")
    p.add_argument("--cache-dir", default="backtest_cache")
    p.add_argument("--out", default="backtest_out")
    p.add_argument("--decisions-only", action="store_true",
                   help="log-only analysis: no market data, no API keys")
    p.add_argument("--log-tz", default=scout_log.LOG_TZ)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = scout_log.load(args.scout_log, args.log_tz)
    if not runs:
        sys.exit(f"No runs parsed from {args.scout_log}")
    end = args.end or (runs[-1].started.date() - timedelta(days=1))
    start = args.start or (pd.Timestamp(end) - pd.DateOffset(months=3) + timedelta(days=1)).date()
    print(f"Window {start} .. {end} | {len(runs)} runs in log "
          f"({runs[0].started.date()} .. {runs[-1].started.date()})")

    # --- rules verification -------------------------------------------------
    problems = rules.verify_against_fleet(args.fleet_repo)
    if problems:
        print("Strategy rules NOT verified against the fleet source:")
        for p in problems:
            print("   -", p)
        if not args.allow_unverified_rules and not args.decisions_only:
            sys.exit("Refusing to run on unverified rules. Point --fleet-repo at a current "
                     "trading-bot-fleet checkout, or pass --allow-unverified-rules.")
    rules_status = "verified against " + args.fleet_repo if not problems else \
        "NOT VERIFIED:\n" + "\n".join(f"  - {p}" for p in problems)

    lo = datetime.combine(start, dtime(0), tzinfo=ET)
    hi = datetime.combine(end, dtime(23, 59), tzinfo=ET)
    window_runs = schedule.published_runs(runs, lo, hi)
    flat = schedule.flat_confidence([r for r in window_runs if lo <= r.scout_done <= hi])

    if args.decisions_only:
        sess = approx_sessions(start, end)
    else:
        from . import data
        sess = data.sessions(start, end)

    llm = schedule.llm_schedule(window_runs, name="C_llm")
    b_pub = schedule.all_candidates_schedule(window_runs, flat, "publish", name="B_all_publish_time")
    b_scan = schedule.all_candidates_schedule(window_runs, flat, "scanner", name="B_all_scanner_time")

    report = [f"# LLM gate ablation backtest\n",
              f"Window **{start} .. {end}** ({len(sess)} sessions). "
              f"Generated {datetime.now():%Y-%m-%d %H:%M}.\n",
              f"- Strategy rules: {rules_status}",
              f"- Flat confidence for no-LLM arms: **{flat}** (median approved confidence in window)",
              f"- Starting equity ${args.equity:,.0f}, slippage {args.slippage_bps} bps/side, "
              f"regime SIDEWAYS / VIX 18 held constant\n"]

    # --- the gate itself (log only) -----------------------------------------
    ds = analysis.decision_stats(runs, start, end)
    lat = analysis.latency_stats(runs, start, end)
    report += ["## 1. What the LLM gate did (from the log)\n",
               f"{ds.get('n', 0)} candidate decisions. Median LLM score across all candidates: "
               f"{_fmt(ds.get('typical_llm'))}.\n",
               "### Per bucket\n", _df_table(ds.get("per_bucket")),
               "\n`llm_needed` is the LLM score (0-1) a candidate's tech score leaves it needing to "
               "clear 0.66. `no_news_pct`: candidates with no news or social data at all. Every "
               "missing source is scored 0.5, which caps the composite at 0.3*tech + 0.35 (0.65 "
               f"even with perfect technicals), so those are always rejected: "
               f"{ds.get('no_news_approved', 0)} of {ds.get('no_news_total', 0)} were approved.\n",
               "### Approval % by month\n", _df_table(ds.get("per_month")),
               "\n### Weighting vs judgment\n",
               "Candidates whose tech score leaves them needing an LLM score above the typical one, "
               "and approval rates on each side of that line. Where the line decides nearly everything, "
               "the weighting is rejecting candidates, not the LLM's reading of the news.\n",
               _df_table(ds.get("mechanical")),
               "\n### LLM outages\n",
               f"{_fmt(ds.get('ai_error_pct'), 1)}% of candidates had a failed LLM call. "
               "`ask_llama` used to score a failed call 0.0, so a run where LM Studio was down "
               "rejected everything and still published empty buckets, which the fleet treats as a "
               "deliberate stand-by. These runs are in every LLM arm (C, C_flat, and D, which matches "
               "the LLM's count of zero), because they happened. The scout now scores a failed call "
               "as missing and refuses to publish a run where more than half the calls failed. The "
               "signal tests in section 5 exclude failed calls, because a failed call is not a "
               "judgment.\n",
               _table(ds.get("outages", []), ["run_start", "candidates", "ai_error_pct", "approved", "published"])
               if ds.get("outages") else "_No outage runs in the window._",
               "\n### News outages\n",
               "Runs where more than half the candidates had no news in any tier. The scout now "
               "refuses to publish these. Before that guard they published whatever the few covered "
               "names produced. A run listed here that looks like a normal day is a false positive of "
               "the guard's threshold.\n",
               _table(ds.get("news_outages", []), ["run_start", "candidates", "no_news_pct", "approved", "published"])
               if ds.get("news_outages") else "_No news-outage runs in the window._",
               f"\n### Latency\nScanner finished -> targets published: median {_fmt(lat.get('median_min'), 0)} min, "
               f"p10 {_fmt(lat.get('p10_min'), 0)}, p90 {_fmt(lat.get('p90_min'), 0)}, "
               f"max {_fmt(lat.get('max_min'), 0)} ({lat.get('runs', 0)} runs).\n"]
    dt_llm = analysis.dead_time(llm, sess)
    dt_scan = analysis.dead_time(b_scan, sess)
    report += [f"### Dead time\nShare of session time in which the bots had nothing to enter: no file yet, "
               f"`updated` > 24h old, or a published file with every equity bucket empty (the outage runs "
               f"above): **{_fmt(dt_llm['empty_pct'], 1)}%** as published, "
               f"{_fmt(dt_scan['empty_pct'], 1)}% if targets had been published at scanner time.\n"]

    if args.decisions_only:
        report.append("\n_Decisions-only run: no market data, so no simulations and no forward returns. "
                      "Session calendar approximated (weekdays minus NYSE holidays, no early closes)._\n")
        _write(out, report)
        return 0

    # --- market data ----------------------------------------------------------
    from . import data, sim
    fetch_start = datetime.combine(start - timedelta(days=21), dtime(0), tzinfo=ET)
    fetch_end = datetime.combine(end + timedelta(days=10), dtime(23, 59), tzinfo=ET)
    fetch_end = min(fetch_end, datetime.now(ET) - timedelta(minutes=20))
    session_dates = [o.date() for o, _ in sess]

    print("Ranking the point-in-time top-N universe (daily bars)...")
    universe = data.equity_universe()
    daily = data.bars_daily(universe + ["SPY"], fetch_start - timedelta(days=20), fetch_end, args.cache_dir)
    top = data.topn_lists(daily, session_dates, n=args.top_n)
    arm_a = schedule.topn_schedule(top, flat, name=f"A_top{args.top_n}")

    syms = set(b_pub.symbols()) | set(arm_a.symbols()) | {"SPY"}
    print(f"Fetching 15m bars for {len(syms)} symbols...")
    bars = data.bars_15m(sorted(syms), fetch_start, fetch_end, args.cache_dir)
    market = sim.Market(bars)
    missing = sorted(s for s in syms if not market.has(s))
    if missing:
        print(f"   {len(missing)} symbols returned no bars: {', '.join(missing[:20])}"
              f"{' ...' if len(missing) > 20 else ''}")

    spy = daily.get("SPY")
    spy_ret = None
    if spy is not None and not spy.empty:
        idx = pd.DatetimeIndex(spy.index).tz_convert(ET).date
        spy_ret = pd.Series(spy["close"].to_numpy(), index=idx).pct_change()

    simulator = sim.Simulator(market, sess, args.equity, args.slippage_bps, args.seed)
    arms = [arm_a, b_scan, b_pub,
            schedule.llm_schedule(window_runs, flat_conf=flat, name="C_llm_flat_conf"), llm]
    results = {}
    for sc in arms:
        print(f"Simulating {sc.name}...")
        results[sc.name] = simulator.run(sc)

    print(f"Simulating {args.null_draws} random-selection draws (arm D)...")
    null = []
    for k in range(args.null_draws):
        r = simulator.run(schedule.random_schedule(window_runs, flat, seed=args.seed * 100_000 + k,
                                                   name=f"D_random_{k}"))
        m = analysis.arm_metrics(r, spy_ret)
        null.append(m)
        if (k + 1) % 25 == 0:
            print(f"   {k + 1}/{args.null_draws}")

    metrics = [analysis.arm_metrics(results[s.name], spy_ret) for s in arms]
    cols = ["arm", "trades", "total_pnl", "return_pct", "win_rate", "avg_trade_ret_pct",
            "profit_factor", "max_drawdown_pct", "sharpe", "avg_exposure_pct",
            "return_on_deployed_pct", "beta_vs_spy", "alpha_ann_pct", "worst5pct_trade_ret_pct"]
    if null:
        nd = pd.DataFrame(null)
        metrics.append(dict(arm=f"D_random (median of {len(null)})",
                            **{c: float(nd[c].median()) for c in cols[1:] if c in nd}))
    report += ["## 2. Fleet simulations\n", _table(metrics, cols),
               "\n`return_on_deployed_pct` = window P&L divided by the capital the arm kept "
               "deployed on average. It is the fair comparison between an arm that trades a lot and "
               "one that mostly sits in cash. `avg_exposure_pct` is how much of the account each "
               "arm kept deployed on average.\n", "### Per bot\n"]
    pb = []
    for s in arms:
        for bot, m in analysis.by_bot(results[s.name]).items():
            pb.append(dict(arm=s.name, bot=bot, **m))
    report.append(_table(pb, ["arm", "bot", "trades", "pnl", "win_rate", "return_on_deployed_pct"]))

    # --- decomposition -----------------------------------------------------------
    R = results
    steps = [
        ("Universe: top-N vs scanner (no LLM, scanner time)", "A_top%d" % args.top_n, "B_all_scanner_time"),
        ("Latency: scanner-time vs publish-time (no LLM)", "B_all_scanner_time", "B_all_publish_time"),
        ("LLM selection (flat sizing)", "C_llm_flat_conf", "B_all_publish_time"),
        ("LLM sizing (confidence-scaled vs flat)", "C_llm", "C_llm_flat_conf"),
        ("Whole LLM stage vs deterministic pipeline", "C_llm", "B_all_scanner_time"),
    ]
    dec = []
    for label, a, b in steps:
        bs = analysis.block_bootstrap_diff(R[a], R[b])
        dec.append(dict(step=label, arms=f"{a} - {b}", total_pnl_diff=bs.get("total"),
                        mean_daily_diff=bs["mean"], ci95_lo=bs["lo"], ci95_hi=bs["hi"], days=bs["days"]))
    report += ["\n## 3. Decomposition (paired daily P&L, 5-day block bootstrap)\n",
               _table(dec, ["step", "arms", "total_pnl_diff", "mean_daily_diff", "ci95_lo", "ci95_hi", "days"]),
               "\nAn interval that contains 0 means three months cannot tell the two apart.\n"]

    # --- the null ----------------------------------------------------------------
    cf = analysis.arm_metrics(R["C_llm_flat_conf"], spy_ret)
    nrows = []
    for key in ("total_pnl", "return_on_deployed_pct", "sharpe", "trades"):
        vals = [m[key] for m in null]
        nrows.append(dict(metric=key, llm_flat=cf[key],
                          null_p5=float(pd.Series(vals).quantile(.05)) if vals else float("nan"),
                          null_median=float(pd.Series(vals).median()) if vals else float("nan"),
                          null_p95=float(pd.Series(vals).quantile(.95)) if vals else float("nan"),
                          llm_percentile=analysis.null_percentile(cf[key], vals)))
    report += [f"## 4. Is it the choices, or just fewer of them? ({len(null)} random draws)\n",
               "Each draw approves the same number of candidates per run and bucket as the LLM did, "
               "chosen at random, with the same timing and flat sizing. Compared with "
               "`C_llm_flat_conf` so that sizing is out of the picture.\n",
               _table(nrows, ["metric", "llm_flat", "null_p5", "null_median", "null_p95", "llm_percentile"]),
               "\nHow to read it: an `llm_percentile` between 5 and 95 on P&L and on deployed return "
               "means the LLM's picks cannot be told apart from random picks at the same rate. Any "
               "edge it has over the all-candidates arm then comes from trading fewer names, and a "
               "deterministic cap can do that without an LLM. Above 95 means the picks themselves carry "
               "information. `trades` shows whether the random draws traded as often as the LLM arm.\n"]

    # --- candidate-level ---------------------------------------------------------
    fwd = analysis.candidate_forward_returns(window_runs, market, sess, start, end)
    avr = analysis.approved_vs_rejected(fwd) if len(fwd) else pd.DataFrame()
    ic = analysis.rank_ic(fwd) if len(fwd) else pd.DataFrame()
    ld = analysis.latency_drift(fwd) if len(fwd) else pd.DataFrame()
    report += ["## 5. Candidate forward returns (no simulator)\n",
               f"{len(fwd)} candidate-days (first appearance per symbol/bucket/day)"
               f"{', of which %d had a failed LLM call and are excluded below' % int(fwd['ai_error'].sum()) if len(fwd) else ''}. "
               "Returns signed by bucket direction; entry at the first session bar after publication.\n",
               "### Approved vs rejected (95% CI clustered by day)\n", _df_table(avr),
               "\n### Rank IC (Fama-MacBeth over run x bucket cross-sections)\n",
               "`llm_resid_ic` is the LLM's information beyond the tech score. |t| < 2 is no evidence "
               "of a signal.\n", _df_table(ic),
               "\n### What the LLM's runtime cost\nSigned move from scanner completion to publication "
               "(positive = the candidate moved in the trade's favour before the fleet could act, "
               "i.e. the entry got worse by waiting).\n", _df_table(ld)]
    if args.shadow_votes:
        report += ["\n### Shadow specialist votes\n",
                   _df_table(analysis.shadow_vote_returns(fwd, args.shadow_votes))]

    report += ["\n## 6. Limitations\n",
               "- Regime/VIX gating, CFO reallocation, CAPITAL_CRUNCH and wheel_bot are not simulated "
               "(identical in every arm, so they shift levels, not differences).",
               "- 15m-bar replay: entries fill at the next bar's open, stops through bar high/low.",
               "- Top-N universe is built from today's active asset list (small survivorship bias).",
               "- Three months is one regime. Treat intervals that include 0 as \"no answer\", not as "
               "\"no difference\".\n"]
    _write(out, report)

    # --- artifacts ---------------------------------------------------------------
    for name, res in results.items():
        _trades_csv(out / f"trades_{name}.csv", res.trades)
    pd.DataFrame(metrics).to_csv(out / "arm_metrics.csv", index=False)
    pd.DataFrame(null).to_csv(out / "null_draws.csv", index=False)
    if len(fwd):
        fwd.to_csv(out / "candidate_forward_returns.csv", index=False)
    print(f"Wrote {out / 'report.md'}")
    return 0


def _trades_csv(path, trades):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot", "symbol", "side", "qty", "entry_time", "entry_price", "exit_time",
                    "exit_price", "pnl", "ret", "reason", "confidence", "run_id", "entry_type"])
        for t in trades:
            d = asdict(t)
            w.writerow([d[k] for k in ("bot", "symbol", "side", "qty", "entry_time", "entry_price",
                                       "exit_time", "exit_price", "pnl", "ret", "reason",
                                       "confidence", "run_id", "entry_type")])


def _write(out, report):
    p = out / "report.md"
    p.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
