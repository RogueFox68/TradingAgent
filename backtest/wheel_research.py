"""Wheel backtest: does wheel_bot's strategy earn its 38% of the account?

    python -m backtest.wheel_research --fleet-repo ..\\trading-bot-fleet

Run from the TradingAgent directory on the Corsair (config.py supplies the
Alpaca keys). See backtest/README.md, "Wheel backtest".

THE GRID BELOW IS THE PRE-REGISTRATION (2026-09-24), in the same sense as
research.HYPOTHESES: settings are chosen per quarter from earlier quarters
only, and the pass bar is walkforward.PASS_BAR, unchanged.

Universe: the scanner's own wheel rule applied point in time to its own
liquidity rule (top 400 by dollar volume, $15-$1000), no LLM - the wheel as
the fleet would run it on the deterministic scanner.
"""
import argparse
import itertools
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from . import options_data as O
from . import rules
from . import walkforward as W
from . import wheel as WH
from .run_backtest import _fmt, _table
from .schedule import ET

# Option history on Alpaca starts in February 2024.
OPTIONS_START = date(2024, 2, 1)

GRID = dict(otm=[0.03, 0.05, 0.08], take_profit=[0.50, None], gates=[True, False])
LIVE_LIKE = dict(otm=0.03, take_profit=0.50, gates=True)
WHY = ("Selling puts earns the gap between implied and realised volatility, the one "
       "strategy in the fleet with a documented source of return. otm 0.03 is closest to "
       "live: wheel_bot sells 0.05 x (1.5 - confidence) OTM and approved wheel "
       "confidence sits around 0.85-0.95. gates = the fleet's regime/VIX entry gates plus "
       "the VIX>28 stop; off = no gates, no stop.")

_OCC = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def trial_name(p):
    tp = "none" if p["take_profit"] is None else f"{p['take_profit']:.0%}"
    return f"wheel[otm={p['otm']:.0%},tp={tp},gates={'on' if p['gates'] else 'off'}]"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2024, 3, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    p.add_argument("--capital", type=float, default=100_000 * 0.38 * 0.95,
                   help="the wheel's sleeve: 38%% allocation x 95%% after the unallocated reserve")
    p.add_argument("--liquid-n", type=int, default=400)
    p.add_argument("--k", type=int, default=10, help="wheel candidates per day (the scanner's top 10)")
    p.add_argument("--slip", type=float, default=0.05,
                   help="fraction of the option price lost per fill (spread + slippage)")
    p.add_argument("--fee", type=float, default=0.05, help="$ per contract per fill")
    p.add_argument("--vix-csv", help="local copy of CBOE's VIX_History.csv")
    p.add_argument("--min-train-folds", type=int, default=1)
    p.add_argument("--no-delisted", action="store_true")
    p.add_argument("--fleet-repo", default=str(Path(__file__).resolve().parents[2] / "trading-bot-fleet"))
    p.add_argument("--allow-unverified-rules", action="store_true")
    p.add_argument("--cache-dir", default="backtest_cache")
    p.add_argument("--out", default="backtest_wheel_out")
    return p.parse_args(argv)


def _dated(df):
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_convert(ET) if idx.tz is not None else idx
    d = df.copy()
    d.index = pd.Index(idx.date)
    return d[~d.index.duplicated(keep="last")].sort_index()


def make_loaders(raw_close, cache_dir, t1):
    def chain_loader(u, day):
        q0 = date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
        q1 = (pd.Timestamp(q0) + pd.offsets.QuarterEnd(0)).date()
        s = raw_close.get(u)
        if s is None or s.empty:
            return pd.DataFrame(columns=O.CHAIN_COLUMNS)
        w = s[(s.index >= q0 - timedelta(days=10)) & (s.index <= q1)]
        if w.empty:
            return pd.DataFrame(columns=O.CHAIN_COLUMNS)
        # a data-fetch band only; pick_contract decides from day t's price
        return O.chain(u, q0 + timedelta(days=20), q1 + timedelta(days=50),
                       round(0.6 * float(w.min()), 2), round(1.4 * float(w.max()), 2), cache_dir)

    def bar_loader(symbols):
        out = {}
        for sym in symbols:
            m = _OCC.match(sym)
            if not m:
                out[sym] = pd.DataFrame()
                continue
            expiry = datetime.strptime(m.group(2), "%y%m%d").date()
            start = max(OPTIONS_START, expiry - timedelta(days=100))
            end = min(expiry + timedelta(days=1), t1.date())
            out.update(O.option_bars([sym], start, end, cache_dir))
        return out

    return chain_loader, bar_loader


def main(argv=None):
    from . import data
    args = parse_args(argv)
    if args.start < OPTIONS_START:
        sys.exit(f"Alpaca option history starts {OPTIONS_START}; --start cannot be earlier.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    problems = rules.verify_wheel_against_fleet(args.fleet_repo)
    if problems:
        for p in problems:
            print("   -", p)
        if not args.allow_unverified_rules:
            sys.exit("The wheel's rules could not be verified against the fleet source. Point "
                     "--fleet-repo at a trading-bot-fleet checkout, or pass --allow-unverified-rules.")

    t1 = min(datetime.combine(args.end, dtime(23, 59), tzinfo=ET), datetime.now(ET) - timedelta(minutes=20))
    t0 = datetime.combine(args.start - timedelta(days=460), dtime(0), tzinfo=ET)

    print("Universe and daily bars (adjusted, for the scanner's indicators)...")
    universe = data.equity_universe(include_inactive=not args.no_delisted)
    daily_adj = data.bars_daily(universe + ["SPY"], t0, t1, args.cache_dir, adjustment="all")
    spy_adj = daily_adj.get("SPY")
    if spy_adj is None or spy_adj.empty:
        sys.exit("No SPY daily bars.")
    cal = [d for d in _dated(spy_adj).index if d >= args.start - timedelta(days=5)]
    days = [d for d in cal if args.start <= d <= args.end]
    print(f"Ranking the top-{args.liquid_n} liquid names and the scanner's wheel candidates...")
    members = data.topn_lists(daily_adj, cal, n=args.liquid_n, min_price=15.0)
    cands = WH.wheel_candidates(daily_adj, days, members, k=args.k)
    underlyings = sorted({s for v in cands.values() for s in v})
    del daily_adj
    print(f"   {len(underlyings)} distinct wheel candidates over {len(days)} sessions")

    raw = data.bars_daily(underlyings + ["SPY"], t0, t1, args.cache_dir)   # RAW: strikes are raw
    raw_close = {s: _dated(df)["close"] for s, df in raw.items() if df is not None and not df.empty}
    spy_raw = _dated(raw["SPY"])
    regime = WH.regime_series(spy_raw)
    vix = O.vix_history(args.cache_dir, args.vix_csv)
    spy_r = _dated(spy_adj)["close"].pct_change()
    spy_r.index = pd.DatetimeIndex(pd.to_datetime(list(spy_r.index)))

    chain_loader, bar_loader = make_loaders(raw_close, args.cache_dir, t1)
    book = WH.OptionBook(chain_loader, bar_loader)

    trials, results = {}, {}
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        name = trial_name(p)
        wp = WH.WheelParams(otm=p["otm"], take_profit=p["take_profit"], gates=p["gates"],
                            slip=args.slip, fee=args.fee)
        res = WH.WheelSim(days, raw_close, book, cands, wp, args.capital, regime, vix).run()
        trials[name] = res.returns()
        results[name] = res
        c = res.counts()
        print(f"   {name}: Sharpe {W.sharpe(trials[name]):.2f}, "
              f"{c.get('sell_put', 0)} puts, {c.get('assigned', 0)} assigned, "
              f"{c.get('skip_no_trade', 0)} skipped (no trade)")

    report = build_report(args, trials, results, spy_r, book, days, cands, raw_close, regime, vix)
    (out / "wheel_report.md").write_text("\n".join(report), encoding="utf-8")
    pd.DataFrame(trials).to_csv(out / "wheel_trial_daily_returns.csv")
    live = results[trial_name(LIVE_LIKE)]
    pd.DataFrame([e.__dict__ for e in live.events]).to_csv(out / "wheel_events_live_like.csv", index=False)
    print(f"Wrote {out / 'wheel_report.md'}")
    return 0


def _coverage(res):
    c = res.counts()
    sells = c.get("sell_put", 0) + c.get("sell_call", 0)
    tried = sells + c.get("skip_no_trade", 0)
    return dict(puts=c.get("sell_put", 0), calls=c.get("sell_call", 0),
                skipped_no_trade=c.get("skip_no_trade", 0),
                fill_rate_pct=100 * sells / tried if tried else float("nan"),
                closes=c.get("close", 0), rolls=c.get("roll_open", 0),
                rolls_aborted=c.get("roll_aborted", 0), assigned=c.get("assigned", 0),
                called_away=c.get("called_away", 0), expired=c.get("expired", 0),
                paused_days=res.paused_days)


def build_report(args, trials, results, spy_r, book, days, cands, raw_close, regime, vix):
    oos, choices = W.walk_forward(trials, "Q", args.min_train_folds)
    lines = ["# Wheel backtest: walk-forward, out-of-sample\n",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. {args.start} .. {args.end} "
             f"({len(days)} sessions). Sleeve capital ${args.capital:,.0f}. Fills at the "
             f"contract's traded VWAP less {args.slip:.0%} of the option price, plus "
             f"${args.fee:.2f}/contract. Underlyings: the scanner's wheel rule on its top-"
             f"{args.liquid_n} liquid names, top {args.k} a day, no LLM.\n",
             f"_{WHY}_\n"]
    if oos.empty:
        return lines + ["**Not enough quarters for an out-of-sample record.**"]
    st = W.summarize(oos, "Q")
    spy = spy_r.reindex(oos.index).dropna()
    spy_st = W.summarize(spy, "Q")
    v = W.verdict(st, spy_st["sharpe"])
    passed = all(p for _, p, _ in v)
    live = results[trial_name(LIVE_LIKE)]
    cov = _coverage(live)
    lines += ["## Verdict\n",
              ("**PASSES the pre-registered bar - candidate for continued paper trading.**" if passed
               else "**Does not pass the pre-registered bar.**"),
              ""]
    if cov["fill_rate_pct"] < 50:
        lines.append(f"**Data warning:** only {cov['fill_rate_pct']:.0f}% of intended sells found a "
                     "traded price on the fill day in the live-like trial. The record describes the "
                     "contracts that traded, which is a biased subset; treat it as indicative.\n")
    rows = [dict(series="wheel (walk-forward OOS)", **{k: x for k, x in st.items() if k != "fold_returns"}),
            dict(series="SPY buy-and-hold, same days", **{k: x for k, x in spy_st.items() if k != "fold_returns"})]
    lines += ["## Out-of-sample record (the result)\n",
              _table(rows, ["series", "days", "ann_return_pct", "ann_vol_pct", "sharpe", "psr",
                            "max_dd_pct", "positive_folds"]), "",
              _table([dict(criterion=c, result="PASS" if p else "fail", value=d) for c, p, d in v],
                     ["criterion", "result", "value"]), "",
              "### Quarter by quarter\n",
              _table([dict(fold=f, chosen=c, train_sharpe=s, oos_return_pct=st["fold_returns"].get(f),
                           spy_return_pct=spy_st["fold_returns"].get(f)) for f, c, s in choices],
                     ["fold", "chosen", "train_sharpe", "oos_return_pct", "spy_return_pct"]), ""]

    # the live-like trial, what actually happened in it
    lr = trials[trial_name(LIVE_LIKE)]
    lst = W.summarize(lr, "Q")
    lines += ["## The live-like trial, full period\n",
              f"`{trial_name(LIVE_LIKE)}` - wheel_bot's settings. Not the result (it was not chosen "
              "out of sample); shown because it is what the fleet runs.\n",
              _table([dict(series="live-like", **{k: x for k, x in lst.items() if k != "fold_returns"})],
                     ["series", "days", "ann_return_pct", "ann_vol_pct", "sharpe", "psr", "max_dd_pct",
                      "positive_folds"]), "",
              "**What it did:**\n",
              _table([cov], list(cov)), "",
              "`skipped_no_trade`: sells that found no traded price on the fill day (wheel_bot would "
              "have skipped on a wide spread). `paused_days`: VIX > 28, the process stopped.\n"]
    gated_days = sum(1 for d in days if (regime.reindex([d]).iloc[0] in rules.WHEEL_GATED_REGIMES)
                     or (vix[vix.index <= d].iloc[-1] if len(vix[vix.index <= d]) else 0) > rules.WHEEL_VIX_GATE)
    lines.append(f"New puts were gated on **{gated_days} of {len(days)} sessions** "
                 f"({100 * gated_days / len(days):.0f}%): SPY below its 20-day EMA, or VIX above 22.\n")

    # cost sensitivity for the live-like trial (full period, labelled)
    lines += ["## Cost sensitivity (live-like trial, full period - not the result)\n",
              "Option spreads are wide, so the edge has to survive them. Same trial re-run at "
              "other spread assumptions.\n"]
    rows = []
    for slip in (0.0, args.slip, 0.10):
        if slip == args.slip:
            r = lr
        else:
            wp = WH.WheelParams(otm=LIVE_LIKE["otm"], take_profit=LIVE_LIKE["take_profit"],
                                gates=LIVE_LIKE["gates"], slip=slip, fee=args.fee)
            r = WH.WheelSim(days, raw_close, book, cands, wp, args.capital, regime, vix).run().returns()
        rows.append(dict(slip=f"{slip:.0%}", ann_return_pct=100 * W.ann_return(r), sharpe=W.sharpe(r),
                         max_dd_pct=100 * W.max_drawdown(r)))
    lines += [_table(rows, ["slip", "ann_return_pct", "sharpe", "max_dd_pct"]), ""]

    # every trial + deflated Sharpe within this family
    names = list(trials)
    sh = {n: W.sharpe(trials[n]) for n in names}
    best = max(names, key=lambda n: sh[n] if not np.isnan(sh[n]) else -99)
    dsr, sr0 = W.deflated_sharpe(trials[best], list(sh.values()))
    lines += ["## Every trial, full period (transparency only - NOT the result)\n",
              f"Best: `{best}`, Sharpe {_fmt(sh[best])}. Luck benchmark for {len(names)} trials: "
              f"Sharpe {_fmt(sr0)}. **Deflated Sharpe: {_fmt(dsr, 3)}** (below 0.95: not "
              "distinguishable from the best of that many zero-skill trials).\n",
              _table(sorted([dict(trial=n, sharpe=sh[n], ann_return_pct=100 * W.ann_return(trials[n]),
                                  max_dd_pct=100 * W.max_drawdown(trials[n]), **_coverage(results[n]))
                             for n in names], key=lambda r: -(r["sharpe"] if not np.isnan(r["sharpe"]) else -99)),
                     ["trial", "sharpe", "ann_return_pct", "max_dd_pct", "puts", "assigned",
                      "fill_rate_pct", "paused_days"]), "",
              "## Limitations\n",
              "- Prices are TRADES, not quotes (Alpaca keeps no option quote history): a fill is the "
              "day's VWAP less a spread haircut, and a contract that did not trade that day cannot "
              "be filled. Contracts that trade are the liquid ones, so fill-rate matters.",
              "- Decisions use day t's close and fill on day t+1; the live bot acts intraday every 15 min.",
              "- No interest on idle collateral (a real cash-secured put earns T-bill yield on it; "
              "Alpaca paper does not), which understates the wheel against SPY by roughly the cash "
              "yield times the idle share.",
              "- The scanner's earnings guard is not applied (no historical earnings calendar); "
              "early exercise and splits during a held contract are not modelled.",
              "- About 2.5 years of option history: one regime-mix, few quarters."]
    return lines


if __name__ == "__main__":
    sys.exit(main())
