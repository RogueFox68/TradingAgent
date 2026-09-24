"""Regression tests for backtest.research / daily / walkforward.

What they pin is the set of ways a research result can be flattered without
anyone noticing: a signal that trades on its own day's close, a
walk-forward that peeks at the fold it is scored on, costs that skip the
overlap, a Sharpe test that ignores how many things were tried.
"""
import math
import tempfile
import unittest
from datetime import date, datetime, time as dtime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backtest import daily as D
from backtest import walkforward as W
from backtest import schedule, sim

import test_backtest as TB


def _panel(opens, closes=None, members=None, spy=None):
    idx = pd.Index(pd.bdate_range("2021-01-04", periods=len(next(iter(opens.values())))).date)
    o = pd.DataFrame(opens, index=idx, dtype=float)
    c = pd.DataFrame(closes if closes is not None else opens, index=idx, dtype=float)
    m = members or {d: list(o.columns) for d in idx}
    return D.Panel(o, c, m, pd.Series(spy, index=idx, dtype=float) if spy is not None else pd.Series(dtype=float))


class CohortEngineTest(unittest.TestCase):
    def test_signal_at_close_trades_from_next_open(self):
        p = _panel({"A": [10, 10, 10, 10, 11, 11, 11]})
        Wt = pd.DataFrame(0.0, index=p.dates, columns=p.open.columns)
        Wt.iloc[2, 0] = 1.0                       # signal at close of day 2
        r = D.run_cohorts(p, Wt, hold=1, step=1, cost_bps=0)
        # held only on day 3: open[3] -> open[4] = +10%; nothing on day 2
        self.assertEqual(list(np.round(r.to_numpy(), 6)), [0, 0, 0, 0.1, 0, 0, 0])

    def test_turnover_costs_entry_and_exit(self):
        p = _panel({"A": [10] * 6})
        Wt = pd.DataFrame(0.0, index=p.dates, columns=p.open.columns)
        Wt.iloc[1, 0] = 1.0
        r = D.run_cohorts(p, Wt, hold=2, step=2, cost_bps=10)
        # one full-weight cohort: in on day 2, out on day 4, 10 bps each way
        self.assertAlmostEqual(r.iloc[2], -0.001)
        self.assertAlmostEqual(r.iloc[4], -0.001)
        self.assertAlmostEqual(r.sum(), -0.002)

    def test_overlapping_cohorts_hold_full_weight_without_churn(self):
        p = _panel({"A": [10] * 8})
        Wt = pd.DataFrame(1.0, index=p.dates, columns=p.open.columns)   # formed every day
        r = D.run_cohorts(p, Wt, hold=3, step=1, cost_bps=10)
        # ramps in over 3 days (1/3 each), then no turnover at all
        self.assertAlmostEqual(r.iloc[1:4].sum(), -0.001)
        self.assertEqual(float(r.iloc[4:].abs().sum()), 0.0)

    def test_missing_next_open_is_a_flat_day(self):
        p = _panel({"A": [10, 10, np.nan, 12]})
        np.testing.assert_array_equal(D.open_to_open(p)["A"].to_numpy(), [0, 0, 0, 0])


class SignalTest(unittest.TestCase):
    def test_reversal_buys_worst_members_only(self):
        p = _panel({"UP": [10, 11, 12, 13], "DOWN": [10, 9, 8, 7], "FLAT": [10] * 4,
                    "OUTSIDER": [10, 5, 3, 1]},
                   members=None)
        p.members = {d: ["UP", "DOWN", "FLAT"] for d in p.dates}
        Wt = D.reversal_weights(p, lookback=2, k=1, long_short=True)
        last = Wt.iloc[-1]
        self.assertEqual(last["DOWN"], 0.5)          # worst member, long
        self.assertEqual(last["UP"], -0.5)           # best member, short
        self.assertEqual(last["OUTSIDER"], 0.0)      # not in the top-N that day

    def test_momentum_skips_the_most_recent_month(self):
        n = 40
        steady = list(np.linspace(10, 20, n))                        # strong, earlier
        late = [10.0] * (n - 5) + [30, 40, 50, 60, 70]                # all gain in last 5 days
        p = _panel({"STEADY": steady, "LATE": late})
        Wt = D.momentum_weights(p, lookback=30, skip=6, k=1, spy_filter=False, step=1)
        self.assertEqual(Wt.iloc[-1]["STEADY"], 1.0)
        self.assertEqual(Wt.iloc[-1]["LATE"], 0.0)

    def test_spy_filter_goes_to_cash(self):
        n = 260
        spy = list(np.linspace(200, 400, n - 10)) + [100] * 10     # crashes below its SMA200
        p = _panel({"A": list(np.linspace(10, 20, n)), "B": list(np.linspace(20, 10, n))}, spy=spy)
        on = D.momentum_weights(p, 50, 5, 1, spy_filter=True, step=1)
        off = D.momentum_weights(p, 50, 5, 1, spy_filter=False, step=1)
        self.assertEqual(float(on.iloc[-1].sum()), 0.0)
        self.assertEqual(float(off.iloc[-1].sum()), 1.0)
        self.assertEqual(float(on.iloc[-20].sum()), 1.0)


class WalkForwardTest(unittest.TestCase):
    def test_selection_uses_only_earlier_folds(self):
        idx = pd.bdate_range("2019-01-01", "2021-12-31")
        rng = np.random.default_rng(0)
        good_early = pd.Series(rng.normal(0.002, 0.01, len(idx)), index=idx)
        lucky_late = pd.Series(rng.normal(-0.002, 0.01, len(idx)), index=idx)
        lucky_late[idx.year == 2021] = 0.05          # spectacular, but only in the test fold
        oos, choices = W.walk_forward({"good_early": good_early, "lucky_late": lucky_late}, "Y")
        self.assertEqual([c[0] for c in choices], ["2020", "2021"])   # 2019 trains only
        self.assertEqual([c[1] for c in choices], ["good_early", "good_early"])
        pd.testing.assert_series_equal(oos, good_early[idx.year >= 2020], check_names=False)

    def test_psr_and_deflation(self):
        rng = np.random.default_rng(1)
        noise = pd.Series(rng.normal(0, 0.01, 2000))
        noise = noise - noise.mean()                 # exactly zero Sharpe
        self.assertAlmostEqual(W.psr(noise), 0.5, places=6)
        strong = pd.Series(rng.normal(0.003, 0.01, 2000))
        self.assertGreater(W.psr(strong), 0.999)
        # the best of 50 noise trials looks good alone, and deflates away
        trials = [pd.Series(rng.normal(0, 0.01, 500)) for _ in range(50)]
        srs = [W.sharpe(t) for t in trials]
        best = trials[int(np.argmax(srs))]
        self.assertGreater(max(srs), 0.8)
        dsr, sr0 = W.deflated_sharpe(best, srs)
        self.assertGreater(sr0, 0.8)
        self.assertLess(dsr, 0.95)

    def test_verdict_needs_every_criterion(self):
        good = dict(sharpe=1.2, psr=0.99, positive_folds=0.8,
                    fold_returns={"2021": 1, "2022": 1, "2023": 1, "2024": 1})
        self.assertTrue(all(p for _, p, _ in W.verdict(good, spy_sharpe=0.9)))
        self.assertFalse(all(p for _, p, _ in W.verdict(good, spy_sharpe=1.5)))
        short = dict(good, fold_returns={"2024": 1})
        self.assertFalse(all(p for _, p, _ in W.verdict(short, spy_sharpe=0.9)))
        nan = dict(sharpe=float("nan"), psr=float("nan"), positive_folds=float("nan"))
        self.assertFalse(any(p for _, p, _ in W.verdict(nan, spy_sharpe=0.5)))


class SimVariantTest(unittest.TestCase):
    def test_inverted_trend_fades_a_bullish_cross(self):
        m = TB._market({"S": TB._bars(TB.DAYS, TB._flat(50.0))})
        i = TB._idx(m, "S", (10, 0), d=TB.DAYS[1])
        m.sym["S"]["adx"][i] = 30.0
        m.sym["S"]["bull_cross"][i] = True
        m.sym["S"]["bear_cross"][i + 4] = True           # the signal later reverses
        tg = {"trend_targets": {"S": 0.68}, "short_targets": {"S": 0.68},
              "survivor_targets": {}, "wheel_targets": {}}
        sessions = [TB._session(d) for d in TB.DAYS]
        plain = sim.Simulator(m, sessions, slippage_bps=0).run(
            TB.FixedSchedule(tg, both_directions=True)).trades
        inv = sim.Simulator(m, sessions, slippage_bps=0, variant={"trend_invert": True}).run(
            TB.FixedSchedule(tg, both_directions=True)).trades
        self.assertEqual(plain[0].side, "long")
        self.assertEqual(inv[0].side, "short")
        # a faded bullish signal exits when the signal turns bearish
        self.assertEqual(inv[0].reason, "Bearish Crossover")

    def test_variant_exit_levels_and_bot_selection(self):
        def path(t):
            if t.hour == 11 and t.minute == 0:
                return (100.0, 100.0, 97.5, 99.0)        # -2.5%: inside -3%, through -2%
            return (100.0,) * 4
        m = TB._market({"S": TB._bars(TB.DAYS[:1], path)})
        m.sym["S"]["rsi"][TB._idx(m, "S", (10, 0))] = 30.0
        tg = {"survivor_targets": {"S": 0.68}, "trend_targets": {}, "short_targets": {},
              "wheel_targets": {}}
        sess = [TB._session(TB.DAYS[0])]
        live = sim.Simulator(m, sess, slippage_bps=0).run(TB.FixedSchedule(tg)).trades
        tight = sim.Simulator(m, sess, slippage_bps=0,
                              variant={"surv_stop": -0.02, "surv_tp": 0.03}).run(TB.FixedSchedule(tg)).trades
        self.assertNotEqual(live[0].reason, "Stop Loss")
        self.assertEqual(tight[0].reason, "Stop Loss")
        self.assertAlmostEqual(tight[0].exit_price, 98.0)
        none = sim.Simulator(m, sess, slippage_bps=0, variant={"bots": ("trend_bot",)}).run(
            TB.FixedSchedule(tg)).trades
        self.assertEqual(none, [])


class ResearchEndToEndTest(unittest.TestCase):
    """Drives research.main with Alpaca stubbed (fleet rule 25)."""

    def test_full_run(self):
        from backtest import data, research
        days = list(pd.bdate_range("2017-06-01", "2021-06-30").date)
        syms = [f"S{k:02d}" for k in range(30)] + ["SPY"]
        rng = np.random.default_rng(5)
        px = {s: 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(days)))) for s in syms}

        def daily(symbols, *a, **k):
            idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days])
            return {s: pd.DataFrame({"open": px[s], "high": px[s] * 1.01, "low": px[s] * 0.99,
                                     "close": px[s], "volume": 1e6 * (1 + int(s[1:]) if s != "SPY" else 1e8)},
                                    index=idx) for s in symbols if s in px}

        iday = [d for d in days if d >= date(2020, 12, 1)]

        def bars15(symbols, *a, **k):
            out = {}
            for s in symbols:
                walk = iter(px[s][-len(iday) * 26:] if len(px[s]) >= len(iday) * 26
                            else np.resize(px[s], len(iday) * 26))
                out[s] = TB._bars(iday, lambda t, w=walk: (lambda p: (p, p * 1.004, p * 0.996, p))(next(w)))
            return out

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(data, "equity_universe", lambda include_inactive=False: syms[:-1]), \
                mock.patch.object(data, "bars_daily", daily), \
                mock.patch.object(data, "bars_15m", bars15), \
                mock.patch.object(data, "sessions",
                                  lambda a, b: [TB._session(d) for d in iday if a <= d <= b]):
            rc = research.main(["--start", "2019-01-01", "--end", "2021-06-30",
                                "--intraday-start", "2021-01-04", "--top-n", "25",
                                "--allow-unverified-rules", "--fleet-repo", str(TB.FLEET),
                                "--out", str(Path(tmp) / "out")])
            self.assertEqual(rc, 0)
            rep = (Path(tmp) / "out" / "research_report.md").read_text(encoding="utf-8")
            for fam in research.HYPOTHESES:
                self.assertIn(f"### {fam}", rep)
            self.assertIn("## Verdict", rep)
            self.assertIn("Deflated Sharpe", rep)
            n_trials = sum(len(list(research.grid_points(h["grid"]))) for h in research.HYPOTHESES.values())
            self.assertIn(f"{n_trials} trials in total", rep)
            oos = pd.read_csv(Path(tmp) / "out" / "oos_daily_returns.csv", index_col=0)
            self.assertIn("reversal_daily", oos.columns)
            self.assertIn("momentum_daily", oos.columns)
            self.assertNotIn("never traded", rep)
            # two OOS quarters of intraday history is below the 4-fold minimum
            self.assertIn("| at least 4 OOS folds | fail |", rep)


if __name__ == "__main__":
    unittest.main()
