"""Regression tests for the LLM-gate ablation backtest (backtest/).

The simulator's job is to hold everything but the target list constant, so
most of these pin the places an arm could quietly get an advantage it would
not have had live: seeing a bar before it completes, reading a stale targets
file as live, trading after the entry cutoff, or sizing past its budget.
"""
import ast
import importlib.util
import os
import unittest
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest import analysis, rules, schedule, scout_log, sim

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
FLEET = Path(os.environ.get("FLEET_REPO", HERE.parent / "trading-bot-fleet"))

LOG = """\
==========================================================
[Mon 09/14/2026  8:30:00.11] 🚀 STARTING DAILY TRADING SEQUENCE (Fixed)
[Mon 09/14/2026  8:30:00.12] Phase 1: Launching Market Scanner...
[Mon 09/14/2026  8:32:10.00] ✅ Scanner Complete. Dragnet file ready.
[Mon 09/14/2026  8:32:10.01] Phase 2: Launching Sector Scout...
2. Deep Diving Candidates...
   👉 Analyzing trend_targets...
      ✅ AAA  | Conf: 0.70 [Tech: 1.00 | T1: N/A | T2: 0.80 | T3: 0.60 | Soc: 0.55]
   [!] AI Error on BBB: Expecting value: line 1 column 1 (char 0)
      ❌ BBB  | Conf: 0.60 [Tech: 0.90 | T1: N/A | T2: N/A | T3: 0.00 | Soc: 0.70]
   👉 Analyzing survivor_targets...
      ❌ CCC  | Conf: 0.52 [Tech: 0.50 | T1: N/A | T2: N/A | T3: 0.80 | Soc: 0.60]
      ✅ DDD  | Conf: 0.67 [Tech: 0.60 | T1: 0.90 | T2: 0.80 | T3: N/A | Soc: 0.70]
   👉 Analyzing short_targets...
      ❌ EEE  | Conf: 0.65 [Tech: 1.00 | T1: N/A | T2: N/A | T3: N/A | Soc: N/A]
   👉 Analyzing wheel_targets...
      ✅ WWW  | Conf: 0.80 [Tech: 0.95 | T1: N/A | T2: N/A | T3: 0.80 | Soc: 0.80]
4. Beaming active_targets.json to Beelink...
   ✅ Transfer Complete (Attempt 1).
[Mon 09/14/2026  9:27:00.00] ✅ Scout Complete. Targets file updated.
[Mon 09/14/2026  9:27:00.01] 🏁 MISSION COMPLETE.
==========================================================
[Mon 09/14/2026 12:00:00.00] 🚀 STARTING DAILY TRADING SEQUENCE (Fixed)
[Mon 09/14/2026 12:02:00.00] ✅ Scanner Complete. Dragnet file ready.
[Mon 09/14/2026 12:02:00.01] Phase 2: Launching Sector Scout...
   👉 Analyzing trend_targets...
      ✅ ZZZ  | Conf: 0.75 [Tech: 1.00 | T1: 0.90 | T2: N/A | T3: N/A | Soc: N/A]
[Mon 09/14/2026 12:40:00.00] ❌ Scout failed. Targets file may be stale.
"""


def runs():
    return scout_log.parse(LOG.splitlines())


class ScoutLogParseTest(unittest.TestCase):
    def test_runs_times_and_publication(self):
        r = runs()
        self.assertEqual(len(r), 2)
        first, second = r
        # Windows pads single-digit hours with a space; the log clock is CT.
        self.assertEqual(first.started, datetime(2026, 9, 14, 8, 30, 0, 110000, tzinfo=ZoneInfo("America/Chicago")))
        self.assertTrue(first.published)
        self.assertEqual(first.llm_latency_s, (54 * 60 + 50))
        # A failed scout never reached the Beelink, whatever it analysed.
        self.assertFalse(second.published)

    def test_candidates_emoji_is_authoritative(self):
        c = {x.symbol: x for x in runs()[0].candidates}
        self.assertEqual(set(c), {"AAA", "BBB", "CCC", "DDD", "EEE", "WWW"})
        self.assertTrue(c["AAA"].approved)
        # DDD's 0.67 and EEE's 0.65 straddle 0.66, but the log's verdict is used
        # as-is: the threshold changed during the log's life.
        self.assertTrue(c["DDD"].approved)
        self.assertFalse(c["EEE"].approved)
        self.assertEqual(c["DDD"].bucket, "survivor_targets")
        self.assertIsNone(c["AAA"].t1)
        self.assertEqual(c["AAA"].t2, 0.80)
        self.assertTrue(c["BBB"].ai_error)
        self.assertFalse(c["AAA"].ai_error)

    def test_llm_score_matches_current_weighting(self):
        c = {x.symbol: x for x in runs()[0].candidates}["AAA"]
        # conf = .3*tech + .7*llm under the current weights (N/A = 0.5)
        self.assertAlmostEqual(0.3 * c.tech + 0.7 * c.llm_score(), 0.30 + 0.15 + 0.16 + 0.06 + 0.055, places=6)


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.r = [x for x in runs() if x.published]

    def t(self, *a):
        return datetime(*a, tzinfo=ET)

    def test_llm_schedule_visible_only_after_publication(self):
        s = schedule.llm_schedule(self.r)
        # scout_done 09:27 CT = 10:27 ET
        self.assertEqual(s.at(self.t(2026, 9, 14, 10, 26)), schedule.EMPTY)
        tg = s.at(self.t(2026, 9, 14, 10, 28))
        self.assertEqual(set(tg["trend_targets"]), {"AAA"})
        self.assertEqual(tg["trend_targets"]["AAA"], 0.70)
        self.assertEqual(set(tg["survivor_targets"]), {"DDD"})

    def test_stale_file_reads_empty_after_24h_from_scout_START(self):
        s = schedule.llm_schedule(self.r)
        # `updated` = Phase 2 launch 08:32:10 CT = 09:32:10 ET
        self.assertTrue(s.at(self.t(2026, 9, 15, 9, 32, 0))["trend_targets"])
        self.assertEqual(s.at(self.t(2026, 9, 15, 9, 33, 0)), schedule.EMPTY)

    def test_scanner_time_arm_is_earlier_and_complete(self):
        s = schedule.all_candidates_schedule(self.r, 0.68, "scanner")
        tg = s.at(self.t(2026, 9, 14, 9, 33))   # scanner done 08:32:10 CT
        self.assertEqual(set(tg["trend_targets"]), {"AAA", "BBB"})
        self.assertEqual(tg["trend_targets"]["BBB"], 0.68)

    def test_random_arm_keeps_the_llms_count_per_bucket(self):
        for seed in range(20):
            s = schedule.random_schedule(self.r, 0.68, seed)
            tg = s.snapshots[0].targets
            self.assertEqual(len(tg["trend_targets"]), 1)
            self.assertEqual(len(tg["survivor_targets"]), 1)
            self.assertEqual(len(tg["short_targets"]), 0)
            self.assertEqual(len(tg["wheel_targets"]), 1)
            self.assertEqual(s.snapshots[0].effective, self.r[0].scout_done)

    def test_topn_arm_shares_one_list(self):
        s = schedule.topn_schedule({date(2026, 9, 14): ["X", "Y"]}, 0.68)
        self.assertFalse(s.survivor_blacklist)
        self.assertTrue(s.both_directions)
        self.assertEqual(set(s.at(self.t(2026, 9, 14, 9, 30))["short_targets"]), {"X", "Y"})


# --- synthetic market ----------------------------------------------------------

def _session(d):
    return (pd.Timestamp(datetime.combine(d, dtime(9, 30)), tz=ET),
            pd.Timestamp(datetime.combine(d, dtime(16, 0)), tz=ET))


def _bars(days, price_fn):
    """Regular-session 15m bars; price_fn(ts) -> (open, high, low, close)."""
    rows, idx = [], []
    for d in days:
        o, c = _session(d)
        t = o
        while t < c:
            rows.append(price_fn(t))
            idx.append(t.tz_convert("UTC"))
            t += pd.Timedelta(minutes=15)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx))
    df["volume"] = 1e6
    return df


def _flat(p=100.0):
    return lambda t: (p, p, p, p)


class FixedSchedule(schedule.TargetSchedule):
    def __init__(self, targets, **kw):
        super().__init__("fixed", [], **kw)
        self.targets = targets

    def at(self, t):
        return self.targets

    def run_at(self, t):
        return 0


DAYS = [date(2026, 9, 14), date(2026, 9, 15)]


def _market(bars):
    m = sim.Market(bars)
    for a in m.sym.values():      # neutral indicators unless a test sets them
        a["rsi"][:] = 50.0
        a["adx"][:] = 10.0
        a["bull_cross"][:] = False
        a["bear_cross"][:] = False
        a["aligned_long"][:] = False
        a["aligned_short"][:] = False
    return m


def _idx(m, s, hhmm, d=DAYS[0]):
    t = pd.Timestamp(datetime.combine(d, dtime(*hhmm)), tz=ET)
    return int(np.searchsorted(m.sym[s]["t"], t.value))


class SimulatorTest(unittest.TestCase):
    def _run(self, m, targets, days=DAYS, **kw):
        s = sim.Simulator(m, [_session(d) for d in days], slippage_bps=0.0, **kw)
        return s.run(FixedSchedule(targets, **{k: v for k, v in {}.items()}))

    def surv(self, *syms):
        return {"survivor_targets": {s: 0.68 for s in syms}, "trend_targets": {},
                "short_targets": {}, "wheel_targets": {}}

    def test_signal_on_completed_bar_fills_at_next_open(self):
        m = _market({"S": _bars(DAYS, lambda t: (100 + t.minute / 100,) * 4)})
        i = _idx(m, "S", (10, 0))           # the bar 10:00-10:15
        m.sym["S"]["rsi"][i] = 30.0
        tr = self._run(m, self.surv("S"), days=DAYS[:1]).trades
        self.assertEqual(len(tr), 1)
        # decided at 10:15 from the completed 10:00 bar; filled at the 10:15 open
        self.assertEqual(tr[0].entry_time, datetime(2026, 9, 14, 10, 15, tzinfo=ET))
        self.assertAlmostEqual(tr[0].entry_price, m.sym["S"]["open"][i + 1])

    def test_no_lookahead_into_the_forming_bar(self):
        m = _market({"S": _bars(DAYS[:1], _flat())})
        last = len(m.sym["S"]["t"]) - 1
        m.sym["S"]["rsi"][last] = 20.0     # only the final bar ever dips
        self.assertEqual(self._run(m, self.surv("S"), days=DAYS[:1]).trades, [])

    def test_no_entries_from_1400(self):
        m = _market({"S": _bars(DAYS[:1], _flat())})
        m.sym["S"]["rsi"][_idx(m, "S", (13, 45)):] = 30.0   # decisions from 14:00 on
        self.assertEqual(self._run(m, self.surv("S"), days=DAYS[:1]).trades, [])

    def test_intrabar_stop_fills_at_stop_and_gap_fills_at_open(self):
        def path(t):
            if t.hour == 11 and t.minute == 0:
                return (100.0, 100.0, 96.0, 99.0)         # trades through -3%
            return (100.0, 100.0, 100.0, 100.0)
        m = _market({"S": _bars(DAYS[:1], path)})
        m.sym["S"]["rsi"][_idx(m, "S", (10, 0))] = 30.0
        tr = self._run(m, self.surv("S"), days=DAYS[:1]).trades
        self.assertEqual(tr[0].reason, "Stop Loss")
        self.assertAlmostEqual(tr[0].exit_price, 97.0)

        def gap(t):
            if t.hour == 11 and t.minute == 0:
                return (90.0, 91.0, 89.0, 90.0)           # opens through the stop
            return (100.0, 100.0, 100.0, 100.0)
        m = _market({"S": _bars(DAYS[:1], gap)})
        m.sym["S"]["rsi"][_idx(m, "S", (10, 0))] = 30.0
        tr = self._run(m, self.surv("S"), days=DAYS[:1]).trades
        self.assertEqual(tr[0].reason, "Stop Loss (gap)")
        self.assertAlmostEqual(tr[0].exit_price, 90.0)

    def test_losing_position_is_liquidated_at_1545(self):
        # -1% by the close: hold score 0 (small loss, RSI 50, <6h) -> CLOSE_EOD
        def path(t):
            p = 99.0 if (t.hour, t.minute) >= (12, 0) else 100.0
            return (p, p, p, p)
        m = _market({"S": _bars(DAYS, path)})
        m.sym["S"]["rsi"][_idx(m, "S", (10, 0))] = 30.0
        tr = self._run(m, self.surv("S")).trades
        self.assertEqual(tr[0].reason, "EOD Liquidation")
        self.assertEqual(tr[0].exit_time, datetime(2026, 9, 14, 15, 45, tzinfo=ET))

    def test_size_is_confidence_scaled_and_budget_clipped(self):
        m = _market({"S": _bars(DAYS[:1], _flat(100.0))})
        m.sym["S"]["rsi"][_idx(m, "S", (10, 0))] = 30.0
        tr = self._run(m, self.surv("S"), days=DAYS[:1]).trades
        # 100k * 5% * (0.5 + 0.68) = 5,900 -> 59 shares (under the 10% cap and 19k budget)
        self.assertEqual(tr[0].qty, 59)

        syms = [f"S{k}" for k in range(8)]
        m = _market({s: _bars(DAYS[:1], _flat(100.0)) for s in syms})
        for s in syms:
            m.sym[s]["rsi"][_idx(m, s, (10, 0))] = 30.0
        tr = self._run(m, self.surv(*syms), days=DAYS[:1]).trades
        # survivor budget = 20% * 95% * 100k = 19,000 -> 59 + 59 + 59 + 13 shares, then nothing
        self.assertEqual(sum(t.qty for t in tr), 190)

    def test_one_position_per_symbol_across_bots(self):
        m = _market({"S": _bars(DAYS[:1], _flat())})
        i = _idx(m, "S", (10, 0))
        m.sym["S"]["rsi"][i] = 30.0
        m.sym["S"]["adx"][i] = 30.0
        m.sym["S"]["bull_cross"][i] = True
        both = {"survivor_targets": {"S": 0.68}, "trend_targets": {"S": 0.68},
                "short_targets": {}, "wheel_targets": {}}
        s = sim.Simulator(m, [_session(DAYS[0])], slippage_bps=0.0)
        tr = s.run(FixedSchedule(both, survivor_blacklist=False)).trades
        self.assertEqual(len(tr), 1)

    def test_survivor_blacklist_follows_the_fleet(self):
        m = _market({"S": _bars(DAYS[:1], _flat())})
        m.sym["S"]["rsi"][_idx(m, "S", (10, 0))] = 30.0
        tg = {"survivor_targets": {"S": 0.68}, "trend_targets": {}, "short_targets": {},
              "wheel_targets": {"S": 0.8}}
        self.assertEqual(self._run(m, tg, days=DAYS[:1]).trades, [])

    def test_trend_short_only_via_elif_unless_both_directions(self):
        # day 2, so the 21-bar slow EMA has warmed up
        m = _market({"S": _bars(DAYS, _flat(50.0))})
        i = _idx(m, "S", (10, 0), d=DAYS[1])
        m.sym["S"]["adx"][i] = 30.0
        m.sym["S"]["bear_cross"][i] = True
        tg = {"trend_targets": {"S": 0.68}, "short_targets": {"S": 0.68},
              "survivor_targets": {}, "wheel_targets": {}}
        s = sim.Simulator(m, [_session(d) for d in DAYS], slippage_bps=0.0)
        self.assertEqual(s.run(FixedSchedule(tg)).trades, [])
        tr = s.run(FixedSchedule(tg, both_directions=True)).trades
        self.assertEqual(tr[0].side, "short")

    def test_empty_targets_trade_nothing(self):
        m = _market({"S": _bars(DAYS[:1], _flat())})
        m.sym["S"]["rsi"][:] = 20.0
        self.assertEqual(self._run(m, schedule.EMPTY, days=DAYS[:1]).trades, [])


class AnalysisTest(unittest.TestCase):
    def test_llm_outage_runs_are_named(self):
        r = runs()[0]
        for c in r.candidates:
            c.ai_error = True
        out = analysis.llm_outage_runs([r], date(2026, 9, 14), date(2026, 9, 14))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ai_error_pct"], 100.0)
        self.assertEqual(analysis.llm_outage_runs(runs(), date(2026, 9, 14), date(2026, 9, 14)), [])

    def test_identical_arms_differ_by_zero(self):
        eq = pd.Series([100_100.0, 100_050.0, 100_300.0] * 5,
                       index=[date(2026, 9, 1) + timedelta(days=k) for k in range(15)])
        r = sim.SimResult("x", [], eq, 100_000.0)
        bs = analysis.block_bootstrap_diff(r, r)
        self.assertEqual((bs["mean"], bs["lo"], bs["hi"]), (0.0, 0.0, 0.0))

    def test_null_percentile(self):
        self.assertEqual(analysis.null_percentile(5, [1, 2, 3, 10]), 75.0)
        self.assertTrue(np.isnan(analysis.null_percentile(5, [])))

    def test_forward_returns_are_direction_signed(self):
        r = [x for x in runs() if x.published]
        days = [date(2026, 9, 14), date(2026, 9, 15)]

        def up(t):
            p = 100.0 if t.date() == days[0] else 110.0
            return (p, p, p, p)
        m = sim.Market({s: _bars(days, up) for s in ("AAA", "EEE")})
        fwd = analysis.candidate_forward_returns(r, m, [_session(d) for d in days],
                                                 days[0], days[1], horizons=(1,))
        got = dict(zip(fwd["symbol"], fwd["fwd1"]))
        self.assertAlmostEqual(got["AAA"], 0.10)    # long bucket, rose 10%
        self.assertAlmostEqual(got["EEE"], -0.10)   # short bucket, rose 10%: a loss


class TopNTest(unittest.TestCase):
    def test_ranking_uses_only_prior_sessions(self):
        days = pd.bdate_range("2026-08-01", periods=30).date
        def frame(vol_fn):
            idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days])
            return pd.DataFrame({"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0,
                                 "volume": [vol_fn(d) for d in days]}, index=idx)
        spike_day = days[25]
        daily = {"STEADY": frame(lambda d: 1e6),
                 "SPIKE": frame(lambda d: 1e9 if d == spike_day else 1e3)}
        from backtest import data
        lists = data.topn_lists(daily, [spike_day, days[26]], n=1)
        self.assertEqual(lists[spike_day], ["STEADY"])   # the spike is not known yet
        self.assertEqual(lists[days[26]], ["SPIKE"])


class EndToEndTest(unittest.TestCase):
    """Drives run_backtest.main with Alpaca stubbed out. The pieces are tested
    above; this is the caller, which is where a contract mismatch between
    them would hide (fleet CLAUDE.md rule 25: stub the dependency and drive
    the check anyway)."""

    def test_full_run_writes_report_and_artifacts(self):
        import tempfile
        from unittest import mock
        from backtest import data, run_backtest

        days = list(pd.bdate_range("2026-08-10", "2026-09-18").date)
        rng = np.random.default_rng(11)
        syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "WWW", "SPY", "TOP1", "TOP2"]
        walk = {}
        for s_ in syms:
            steps = rng.normal(0, 0.004, len(days) * 26)
            walk[s_] = 100 * np.exp(np.cumsum(steps))

        def bars15(symbols, *a, **k):
            out = {}
            for s_ in symbols:
                it = iter(walk[s_])
                def fn(t, it=it):
                    p = next(it)
                    return (p, p * 1.002, p * 0.998, p)
                out[s_] = _bars(days, fn) if s_ in walk else pd.DataFrame()
            return out

        def daily(symbols, *a, **k):
            idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days])
            return {s_: pd.DataFrame({"open": 50.0, "high": 50.0, "low": 50.0,
                                      "close": walk[s_][::26][:len(days)],
                                      "volume": 1e6 if s_.startswith("TOP") else 1e3}, index=idx)
                    for s_ in symbols if s_ in walk}

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "scout_log.txt"
            log.write_text(LOG, encoding="utf-8")
            with mock.patch.object(data, "sessions", lambda a, b: [_session(d) for d in days if a <= d <= b]), \
                 mock.patch.object(data, "equity_universe", lambda: ["TOP1", "TOP2"]), \
                 mock.patch.object(data, "bars_daily", daily), \
                 mock.patch.object(data, "bars_15m", bars15):
                rc = run_backtest.main(["--scout-log", str(log), "--start", "2026-09-14",
                                        "--end", "2026-09-15", "--top-n", "2", "--null-draws", "3",
                                        "--allow-unverified-rules", "--fleet-repo", str(FLEET),
                                        "--out", str(Path(tmp) / "out")])
            self.assertEqual(rc, 0)
            out = Path(tmp) / "out"
            report = (out / "report.md").read_text(encoding="utf-8")
            for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
                self.assertIn(section, report)
            for arm in ("A_top2", "B_all_scanner_time", "B_all_publish_time",
                        "C_llm_flat_conf", "C_llm"):
                self.assertIn(arm, report)
                self.assertTrue((out / f"trades_{arm}.csv").exists())
            self.assertEqual(len(pd.read_csv(out / "null_draws.csv")), 3)


class NoLimitWithStartTest(unittest.TestCase):
    """trading-bot-fleet rule 12: `limit` beside `start` returns the OLDEST bars."""

    def test_bar_requests_never_pass_limit(self):
        tree = ast.parse((HERE / "backtest" / "data.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "StockBarsRequest":
                self.assertNotIn("limit", [k.arg for k in node.keywords])


@unittest.skipUnless(importlib.util.find_spec("ta"), "ta not installed")
class IndicatorParityTest(unittest.TestCase):
    def test_matches_ta(self):
        from ta.momentum import RSIIndicator
        from ta.trend import ADXIndicator, EMAIndicator
        from backtest import indicators as I
        rng = np.random.default_rng(3)
        c = pd.Series(100 + rng.normal(0, 1, 500).cumsum())
        h, l = c + rng.uniform(0, 1, 500), c - rng.uniform(0, 1, 500)
        pd.testing.assert_series_equal(I.ema(c, 9), EMAIndicator(c, 9).ema_indicator(), check_names=False)
        pd.testing.assert_series_equal(I.rsi(c, 14), RSIIndicator(c, 14).rsi(), check_names=False)
        diff = (I.adx(h, l, c, 14) - ADXIndicator(h, l, c, 14).adx()).iloc[-300:].abs().max()
        self.assertLess(diff, 1e-3)


@unittest.skipUnless((FLEET / "trend_bot.py").exists(), "trading-bot-fleet checkout not found")
class RulesParityTest(unittest.TestCase):
    def test_pinned_rules_match_fleet_source(self):
        self.assertEqual(rules.verify_against_fleet(FLEET), [])


if __name__ == "__main__":
    unittest.main()
