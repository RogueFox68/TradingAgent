"""Regression tests for the scanner's data sourcing and publish guard.

Written after the 2026-09 yfinance outage exposed two problems in this file:

  1. yfinance was the ONLY bar source and was imported at module scope, so a
     broken install killed the scan before it could do anything about it.
  2. A scan that found nothing still wrote dragnet_candidates.json and reported
     success. Downstream that becomes an empty active_targets.json, SCP'd over
     the Beelink's good targets — and the fleet's 24h staleness check cannot
     catch it, because the file it receives is perfectly fresh. Just empty.

Run: python -m unittest test_market_scanner -v
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

# config.py is gitignored (real keys); the scanner imports it at module scope.
sys.modules.setdefault("config", types.SimpleNamespace(
    API_KEY="k", SECRET_KEY="s", PAPER=True, WEBHOOK_OVERSEER=""))

import market_scanner as ms


def ohlcv(n=260, base=100.0, volume=5_000_000):
    idx = pd.date_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(base, base * 1.4, n), index=idx)
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": pd.Series([float(volume)] * n, index=idx),
    })


class FetchDailyBarsTests(unittest.TestCase):
    def test_prefers_yfinance_when_it_answers(self):
        bars = {"AAPL": ohlcv()}
        with mock.patch.object(ms, "_bars_from_yfinance", return_value=bars), \
             mock.patch.object(ms, "_bars_from_alpaca") as alp:
            got, source = ms.fetch_daily_bars(["AAPL"], 10)
        self.assertEqual(source, "yfinance")
        alp.assert_not_called()

    def test_falls_back_to_alpaca_on_empty_yfinance(self):
        """The 2026-09 shape: yfinance answers, with nothing in it."""
        with mock.patch.object(ms, "_bars_from_yfinance", return_value={}), \
             mock.patch.object(ms, "_bars_from_alpaca",
                               return_value={"AAPL": ohlcv()}) as alp:
            got, source = ms.fetch_daily_bars(["AAPL"], 10)
        self.assertEqual(source, "alpaca")
        self.assertIn("AAPL", got)
        alp.assert_called_once()

    def test_falls_back_when_yfinance_raises(self):
        with mock.patch.object(ms, "_bars_from_yfinance",
                               side_effect=RuntimeError("curl 403")), \
             mock.patch.object(ms, "_bars_from_alpaca",
                               return_value={"AAPL": ohlcv()}):
            got, source = ms.fetch_daily_bars(["AAPL"], 10)
        self.assertEqual(source, "alpaca")

    def test_broken_yfinance_import_is_not_fatal(self):
        with mock.patch.object(ms, "yf", None), \
             mock.patch.object(ms, "_yf_unavailable", False), \
             mock.patch("builtins.__import__", side_effect=ImportError("gone")):
            self.assertEqual(ms._bars_from_yfinance(["AAPL"], 10), {})

    def test_no_symbols_is_not_a_fetch(self):
        with mock.patch.object(ms, "_bars_from_yfinance") as yfm:
            self.assertEqual(ms.fetch_daily_bars([], 10), ({}, "none"))
            yfm.assert_not_called()


class LiquidityFilterTests(unittest.TestCase):
    def test_yfinance_path_uses_the_absolute_threshold(self):
        bars = {"BIG": ohlcv(volume=9_000_000), "SMALL": ohlcv(volume=1_000)}
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=(bars, "yfinance")):
            self.assertEqual(ms.filter_by_volume(["BIG", "SMALL"]), ["BIG"])

    def test_price_band_is_enforced(self):
        bars = {"CHEAP": ohlcv(base=2.0, volume=9_000_000),
                "OK": ohlcv(base=100.0, volume=9_000_000)}
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=(bars, "yfinance")):
            self.assertEqual(ms.filter_by_volume(["CHEAP", "OK"]), ["OK"])

    def test_alpaca_path_ranks_instead_of_thresholding(self):
        """Alpaca's free feed is IEX-only, so its Volume is a single venue's
        share. Applying MIN_VOLUME to it would reject the entire market; the
        rank-based filter is scale-invariant and needs no calibration."""
        bars = {"A": ohlcv(volume=90_000), "B": ohlcv(volume=40_000),
                "C": ohlcv(volume=1_000)}
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=(bars, "alpaca")), \
             mock.patch.object(ms, "ALPACA_LIQUIDITY_RANK", 2):
            got = ms.filter_by_volume(["A", "B", "C"])
        self.assertEqual(set(got), {"A", "B"})
        self.assertNotIn("C", got)

    def test_alpaca_path_keeps_the_seed_list(self):
        seed = ms.SEED_LIST[0]
        bars = {seed: ohlcv(volume=10), "OTHER": ohlcv(volume=90_000)}
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=(bars, "alpaca")), \
             mock.patch.object(ms, "ALPACA_LIQUIDITY_RANK", 1):
            got = ms.filter_by_volume([seed, "OTHER"])
        self.assertIn(seed, got, "institutional seeds survive the rank cut")


class PublishGuardTests(unittest.TestCase):
    """An empty scan must not overwrite the previous one."""

    def setUp(self):
        for name, val in (("is_mission_time", True),
                          ("get_alpaca_client", object())):
            p = mock.patch.object(ms, name, return_value=val)
            p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(ms, "get_market_universe", return_value=["A", "B"])
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(ms, "_alert")
        self.alert = p.start(); self.addCleanup(p.stop)

    def _run(self, results):
        with mock.patch.object(ms, "filter_by_volume", return_value=["A", "B"]), \
             mock.patch.object(ms, "analyze_technicals", return_value=results), \
             mock.patch.object(ms, "get_earnings_date", return_value=True), \
             mock.patch("builtins.open", mock.mock_open()) as op, \
             mock.patch.object(ms.json, "dump") as dump:
            try:
                ms.run_dragnet()
                raised = None
            except SystemExit as e:
                raised = e.code
        return raised, dump

    def test_empty_scan_aborts_without_writing(self):
        code, dump = self._run([])
        self.assertEqual(code, 1)
        dump.assert_not_called()
        self.assertTrue(self.alert.called, "an aborted scan must be LOUD")

    def test_thin_scan_aborts(self):
        results = [{"symbol": f"S{i}", "type": "trend_targets", "score": 30.0}
                   for i in range(ms.MIN_VIABLE_CANDIDATES - 1)]
        code, dump = self._run(results)
        self.assertEqual(code, 1)
        dump.assert_not_called()

    def test_healthy_scan_writes(self):
        results = [{"symbol": f"S{i}", "type": "trend_targets", "score": 30.0}
                   for i in range(ms.MIN_VIABLE_CANDIDATES)]
        code, dump = self._run(results)
        self.assertIsNone(code)
        dump.assert_called_once()
        self.assertFalse(self.alert.called)


class TechnicalsTests(unittest.TestCase):
    def test_no_bars_from_any_source_yields_nothing(self):
        with mock.patch.object(ms, "fetch_daily_bars", return_value=({}, "none")):
            self.assertEqual(ms.analyze_technicals(["AAPL"]), [])

    def test_short_history_is_skipped_not_fatal(self):
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=({"AAPL": ohlcv(n=50)}, "yfinance")):
            self.assertEqual(ms.analyze_technicals(["AAPL"]), [])

    def test_a_downtrend_is_a_short_target(self):
        """A price below its SMA200 is the one unambiguous bucket, so it is the
        one worth asserting: it proves the indicator maths runs end to end on a
        DataFrame in the shared bar shape, whichever provider produced it."""
        n = 260
        idx = pd.date_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(200.0, 100.0, n), index=idx)
        df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                           "Close": close,
                           "Volume": pd.Series([5e6] * n, index=idx)})
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=({"AAPL": df}, "alpaca")):
            out = ms.analyze_technicals(["AAPL"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["symbol"], "AAPL")
        self.assertEqual(out[0]["type"], "short_targets")

    def test_a_ticker_with_broken_data_does_not_abort_the_scan(self):
        """The old bare `except: continue` hid the difference between one odd
        ticker and the maths being broken for every ticker."""
        good = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0],
                             "Close": [1.0], "Volume": [1.0]})
        n = 260
        idx = pd.date_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(200.0, 100.0, n), index=idx)
        ok_df = pd.DataFrame({"Open": close, "High": close + 1,
                              "Low": close - 1, "Close": close,
                              "Volume": pd.Series([5e6] * n, index=idx)})
        with mock.patch.object(ms, "fetch_daily_bars",
                               return_value=({"BAD": good, "AAPL": ok_df},
                                             "yfinance")):
            out = ms.analyze_technicals(["BAD", "AAPL"])
        self.assertEqual([c["symbol"] for c in out], ["AAPL"])


if __name__ == "__main__":
    unittest.main()
