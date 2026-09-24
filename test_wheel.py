"""Regression tests for the wheel backtest (backtest/wheel*.py, options_data).

The wheel's P&L is decided by a handful of events - a fill, an expiry, an
assignment, a roll, a gate - so each one is pinned on synthetic chains where
the right answer is known in advance. The failure modes guarded against are
the flattering ones: a fill at a price that never traded, a contract chosen
with tomorrow's price, a gated day that still sold a put, an assignment that
never happened.
"""
import os
import sys
import tempfile
import types
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backtest import options_data as O
from backtest import rules
from backtest import wheel as WH

HERE = Path(__file__).resolve().parent
FLEET = Path(os.environ.get("FLEET_REPO", HERE.parent / "trading-bot-fleet"))

DAYS = list(pd.bdate_range("2025-01-06", "2025-04-30").date)


def occ(u, exp, kind, strike):
    return f"{u}{exp:%y%m%d}{kind}{int(round(strike * 1000)):08d}"


def fridays(a, b):
    d = a
    while d.weekday() != 4:
        d += timedelta(days=1)
    while d <= b:
        yield d
        d += timedelta(days=7)


def make_chain(u, strikes=range(80, 121)):
    rows = [(occ(u, e, k, s), k, float(s), e) for e in fridays(date(2025, 1, 1), date(2025, 7, 31))
            for k in "PC" for s in strikes]
    return pd.DataFrame(rows, columns=O.CHAIN_COLUMNS)


class Book(WH.OptionBook):
    """Synthetic book: every contract trades every day at `price_fn(sym, day)`
    unless it returns None."""

    def __init__(self, chains, price_fn):
        super().__init__(lambda u, d: chains.get(u, pd.DataFrame(columns=O.CHAIN_COLUMNS)), None)
        self.price_fn = price_fn

    def bar(self, sym, day):
        px = self.price_fn(sym, day)
        return None if px is None else dict(vwap=px, close=px)


def closes(price_fn, days=DAYS):
    return pd.Series([price_fn(d) for d in days], index=days, dtype=float)


def sim(book, prices, cands, params=None, capital=50_000, regime=None, vix=None):
    params = params or WH.WheelParams(slip=0.0, fee=0.0)
    regime = regime if regime is not None else pd.Series("SIDEWAYS", index=DAYS, dtype=object)
    vix = vix if vix is not None else pd.Series(15.0, index=DAYS)
    return WH.WheelSim(DAYS, prices, book, cands, params, capital, regime, vix).run()


def always(u, days=DAYS):
    return {d: [u] for d in days}


class PickContractTest(unittest.TestCase):
    def test_strictly_otm_closest_to_target_in_window(self):
        ch = make_chain("AAA")
        c = WH.pick_contract(ch, "P", 100.0, 0.05, date(2025, 1, 7), 25, 45)
        self.assertEqual(c["strike"], 95.0)
        self.assertEqual(c["type"], "P")
        dte = (c["expiry"] - date(2025, 1, 7)).days
        self.assertTrue(25 <= dte <= 45)
        self.assertEqual(c["expiry"], min(e for e in fridays(date(2025, 2, 1), date(2025, 2, 21))))
        call = WH.pick_contract(ch, "C", 100.0, 0.05, date(2025, 1, 7), 25, 45)
        self.assertEqual(call["strike"], 105.0)
        # an at-the-money strike is never OTM
        self.assertIsNone(WH.pick_contract(ch[ch["strike"] == 100.0], "P", 100.0, 0.05,
                                           date(2025, 1, 7), 25, 45))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.chains = {"AAA": make_chain("AAA")}

    def test_put_fills_next_day_at_vwap_less_slip_then_rolls(self):
        prices = {"AAA": closes(lambda d: 100.0)}
        book = Book(self.chains, lambda s, d: 1.00)
        cands = {DAYS[0]: ["AAA"]}
        p = WH.WheelParams(slip=0.05, fee=0.05, take_profit=None)
        res = sim(book, prices, cands, p)
        ev = [e for e in res.events if e.kind == "sell_put"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].day, DAYS[1])                  # decided at close t, filled t+1
        self.assertAlmostEqual(ev[0].price, 0.95)             # VWAP less 5%
        # rolled at DTE <= 10 while OTM (the fleet rolls, it does not let it ride)
        self.assertIn("roll_open", [e.kind for e in res.events])

    def test_no_trade_on_fill_day_means_no_position(self):
        prices = {"AAA": closes(lambda d: 100.0)}
        book = Book(self.chains, lambda s, d: None)
        res = sim(book, prices, {DAYS[0]: ["AAA"]})
        self.assertEqual(res.counts().get("sell_put", 0), 0)
        self.assertGreaterEqual(res.counts().get("skip_no_trade", 0), 1)
        self.assertEqual(float(res.equity.iloc[-1]), 50_000.0)

    def test_assignment_then_covered_call_even_when_gated(self):
        # stock collapses below the strike and stays there; the option never
        # trades after the sale, so neither the backstop nor a roll can fill
        sold = {}

        def opt_px(sym, d):
            if sym in sold:
                return None
            if d == DAYS[1]:
                sold[sym] = True
                return 1.0
            return 1.0 if sym[9] == "C" else None
        prices = {"AAA": closes(lambda d: 100.0 if d <= DAYS[1] else 80.0)}
        book = Book(self.chains, opt_px)
        regime = pd.Series("BEAR_TREND", index=DAYS, dtype=object)
        regime.iloc[0] = "SIDEWAYS"                            # only day 0 allows a put
        res = sim(book, prices, always("AAA"), regime=regime)
        kinds = [e.kind for e in res.events]
        self.assertEqual(kinds.count("sell_put"), 1)
        self.assertIn("assigned", kinds)
        a = kinds.index("assigned")
        self.assertIn("sell_call", kinds[a:])                  # gate-exempt covered call

    def test_take_profit_at_half_the_premium(self):
        sold_day = DAYS[1]

        def opt_px(sym, d):
            return 2.0 if d <= sold_day else 0.9
        prices = {"AAA": closes(lambda d: 100.0)}
        res = sim(Book(self.chains, opt_px), prices, {DAYS[0]: ["AAA"]})
        close = [e for e in res.events if e.kind == "close"]
        self.assertEqual(close[0].note, "take profit")
        self.assertEqual(close[0].day, DAYS[3])                # mark 0.9 at DAYS[2] close -> fill DAYS[3]
        self.assertAlmostEqual(float(res.equity.iloc[-1]), 50_000 + (2.0 - 0.9) * 100)

    def test_itm_near_expiry_is_closed_not_assigned(self):
        prices = {"AAA": closes(lambda d: 100.0 if d <= DAYS[15] else 90.0)}
        p = WH.WheelParams(slip=0.0, fee=0.0, take_profit=None, stale_roll_dte=0)
        res = sim(Book(self.chains, lambda s, d: 1.0), prices, {DAYS[0]: ["AAA"]}, p)
        closes_ = [e for e in res.events if e.kind == "close"]
        self.assertTrue(closes_ and closes_[0].note == "expiry backstop (ITM)")
        self.assertNotIn("assigned", [e.kind for e in res.events])

    def test_roll_aborts_when_the_new_leg_has_no_trade(self):
        first = {}

        def opt_px(sym, d):
            first.setdefault(sym, d)
            return 1.0 if first[sym] == DAYS[1] else None      # only the first contract ever trades
        prices = {"AAA": closes(lambda d: 100.0)}
        p = WH.WheelParams(slip=0.0, fee=0.0, take_profit=None)
        res = sim(Book(self.chains, opt_px), prices, {DAYS[0]: ["AAA"]}, p)
        kinds = [e.kind for e in res.events]
        self.assertIn("roll_aborted", kinds)
        self.assertNotIn("roll_open", kinds)
        self.assertIn("expired", kinds)                        # rode to expiry OTM


class GateTest(unittest.TestCase):
    def setUp(self):
        self.book = Book({"AAA": make_chain("AAA")}, lambda s, d: 1.0)
        self.prices = {"AAA": closes(lambda d: 100.0)}

    def test_bear_regime_and_vix_block_new_puts(self):
        for regime, vix in (("BEAR_TREND", 15.0), ("SIDEWAYS", 23.0)):
            res = sim(self.book, self.prices, always("AAA"),
                      regime=pd.Series(regime, index=DAYS, dtype=object),
                      vix=pd.Series(vix, index=DAYS))
            self.assertEqual(res.counts().get("sell_put", 0), 0, (regime, vix))
        off = sim(self.book, self.prices, always("AAA"),
                  WH.WheelParams(slip=0.0, fee=0.0, gates=False),
                  regime=pd.Series("BEAR_TREND", index=DAYS, dtype=object))
        self.assertGreater(off.counts().get("sell_put", 0), 0)

    def test_vix_above_28_stops_management(self):
        prices = {"AAA": closes(lambda d: 100.0)}
        book = Book({"AAA": make_chain("AAA")}, lambda s, d: 2.0 if d <= DAYS[1] else 0.5)
        vix = pd.Series(15.0, index=DAYS)
        vix.iloc[2:6] = 30.0                                   # stopped while the profit is there
        res = sim(book, prices, {DAYS[0]: ["AAA"]}, vix=vix)
        close = [e for e in res.events if e.kind == "close"][0]
        self.assertEqual(close.day, DAYS[7])                   # decided DAYS[6], first un-paused close
        self.assertEqual(res.paused_days, 4)

    def test_collateral_budget(self):
        chains = {u: make_chain(u) for u in ("AAA", "BBB", "CCC")}
        book = Book(chains, lambda s, d: 1.0)
        prices = {u: closes(lambda d: 100.0) for u in chains}
        # $95 strike x 100 = $9,500 collateral each; $20k fits two
        res = sim(book, prices, {DAYS[0]: ["AAA", "BBB", "CCC"]}, capital=20_000)
        self.assertEqual(res.counts().get("sell_put", 0), 2)


class InputsTest(unittest.TestCase):
    def test_regime_rule(self):
        n = 80
        up = pd.DataFrame({"close": np.linspace(100, 140, n)}, index=DAYS[:n])
        up["high"], up["low"] = up["close"] + 0.5, up["close"] - 0.5
        self.assertEqual(WH.regime_series(up).iloc[-1], "BULL_TREND")
        down = up.copy()
        down["close"] = np.linspace(140, 100, n)
        down["high"], down["low"] = down["close"] + 0.5, down["close"] - 0.5
        self.assertEqual(WH.regime_series(down).iloc[-1], "BEAR_TREND")

    def test_regime_math_matches_the_scanner(self):
        sys.modules.setdefault("config", types.SimpleNamespace(API_KEY="x", SECRET_KEY="x", PAPER=True))
        try:
            import market_scanner
        except Exception as e:                                 # pragma: no cover
            self.skipTest(f"market_scanner not importable: {e}")
        rng = np.random.default_rng(2)
        c = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
        h, l = c + rng.uniform(0, 1, 300), c - rng.uniform(0, 1, 300)
        TM = market_scanner.TechnicalMath
        pd.testing.assert_series_equal(WH._adx_scanner(h.copy(), l.copy(), c), TM.get_adx(h.copy(), l.copy(), c),
                                       check_names=False)
        pd.testing.assert_series_equal(WH._rsi_scanner(c), TM.get_rsi(c), check_names=False)

    def test_wheel_candidates_rule(self):
        n = 260
        days = list(pd.bdate_range("2024-01-01", periods=n).date)
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days])
        rng = np.random.default_rng(0)
        calm = 50 + np.linspace(0, 10, n) + rng.normal(0, 0.3, n)  # above SMA200, mild
        falling = np.linspace(80, 40, n)                             # below SMA200
        def frame(c):
            return pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c,
                                 "volume": 1e6}, index=idx)
        daily = {"CALM": frame(calm), "FALL": frame(falling)}
        members = {d: ["CALM", "FALL"] for d in days}
        out = WH.wheel_candidates(daily, days[-5:], members)
        self.assertTrue(all("FALL" not in v for v in out.values()))

    def test_vix_csv(self):
        s = O.parse_vix_csv("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2025,17.2,18.1,16.9,17.9\n01/03/2025,17.9,18,16,16.5\n")
        self.assertEqual(s[date(2025, 1, 3)], 16.5)


class CoveringCacheTest(unittest.TestCase):
    def test_sub_window_reuses_a_wider_cached_file(self):
        from backtest import data
        with tempfile.TemporaryDirectory() as tmp:
            idx = pd.DatetimeIndex(pd.date_range("2020-01-01", "2020-12-31", tz="UTC"))
            df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
            p = data._cache_path(tmp, "1d_all", "AAA", datetime(2020, 1, 1), datetime(2020, 12, 31))
            p.parent.mkdir(parents=True)
            df.to_pickle(p)
            index = data._cache_index(tmp, "1d_all")
            got = data._covering_cache(index, "AAA", datetime(2020, 3, 1), datetime(2020, 3, 31))
            self.assertEqual(len(got), 31)
            self.assertIsNone(data._covering_cache(index, "AAA", datetime(2019, 3, 1),
                                                   datetime(2020, 3, 31)))

    def test_cache_directory_is_listed_once_not_per_symbol(self):
        """The first version globbed the cache per symbol: ~15,000 symbols x
        ~15,000 files, which looked like a hang on the Corsair."""
        import os as _os
        from backtest import data
        syms = [f"S{i:04d}" for i in range(3000)]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "1d_all"
            d.mkdir()
            idx = pd.DatetimeIndex(pd.date_range("2020-01-01", "2020-12-31", tz="UTC"))
            df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
            df.to_pickle(d / "S0000_20200101_20201231.pkl")
            for s_ in syms[1:]:
                (d / f"{s_}_20200101_20201231.pkl").write_bytes((d / "S0000_20200101_20201231.pkl").read_bytes())
            calls = []
            real = _os.scandir

            def counting(path):
                calls.append(path)
                return real(path)
            with mock.patch.object(data.os, "scandir", counting), \
                    mock.patch.object(data, "data_client", lambda: self.fail("fetched: the cache covered it")):
                out = data.bars_daily(syms, datetime(2020, 3, 1), datetime(2020, 3, 31), tmp,
                                      adjustment="all")
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(out), 3000)
            self.assertEqual(len(out["S2999"]), 31)


@unittest.skipUnless((FLEET / "wheel_bot.py").exists(), "trading-bot-fleet checkout not found")
class WheelRulesParityTest(unittest.TestCase):
    def test_pinned_wheel_rules_match_fleet_source(self):
        self.assertEqual(rules.verify_wheel_against_fleet(FLEET), [])


class WheelEndToEndTest(unittest.TestCase):
    """Drives wheel_research.main with Alpaca and CBOE stubbed."""

    def test_full_run(self):
        from backtest import data, wheel_research as WR
        days = list(pd.bdate_range("2023-01-02", "2025-06-30").date)
        rng = np.random.default_rng(9)
        syms = ["AAA", "BBB", "CCC", "SPY"]
        px = {s: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days)))) for s in syms}
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=4) for d in days])

        def daily(symbols, *a, **k):
            return {s: pd.DataFrame({"open": px[s], "high": px[s] * 1.01, "low": px[s] * 0.99,
                                     "close": px[s], "volume": 1e7}, index=idx)
                    for s in symbols if s in px}

        def chain(u, a, b, lo, hi, cache):
            rows = [(occ(u, e, k, s), k, float(s), e) for e in fridays(a, b) for k in "PC"
                    for s in range(int(lo), int(hi) + 1)]
            return pd.DataFrame(rows, columns=O.CHAIN_COLUMNS)

        def bars(symbols, start, end, cache):
            out = {}
            for s in symbols:
                d = [x for x in days if start <= x <= end]
                out[s] = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                       "volume": 10, "vwap": 1.0}, index=pd.Index(d))
            return out

        vix = pd.Series(18.0, index=days)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(data, "equity_universe", lambda include_inactive=False: syms[:-1]), \
                mock.patch.object(data, "bars_daily", daily), \
                mock.patch.object(O, "chain", chain), \
                mock.patch.object(O, "option_bars", bars), \
                mock.patch.object(O, "vix_history", lambda cache, csv=None: vix), \
                mock.patch.object(WH, "wheel_candidates",
                                  lambda daily, ds, members, k=10: {d: ["AAA", "BBB"] for d in ds}):
            rc = WR.main(["--start", "2024-03-01", "--end", "2025-06-30", "--allow-unverified-rules",
                          "--fleet-repo", str(FLEET), "--out", str(Path(tmp) / "out")])
            self.assertEqual(rc, 0)
            rep = (Path(tmp) / "out" / "wheel_report.md").read_text(encoding="utf-8")
            for s in ("## Verdict", "## Out-of-sample record", "## The live-like trial",
                      "## Cost sensitivity", "Deflated Sharpe"):
                self.assertIn(s, rep)
            ev = pd.read_csv(Path(tmp) / "out" / "wheel_events_live_like.csv")
            self.assertIn("sell_put", set(ev["kind"]))


if __name__ == "__main__":
    unittest.main()
