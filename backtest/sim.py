"""Bar-replay simulator for trend_bot + survivor_bot against a TargetSchedule.

One shared simulated account, both bots, their live entry/exit rules
(backtest.rules), and whatever target list the schedule says the Beelink
held at each moment. Everything that is not the target list is held
constant across arms, so differences between arms are differences the target
list caused.

Clock. The live bots cycle every 60s during 09:30-16:00 ET on 15m bars. The
simulator steps once per 15m bar:

  at t (a bar START, 09:30 ... last bar before the close):
    1. decide from the bars COMPLETED by t (start <= t - 15m) - what a live
       cycle at t reads - exits first, then entries (entries only before
       14:00, fleet_bot.is_eod_skip_entry);
    2. fill those orders at the OPEN of the bar starting at t, plus slippage;
    3. run stop-loss / take-profit through that bar's high/low. The live bots
       check a live quote every minute; intrabar high/low is the 15m-bar
       equivalent. A bar that OPENS through a level fills at the open (a gap
       is not a fill at the stop). If one bar spans both levels the stop is
       assumed to fill first - the conservative reading.

What is NOT modelled, identically in every arm (see README "Limitations"):
market regime / VIX gating and sizing (a neutral SIDEWAYS / VIX 18 is used),
CFO dynamic reallocation, CAPITAL_CRUNCH, ownership conflicts with wheel_bot,
failed-order cooldowns, borrow cost on shorts, and partial fills.
"""
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import indicators, rules
from .schedule import ET

BAR = pd.Timedelta(minutes=15)
BAR_NS = BAR.value
STALE_MAX_AGE_S = 45 * 60                  # utils.bars_are_fresh: 15m x 3
PRIOR_SESSION_GAP_S = 4.5 * 24 * 3600      # utils.PRIOR_SESSION_GAP_SECONDS


@dataclass
class Position:
    bot: str
    symbol: str
    side: str            # "long" | "short"
    qty: int
    entry_price: float
    entry_time: datetime
    confidence: float
    run_id: int
    entry_type: str


@dataclass
class Trade:
    bot: str
    symbol: str
    side: str
    qty: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    ret: float
    reason: str
    confidence: float
    run_id: int
    entry_type: str

    @property
    def hold_hours(self):
        return (self.exit_time - self.entry_time).total_seconds() / 3600.0

    @property
    def capital_days(self):
        """Entry notional x calendar days held: the capital this trade tied
        up, overnight included. The denominator for exposure-adjusted return."""
        return abs(self.qty * self.entry_price) * max(self.hold_hours, 0.25) / 24.0


@dataclass
class SimResult:
    arm: str
    trades: list
    daily_equity: pd.Series          # equity at each session close
    start_equity: float
    params: dict = field(default_factory=dict)


class Market:
    """Per-symbol numpy views of 15m bars with the bots' indicators attached.

    Indicators are computed over the whole cached series, not re-computed on
    the bots' trailing 200/500-bar window at every step; see
    backtest.indicators for why the two agree.
    """

    def __init__(self, bars):
        self.sym = {}
        for s, df in bars.items():
            if df is None or df.empty:
                continue
            df = df.sort_index()
            d = indicators.add_trend_indicators(df, rules.TREND["FAST_EMA"], rules.TREND["SLOW_EMA"])
            d = indicators.add_rsi(d, rules.SURVIVOR["RSI_WINDOW"])
            fast, slow = d["ema_fast"], d["ema_slow"]
            above = (fast > slow)
            below = (fast < slow)
            nb = rules.TREND["MOMENTUM_BARS"] + 1
            d["bull_cross"] = above & (fast.shift(1) <= slow.shift(1))
            d["bear_cross"] = below & (fast.shift(1) >= slow.shift(1))
            d["aligned_long"] = above.astype(float).rolling(nb).min() == 1
            d["aligned_short"] = below.astype(float).rolling(nb).min() == 1
            d["pullback"] = (d["close"] - fast).abs() / fast
            idx = d.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            self.sym[s] = dict(
                # pandas 3 defaults datetimes to microsecond resolution, so asi8
                # is NOT nanoseconds unless the unit is pinned. Every lookup
                # below compares against Timestamp.value, which always is.
                t=idx.tz_convert("UTC").as_unit("ns").asi8,
                **{c: d[c].to_numpy(dtype=float, copy=True) for c in
                   ("open", "high", "low", "close", "ema_fast", "ema_slow", "adx", "rsi", "pullback")},
                **{c: d[c].to_numpy(dtype=bool, copy=True) for c in
                   ("bull_cross", "bear_cross", "aligned_long", "aligned_short")},
            )

    def has(self, s):
        return s in self.sym

    def latest(self, s, t_ns):
        """Index of the newest bar COMPLETED by t, or -1."""
        a = self.sym.get(s)
        if a is None:
            return -1
        return int(np.searchsorted(a["t"], t_ns - BAR_NS, side="right")) - 1

    def bar_at(self, s, t_ns):
        a = self.sym.get(s)
        if a is None:
            return -1
        i = int(np.searchsorted(a["t"], t_ns, side="left"))
        return i if i < len(a["t"]) and a["t"][i] == t_ns else -1

    def fill_price(self, s, t_ns):
        """Open of the bar starting at t, else the latest close (IEX bars are
        sparse; a market order still fills)."""
        i = self.bar_at(s, t_ns)
        if i >= 0:
            return self.sym[s]["open"][i]
        j = self.latest(s, t_ns)
        return self.sym[s]["close"][j] if j >= 0 else None

    def mark(self, s, t_ns):
        j = self.latest(s, t_ns + BAR_NS)   # include the bar starting at t
        return self.sym[s]["close"][j] if j >= 0 else None


def _fresh(market, s, i, t, session_open):
    """utils.bars_are_fresh with the session-open widening."""
    if i < 0:
        return False
    age = (t.value - market.sym[s]["t"][i]) / 1e9
    elapsed = (t - session_open).total_seconds()
    max_age = STALE_MAX_AGE_S
    if 0 <= elapsed < max_age:
        max_age = elapsed + PRIOR_SESSION_GAP_S
    return age <= max_age


class Simulator:
    def __init__(self, market, sessions, start_equity=100_000.0, slippage_bps=5.0,
                 seed=0, regime="SIDEWAYS", vix=18.0):
        """sessions: list of (open_ts, close_ts) tz-aware pandas Timestamps."""
        self.m = market
        self.sessions = sessions
        self.start_equity = float(start_equity)
        self.slip = slippage_bps / 1e4
        self.seed = seed
        self.regime = regime
        self.vix = vix

    # -- account -------------------------------------------------------------
    def _equity(self, t_ns):
        eq = self.cash
        for p in self.pos.values():
            px = self.m.mark(p.symbol, t_ns)
            px = p.entry_price if px is None else px
            eq += p.qty * px if p.side == "long" else -p.qty * px
        return eq

    # Budget and exposure are read at decision time t, from bars completed by
    # t - never from the bar the order is about to fill in.
    def _bot_used(self, bot, t_ns):
        used = 0.0
        for p in self.pos.values():
            if p.bot == bot:
                px = self.m.mark(p.symbol, t_ns - BAR_NS) or p.entry_price
                used += abs(p.qty * px)
        return used

    def _short_exposure(self, t_ns):
        return sum(abs(p.qty * (self.m.mark(p.symbol, t_ns - BAR_NS) or p.entry_price))
                   for p in self.pos.values() if p.bot == "trend_bot" and p.side == "short")

    def _open(self, bot, s, side, qty, raw_px, t, conf, run_id, entry_type):
        px = raw_px * (1 + self.slip) if side == "long" else raw_px * (1 - self.slip)
        self.cash += -qty * px if side == "long" else qty * px
        self.pos[s] = Position(bot, s, side, qty, px, t.to_pydatetime(), conf, run_id, entry_type)

    def _close(self, s, raw_px, t, reason):
        p = self.pos.pop(s)
        if p.side == "long":
            px = raw_px * (1 - self.slip)
            self.cash += p.qty * px
            pnl = p.qty * (px - p.entry_price)
            ret = px / p.entry_price - 1
        else:
            px = raw_px * (1 + self.slip)
            self.cash -= p.qty * px
            pnl = p.qty * (p.entry_price - px)
            ret = p.entry_price / px - 1
        self.trades.append(Trade(p.bot, s, p.side, p.qty, p.entry_time, p.entry_price,
                                 t.to_pydatetime(), px, pnl, ret, reason, p.confidence,
                                 p.run_id, p.entry_type))

    def _size(self, bot, price, conf, risk, max_pos_pct, size_mult, equity, t_ns):
        """fleet_bot.size_position: confidence-scaled, clipped to the bot's
        remaining CFO budget and its max position share."""
        budget = rules.BASE_ALLOCATIONS[bot] * (1 - rules.UNALLOCATED_RESERVE) * equity
        available = budget - self._bot_used(bot, t_ns)
        if available <= 0:
            return 0
        risk_amt = min(equity * risk * (0.5 + conf) * size_mult, available)
        qty = int(risk_amt / price)
        return max(0, min(qty, int(equity * max_pos_pct / price)))

    # -- per-step logic ------------------------------------------------------
    def _manage(self, p, t, t_ns, tstr):
        """Decision-time exits for one position, in the bots' order: max-hold
        backstop, stop/target, signal exit, then the tiered-hold EOD policy."""
        a = self.m.sym.get(p.symbol)
        i = self.m.latest(p.symbol, t_ns)
        if a is None or i < 0:
            return None
        price = a["close"][i]
        ok = _fresh(self.m, p.symbol, i, t, self._session_open)
        long = p.side == "long"
        pnl_pct = (price - p.entry_price) / p.entry_price if long \
            else (p.entry_price - price) / p.entry_price
        hours = (t.to_pydatetime() - p.entry_time).total_seconds() / 3600.0

        if p.bot == "trend_bot":
            ind = {}
            if ok and not np.isnan(a["adx"][i]):
                intact = a["ema_fast"][i] > a["ema_slow"][i] if long else a["ema_fast"][i] < a["ema_slow"][i]
                ind = {"adx": a["adx"][i], "ema_trend_intact": bool(intact)}
            stop, tp = rules.TREND["STOP_LOSS"], rules.TREND["TAKE_PROFIT"]
        else:
            ind = {"rsi": a["rsi"][i]} if ok and not np.isnan(a["rsi"][i]) else {}
            stop, tp = rules.SURVIVOR_STOP_LOSS, rules.SURVIVOR_TAKE_PROFIT

        tier = rules.hold_tier(p.bot, rules.hold_score(p.bot, pnl_pct, ind, hours, self.regime, self.vix))
        max_days = rules.MAX_HOLD_DAYS.get(tier)
        if max_days is not None and hours / 24.0 >= max_days:
            return "Max Hold Exceeded"

        if p.bot == "trend_bot":
            if pnl_pct <= stop:
                return "Stop Loss"
            if pnl_pct >= tp:
                return "Take Profit"
            if ok and long and a["bear_cross"][i] and not a["bull_cross"][i]:
                return "Bearish Crossover"
            if ok and not long and a["bull_cross"][i] and not a["bear_cross"][i]:
                return "Bullish Crossover"
        else:
            if ok and not np.isnan(a["rsi"][i]) and a["rsi"][i] > rules.SURVIVOR["RSI_SELL"]:
                return "RSI Overbought"
            if pnl_pct > tp:
                return "Take Profit (+5%)"
            if pnl_pct < stop:
                return "Stop Loss (-3%)"

        held_overnight = tstr >= rules.EOD_EVAL_FROM and tier != "CLOSE_EOD"
        if tstr >= rules.EOD_CLOSE_FROM and not held_overnight:
            return "EOD Liquidation"
        return None

    def _intrabar(self, p, t_ns, t):
        """Stop / target through the bar starting at t. Returns True if closed."""
        a = self.m.sym.get(p.symbol)
        i = self.m.bar_at(p.symbol, t_ns)
        if a is None or i < 0:
            return False
        o, h, l = a["open"][i], a["high"][i], a["low"][i]
        e = p.entry_price
        if p.bot == "trend_bot":
            sl, tp = rules.TREND["STOP_LOSS"], rules.TREND["TAKE_PROFIT"]
        else:
            sl, tp = rules.SURVIVOR_STOP_LOSS, rules.SURVIVOR_TAKE_PROFIT
        if p.side == "long":
            stop_px, tp_px = e * (1 + sl), e * (1 + tp)
            if o <= stop_px:
                self._close(p.symbol, o, t, "Stop Loss (gap)")
            elif o >= tp_px:
                self._close(p.symbol, o, t, "Take Profit (gap)")
            elif l <= stop_px:
                self._close(p.symbol, stop_px, t, "Stop Loss")
            elif h >= tp_px:
                self._close(p.symbol, tp_px, t, "Take Profit")
            else:
                return False
        else:
            stop_px, tp_px = e * (1 - sl), e * (1 - tp)
            if o >= stop_px:
                self._close(p.symbol, o, t, "Stop Loss (gap)")
            elif o <= tp_px:
                self._close(p.symbol, o, t, "Take Profit (gap)")
            elif h >= stop_px:
                self._close(p.symbol, stop_px, t, "Stop Loss")
            elif l <= tp_px:
                self._close(p.symbol, tp_px, t, "Take Profit")
            else:
                return False
        return True

    def _survivor_entries(self, targets, sched, t, t_ns, equity, rng):
        cand = dict(targets.get("survivor_targets", {}))
        if sched.survivor_blacklist:
            black = set(targets.get("trend_targets", {})) | set(targets.get("wheel_targets", {}))
            cand = {s: c for s, c in cand.items() if s not in black}
        syms = list(cand)
        rng.shuffle(syms)
        R = rules.SURVIVOR
        for s in syms:
            if s in self.pos or s in self._exited_now or not self.m.has(s):
                continue
            a = self.m.sym[s]
            i = self.m.latest(s, t_ns)
            if not _fresh(self.m, s, i, t, self._session_open) or np.isnan(a["rsi"][i]):
                continue
            price, r = a["close"][i], a["rsi"][i]
            if price < R["MIN_PRICE"] or r >= R["RSI_BUY"]:
                continue
            # Gate is (SMA200 uptrend OR scout-approved); every symbol scanned
            # here IS on the list, so the SMA200 branch never decides.
            qty = self._size("survivor_bot", price, cand[s], R["RISK_PER_TRADE"],
                             R["MAX_POSITION_PCT"], 1.0, equity, t_ns)
            fill = self.m.fill_price(s, t_ns)
            if qty > 0 and fill:
                self._open("survivor_bot", s, "long", qty, fill, t, cand[s],
                           sched.run_at(t), "Dip")

    def _trend_entries(self, targets, sched, t, t_ns, equity, rng):
        longs = targets.get("trend_targets", {})
        shorts = targets.get("short_targets", {})
        syms = list(set(longs) | set(shorts))
        rng.shuffle(syms)
        R = rules.TREND
        short_exp = self._short_exposure(t_ns)
        for s in syms:
            if s in self.pos or s in self._exited_now or not self.m.has(s):
                continue
            a = self.m.sym[s]
            i = self.m.latest(s, t_ns)
            if i < 1 or not _fresh(self.m, s, i, t, self._session_open) or np.isnan(a["adx"][i]) \
                    or np.isnan(a["ema_slow"][i - 1]):
                continue
            if s in longs:
                sides = ["long", "short"] if (sched.both_directions and s in shorts) else ["long"]
            else:
                sides = ["short"]
            for side in sides:
                opened = self._try_trend(s, side, a, i, longs if side == "long" else shorts,
                                         sched, t, t_ns, equity, short_exp)
                if opened:
                    if side == "short":
                        short_exp += opened
                    break

    def _try_trend(self, s, side, a, i, tmap, sched, t, t_ns, equity, short_exp):
        R = rules.TREND
        price, adx = a["close"][i], a["adx"][i]
        long = side == "long"
        if price < (R["MIN_PRICE_LONG"] if long else R["MIN_PRICE_SHORT"]):
            return 0
        if not long and short_exp >= equity * R["MAX_SHORT_EXPOSURE"]:
            return 0
        if adx <= rules.TREND_ADX_MIN:
            return 0
        size_mult = 1.0
        if long and self.regime == "BEAR_TREND":
            size_mult = 0.5
        elif not long and self.regime == "BULL_TREND":
            size_mult = 0.5
        cross = a["bull_cross"][i] if long else a["bear_cross"][i]
        aligned = a["aligned_long"][i] if long else a["aligned_short"][i]
        momentum = aligned and a["pullback"][i] <= R["MOMENTUM_PULLBACK_PCT"] and adx > R["MOMENTUM_ADX_MIN"]
        if cross:
            entry_type = "Crossover"
        elif momentum:
            entry_type = "Momentum"
            size_mult *= R["MOMENTUM_SIZE_MULT"]
        else:
            return 0
        conf = tmap.get(s, 0.5)
        qty = self._size("trend_bot", price, conf, R["RISK_PER_TRADE"], R["MAX_POSITION_PCT"],
                         size_mult, equity, t_ns)
        fill = self.m.fill_price(s, t_ns)
        if qty <= 0 or not fill:
            return 0
        self._open("trend_bot", s, side, qty, fill, t, conf, sched.run_at(t), entry_type)
        return qty * fill

    # -- main loop -----------------------------------------------------------
    def run(self, sched):
        self.cash = self.start_equity
        self.pos = {}
        self.trades = []
        rng = random.Random(self.seed)
        closes = {}
        for session_open, session_close in self.sessions:
            self._session_open = session_open
            t = session_open
            while t < session_close:
                t_ns = t.value
                tstr = t.tz_convert(ET).strftime("%H:%M")
                equity = self._equity(t_ns - BAR_NS)
                # 1-2. decide at t from completed bars, fill at this bar's open.
                # A symbol sold this step is not re-bought in the same step:
                # live, the sell is still settling when the next cycle looks.
                self._exited_now = set()
                for s in list(self.pos):
                    reason = self._manage(self.pos[s], t, t_ns, tstr)
                    if reason:
                        px = self.m.fill_price(s, t_ns)
                        if px:
                            self._close(s, px, t, reason)
                            self._exited_now.add(s)
                if tstr < rules.NO_ENTRY_AFTER:
                    targets = sched.at(t.to_pydatetime())
                    bots = [self._trend_entries, self._survivor_entries]
                    rng.shuffle(bots)
                    for enter in bots:
                        enter(targets, sched, t, t_ns, equity, rng)
                # 3. intrabar stops / targets
                for s in list(self.pos):
                    self._intrabar(self.pos[s], t_ns, t)
                t = t + BAR
            closes[session_close.tz_convert(ET).date()] = self._equity(session_close.value - BAR_NS)
        # Mark whatever is still open at the last close, labelled as such.
        if self.sessions:
            end = self.sessions[-1][1]
            for s in list(self.pos):
                px = self.m.mark(s, end.value - BAR_NS)
                if px:
                    # reverse the slippage _close applies: an unrealised mark
                    # is not a fill
                    p = self.pos[s]
                    adj = px / (1 - self.slip) if p.side == "long" else px / (1 + self.slip)
                    self._close(s, adj, end, "open_at_end")
        return SimResult(sched.name, self.trades, pd.Series(closes, dtype=float),
                         self.start_equity,
                         dict(slippage_bps=self.slip * 1e4, seed=self.seed,
                              regime=self.regime, vix=self.vix))
