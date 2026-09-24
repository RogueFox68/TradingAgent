"""Daily simulator for wheel_bot: cash-secured puts, assignment, covered calls.

wheel_bot's rules (trading-bot-fleet/wheel_bot.py), restated and checked by
rules.verify_wheel_against_fleet:
  entry   sell a put ~`otm` below the price, 25-45 DTE, premium >= $0.10,
          if the collateral (strike x 100) fits the budget and the entry
          gates allow it (SPY regime not BEAR_TREND, VIX <= 22);
          on 100+ wheel-owned shares, sell a call ~`otm` above instead
          (covered calls are exempt from the gates)
  manage  take profit at 50% premium captured; an ITM contract at DTE <= 5
          is closed outright; at DTE <= 10 roll to a fresh 25-45 DTE contract
  pause   VIX > 28: the commander STOPS the process, so nothing is managed
          at all until it comes back

Clock: every decision is made from day t's CLOSE (stock close, VIX close,
option close) and filled on day t+1 at that contract's traded VWAP, less a
spread haircut `slip` (a fraction of the option price) and a per-contract
fee. Alpaca has no historical option quotes, so a contract that did not
trade on t+1 has no price, and the order does not happen - the same outcome
as wheel_bot finding the spread too wide (it skips that ticker for the
cycle). Expiry settles at intrinsic value against the underlying's close on
expiration day: an ITM put becomes 100 shares bought at the strike, an ITM
call delivers 100 shares away at the strike.

Everything is in RAW (unadjusted) prices, because strikes are. A split
during a held contract is not modelled.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import rules

CONTRACT = 100
_W = rules.WHEEL


@dataclass
class WheelParams:
    otm: float = _W["TARGET_OTM_PCT"]
    take_profit: float | None = _W["TAKE_PROFIT_PCT"]
    gates: bool = True              # regime/VIX entry gates AND the VIX>28 pause
    min_dte: int = _W["MIN_DTE"]
    max_dte: int = _W["MAX_DTE"]
    stale_roll_dte: int = _W["STALE_ROLL_DTE"]
    force_close_dte: int = _W["FORCE_CLOSE_DTE"]
    min_premium: float = _W["MIN_PREMIUM"]
    vix_gate: float = float(rules.WHEEL_VIX_GATE)
    vix_pause: float = rules.FLEET_VIX_PAUSE
    gated_regimes: tuple = rules.WHEEL_GATED_REGIMES
    slip: float = 0.05              # fraction of option price paid away per fill
    fee: float = 0.05               # $ per contract per fill


@dataclass
class OptPos:
    symbol: str
    underlying: str
    kind: str          # "P" | "C"
    strike: float
    expiry: date
    entry: float       # premium received per share, net of slippage
    opened: date
    mark: float


@dataclass
class Event:
    day: date
    kind: str          # sell_put / sell_call / close / roll_close / roll_open / assigned / called_away / expired / skip_no_trade / roll_aborted
    underlying: str
    symbol: str = ""
    price: float = 0.0
    cash: float = 0.0
    note: str = ""


class OptionBook:
    """Chains and bars, loaded lazily through callables so tests can feed
    synthetic data and the CLI can feed Alpaca."""

    def __init__(self, chain_loader, bar_loader):
        self._chain_loader = chain_loader    # (underlying, day) -> DataFrame[symbol,type,strike,expiry]
        self._bar_loader = bar_loader        # ([symbols]) -> {symbol: bars by date}
        self._chains, self._bars = {}, {}
        self.chain_loads = self.bar_loads = 0   # loader calls (cache or network)

    def chain(self, u, day):
        key = (u, day.year, (day.month - 1) // 3)
        if key not in self._chains:
            self._chains[key] = self._chain_loader(u, day)
            self.chain_loads += 1
        return self._chains[key]

    def bars(self, sym):
        if sym not in self._bars:
            self._bars.update(self._bar_loader([sym]))
            self._bars.setdefault(sym, pd.DataFrame())
            self.bar_loads += 1
        return self._bars[sym]

    def bar(self, sym, day):
        b = self.bars(sym)
        if b is None or b.empty or day not in b.index:
            return None
        row = b.loc[day]
        vwap = row.get("vwap")
        if vwap is None or not np.isfinite(vwap) or vwap <= 0:
            vwap = row.get("close")
        return dict(vwap=float(vwap), close=float(row["close"]))


def pick_contract(chain_df, kind, price, otm, exec_day, min_dte, max_dte):
    """wheel_bot.find_best_contract: in the DTE window, strictly OTM, the
    strike whose distance from the price is closest to `otm`. Ties go to
    the nearer expiry, then the lower strike (the live bot takes the API's
    first; this makes the choice deterministic)."""
    if chain_df is None or chain_df.empty:
        return None
    lo, hi = exec_day + timedelta(days=min_dte), exec_day + timedelta(days=max_dte)
    c = chain_df[(chain_df["type"] == kind) & (chain_df["expiry"] >= lo) & (chain_df["expiry"] <= hi)]
    c = c[c["strike"] < price] if kind == "P" else c[c["strike"] > price]
    if c.empty:
        return None
    score = ((price - c["strike"]).abs() / price - otm).abs()
    c = c.assign(score=score).sort_values(["score", "expiry", "strike"])
    if c.iloc[0]["score"] >= 1.0:          # live best_score starts at 1.0
        return None
    return c.iloc[0]


class WheelSim:
    def __init__(self, days, stock_close, book, candidates, params, capital,
                 regime=None, vix=None):
        """days: trading dates. stock_close: {symbol: Series by date} RAW.
        candidates: {date: [symbols]} (known at that day's close).
        regime / vix: Series by date (required when params.gates)."""
        self.days = list(days)
        self.px = stock_close
        self.book = book
        self.cands = candidates
        self.p = params
        self.capital = float(capital)
        self.regime = regime if regime is not None else pd.Series(dtype=object)
        self.vix = vix if vix is not None else pd.Series(dtype=float)
        if params.gates and (self.vix.empty or self.regime.empty):
            raise ValueError("gated wheel needs VIX and regime series")

    # -- helpers ---------------------------------------------------------------
    def _price(self, u, day):
        s = self.px.get(u)
        if s is None or s.empty:
            return None
        s = s[s.index <= day]
        return float(s.iloc[-1]) if len(s) else None

    def _gate_value(self, series, day):
        s = series[series.index <= day]
        return s.iloc[-1] if len(s) else None

    def _equity(self, day):
        eq = self.cash
        for u, n in self.shares.items():
            px = self._price(u, day)
            eq += n * (px or 0.0)
        for o in self.opts.values():
            eq -= o.mark * CONTRACT
        return eq

    def _commitment(self, day):
        """wheel_bot.try_open_position's total_commitment: short puts at
        strike collateral, everything else at market value."""
        total = 0.0
        for o in self.opts.values():
            total += o.strike * CONTRACT if o.kind == "P" else o.mark * CONTRACT
        for u, n in self.shares.items():
            total += abs(n * (self._price(u, day) or 0.0))
        return total

    def _fill_sell(self, vwap):
        return vwap * (1 - self.p.slip)

    def _fill_buy(self, vwap):
        return vwap * (1 + self.p.slip)

    def _log(self, *a, **k):
        self.events.append(Event(*a, **k))

    # -- execution (day t+1) ------------------------------------------------
    def _execute(self, day, orders):
        for o in orders:
            kind = o["kind"]
            if kind in ("sell_put", "sell_call"):
                c = o["contract"]
                bar = self.book.bar(c["symbol"], day)
                if bar is None:
                    self._log(day, "skip_no_trade", o["u"], c["symbol"], note=kind)
                    continue
                prem = self._fill_sell(bar["vwap"])
                if prem < self.p.min_premium:
                    self._log(day, "skip_no_trade", o["u"], c["symbol"], note="premium below minimum")
                    continue
                self.cash += prem * CONTRACT - self.p.fee
                self.opts[c["symbol"]] = OptPos(c["symbol"], o["u"], c["type"], float(c["strike"]),
                                                c["expiry"], prem, day, bar["close"])
                self._log(day, kind, o["u"], c["symbol"], prem)
            elif kind == "close":
                pos = self.opts.get(o["symbol"])
                bar = self.book.bar(o["symbol"], day) if pos else None
                if bar is None:
                    continue                      # retried next close
                px = self._fill_buy(bar["vwap"])
                self.cash -= px * CONTRACT + self.p.fee
                del self.opts[o["symbol"]]
                self._log(day, "close", pos.underlying, pos.symbol, px, note=o["reason"])
            elif kind == "roll":
                pos = self.opts.get(o["symbol"])
                if pos is None:
                    continue
                new = o["contract"]
                nb = self.book.bar(new["symbol"], day)
                ob = self.book.bar(o["symbol"], day)
                # wheel_bot prices the new leg BEFORE closing the old one and
                # aborts the roll if it cannot (roll_option_position step 1)
                if nb is None or self._fill_sell(nb["vwap"]) < self.p.min_premium or ob is None:
                    self._log(day, "roll_aborted", pos.underlying, pos.symbol)
                    continue
                px = self._fill_buy(ob["vwap"])
                self.cash -= px * CONTRACT + self.p.fee
                del self.opts[o["symbol"]]
                self._log(day, "roll_close", pos.underlying, pos.symbol, px)
                prem = self._fill_sell(nb["vwap"])
                self.cash += prem * CONTRACT - self.p.fee
                self.opts[new["symbol"]] = OptPos(new["symbol"], pos.underlying, pos.kind,
                                                  float(new["strike"]), new["expiry"], prem, day, nb["close"])
                self._log(day, "roll_open", pos.underlying, new["symbol"], prem)

    def _settle_expiries(self, day):
        for sym, o in list(self.opts.items()):
            if o.expiry > day:
                continue
            px = self._price(o.underlying, o.expiry)
            if px is None:
                continue
            del self.opts[sym]
            if o.kind == "P" and px < o.strike:
                self.cash -= o.strike * CONTRACT
                self.shares[o.underlying] = self.shares.get(o.underlying, 0) + CONTRACT
                self._log(day, "assigned", o.underlying, sym, o.strike)
            elif o.kind == "C" and px > o.strike:
                self.cash += o.strike * CONTRACT
                self.shares[o.underlying] = self.shares.get(o.underlying, 0) - CONTRACT
                if self.shares[o.underlying] <= 0:
                    self.shares.pop(o.underlying)
                self._log(day, "called_away", o.underlying, sym, o.strike)
            else:
                self._log(day, "expired", o.underlying, sym)

    # -- decisions (day t close) --------------------------------------------
    def _decide(self, day, next_day):
        p = self.p
        orders = []
        busy = set()
        for sym, o in self.opts.items():
            busy.add(o.underlying)
            dte = (o.expiry - next_day).days
            under = self._price(o.underlying, day)
            capture = (o.entry - o.mark) / o.entry if o.entry > 0 else 0.0
            if p.take_profit is not None and capture >= p.take_profit:
                orders.append(dict(kind="close", symbol=sym, reason="take profit"))
                continue
            itm = under is not None and ((o.kind == "P" and under < o.strike) or
                                         (o.kind == "C" and under > o.strike))
            if itm and dte <= p.force_close_dte:
                orders.append(dict(kind="close", symbol=sym, reason="expiry backstop (ITM)"))
                continue
            if dte <= p.stale_roll_dte and under is not None:
                new = pick_contract(self.book.chain(o.underlying, day), o.kind, under, p.otm,
                                    next_day, p.min_dte, p.max_dte)
                if new is None:
                    self._log(day, "roll_aborted", o.underlying, sym, note="no contract")
                else:
                    orders.append(dict(kind="roll", symbol=sym, contract=new))

        gated = False
        if p.gates:
            gated = (self._gate_value(self.regime, day) in p.gated_regimes or
                     (self._gate_value(self.vix, day) or 0) > p.vix_gate)
        budget = self._equity(day)
        commitment = self._commitment(day)
        owned = [u for u, n in self.shares.items() if n >= CONTRACT]
        todo = list(self.cands.get(day, [])) + [u for u in owned if u not in self.cands.get(day, [])]
        for u in todo:
            if u in busy:
                continue
            price = self._price(u, day)
            if not price:
                continue
            if self.shares.get(u, 0) >= CONTRACT:
                c = pick_contract(self.book.chain(u, day), "C", price, p.otm, next_day, p.min_dte, p.max_dte)
                if c is not None:
                    orders.append(dict(kind="sell_call", u=u, contract=c))
                    busy.add(u)
                continue
            if gated:
                continue
            # Budget pre-check BEFORE the chain lookup, which is an API call
            # per underlying per quarter: with a ~$36k sleeve most candidates
            # can never fit, and fetching their chains made the first real run
            # look hung. The chosen strike sits near price x (1 - otm); unless
            # no strike exists within 10 points of that, its collateral is at
            # least this much, so skipping here loses nothing. The exact check
            # on the real strike still follows.
            if commitment + price * max(0.0, 1 - p.otm - 0.10) * CONTRACT > budget:
                continue
            c = pick_contract(self.book.chain(u, day), "P", price, p.otm, next_day, p.min_dte, p.max_dte)
            if c is None:
                continue
            collateral = float(c["strike"]) * CONTRACT
            if commitment + collateral > budget:
                continue
            commitment += collateral
            orders.append(dict(kind="sell_put", u=u, contract=c))
            busy.add(u)
        return orders

    # -- main loop -------------------------------------------------------------
    def run(self, progress=None):
        """progress: optional callable(day, book) called at each new month."""
        self.cash = self.capital
        self.shares, self.opts, self.events = {}, {}, []
        pending = []
        equity = {}
        self.paused_days = 0
        month = None
        for i, day in enumerate(self.days):
            if progress and (day.year, day.month) != month:
                month = (day.year, day.month)
                progress(day, self.book)
            paused = self.p.gates and (self._gate_value(self.vix, day) or 0) > self.p.vix_pause
            if pending:
                self._execute(day, pending)
                pending = []
            self._settle_expiries(day)
            for o in self.opts.values():
                b = self.book.bar(o.symbol, day)
                if b is not None:
                    o.mark = b["close"]
                else:
                    # no trade today: keep the last mark, but never below
                    # intrinsic value, which a stale mark would understate
                    under = self._price(o.underlying, day) or 0.0
                    intrinsic = max(0.0, o.strike - under) if o.kind == "P" else max(0.0, under - o.strike)
                    o.mark = max(o.mark, intrinsic)
            equity[day] = self._equity(day)
            if i + 1 < len(self.days):
                if paused:
                    self.paused_days += 1
                else:
                    pending = self._decide(day, self.days[i + 1])
        eq = pd.Series(equity, dtype=float)
        return WheelResult(eq, self.events, self.capital, self.paused_days)


@dataclass
class WheelResult:
    equity: pd.Series
    events: list
    capital: float
    paused_days: int

    def returns(self):
        eq = self.equity.sort_index()
        r = eq / pd.concat([pd.Series([self.capital]), eq.iloc[:-1]]).to_numpy() - 1
        r.index = pd.DatetimeIndex(pd.to_datetime(list(r.index)))
        return r

    def counts(self):
        out = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out


# --- inputs derived from daily bars ---------------------------------------------

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _adx_scanner(high, low, close, window=14):
    """market_analyst / market_scanner TechnicalMath.get_adx, restated
    (the regime rule reads THIS ADX, not ta's)."""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                   axis=1).max(axis=1).replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1 / window, adjust=False).mean() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def _rsi_scanner(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))


def regime_series(spy):
    """market_analyst._classify_regime on SPY daily bars, per day:
    below EMA20 -> BEAR_TREND; above EMA20 with ADX > 25 -> BULL_TREND;
    else SIDEWAYS. `spy` has high/low/close indexed by date."""
    close = spy["close"]
    ema20 = _ema(close, 20)
    adx = _adx_scanner(spy["high"], spy["low"], close)
    out = pd.Series("SIDEWAYS", index=close.index, dtype=object)
    out[(close > ema20) & (adx > 25)] = "BULL_TREND"
    out[close < ema20] = "BEAR_TREND"
    return out


def wheel_candidates(daily, days, members, k=10, min_price=15.0, max_price=1000.0):
    """market_scanner.analyze_technicals' WHEEL bucket, point in time:
    price > SMA200, 40 <= RSI(14) <= 55, ADX(14) < 25, ranked by 50 - RSI,
    top k, among that day's liquid names (`members`, the scanner's own
    Alpaca-path rule: top 400 by dollar volume). The scanner's earnings
    guard needs yfinance calendars that have no history; it is not applied."""
    feats = {}
    for s in {x for v in members.values() for x in v}:
        df = daily.get(s)
        if df is None or df.empty or len(df) < 205:
            continue
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
        d = df.copy()
        d.index = pd.Index(idx.date)
        d = d[~d.index.duplicated(keep="last")]
        c = d["close"]
        feats[s] = pd.DataFrame(dict(close=c, sma200=c.rolling(200).mean(), rsi=_rsi_scanner(c),
                                     adx=_adx_scanner(d["high"], d["low"], c)))
    out = {}
    for day in days:
        rows = []
        for s in members.get(day, []):
            f = feats.get(s)
            if f is None or day not in f.index:
                continue
            r = f.loc[day]
            if not (min_price <= r.close <= max_price):
                continue
            if r.close > r.sma200 and 40 <= r.rsi <= 55 and r.adx < 25:
                rows.append((50 - r.rsi, s))
        out[day] = [s for _, s in sorted(rows, reverse=True)[:k]]
    return out
