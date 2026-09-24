"""The fleet's trend_bot / survivor_bot rules, pinned, and checked against source.

The backtest cannot import the bots: every bot module builds a live FleetBot
(Alpaca clients, Discord, config.py) at import time, and the fleet repo lives
on the Beelink, not the Corsair. So the rules are restated here - and a
restatement drifts. `verify_against_fleet` reads the fleet's source (never
imports it) and reports every constant or inline threshold that no longer
matches. run_backtest refuses to run on a mismatch unless told otherwise, and
the report states whether the rules were verified at all.

Held constant across every arm on purpose: the question is what the TARGET
LIST does, so the strategy that trades it must not change between arms.
"""
import ast
from pathlib import Path

# --- trend_bot.py ---------------------------------------------------------
TREND = dict(
    FAST_EMA=9,
    SLOW_EMA=21,
    RISK_PER_TRADE=0.02,
    MAX_POSITION_PCT=0.15,
    MIN_PRICE_LONG=5.00,
    MIN_PRICE_SHORT=10.00,
    MAX_SHORT_EXPOSURE=0.30,
    MOMENTUM_BARS=5,
    MOMENTUM_ADX_MIN=25,
    MOMENTUM_PULLBACK_PCT=0.01,
    MOMENTUM_SIZE_MULT=0.8,
    STOP_LOSS=-0.05,
    TAKE_PROFIT=0.08,
)
TREND_ADX_MIN = 20          # inline: `if local_adx <= 20:` skip

# --- survivor_bot.py ------------------------------------------------------
SURVIVOR = dict(
    RSI_BUY=38,
    RSI_SELL=70,
    RSI_WINDOW=14,
    RISK_PER_TRADE=0.05,
    MIN_PRICE=5.00,
    MAX_POSITION_PCT=0.10,
)
SURVIVOR_TAKE_PROFIT = 0.05  # inline: `elif pct_gain > 0.05:`
SURVIVOR_STOP_LOSS = -0.03   # inline: `elif pct_gain < -0.03:`

# --- fleet_bot.py (EOD windows, ET) ----------------------------------------
NO_ENTRY_AFTER = "14:00"
EOD_EVAL_FROM = "15:30"
EOD_CLOSE_FROM = "15:45"

# --- tiered_hold.py -------------------------------------------------------
HOLD_THRESHOLDS = {
    "trend_bot": {"HOLD_SWING": 50, "HOLD_OVERNIGHT": 25},
    "survivor_bot": {"HOLD_SWING": 55, "HOLD_OVERNIGHT": 30},
}
MAX_HOLD_DAYS = {"HOLD_OVERNIGHT": 3, "HOLD_SWING": 7}

# --- bot_config.template.json cfo_settings ----------------------------------
BASE_ALLOCATIONS = {"trend_bot": 0.28, "survivor_bot": 0.20}
UNALLOCATED_RESERVE = 0.05

# --- utils.load_and_validate_targets --------------------------------------
TARGETS_MAX_AGE_S = 86400


def hold_score(bot, pnl_pct, indicators, hours_held, regime="SIDEWAYS", vix=18.0):
    """tiered_hold.calculate_hold_score, restated. The bots never pass
    entry_confidence, so factor 3 is always zero and confidence cannot reach
    the hold decision - reproduced by omitting it."""
    score = 0
    if pnl_pct > 0.02:
        score += 30
    elif pnl_pct > 0.005:
        score += 20
    elif pnl_pct > -0.005:
        score += 10
    elif pnl_pct > -0.02:
        score += 0
    else:
        score -= 20
    if bot == "trend_bot":
        adx = indicators.get("adx", 0)
        if adx > 25 and indicators.get("ema_trend_intact", False):
            score += 25
        elif adx > 20:
            score += 10
    else:
        r = indicators.get("rsi", 50)
        if r < 35 and pnl_pct < 0.03:
            score += 25
        elif r < 45:
            score += 10
    if hours_held is not None:
        if hours_held > 48:
            score += 15
        elif hours_held > 24:
            score += 10
        elif hours_held > 6:
            score += 5
    if regime == "CRITICAL_VOLATILITY":
        score -= 30
    elif regime == "BEAR_TREND":
        score -= 15
    if vix > 30:
        score -= 15
    elif vix > 25:
        score -= 10
    return score


def hold_tier(bot, score):
    t = HOLD_THRESHOLDS[bot]
    if score >= t["HOLD_SWING"]:
        return "HOLD_SWING"
    if score >= t["HOLD_OVERNIGHT"]:
        return "HOLD_OVERNIGHT"
    return "CLOSE_EOD"


# --- verification ---------------------------------------------------------

def _module_constants(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


# Inline thresholds have no name to read, so their exact source text is the
# contract. A reworded line fails verification rather than passing silently.
_INLINE = {
    "trend_bot.py": ["if local_adx <= 20:"],
    "survivor_bot.py": ["elif pct_gain > 0.05:", "elif pct_gain < -0.03:",
                        "if indicators_ok and rsi is not None and rsi > RSI_SELL:",
                        "if rsi >= RSI_BUY:"],
    "fleet_bot.py": ['self.is_eod_eval = "15:30" <= self.time_str < "15:45"',
                     'self.is_eod_close = self.time_str >= "15:45"',
                     'self.is_eod_skip_entry = self.time_str >= "14:00"',
                     "scaler = 0.5 + confidence"],
    "utils.py": ["if age_seconds > 86400:"],
    "tiered_hold.py": ["if pnl_pct > 0.02:    score += 30",
                       "if adx > 25 and ema_intact:",
                       "if rsi < 35 and pnl_pct < 0.03:",
                       "if hours_held > 48: score += 15"],
}


def verify_against_fleet(fleet_repo):
    """List of human-readable mismatches between these rules and the fleet
    source at `fleet_repo`. Empty list = verified."""
    root = Path(fleet_repo)
    problems = []
    for fname, pinned in (("trend_bot.py", TREND), ("survivor_bot.py", SURVIVOR)):
        path = root / fname
        if not path.exists():
            problems.append(f"{fname}: not found under {root}")
            continue
        live = _module_constants(path)
        for k, v in pinned.items():
            if k not in live:
                problems.append(f"{fname}: {k} no longer defined")
            elif live[k] != v:
                problems.append(f"{fname}: {k} = {live[k]!r}, backtest assumes {v!r}")
    th = root / "tiered_hold.py"
    if th.exists():
        live = _module_constants(th)
        for bot, tiers in HOLD_THRESHOLDS.items():
            for tier, v in tiers.items():
                got = live.get("HOLD_THRESHOLDS", {}).get(bot, {}).get(tier)
                if got != v:
                    problems.append(f"tiered_hold.py: HOLD_THRESHOLDS[{bot}][{tier}] = {got!r}, assumed {v!r}")
        for tier, v in MAX_HOLD_DAYS.items():
            got = live.get("OVERNIGHT_STOPS", {}).get(tier, {}).get("max_hold_days")
            if got != v:
                problems.append(f"tiered_hold.py: {tier} max_hold_days = {got!r}, assumed {v!r}")
    for fname, snippets in _INLINE.items():
        path = root / fname
        if not path.exists():
            problems.append(f"{fname}: not found under {root}")
            continue
        text = path.read_text(encoding="utf-8")
        for snip in snippets:
            if snip not in text:
                problems.append(f"{fname}: expected source line not found: {snip!r}")
    return problems


# --- wheel_bot.py (wheel backtest) ---------------------------------------------
WHEEL = dict(
    MIN_DTE=25,
    MAX_DTE=45,
    TARGET_OTM_PCT=0.05,
    MIN_PREMIUM=0.10,
    TAKE_PROFIT_PCT=0.50,
    STALE_ROLL_DTE=10,
    FORCE_CLOSE_DTE=5,
)
WHEEL_GATED_REGIMES = ("BEAR_TREND", "CRITICAL_VOLATILITY")
WHEEL_VIX_GATE = 22
FLEET_VIX_PAUSE = 28.0

_WHEEL_INLINE = {
    # the dynamic OTM, the strict-OTM strike filter and the closest-OTM score
    "wheel_bot.py": ["dynamic_otm = TARGET_OTM_PCT * (1.5 - confidence)",
                     "if side == \"PUT\" and strike >= current_price: continue",
                     "score = abs(pct_otm - target_otm)",
                     "if capture_pct >= TAKE_PROFIT_PCT:",
                     "if is_itm and dte <= FORCE_CLOSE_DTE:",
                     "is_stale = dte <= STALE_ROLL_DTE",
                     "if new_limit_price < MIN_PREMIUM:",
                     "if new_trade_collateral > 0 and total_commitment + new_trade_collateral > my_budget:"],
    "fleet_registry.py": ['gated_when=dict(regimes=("BEAR_TREND", "CRITICAL_VOLATILITY"), vix_above=22)'],
    "market_analyst.py": ["if vix_val > 28.0:",
                          "if price < ema20:",
                          "if price > ema20 and adx > 25:"],
    # a paused bot is STOPPED (pm2 stop): it manages nothing until resumed
    "commander.py": ["elif desired_status == \"paused\" and actual_status == \"online\":",
                     "subprocess.run(['pm2', 'stop', bot_name])"],
}


def verify_wheel_against_fleet(fleet_repo):
    """Mismatches between the wheel backtest's rules and the fleet source."""
    root = Path(fleet_repo)
    problems = []
    path = root / "wheel_bot.py"
    if not path.exists():
        return [f"wheel_bot.py: not found under {root}"]
    live = _module_constants(path)
    for k, v in WHEEL.items():
        if live.get(k) != v:
            problems.append(f"wheel_bot.py: {k} = {live.get(k)!r}, backtest assumes {v!r}")
    for fname, snippets in _WHEEL_INLINE.items():
        p = root / fname
        if not p.exists():
            problems.append(f"{fname}: not found under {root}")
            continue
        text = p.read_text(encoding="utf-8")
        for snip in snippets:
            if snip not in text:
                problems.append(f"{fname}: expected source line not found: {snip!r}")
    return problems
