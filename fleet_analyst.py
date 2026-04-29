"""
Fleet Analyst — Autonomous Trading Fleet Analysis Agent
========================================================
Standalone Python agent that connects to LM Studio (or any OpenAI-compatible
endpoint) and provides tool-augmented analysis of the trading bot fleet.

Designed for the Corsair AI Workstation (Windows, Strix Halo, 96GB).
Queries InfluxDB on the Beelink for live fleet metrics.

Usage:
    python fleet_analyst.py                     # Interactive mode
    python fleet_analyst.py --once "question"   # Single question mode

Requirements:
    pip install openai requests

Configuration:
    Edit the CONFIG section below to match your network.
"""

import json
import os
import sys
import time
import datetime
import argparse
import glob
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Run: pip install openai requests")
    sys.exit(1)

import requests

# ============================================================================
# CONFIG — Edit these to match your environment
# ============================================================================

# LM Studio endpoint (default local)
LM_STUDIO_URL = "http://localhost:1234/v1"

# Model to use — should be the Gemma 4 31B Dense loaded in LM Studio
# Run http://localhost:1234/v1/models in a browser to find the exact string
MODEL_NAME = "google/gemma-4-31b"  # UPDATE THIS to your actual model identifier

# Beelink InfluxDB endpoint
BEELINK_IP = "192.168.5.27"
INFLUX_PORT = 8086
INFLUX_DB = "trading_bots"
INFLUX_TIMEOUT = 15

# Local fleet file paths on the Corsair (Windows)
# Adjust if your trading_desk is in a different location
TRADING_DESK = Path.home()
FLEET_DIR = TRADING_DESK / "trading-bot-fleet"
SCOUT_DIR = TRADING_DESK / "TradingAgent"

# Max conversation turns before auto-summarizing (keeps context manageable)
MAX_HISTORY_TURNS = 20

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are the Fleet Analyst for an autonomous algorithmic trading system running on Alpaca Markets (paper trading). Your job is to monitor, diagnose, and recommend improvements to the fleet.

## Your Environment
- **Execution Node (Beelink S12 Mini):** Runs the trading bot fleet natively under PM2 on Ubuntu. InfluxDB 1.8 + Grafana in Docker. You can query InfluxDB for live metrics.
- **Analysis Node (Corsair AI Workstation):** Where you run. Windows, Strix Halo, 96GB unified memory. Market Scanner and Sector Scout run here, producing active_targets.json that gets SCP'd to the Beelink.

## The Fleet
- **survivor_bot** — Mean-reversion dip buying on leveraged ETFs. Buys oversold bounces, targets +5% take-profit. 15% allocation.
- **trend_bot** — EMA momentum, long AND short capable. Most aggressive bot. 25% allocation.
- **wheel_bot** — Options premium selling (covered calls, cash-secured puts). Gated when VIX > 22 or regime is BEAR_TREND/CRITICAL_VOLATILITY. 40% allocation.
- **crypto_grid** — BTC/ETH/SOL grid trading on Alpaca. Draws from shared USD pool. 5% allocation.
- **moon_bot** — Crypto breakout via Donchian channels. Rarely triggers. 5% allocation.
- **condor_bot** — SIDELINED. Excluded from all allocations. Do not suggest reactivating.
- **accountant** — CFO module. Tracks P&L per bot, logs to InfluxDB every 5 minutes.
- **market_analyst** — Monitors SPY/VIX, sets regime in bot_config.json. Pauses all bots if VIX > 28.
- **commander** — Discord bot, fleet watchdog, auto-restarts crashed bots.

## Key Principles
- Silent failures are fleet killers. Many functions use bare `except: pass`. Always suspect silent errors when data looks wrong or missing.
- When analyzing trades, ALWAYS calculate P&L manually: (sell_price - buy_price) × qty. Never say "no losses" without doing the math.
- The iron condor bot is dead. Don't allocate to it, don't suggest reviving it.
- crypto_grid is an invisible capital consumer — it trades on Alpaca's shared USD pool with no separate account.
- wheel_bot being gated in bear markets is CORRECT architecture, not a bug.
- Old data from before March 2026 was collected during known system issues. Don't use it for performance baselines.

## InfluxDB Measurements
- `trades` — trend_bot trades
- `survivor_trades` — survivor_bot trades
- `wheel_trades` — wheel_bot trades
- `condor_trades` — condor_bot trades (historical only)
- `crypto_trades` — crypto_grid trades
- `breakout_trades` — moon_bot trades
- `bot_performance` — Per-bot P&L snapshots (unrealized_pl, realized_pl, total_pl, allocation)
- `account_stats` — Account equity, cash, buying power
- `market_regime` — SPY price, VIX, ADX, regime classification
- `bot_monitor` — PM2 process health (memory, CPU, restarts, uptime)

## How to Work
1. When asked about fleet performance, query InfluxDB first — don't guess.
2. When asked about code or architecture, read the source files from the workspace.
3. When you find a problem, explain the root cause AND suggest a specific fix with code if appropriate.
4. Be direct. No hedging. If a bot is losing money, say so and say why.
5. When comparing buy/sell pairs, match them by symbol and calculate the actual dollar P&L.
6. Use multiple tools in sequence when needed. For example: check regime, then pull trades, then check bot status — build the full picture before answering.
7. If InfluxDB is unreachable, say so clearly rather than guessing at data.
"""

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def query_influx(query: str) -> str:
    """Run a raw InfluxQL query against the Beelink's trading_bots database."""
    try:
        resp = requests.get(
            f"http://{BEELINK_IP}:{INFLUX_PORT}/query",
            params={"db": INFLUX_DB, "q": query},
            timeout=INFLUX_TIMEOUT
        )
        data = resp.json()
        results = data.get("results", [])
        if not results or "series" not in results[0]:
            return f"Query returned no data.\nQuery: {query}"

        output = []
        for series in results[0]["series"]:
            name = series.get("name", "results")
            tags = series.get("tags", {})
            columns = series.get("columns", [])
            values = series.get("values", [])

            tag_str = ""
            if tags:
                tag_str = " (" + ", ".join(f"{k}={v}" for k, v in tags.items()) + ")"

            output.append(f"--- {name}{tag_str} ---")
            output.append(" | ".join(columns))
            output.append("-" * 60)
            for row in values[-30:]:  # cap at 30 rows
                output.append(" | ".join(str(v) for v in row))
            if len(values) > 30:
                output.append(f"... ({len(values)} total rows, showing last 30)")
        return "\n".join(output)

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to InfluxDB on the Beelink. Is it running? Check that the Beelink is on and InfluxDB Docker container is up."
    except requests.exceptions.Timeout:
        return "ERROR: InfluxDB query timed out. The Beelink may be under heavy load."
    except Exception as e:
        return f"ERROR querying InfluxDB: {str(e)}"


def get_bot_performance() -> str:
    """Get the latest P&L snapshot for all bots."""
    return query_influx(
        "SELECT unrealized_pl, realized_pl, total_pl, allocation "
        "FROM bot_performance ORDER BY time DESC LIMIT 10"
    )


def get_account_stats() -> str:
    """Get the latest account-level stats: equity, cash, buying power."""
    return query_influx(
        "SELECT last(equity), last(cash), last(buying_power) "
        "FROM account_stats WHERE time > now() - 1h"
    )


def get_market_regime() -> str:
    """Get the current market regime reading."""
    return query_influx(
        "SELECT last(regime), last(vix), last(adx), last(price) "
        "FROM market_regime WHERE time > now() - 2h"
    )


def get_recent_trades(measurement: str = "trades", limit: int = 10) -> str:
    """Get recent trades from a specific measurement.

    Args:
        measurement: One of 'trades' (trend_bot), 'survivor_trades',
                     'wheel_trades', 'crypto_trades', 'breakout_trades'
        limit: Number of recent entries (max 50)
    """
    limit = min(int(limit), 50)
    allowed = ["trades", "survivor_trades", "wheel_trades",
               "condor_trades", "crypto_trades", "breakout_trades"]
    if measurement not in allowed:
        return f"Unknown measurement '{measurement}'. Valid options: {allowed}"
    return query_influx(f"SELECT * FROM {measurement} ORDER BY time DESC LIMIT {limit}")


def list_measurements() -> str:
    """List all available InfluxDB measurements in the trading_bots database."""
    return query_influx("SHOW MEASUREMENTS")


def get_bot_health() -> str:
    """Get PM2 process health stats for all bots (memory, CPU, restarts, uptime)."""
    return query_influx(
        "SELECT last(memory), last(cpu), last(restarts), last(uptime) "
        "FROM bot_monitor WHERE time > now() - 1h GROUP BY bot"
    )


def read_active_targets() -> str:
    """Read the current active_targets.json — the live watchlist bots are trading against."""
    for base in [FLEET_DIR, SCOUT_DIR]:
        path = base / "active_targets.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                output = []
                updated = data.get("updated", "unknown")
                status = data.get("status", "unknown")
                output.append(f"Last Updated: {updated}")
                output.append(f"Scan Status: {status}")
                output.append("")

                for strategy in ["trend_targets", "survivor_targets", "wheel_targets",
                                 "condor_targets", "short_targets"]:
                    targets = data.get(strategy, [])
                    output.append(f"--- {strategy.upper()} ({len(targets)} targets) ---")
                    if not targets:
                        output.append("  (empty)")
                    for item in targets:
                        if isinstance(item, dict):
                            sym = item.get("symbol", "?")
                            conf = item.get("confidence", 0)
                            tech = item.get("tech_score", 0)
                            output.append(f"  {sym:<6} | Conf: {conf:.2f} | Tech: {tech:.2f}")
                        else:
                            output.append(f"  {item}")
                    output.append("")

                # File age
                age_hours = (time.time() - os.path.getmtime(str(path))) / 3600
                age_str = f"{age_hours:.1f}h ago"
                if age_hours > 24:
                    age_str += " ⚠️ STALE"
                output.insert(0, f"File Age: {age_str}")

                return "\n".join(output)
            except Exception as e:
                return f"Error reading active_targets.json: {e}"

    return "active_targets.json not found in fleet or scout directories."


def read_bot_config() -> str:
    """Read bot_config.json — who's active, paused, regime, emergency stop status."""
    path = FLEET_DIR / "bot_config.json"
    if not path.exists():
        return f"bot_config.json not found at {path}"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading bot_config.json: {e}"


def read_scout_log(lines: int = 80) -> str:
    """Read the tail of the sector scout log.

    Args:
        lines: Number of lines from the end to return (max 200)
    """
    lines = min(int(lines), 200)
    for base in [SCOUT_DIR, FLEET_DIR]:
        path = base / "scout_log.txt"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                tail = all_lines[-lines:]
                return f"--- scout_log.txt (last {len(tail)} lines) ---\n" + "".join(tail)
            except Exception as e:
                return f"Error reading scout_log.txt: {e}"
    return "scout_log.txt not found."


def list_directory(path: str = "") -> str:
    """List files in the trading workspace directories.

    Args:
        path: Subdirectory to list. Options: 'fleet' (trading-bot-fleet),
              'scout' (TradingAgent), or '' for both top-level.
    """
    targets = []
    if path in ("", "both"):
        targets = [FLEET_DIR, SCOUT_DIR]
    elif path in ("fleet", "trading-bot-fleet"):
        targets = [FLEET_DIR]
    elif path in ("scout", "TradingAgent"):
        targets = [SCOUT_DIR]
    else:
        # Try as a literal path
        literal = TRADING_DESK / path
        if literal.exists():
            targets = [literal]
        else:
            return f"Directory '{path}' not found. Use 'fleet', 'scout', or a path relative to ~/trading_desk/"

    output = []
    for target in targets:
        if not target.exists():
            output.append(f"--- {target.name} --- (NOT FOUND)")
            continue
        output.append(f"--- {target.name} ---")
        try:
            entries = sorted(target.iterdir())
            for entry in entries:
                if entry.name.startswith('.'):
                    continue
                if entry.is_dir():
                    output.append(f"  📁 {entry.name}/")
                else:
                    size_kb = entry.stat().st_size / 1024
                    output.append(f"  📄 {entry.name} ({size_kb:.1f} KB)")
        except PermissionError:
            output.append("  (permission denied)")
        output.append("")

    return "\n".join(output) if output else "No directories found."


def read_file(filepath: str) -> str:
    """Read a source file from the trading workspace.

    Args:
        filepath: Path relative to ~/trading_desk/. Example: 'trading-bot-fleet/survivor_bot.py'
                  or 'TradingAgent/sector_scout_3.py'
    """
    # Try relative to trading desk first
    path = TRADING_DESK / filepath
    if not path.exists():
        # Try as-is
        path = Path(filepath)
    if not path.exists():
        return f"File not found: {filepath}"

    # Safety: only allow reading from trading_desk
    try:
        path.resolve().relative_to(TRADING_DESK.resolve())
    except ValueError:
        return f"Access denied: can only read files under {TRADING_DESK}"

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Truncate very large files
        if len(content) > 15000:
            half = 7000
            content = (content[:half] +
                       f"\n\n... [TRUNCATED — {len(content)} chars total, showing first and last {half}] ...\n\n" +
                       content[-half:])

        return f"--- {filepath} ---\n{content}"
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def run_custom_influx_query(query: str) -> str:
    """Run any InfluxQL query. Use for custom analysis beyond the helper functions.

    Args:
        query: A valid InfluxQL query string. Examples:
               'SELECT * FROM trades WHERE time > now() - 24h'
               'SELECT count(action) FROM survivor_trades GROUP BY action'
               'SHOW TAG VALUES FROM bot_performance WITH KEY = bot'
    """
    # Basic safety check
    dangerous = ["DROP", "DELETE", "ALTER", "CREATE", "INSERT"]
    if any(d in query.upper().split() for d in dangerous):
        return "ERROR: Write/modify queries are not allowed. Read-only access only."
    return query_influx(query)


# ============================================================================
# TOOL DEFINITIONS (OpenAI function-calling format)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_bot_performance",
            "description": "Get the latest P&L snapshot for all bots — unrealized P&L, realized P&L, total P&L, and capital allocation. Call this first when asked about fleet performance.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_stats",
            "description": "Get account-level stats: total equity, available cash, and buying power.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Get the current market regime — SPY price, VIX level, ADX, and regime classification (BULL_TREND, BEAR_TREND, SIDEWAYS, CRITICAL_VOLATILITY).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_trades",
            "description": "Get recent trade entries from a specific bot's trade log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "measurement": {
                        "type": "string",
                        "description": "Which trade log to query. Options: 'trades' (trend_bot), 'survivor_trades', 'wheel_trades', 'crypto_trades', 'breakout_trades'",
                        "enum": ["trades", "survivor_trades", "wheel_trades", "crypto_trades", "breakout_trades"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent entries to return (1-50, default 10)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_measurements",
            "description": "List all available data tables (measurements) in InfluxDB. Use this to discover what data is available.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bot_health",
            "description": "Get PM2 process health for all bots — memory usage, CPU, restart count, and uptime.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_active_targets",
            "description": "Read the current active_targets.json — shows which tickers are approved for each strategy, their confidence scores, and how old the scan is.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_bot_config",
            "description": "Read bot_config.json — shows which bots are active/paused, global settings, emergency stop status, and regime overrides.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_scout_log",
            "description": "Read the tail of the sector scout execution log — shows recent scan results, errors, and approval rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines from the end to show (default 80, max 200)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in the trading workspace. Use to discover available source files before reading them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Which directory to list: 'fleet' (trading-bot-fleet repo), 'scout' (TradingAgent repo), or '' for both."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a source file from the trading workspace. Use for code review, debugging, or understanding bot logic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path relative to ~/trading_desk/. Examples: 'trading-bot-fleet/survivor_bot.py', 'TradingAgent/sector_scout_3.py', 'trading-bot-fleet/utils.py'"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_custom_influx_query",
            "description": "Run a custom InfluxQL query for analysis not covered by the helper functions. Read-only access.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid InfluxQL query. Examples: 'SELECT * FROM trades WHERE time > now() - 24h', 'SELECT count(action) FROM survivor_trades GROUP BY action'"
                    }
                },
                "required": ["query"]
            }
        }
    },
]

# Map function names to implementations
TOOL_MAP = {
    "get_bot_performance": lambda **_: get_bot_performance(),
    "get_account_stats": lambda **_: get_account_stats(),
    "get_market_regime": lambda **_: get_market_regime(),
    "get_recent_trades": lambda **kw: get_recent_trades(
        measurement=kw.get("measurement", "trades"),
        limit=kw.get("limit", 10)
    ),
    "list_measurements": lambda **_: list_measurements(),
    "get_bot_health": lambda **_: get_bot_health(),
    "read_active_targets": lambda **_: read_active_targets(),
    "read_bot_config": lambda **_: read_bot_config(),
    "read_scout_log": lambda **kw: read_scout_log(lines=kw.get("lines", 80)),
    "list_directory": lambda **kw: list_directory(path=kw.get("path", "")),
    "read_file": lambda **kw: read_file(filepath=kw["filepath"]),
    "run_custom_influx_query": lambda **kw: run_custom_influx_query(query=kw["query"]),
}


# ============================================================================
# AGENT LOOP
# ============================================================================

class FleetAnalyst:
    """Agentic tool-calling loop against LM Studio's OpenAI-compatible API."""

    def __init__(self):
        self.client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio"  # LM Studio doesn't need a real key
        )
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.total_tool_calls = 0

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result string."""
        fn = TOOL_MAP.get(name)
        if not fn:
            return f"ERROR: Unknown tool '{name}'"
        try:
            return fn(**arguments)
        except Exception as e:
            return f"ERROR executing {name}: {str(e)}"

    def _trim_history(self):
        """Keep conversation history manageable by trimming old turns."""
        # Count user/assistant pairs (skip system message at index 0)
        user_msgs = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_msgs) > MAX_HISTORY_TURNS:
            # Keep system prompt + last N turns
            keep_from = user_msgs[-MAX_HISTORY_TURNS]
            self.messages = [self.messages[0]] + self.messages[keep_from:]

    def chat(self, user_input: str) -> str:
        """Send a message and handle the full tool-calling loop until final response."""
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        max_tool_rounds = 8  # Safety limit on chained tool calls
        round_num = 0

        while round_num < max_tool_rounds:
            round_num += 1

            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as e:
                error_msg = f"LM Studio API error: {str(e)}"
                print(f"\n  ❌ {error_msg}")
                return error_msg

            choice = response.choices[0]
            message = choice.message

            # If the model wants to call tools
            if message.tool_calls:
                # Add the assistant's tool-call message to history
                self.messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each tool call
                for tc in message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        fn_args = {}

                    self.total_tool_calls += 1
                    print(f"  🔧 {fn_name}({json.dumps(fn_args) if fn_args else ''})")

                    result = self._execute_tool(fn_name, fn_args)

                    # Truncate very large results to avoid blowing context
                    if len(result) > 8000:
                        result = result[:8000] + f"\n... [truncated, {len(result)} chars total]"

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

                # Loop back for next round (model may call more tools or respond)
                continue

            else:
                # No tool calls — this is the final response
                final = message.content or "(empty response)"
                self.messages.append({"role": "assistant", "content": final})
                return final

        # If we hit the safety limit
        return "(Fleet Analyst hit the maximum tool-call depth. Try a more specific question.)"


# ============================================================================
# CONNECTIVITY CHECK
# ============================================================================

def check_connectivity():
    """Quick startup checks."""
    issues = []

    # Check LM Studio
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        models = resp.json()
        model_ids = [m.get("id", "?") for m in models.get("data", [])]
        if model_ids:
            print(f"  ✅ LM Studio: {len(model_ids)} model(s) available")
            for m in model_ids:
                marker = " 👈 (selected)" if m == MODEL_NAME else ""
                print(f"     - {m}{marker}")
            if MODEL_NAME not in model_ids:
                issues.append(
                    f"  ⚠️  MODEL_NAME '{MODEL_NAME}' not found in LM Studio. "
                    f"Update MODEL_NAME in the script to one of: {model_ids}"
                )
        else:
            issues.append("  ⚠️  LM Studio is running but no models are loaded.")
    except Exception:
        issues.append("  ❌ Cannot reach LM Studio. Is it running? Check LM_STUDIO_URL.")

    # Check InfluxDB
    try:
        resp = requests.get(
            f"http://{BEELINK_IP}:{INFLUX_PORT}/ping",
            timeout=5
        )
        if resp.status_code == 204:
            print(f"  ✅ InfluxDB: Beelink reachable at {BEELINK_IP}:{INFLUX_PORT}")
        else:
            issues.append(f"  ⚠️  InfluxDB responded with status {resp.status_code}")
    except Exception:
        issues.append(f"  ❌ Cannot reach InfluxDB at {BEELINK_IP}:{INFLUX_PORT}. Is the Beelink on?")

    # Check local files
    for name, path in [("Fleet repo", FLEET_DIR), ("Scout repo", SCOUT_DIR)]:
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            issues.append(f"  ⚠️  {name} not found at {path}")

    return issues


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fleet Analyst — Trading Bot Fleet Analysis Agent")
    parser.add_argument("--once", type=str, help="Ask a single question and exit")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  🏦 FLEET ANALYST")
    print("  Autonomous Trading Fleet Analysis Agent")
    print("=" * 60)
    print()
    print("Running connectivity checks...")

    issues = check_connectivity()
    print()

    if issues:
        for issue in issues:
            print(issue)
        print()
        resp = input("Issues detected. Continue anyway? (y/n): ").strip().lower()
        if resp != 'y':
            print("Exiting.")
            return

    analyst = FleetAnalyst()

    # Single question mode
    if args.once:
        print(f"\n📊 Question: {args.once}\n")
        response = analyst.chat(args.once)
        print(f"\n{response}\n")
        return

    # Interactive mode
    print("Ready. Ask about the fleet, or type 'quit' to exit.")
    print("Examples:")
    print("  • How is the fleet performing today?")
    print("  • Show me survivor_bot's recent trades")
    print("  • What's the current market regime?")
    print("  • Why isn't wheel_bot trading?")
    print("  • Read survivor_bot.py and check for issues")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Fleet Analyst.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting Fleet Analyst.")
            break
        if user_input.lower() == "reset":
            analyst = FleetAnalyst()
            print("Conversation reset.\n")
            continue

        print()
        response = analyst.chat(user_input)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
