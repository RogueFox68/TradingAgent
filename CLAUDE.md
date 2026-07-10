# CLAUDE.md - TradingAgent (Sector Scout)

> **System Context:** This repo runs on the **Corsair AI Workstation** (AMD Strix Halo,
> 96GB unified memory), the analysis node in a two-machine trading system. The companion
> repo `trading-bot-fleet` runs on the **Beelink S12 Mini** and handles all trade execution.
> See `CORSAIR_ARCHITECTURE.md` in this repo for full hardware and container details.

## Project Overview

TradingAgent is an automated stock scanning and AI analysis system. It scans the US equity
market via the Alpaca API, filters candidates using technical indicators (RSI, ADX, SMA), then
scores them with a **local LM Studio LLM (Gemma 4 21B MoE)** for multi-source sentiment analysis.

The output is a JSON payload (`active_targets.json`) that gets SCP'd to the Beelink execution
node where the trading bot fleet consumes it.

Stocks are categorized into the following strategy buckets:
- **trend_targets** — Strong bullish momentum
- **survivor_targets** — Oversold bounce plays (mean reversion)
- **wheel_targets** — Stable/neutral, good for options premium selling
- **short_targets** — Bearish setups (trend_bot short entries)

(`condor_targets` is deprecated — iron condor is sidelined on the Beelink and is no longer
emitted by the scout.)

## Platform History

This repo originated on a **Windows desktop (MSI Aegis / RTX 5080)** running Llama 3.1 8B via
Ollama, scheduled through Windows Task Scheduler every 30 minutes. The entire
sector scout pipeline was **migrated to the Corsair AI Workstation** and now runs **Gemma 4 21B MoE** via **LM Studio**.

1. **Model upgrade:** 21B MoE provides fast inference with near-dense-model quality.
2. **GPU freedom:** The RTX 5080 is now dedicated to ComfyUI/creative AI work without
   scheduling conflicts.
3. **Frequency reduction:** Runs 3x daily, eliminating
   ~80% of inference cycles with negligible signal loss.

The pipeline runs entirely natively on Windows via Task Scheduler. (A previous brief excursion to Ubuntu/Docker/ROCm was abandoned in favor of Windows stability).

## Project Structure

```
TradingAgent/
├── CLAUDE.md                    # This file
├── CORSAIR_ARCHITECTURE.md      # Hardware, containers, scheduling details
├── market_scanner.py            # Phase 1: Scans full Alpaca universe, outputs dragnet_candidates.json
├── sector_scout_3.py            # Phase 2: AI analysis of candidates, outputs active_targets.json
├── shadow_advisors.py           # Shadow-only specialist votes for equity/options/crypto targets
├── test_parser_logic.py         # Unit tests for LLM JSON response parsing
├── test_shadow_advisors.py      # Unit tests for specialist routing/vote persistence
├── test_scp_logic.py            # Unit tests for SCP transfer retry logic
├── test_scoring_logic.py        # Unit tests for confidence weighting/normalization
├── requirements.txt             # Python dependencies
├── run_scout.bat                # Windows Task Scheduler automation
├── TASK_SCHEDULER_SETUP.md      # Windows Task Scheduler setup notes
│
├── (Gitignored - Runtime state)
│   ├── config.py                # Alpaca API keys, Discord webhooks, PAPER flag
│   ├── active_targets.json      # Generated output — current trading targets
│   ├── shadow_advisor_votes.json  # Generated output — latest shadow specialist votes
│   ├── shadow_advisor_votes.jsonl # Generated output — append-only shadow vote history
│   ├── dragnet_candidates.json  # Generated output — scanned candidates
│   └── scout_log.txt            # Execution log
```

## How It Runs (Corsair)

The pipeline is fully autonomous, triggered by **Windows Task Scheduler** on the Corsair host:

**Schedule:** Monday–Friday, 3x daily (Central Time):
- `08:30` — Market open scan
- `12:00` — Mid-day refresh
- `15:00` — Pre-close scan

**Mechanism:**
```
Task Scheduler → run_scout.bat
```

`run_scout.bat` runs Phase 1 (market_scanner.py) then Phase 2 (sector_scout_3.py) sequentially.

### Phase 1: Market Scanner (`market_scanner.py`)

1. Fetches ~4,800 tradeable assets from Alpaca
2. Downloads price/volume data via yfinance in batches
3. Filters for liquidity (volume > 2M, price $15–$1000)
4. Calculates technical indicators: RSI(14), ADX(14), SMA(20/50/200)
5. Categorizes into strategy buckets based on indicator profiles
6. Outputs top 10 per category to `dragnet_candidates.json`

### Phase 2: Sector Scout (`sector_scout_3.py`)

For each candidate from Phase 1:

1. **Gathers multi-source intelligence:**
   - Tier 1: Elite financial news (yfinance, bucketed by publisher)
   - Tier 2: Mainstream news (yfinance)
   - Tier 3: Specialty/industry news (yfinance)
   - Social: Reddit sentiment (public JSON API)

2. **Scores via Gemma 4 21B MoE** — each source gets its own LLM call with a role-specific
   system prompt (hedge fund analyst for trend, value investor for survivor, options income
   trader for wheel, short seller for shorts)

3. **Calculates composite confidence:**
   ```
   Conf = weighted_average(Tech, T1, T2, T3, Social)
   ```
   Candidates scoring at or above 0.66 are approved.

4. **Validates LLM responses:**
   - Clamps scores to 0.0–1.0
   - Penalizes weak reasoning (<50 chars) by 30%
   - Catches "insufficient data" hedging and assigns 0.5

5. **Writes `active_targets.json`** and **SCP transfers to Beelink**
   - 3 retry attempts with 5-second backoff
   - Discord webhook alert on transfer failure

6. **Runs shadow specialist advisors** (paper-only measurement layer)
   - Equity specialist: trend/survivor/short stock buckets
   - Options specialist: wheel candidates
   - Crypto specialist: crypto-shaped candidates if present (dormant today — the
     scanner emits no crypto candidates, so only two specialist models are exercised)
   - Specialists see raw evidence only (tech score + headlines/social), **never the
     scout's confidence or per-source breakdown** — their votes are benchmarked
     against the scout, and an anchored vote would just measure agreement
   - Uses LM Studio JSON Schema output, then extracts the first valid JSON object
     defensively and makes one repair attempt if parsing still fails
   - Writes `shadow_advisor_votes.json` and appends `shadow_advisor_votes.jsonl`
     (vote snapshots carry an `advisor_failures` count — a wrong model id fails
     every call, and the run summary calls that out loudly). Each vote also records
     model, attempt count, finish reason, and bounded parse-failure diagnostics
   - Does not alter target approval, `active_targets.json`, or Beelink execution

## Tech Stack

- **Language:** Python 3.11
- **LLM:** Gemma 4 21B MoE via LM Studio (Native Windows)
- **Secondary models available:** `qwen2.5-coder` (local dev/scripting on Corsair)
- **Key dependencies:** yfinance, pandas, numpy, alpaca-py, requests
- **Scheduling:** Windows Task Scheduler
- **Transfer:** SCP to Beelink execution node
- **Alerts:** Discord webhooks for failures

## Configuration

Key parameters are in the Python files:
- `MIN_VOLUME`: 2,000,000 (liquidity filter)
- `MIN_PRICE` / `MAX_PRICE`: $15 – $1000
- `LM_STUDIO_URL`: `http://localhost:1234/v1/chat/completions`
- `MODEL_NAME`: `google/gemma-4-26b-a4b` (the model id sent to LM Studio)
- `BEELINK_USER` / `BEELINK_IP` / `BEELINK_PATH`: SCP transfer target — overridable
  in `config.py` so infrastructure coordinates stay out of tracked source
- `ENABLE_SHADOW_ADVISORS`: defaults to `True`; set `False` to skip shadow votes
- `SHADOW_ADVISOR_MODELS`: optional dict mapping `equity_specialist`, `options_specialist`,
  and `crypto_specialist` to smaller local LM Studio models

API credentials are loaded from `config.py` (gitignored) via `import config`
(`API_KEY`, `SECRET_KEY`, `PAPER`, `WEBHOOK_OVERSEER`, optional `BEELINK_*`).

Note: the Beelink-side fleet runs inside the `trading-fleet` Docker container with
the repo live-mounted from `~/bots/repo/` — which is exactly where `BEELINK_PATH`
drops `active_targets.json`, so the file is visible in-container immediately.

## Architecture Notes

- **No build system** — standalone Python scripts, no packaging
- **Test coverage:** `test_parser_logic.py` (LLM JSON response parsing — clean, chatty, broken),
  `test_shadow_advisors.py` (specialist routing, parsing, fallback, persistence),
  `test_scp_logic.py` (SCP transfer retry/backoff), and `test_scoring_logic.py` (confidence
  weighting and technical-score normalization).
- **No linter/formatter configured** — code follows loose PEP 8 style with 4-space indentation
- **Logging:** `scout_log.txt` with emoji-annotated output. Typical run: ~50 tickers analyzed,
  95–98% approval rate, average confidence 0.68–0.72.
- **Transfer is fire-and-forget:** If SCP fails after 3 retries, bots continue using the
  previous `active_targets.json` on the Beelink. A Discord alert fires but the fleet doesn't
  stop.

## Version Control

Strict separation of state and code:

- **Tracked:** `sector_scout_3.py`, `market_scanner.py`, `test_*.py`, `requirements.txt`, `CLAUDE.md`, `CORSAIR_ARCHITECTURE.md`, `run_scout.bat`
- **Untracked:** `config.py`, `keys.json`, `*.log`, `scout_log.txt`, `active_targets.json`,
  `dragnet_candidates.json`

### Commit Style

Same as fleet repo: lowercase, concise, no conventional commit prefixes.

## Key Lessons

- **Scout frequency was massively over-engineered.** Running every 30 minutes produced nearly
  identical target lists. 3x daily captures all meaningful market shifts with negligible signal
  loss and eliminates ~80% of GPU inference time.
- **LLM response parsing must be defensive.** The model occasionally wraps JSON in
  markdown code blocks, adds preamble text, or returns non-JSON entirely. Shadow specialists
  use LM Studio JSON Schema output, a first-valid-object decoder, and one bounded repair attempt;
  do not replace that decoder with a greedy brace regex.
- **Tech score anchors confidence.** The technical indicator score from Phase 1 carries the
  most weight — LLM sentiment is a refinement layer, not the primary signal. Candidates with
  weak technicals rarely survive even with strong sentiment.
- **SCP is the single point of fragile coupling** between the two machines. A network hiccup
  doesn't crash anything (bots use stale targets), but monitoring the transfer success rate
  matters for signal freshness.
