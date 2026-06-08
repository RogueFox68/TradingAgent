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

Stocks are categorized into 5 trading strategies:
- **trend_targets** — Strong bullish momentum
- **survivor_targets** — Oversold bounce plays (mean reversion)
- **wheel_targets** — Stable/neutral, good for options premium selling
- **condor_targets** — Sideways/range-bound (iron condor candidates)
- **short_targets** — Bearish setups (trend_bot short entries)

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
├── test_analyst.py              # Manual single-ticker test script
├── test_parser_logic.py         # Unit tests for LLM JSON response parsing
├── requirements.txt             # Python dependencies
├── run_scout.bat                # Windows Task Scheduler automation
│
├── (Gitignored - Runtime state)
│   ├── keys.json                # Alpaca API credentials
│   ├── active_targets.json      # Generated output — current trading targets
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
3. Filters for liquidity (volume > 1.5M, price $15–$500)
4. Calculates technical indicators: RSI(14), ADX(14), SMA(20/50/200)
5. Categorizes into strategy buckets based on indicator profiles
6. Outputs top 10 per category to `dragnet_candidates.json`

### Phase 2: Sector Scout (`sector_scout_3.py`)

For each candidate from Phase 1:

1. **Gathers multi-source intelligence:**
   - Tier 1: Direct financial news (yfinance)
   - Tier 2: Broader news (GoogleNews)
   - Tier 3: Additional context
   - Social: Reddit/social sentiment

2. **Scores via Gemma 4 21B MoE** — each source gets its own LLM call with a role-specific
   system prompt (hedge fund analyst for trend, value investor for survivor, options income
   trader for wheel, short seller for shorts)

3. **Calculates composite confidence:**
   ```
   Conf = weighted_average(Tech, T1, T2, T3, Social)
   ```
   Candidates scoring above 0.50 are approved.

4. **Validates LLM responses:**
   - Clamps scores to 0.0–1.0
   - Penalizes weak reasoning (<50 chars) by 30%
   - Catches "insufficient data" hedging and assigns 0.5

5. **Writes `active_targets.json`** and **SCP transfers to Beelink**
   - 3 retry attempts with 5-second backoff
   - Discord webhook alert on transfer failure

## Tech Stack

- **Language:** Python 3.11
- **LLM:** Gemma 4 21B MoE via LM Studio (Native Windows)
- **Secondary models available:** `qwen2.5-coder` (local dev/scripting on Corsair)
- **Key dependencies:** yfinance, pandas, numpy, alpaca-py, requests, GoogleNews
- **Scheduling:** Windows Task Scheduler
- **Transfer:** SCP to Beelink execution node
- **Alerts:** Discord webhooks for failures

## Configuration

Key parameters are in the Python files:
- `MIN_VOLUME`: 1,500,000 (liquidity filter)
- `MIN_PRICE` / `MAX_PRICE`: $15 – $500
- `LM_STUDIO_URL`: `http://localhost:1234/v1/chat/completions`
- `MODEL_NAME`: `gemma4-21b`
- `BEELINK_USER` / `BEELINK_IP` / `BEELINK_PATH`: SCP transfer target

API credentials loaded from `keys.json` (gitignored).

## Architecture Notes

- **No build system** — standalone Python scripts, no packaging
- **Test coverage:** `test_parser_logic.py` covers LLM JSON response parsing (clean, chatty,
  and broken responses). `test_analyst.py` is for manual ad-hoc single-ticker testing.
- **No linter/formatter configured** — code follows loose PEP 8 style with 4-space indentation
- **Logging:** `scout_log.txt` with emoji-annotated output. Typical run: ~50 tickers analyzed,
  95–98% approval rate, average confidence 0.68–0.72.
- **Transfer is fire-and-forget:** If SCP fails after 3 retries, bots continue using the
  previous `active_targets.json` on the Beelink. A Discord alert fires but the fleet doesn't
  stop.

## Version Control

Strict separation of state and code:

- **Tracked:** `sector_scout_3.py`, `market_scanner.py`, `test_*.py`, `requirements.txt`, `CLAUDE.md`, `CORSAIR_ARCHITECTURE.md`, `run_scout.bat`
- **Untracked:** `keys.json`, `*.log`, `scout_log.txt`, `active_targets.json`,
  `dragnet_candidates.json`

### Commit Style

Same as fleet repo: lowercase, concise, no conventional commit prefixes.

## Key Lessons

- **Scout frequency was massively over-engineered.** Running every 30 minutes produced nearly
  identical target lists. 3x daily captures all meaningful market shifts with negligible signal
  loss and eliminates ~80% of GPU inference time.
- **LLM response parsing must be defensive.** The model occasionally wraps JSON in
  markdown code blocks, adds preamble text, or returns non-JSON entirely. The regex fallback
  extractor (`re.search(r'\{.*\}', raw_text, re.DOTALL)`) is critical.
- **Tech score anchors confidence.** The technical indicator score from Phase 1 carries the
  most weight — LLM sentiment is a refinement layer, not the primary signal. Candidates with
  weak technicals rarely survive even with strong sentiment.
- **SCP is the single point of fragile coupling** between the two machines. A network hiccup
  doesn't crash anything (bots use stale targets), but monitoring the transfer success rate
  matters for signal freshness.
