# CLAUDE.md

## Project Overview

TradingAgent is an automated stock scanning and AI analysis system. It scans the US equity market via the Alpaca API, filters candidates using technical indicators (RSI, ADX, SMA), then scores them with a local Ollama LLM (Llama 3.1) for sentiment analysis.

Stocks are categorized into 5 trading strategies:
- **trend_targets** - Strong bullish momentum
- **survivor_targets** - Oversold bounce plays
- **wheel_targets** - Stable/neutral bull positions
- **condor_targets** - Sideways/range-bound
- **short_targets** - Bearish setups

## Project Structure

```
market_scanner.py      # Phase 1: Scans full market, outputs dragnet_candidates.json
sector_scout_3.py      # Phase 2: AI analysis of candidates, outputs active_targets.json
test_analyst.py        # Manual single-ticker test script
run_scout.bat          # Windows Task Scheduler automation script
keys.json              # Alpaca API credentials (DO NOT COMMIT secrets)
active_targets.json    # Generated output - current trading targets
dragnet_candidates.json # Generated output - scanned candidates
scout_log.txt          # Execution log
```

## Tech Stack

- **Language**: Python 3.11
- **Key dependencies**: yfinance, pandas, numpy, alpaca-trade-api, requests, ollama, GoogleNews

## How to Run

```bash
# Phase 1: Scan market universe
python market_scanner.py

# Phase 2: AI-powered analysis of scanned candidates
python sector_scout_3.py

# Single ticker test
python test_analyst.py
```

On Windows, `run_scout.bat` runs both phases sequentially and is triggered by Task Scheduler during market hours (Mon-Fri 08:00-15:00 CST).

## Configuration

Key parameters are hard-coded in the Python files:
- `MIN_VOLUME`: 1,500,000 (liquidity filter)
- `MIN_PRICE` / `MAX_PRICE`: $15 - $500
- `OLLAMA_URL`: `http://localhost:11434/api/generate` (local LLM endpoint)
- `MODEL_NAME`: `llama3.1`

API credentials are loaded from `keys.json`.

## Architecture Notes

- **No build system** - standalone Python scripts, no packaging
- **No test framework** - `test_analyst.py` is for manual ad-hoc testing
- **No linter/formatter configured** - code follows loose PEP 8 style with 4-space indentation
- Results are deployed to a remote machine (Beelink) via SCP
- Logging goes to `scout_log.txt` with emoji-annotated output
