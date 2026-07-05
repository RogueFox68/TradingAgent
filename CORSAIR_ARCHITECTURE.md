# ARCHITECTURE: Corsair AI Workstation ("The Brain")

## 1. Hardware & Environment
* **Role:** Market Analysis, Sentiment Scoring, Target Generation, and Fleet Analysis.
* **CPU/APU:** AMD Strix Halo
* **Memory:** 96GB Unified Memory (VRAM/RAM shared)
* **GPU:** NVIDIA RTX 5080 (16GB VRAM — used for CUDA-accelerated specialist tasks)
* **OS:** Windows (returned from a failed Ubuntu/ROCm expedition)

## 2. AI & LLM Infrastructure
The AI pipeline runs locally to process live market data without external API costs or rate limits.
* **Backend:** LM Studio (replaced Ollama after the Linux migration)
* **Primary Model:** Gemma 4 21B MoE
  * *Architecture:* Mixture-of-Experts — activates ~3.8B parameters per token, yielding fast inference with near-dense-model quality.
  * *Resource Footprint:* ~15-18GB during inference (vs. ~80GB for the previous Llama 3.3 70B).
  * *Purpose:* Reads live Reddit sentiment, news headlines, and Alpaca technical data to score and categorize stock tickers. Powers Sector Scout and Market Scanner analysis.
* **Fleet Analyst Model:** TBD — needs to be rebuilt from scratch after being lost in the Linux migration. Candidate: Gemma 4 31B Dense (better reasoning depth for analyst role).

### Model History
| Era | Model | Backend | OS | Notes |
|-----|-------|---------|-----|-------|
| v1 | Llama 3.1 8B | Ollama | Windows | Original prototype |
| v2 | Llama 3.3 70B | Ollama (Docker) | Ubuntu | ~80GB VRAM, ROCm/AMD stack |
| v3 | Gemma 4 31B Dense | Ollama (Docker) | Ubuntu | Short-lived, Linux abandoned |
| v4 (current) | Gemma 4 21B MoE | LM Studio | Windows | Fast, efficient, stable |

## 3. Deployed Services

### Market Scanner (`market_scanner.py`)
* Scans the active US-equity universe from Alpaca (seed list + all tradeable/marginable/shortable)
* Filters for liquidity (volume > 2M, price $15–$1000)
* Calculates technical indicators: RSI(14), ADX(14), SMA(20/50/200)
* Categorizes into strategy buckets (trend, survivor, wheel, short)
* Outputs top 10 per category to `dragnet_candidates.json`

### Sector Scout (`sector_scout_3.py`)
* Analyzes candidates from Market Scanner using multi-source intelligence:
  - Tier 1: Elite financial news (yfinance, bucketed by publisher)
  - Tier 2: Mainstream news (yfinance)
  - Tier 3: Specialty/industry news (yfinance)
  - Social: Reddit sentiment (public JSON API)
* Scores via Gemma 4 21B MoE with role-specific system prompts per strategy
* Composite confidence: `weighted_average(Tech, T1, T2, T3, Social)`
* Approval threshold: 0.66
* Writes `active_targets.json` and transfers to Beelink via SCP

### Fleet Analyst (PLANNED — needs rebuild)
* Was previously running via Open WebUI with custom tools on Linux
* Lost in the Linux → Windows migration
* Needs: InfluxDB querying, fleet status reading, workspace analysis
* Model candidate: Gemma 4 31B Dense (reasoning depth > MoE for analyst role)

## 4. Automation & Scheduling
* **Schedule:** TID (Three Intraday) — runs Monday through Friday, 3 times daily during market hours (Central Time)
* **Mechanism:** Windows Task Scheduler (replaced Linux systemd timers)
* Both Market Scanner and Sector Scout run sequentially each session

## 5. Version Control & Data Separation
Source code tracked via Git (GitHub). Strict separation of state and code.
* **Tracked Files:** `market_scanner.py`, `sector_scout_3.py`, `requirements.txt`, `test_parser_logic.py`, `test_scp_logic.py`, `test_scoring_logic.py`
* **Untracked (Ignored) Files:**
  * `keys.json` (Alpaca API credentials)
  * `*.log` and `scout_log.txt` (Execution logs)
  * `active_targets.json` (Live payload for the Beelink)
  * `dragnet_candidates.json` (Raw ticker list)

## 6. The Output Payload
Once Gemma 4 21B MoE finishes analyzing the market, Sector Scout writes `active_targets.json`. This JSON file is transferred to the Beelink Execution Node via SCP and acts as the bridge payload that the trading bots monitor and trade against.

## 7. Multi-Machine Topology

```
┌─────────────────────────────────────┐
│  Corsair AI Workstation ("Brain")   │
│  Windows · Strix Halo · 96GB       │
│  LM Studio + Gemma 4 21B MoE       │
│                                     │
│  Market Scanner → Sector Scout      │
│         ↓ active_targets.json       │
│         ↓ (SCP transfer)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Beelink S12 Mini ("Executor")     │
│  Ubuntu · PM2 · Docker             │
│                                     │
│  survivor_bot · trend_bot           │
│  wheel_bot · crypto_grid            │
│  moon_bot · commander               │
│  accountant · market_analyst        │
│                                     │
│  InfluxDB 1.8 · Grafana            │
│  WireGuard · Discord Bot            │
└─────────────────────────────────────┘
```

## 8. Known Issues & Tech Debt
* Fleet Analyst needs full rebuild on Windows/LM Studio
* `CORSAIR_ARCHITECTURE.md` in the project repo was severely outdated (referenced Ubuntu, Ollama, Llama 3.3 70B, ROCm, Docker containers) — replaced with this version
* ~~Bare `except: pass` in InfluxDB write functions across the Beelink fleet~~ — resolved
  fleet-side: writes check status codes, and `error_watchdog` ships the fleet error
  registry into InfluxDB (`bot_error_events`)