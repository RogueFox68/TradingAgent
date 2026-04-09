# ARCHITECTURE: Corsair AI Workstation ("The Brain")

## 1. Hardware & Environment
* **Role:** Market Analysis, Sentiment Scoring, and Target Generation.
* **CPU/APU:** AMD Strix Halo
* **Memory:** 96GB Unified Memory (VRAM/RAM shared)
* **OS:** Ubuntu Linux (Bare Metal)
* **GPU Compute:** AMD ROCm stack leveraging `/dev/kfd` and `/dev/dri` pass-through.

## 2. AI & LLM Infrastructure
The AI pipeline runs locally to process live market data without external API costs or rate limits.
* **Backend:** Ollama (Dockerized)
* **Active Market Analyst Model:** `llama3.3:70b` (70 Billion parameters)
  * *Resource Footprint:* Consumes ~80GB VRAM during inference.
  * *Purpose:* Reads live Reddit sentiment, news headlines, and Alpaca technical data to score and categorize stock tickers.
* **Secondary Models Available:** `qwen2.5-coder` (for local development/scripting).

## 3. Container Topology
The system uses isolated Docker containers to prevent dependency collisions.
* **Container 1: `ollama_backend`**
  * Passes through AMD hardware via `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
  * Exposes port `11434` to the host network.
* **Container 2: `sector_scout_bot`**
  * A lightweight Python 3.11 environment running `alpaca-py`.
  * Executes the `sector_scout_3.py` script.
  * Uses a live volume mount (`-v ~/trading_desk/TradingAgent/:/app`) so code updates on the host are immediately reflected inside the container.

## 4. Automation & Scheduling
The pipeline is fully autonomous and decoupled from Docker's internal scheduling. It uses native Linux `systemd` timers to fire the execution script.
* **Schedule:** Runs Monday through Friday, 3 times daily (Central Time):
  * `08:30:00` (Market Open)
  * `12:00:00` (Mid-day / Lunch)
  * `15:00:00` (Market Close)
* **Mechanism:** A systemd timer triggers a service that runs `docker exec -it sector_scout_bot ./run_scout.sh`.

## 5. Version Control & Data Separation
The source code is tracked via Git (GitHub via SSH), but strict separation of state and code is enforced.
* **Tracked Files:** `sector_scout_3.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, systemd templates.
* **Untracked (Ignored) Files:**
  * `keys.json` (Alpaca API credentials)
  * `*.log` and `scout_log.txt` (Execution logs)
  * `active_targets.json` (Live payload for the Beelink)
  * `dragnet_candidates.json` (Raw ticker list)

## 6. The Output Payload
Once the Llama 70B model finishes analyzing the market, the `sector_scout_bot` container writes the final results directly to the host drive at `~/trading_desk/TradingAgent/active_targets.json`. This JSON file acts as the bridge payload that the Beelink Execution Node will monitor and trade against.