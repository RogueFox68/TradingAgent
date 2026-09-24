# LLM-gate ablation backtest

**Question:** does the LLM stage of the scout help the fleet? If the LLM-gated fleet does
better, is that because of **what** it picks, or only because it trades **less**?

Comparing the live fleet against a top-100 technical backtest can't answer either part. It
changes the scanner and the LLM at the same time. It measures the LLM against live P&L that
also carries unrelated bugs (survivor's stale bars, stops disabled from 15:30). And it has
no control for trading less. This harness replays the fleet's own trend_bot and survivor_bot
rules against several target lists that differ in exactly one respect each.

## Arms

All arms use the same simulator, rules, costs, starting equity and CFO budgets.

| Arm | Target list | Confidence (sizing) | Visible from |
|---|---|---|---|
| `A_topN` | Point-in-time top-N by trailing 20-day dollar volume | flat | the open |
| `B_all_scanner_time` | Every scanner candidate (LLM removed) | flat | scanner finish |
| `B_all_publish_time` | Every scanner candidate | flat | LLM publish |
| `C_llm_flat_conf` | LLM-approved only | flat | LLM publish |
| `C_llm` | LLM-approved only (what the fleet got) | LLM's | LLM publish |
| `D_random_k` (x500) | Random picks, **same count per run and bucket** as the LLM | flat | LLM publish |

"Flat" is the median approved confidence in the window, so average position size matches
across arms.

Each pair of adjacent arms isolates one step, reported as a paired daily-P&L difference with a
block-bootstrap 95% interval:

- **Universe** `A - B_scanner`: a broad top-N list vs the deterministic scanner.
- **Latency** `B_scanner - B_publish`: what waiting for the LLM costs (median 55 min).
- **LLM selection** `C_flat - B_publish`: the LLM's filter, sizing held flat.
- **LLM sizing** `C - C_flat`: confidence-scaled sizing.
- **Trading less vs choosing well**: `C_flat` placed in the distribution of `D`. Between the
  5th and 95th percentile means the picks can't be told apart from random picks at the same
  rate.

A separate test doesn't use the simulator at all (report section 5). It takes every candidate
the scout ever analysed, approved or rejected, and the return that followed. It reports
approved-minus-rejected spreads with day-clustered intervals, and Fama-MacBeth rank ICs for
the tech score, the LLM score, and the **LLM score after removing what the tech score already
explains** (`llm_resid_ic`). That last one is the most direct answer to "does the LLM know
anything the technicals don't?", and it has ~6,000 observations instead of a few hundred
trades.

## Where the LLM's decisions come from

`scout_log.txt`. `run_scout.bat` appends to it, and the scout prints every candidate with
its approve/reject mark and its sub-scores. The mark is used as-is, never re-derived:
the threshold was about 0.50 until mid-June 2026 and 0.66 since. The LLM can't be re-run after the
fact, because yfinance only serves current headlines.

The targets file is modelled as the fleet reads it (`utils.load_and_validate_targets`). The
newest *transferred* run wins. A file whose `updated` stamp (set when the scout **starts**)
is over 24h old reads as empty. That is why the first ~hour of every Monday trades nothing.

LLM outages are kept as they happened. `ask_llama` used to score a failed call 0.0, so when
LM Studio was down every candidate was rejected and an empty "success" file was published.
The report lists those runs. The random arm matches the LLM's approval count of zero on them.
The candidate-level signal tests exclude failed calls, because a failed call is not a
judgment. The scout now scores a failed call as missing. It prints the call as `N/A` and
names it in a `(LLM failed: T1, T2)` suffix, which the parser reads into
`Candidate.failed`. It also refuses to publish a run where more than half the calls failed
or more than half the candidates had no news. The report lists news-outage runs as well
(`analysis.news_outage_runs`). On pre-guard history, a normal-looking run in that list is a
false positive of the guard's 0.5 threshold.

## Running it on the Corsair

From the TradingAgent directory, with the project venv (`config.py` supplies the Alpaca
keys; `ta` is **not** required):

```bat
:: 1. What the gate did, from the log alone (seconds, no network)
venv\Scripts\python -m backtest.run_backtest --scout-log scout_log.txt --decisions-only

:: 2. The full backtest (downloads and caches bars on first run)
venv\Scripts\python -m backtest.run_backtest --scout-log scout_log.txt ^
    --shadow-votes shadow_advisor_votes.jsonl --fleet-repo ..\trading-bot-fleet
```

- Defaults: the window is the 3 months ending the day before the log's last run
  (`--start` / `--end` to override), `--top-n 100`, `--null-draws 500`, `--equity 100000`,
  `--slippage-bps 5`.
- **Rules check.** The strategy rules are restated in `backtest/rules.py`, because the bot
  modules can't be imported off the Beelink. On every run they are checked against the
  fleet's **source** (read, not imported). The run refuses to start on any mismatch. Clone
  `trading-bot-fleet` beside this repo, or pass `--allow-unverified-rules`; the report says
  which was used.
- **Data.** Bars are cached under `backtest_cache/` (gitignored). The first run fetches daily
  bars for the whole active universe (to rank the top-N) and 15m bars for every candidate and
  top-N name.
- **Outputs.** `backtest_out/report.md`, a `trades_<arm>.csv` per arm, `arm_metrics.csv`,
  `null_draws.csv` and `candidate_forward_returns.csv`.

Tests: `python -m unittest test_backtest`.

## Simulator conventions

- It steps once per 15m bar during the regular session. Each decision reads only bars
  **completed** by that time and fills at the next bar's open. Stops and targets run through
  each bar's high/low; a bar that opens through a level fills at the open, and if one bar
  spans both levels the stop fills first.
- Budgets follow `fleet_bot.size_position`: `equity x risk x (0.5 + confidence)`, clipped to
  the bot's CFO budget (`base_allocation x 0.95 x equity`) and its max position share.
- The survivor blacklist and trend's long-before-short `elif` are reproduced. `A_topN` has a
  single shared list, so it turns off the blacklist (which would otherwise empty survivor) and
  lets trend take either direction.
- Tiered-hold scoring decides the 15:45 sweep, and max-hold (3d/7d) applies.

## Limitations (identical in every arm)

- Not simulated: regime/VIX gating and sizing (fixed at SIDEWAYS / VIX 18), CFO reallocation,
  CAPITAL_CRUNCH, and wheel_bot and its ownership interactions.
- Borrow cost on shorts and partial fills are not modelled.
- The top-N universe starts from today's active asset list, so names delisted mid-window are
  missing. This is a small survivorship bias.
- Three months is one market regime. An interval that includes 0 means "no answer", not "no
  difference".
