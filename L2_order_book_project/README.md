# Track 3 — Crypto Order Book Microstructure (L2) — WIP (Project Closed)

## ⚠️ Disclaimer (read first)
**FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.**

This repository contains **experimental trading research code**. Nothing here is production-grade, nothing is an offer/solicitation, and it may be wrong, incomplete, or misleading.  
**Do not trade real money based on this code. Use at your own risk.**

---

## Goal
Build a full microstructure research pipeline based on real-time exchange market data:

**WebSocket data → normalized event stream → L2 book reconstruction → microstructure features → forward labels → predictive models → evaluation + (paper) execution simulation**

---

## Why this matters
Microstructure research forces you to handle real-world trading-system constraints:

- **Event-driven systems** (not bar-based)
- **Latency + data quality issues**
- **Stateful L2 book maintenance**
- **Microstructure features** (imbalance, OFI, spreads, depth)
- **Realistic evaluation** (slippage, fees, queueing assumptions)

---

## What we compute

### Book state / microstructure features (implemented + planned)
- Best bid/ask, **spread**
- **Midprice**, **microprice**
- **Imbalance** (top 1 / top k) *(top-of-book implemented; multi-level planned)*
- **Order Flow Imbalance (OFI)** variants *(planned)*
- Short-horizon volatility proxies *(implemented)*
- Event intensity (updates/sec), burstiness *(planned/partial)*

### Targets / labels (implemented)
- Future midprice move over horizon (**classification**, 3-class: -1/0/+1)
- Future return / microprice change (**regression**)
- Triple-barrier style labels *(optional / planned)*

### Models (implemented + planned)
- Baselines: **logistic regression**, **linear/ridge**
- Tree models *(explored; expansion planned)*
- Lightweight neural baselines (**MLP / 1D conv**) *(planned)*
- Strict time-split evaluation + **walk-forward** *(core principle; partial tooling)*

### Evaluation (implemented + planned)
- Classification metrics (accuracy, balanced accuracy, macro-F1, confusion matrix)
- **Information coefficient (IC)**
- Turnover + trade count constraints
- Execution-aware simulation with **fees + spread/slippage assumptions** *(in progress → see status)*

---

## Project status (Closed)
This project is now **closed** as a completed research spike / learning project.

### Summary (what I built)
I built a high-frequency research pipeline on crypto Level-2 order book data:

- **Streaming data capture** from exchange WebSocket (snapshots + deltas)
- **Stateful L2 book reconstruction** (top-of-book time series from event stream)
- **Feature engineering** for microstructure signals (spread/mid/microprice, imbalance, short-horizon returns/volatility, change-rate flags)
- **Forward label generation** over multiple horizons (e.g., 10s → 600s)
- **Baseline modeling** with strict chronological splits and leakage guards
- **Cost-aware evaluation tooling**, including simplified execution frictions:
  - maker/taker fee scenarios
  - spread costs
  - trade gating via cost-aware thresholds
  - non-overlapping “hold-to-horizon” trade simulation (to avoid overlapping label/trade bias)

### Key learnings (why this project was valuable)
- **Costs dominate at short horizons.** Even when IC is non-zero, realistic taker + spread costs can wipe out almost all “tradable” opportunities.
- **Overlapping-trade backtests are misleading.** Allowing re-entry every second for a 600s horizon can create fake performance. Enforcing non-overlap collapses trades (and reveals reality).
- **Leakage is easy to introduce.** Including *_fwd or future-derived columns in features can silently produce “perfect” results. Strong feature filtering and embargo/purge logic are mandatory.
- **Accuracy is not the goal.** Microstructure problems are heavily imbalanced (tons of “no move”). Classification metrics can look fine while PnL is negative.
- **Data volume matters.** A few days of data is not enough for stable conclusions at longer horizons with realistic constraints (non-overlap + cost thresholds).
- **The pipeline pieces are reusable.** The reconstruction + labeling + strict evaluation framework transfers well to other market data research.

### Why I’m closing it
The pipeline achieved its purpose: validate feasibility, expose pitfalls (costs, leakage, overlap), and produce a reusable research scaffold.  
Pushing it toward production-grade execution simulation and profitable strategies would require significantly more data, more realistic execution modeling (queue position, latency, partial fills), and longer iteration cycles.

---

## Repo structure (high level)
*(Paths may evolve as this repo is reorganized.)*

- `L2_order_book_project/data/raw/`  
  WebSocket run folders with `part_*.parquet`
- `L2_order_book_project/data/processed/`  
  Reconstructed top-of-book (`tob_*`) including an `ALL` file
- `L2_order_book_project/data/datasets/`  
  Feature + label datasets per horizon (`tob_dataset_*_<horizon>.parquet`)
- `L2_order_book_project/data/reports/`  
  JSON reports from baseline + cost-aware experiments
- `L2_order_book_project/scripts/`  
  End-to-end scripts (reconstruction → dataset → baselines → reports)

---

## Notes
If you’re reading this repo later: treat it as a **research notebook in code form**, not a trading system.

Again: **not financial advice**.
