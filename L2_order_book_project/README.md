# Track 3 — Crypto Order Book Microstructure (L2) — WIP

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

## What we compute (planned)

### Book state / microstructure features
- Best bid/ask, **spread**
- **Midprice**, **microprice**
- Depth at top **N levels**
- **Imbalance** (top 1 / top k)
- **Order Flow Imbalance (OFI)** variants
- Short-horizon volatility proxies
- Event intensity (updates/sec), burstiness

### Targets / labels
- Future midprice move over horizon (**classification**)
- Future return / microprice change (**regression**)
- Triple-barrier style labels (optional)

### Models
- Baselines: **logistic regression**, **linear/ridge**, **tree models**
- Lightweight neural baselines (**MLP / 1D conv**)
- Strict time-split evaluation + **walk-forward**

### Evaluation
- Out-of-sample AUC / logloss, calibration
- **Information coefficient (IC)**
- Turnover + trade count constraints
- Simple execution sim with **fees + slippage assumptions**
