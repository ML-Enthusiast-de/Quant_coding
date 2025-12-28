FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.
This repository contains experimental trading research code. Nothing here is production-grade, nothing is an offer/solicitation, and it may be wrong, incomplete, or misleading.
Do not trade real money based on this code. Use at your own risk.

Track 3 — Crypto Order Book Microstructure (WIP)
Goal

Build a full microstructure research pipeline based on real-time exchange market data:

WebSocket data → normalized event stream → L2 book reconstruction → microstructure features → forward labels → predictive models → evaluation + (paper) execution simulation

Why this matters


event-driven systems

latency + data quality issues

stateful L2 book maintenance

microstructure features (imbalance, OFI, spreads, depth)

realistic evaluation (slippage, fees, queueing assumptions)


What we compute (planned)
Book state / microstructure features

best bid/ask, spread

midprice, microprice

depth at top N levels

imbalance (top 1 / top k)

order flow imbalance (OFI) variants

short-horizon volatility proxies

event intensity (updates/sec), burstiness

Targets / labels (planned)

future midprice move over horizon (classification)

future return / microprice change (regression)

triple-barrier style labels (optional)

Models (planned)

baselines: logistic regression, linear/ridge, tree models

lightweight neural baselines (MLP / 1D conv)

strict time-split evaluation + walk-forward

Evaluation (planned)

out-of-sample AUC / logloss, calibration

information coefficient (IC)

turnover + trade count constraints

simple execution sim with fees + slippage assumptions