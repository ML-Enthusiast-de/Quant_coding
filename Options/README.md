# SPY Option Surface Research Lab (EOD)

Public, research-only repo for building daily **SPY implied volatility surfaces** from end-of-day option chain data, validating data quality and no-arbitrage constraints, fitting a smooth surface, and researching **surface dynamics** (ATM level, skew, term structure) with simple forecast + hedged PnL proxies.

> **Disclaimer (Research Only):**  
> This project is for educational and research purposes only. Nothing here is investment advice, a recommendation, or an offer to buy/sell any security or derivative. Any “PnL” or “strategy” results are illustrative backtest proxies that may be unrealistic and are not indicative of future performance. Data quality varies; results depend strongly on cleaning choices and assumptions.

---

## Why this repo exists

Index / equity options surfaces are a canonical playground for:
- no-arbitrage reasoning and market microstructure hygiene
- surface modeling (smile + term structure)
- Greeks / hedging intuition
- disciplined quantitative research workflows

This repo starts with **EOD (end-of-day)** SPY option chains to build a robust pipeline without needing intraday infrastructure.

---

## Project goals

1. **Data → trusted surface points**  
   Convert raw option chains into a clean dataset and compute implied vol (IV) points.

2. **Validate + clean** (planned)  
   Identify and handle static / calendar arbitrage violations, track how often they occur, and improve surface stability.

3. **Fit a smooth surface** (planned)  
   Turn scattered IV points into a stable function (e.g., splines / SSVI / SABR-family approaches).

4. **Research surface dynamics** (planned)  
   Model and forecast daily changes in:
   - ATM level
   - skew (e.g., 25d put – 25d call)
   - curvature
   - term structure slope

5. **“Does it pay?” via hedged PnL proxy** (planned)  
   Evaluate forecasts with simple, cost-aware hedged PnL proxies (e.g., straddle, risk reversal, calendar spread).

---

## Current status

✅ **Phase 1 implemented:** build dataset + implied vol extraction (EOD)  
⬜ Phase 2–5 planned (no-arb checks, surface fitting, forecasting, hedged PnL proxy)

---

## Data

This repo expects **locally downloaded historical** SPY option chain data (no daily scraping).
A convenient starting point is a Kaggle-style dataset with columns like:

- `QUOTE_DATE`, `EXPIRE_DATE`, `DTE`, `UNDERLYING_LAST`, `STRIKE`
- `C_BID`, `C_ASK`, `P_BID`, `P_ASK`
- (optional) `C_VOLUME`, `P_VOLUME`, and vendor greeks / IV

