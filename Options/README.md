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

2. **Validate + clean**  
   Identify static no-arbitrage issues (monotonicity/convexity across strikes), quantify how often they occur, and improve stability.

3. **Fit a smooth surface**  
   Turn scattered IV points into stable daily smiles/surfaces (initially a simple smooth fit on a grid).

4. **Research surface dynamics**  
   Extract daily “surface factors” and forecast changes in:
   - ATM level (vol level)
   - skew (e.g., 25d risk reversal)
   - curvature (e.g., 25d butterfly)
   - term structure slopes

5. **“Does it pay?” via hedged PnL proxy** *(planned)*  
   Evaluate forecasts with simple, cost-aware hedged PnL proxies (straddles, risk reversals, calendar spreads).

---

## Current status (what’s already implemented)

✅ **Phase 1: Dataset + IV extraction**  
- Built:
  - `data/processed/spy_chain_clean.parquet`
  - `data/processed/spy_iv_points.parquet`

✅ **Phase 2: No-arbitrage diagnostics (calls)**  
- Built:
  - `data/processed/no_arb_summary.parquet`
- Reports monotonicity + convexity violation rates per (quote_date, expiry) call chain.

✅ **Phase 3: Smooth smile fitting → daily grid output**
- Built:
  - `data/processed/surface_smile_grid.parquet`
- Output is a daily per-expiry smile sampled on a log-moneyness grid `k = log(K/F)`.

✅ **Phase 4: Surface factors + forecasting baselines**
- Built:
  - `data/processed/surface_factors.parquet`
    - “Exact tenor” factors (interpolated in T) and
    - “Closest maturity” factors with `gap_days_*` and `T_used_*` diagnostics.
  - `data/processed/factor_forecast_report.csv` / `.parquet` (baselines: RW / EWMA / AR(1))

⬜ **Phase 5: Hedged PnL proxy** *(planned next)*

---

## Data

This repo expects **locally downloaded historical** SPY option chain data (no daily scraping).

A convenient starting point is a Kaggle-style dataset with columns like:

- `QUOTE_DATE`, `EXPIRE_DATE`, `DTE`, `UNDERLYING_LAST`, `STRIKE`
- `C_BID`, `C_ASK`, `P_BID`, `P_ASK`
- optional vendor fields (volume, greeks, vendor IV)

> Note: vendor IV/greeks are not trusted by default. This pipeline computes implied vol via Black–Scholes inversion using mid quotes and an estimated forward from put–call parity.

---

## Pipeline overview (what each stage does)

### 1) Build clean chain + IV points
**Script:** `src/build_option_surface_dataset.py`  
**Outputs:**
- `spy_chain_clean.parquet` (cleaned, long-form quotes)
- `spy_iv_points.parquet` (IV + log-moneyness + forward delta)

**What happens:**
- normalize Kaggle column headers
- compute mid quotes and filter bad markets (spread, tiny mids, etc.)
- estimate forward `F` and discount factor `DF` per (date, expiry) from put–call parity
- compute implied vol by inverting Black–Scholes

---

### 2) No-arbitrage diagnostics (calls)
**Script:** `src/no_arb_report.py` *(or equivalent script name in this repo)*  
**Output:** `no_arb_summary.parquet`

**Checks:**
- monotonicity in strike for call prices
- convexity in strike (butterfly no-arb)

This does **not** “fix” quotes yet — it measures where the market data (or the cleaning) breaks theoretical constraints.

---

### 3) Fit smooth smile(s) and create a daily grid
**Script:** `src/fit_surface_daily.py` *(or equivalent script name in this repo)*  
**Output:** `surface_smile_grid.parquet`

**What happens:**
- take scattered IV points per expiry
- fit a stable smile shape (initial simple approach)
- evaluate the smile on a standard grid of `k = log(K/F)` so every day has comparable coordinates

---

### 4) Extract surface factors + forecast baselines
**Scripts:**
- `src/surface_factors.py` → `surface_factors.parquet`
- `src/forecast_surface_factors.py` → forecast report

**Factors include (examples):**
- ATM vol at target tenors (exact interpolation)  
- 25d Risk Reversal: `IV(25d put) – IV(25d call)`
- 25d Butterfly: `0.5*(IV_put25 + IV_call25) – IV_ATM`
- term slope metrics

**Important:**  
Because this dataset has maturity gaps, the script also outputs:
- `gap_days_30d`, `gap_days_60d`, `gap_days_90d`
- `T_used_60d`, `T_used_90d`
so you can filter to “meaningful tenor matches” for rigorous research.

---

## How to run

From the repo root (Windows):

```bat
python Options\src\build_option_surface_dataset.py
python Options\src\no_arb_report.py
python Options\src\fit_surface_daily.py
python Options\src\surface_factors.py
python Options\src\forecast_surface_factors.py
