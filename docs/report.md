# European Power Fair Value: Forecasting Day-Ahead and Prompt Curve Translation
**Candidate:** [Your Name] | **Email:** [Your Email]

---

## 1. Methodology Overview
This project delivers a resilient, end-to-end quantitative prototype that forecasts hourly Day-Ahead electricity prices for the German (DE_LU) bidding zone. The pipeline fetches data directly from the official **ENTSO-E API** (with auto-failover to Energy-Charts API and local caches to ensure 100% uptime). 

Three core feature sets were engineered:
1. Hourly Day-Ahead Prices (Target)
2. Actual Total Load (Demand proxy)
3. Wind & Solar Generation Forecasts (Supply proxy)

The pipeline incorporates automated Quality Assurance (QA) handling DST transitions gracefully, replacing extreme anomalies via forward filling while preserving true negative prices that reflect genuine renewable oversupply.

## 2. Forecasting & Validation Rigor
To robustly predict prices, the system implements a suite of progressively advanced models tested across an expanding 60-day rolling window:
- **Baseline Models:** Naive Persistence (Same-hour yesterday) and Ridge Regression (ARX).
- **Advanced Models:** LightGBM and XGBoost, configured to output probabilistic quantiles (q10, q50, q90).
- **Ensemble:** A weighted ensemble that blends LightGBM and XGBoost based on rolling RMSE performance.

### Validation Metrics (Latest End-to-End Run)
* Evaluated iteratively to prevent lookahead bias. Symmetric Mean Absolute Percentage Error (sMAPE) handles instances where real prices hit exactly zero or negative.

| Model | sMAPE (%) | MAE (€/MWh) |
|---|---|---|
| **Naive Persistence (Baseline)** | 30.92% | 24.23 |
| **Ridge Regression (Baseline)** | 24.85% | 16.64 |
| **Linear Regression (OLS)**| 21.58% | 15.48 |
| **MLP (Neural Network)**| 22.50% | 21.62 |
| **LightGBM** | 16.84% | 11.36 |
| **XGBoost** | 16.39% | 10.88 |
| **CatBoost** | 16.93% | 11.44 |
| **Ensemble (Final)** | **16.40%** | **11.02** |

*Note: While standalone XGBoost marginally outperformed the Ensemble on this specific historical slice, the Weighted Ensemble is retained as the final production model to reduce variance and ensure robust performance across unseen market regime shifts. All quantile models maintained strict monotonicity (Zero quantile crossings), ensuring calibrated risk intervals.*

### Feature Ablation Study
To quantify the value of our fundamental features, an ablation study was conducted using the tuned LightGBM model over an identical 60-day walk-forward process:

| Feature Set | sMAPE (%) | MAE (€/MWh) |
|---|---|---|
| **1. Base** (Time + Historical Prices) | 25.27% | 20.19 |
| **2. Base + Demand** | 18.38% | 12.52 |
| **3. Base + Supply** | 18.57% | 12.53 |
| **4. Full Engine** | **16.97%** | **11.08** |

*Finding: Adding fundamentally driven data (load or wind/solar) independently cuts error drop by ~7%, and utilizing the full system yields the most optimal predictive accuracy.*

## 3. Prompt Curve Translation & Trading Relevance
The hourly ensemble forecasts are systematically aggregated into a tradable **DA-to-Curve View** (combining Baseload and Peakload averages for the prompt week).

**Guidance on Usage:**
The `curve_translator.py` module compares the model's Fair Value (Base q50) against mocked forward prices to extract a **Risk Premium (RP)**. 
- **Execution:** We allocate a `Strong Buy` / `Strong Sell` conviction when the traded forward price falls entirely outside our model's 10th-90th percentile confidence bands.
- **Invalidation Criteria:** Models explicitly reject signals under three strict heuristics: 
    1. The confidence interval is too wide (High market uncertainty).
    2. Actual prices enter extreme regimes (>$200 or <$-10) triggering a regime shift assumption.
    3. The calculated Risk Premium is too small (< €2/MWh) to justify friction and slippage costs.

*(See `/docs/figures/curve_view.png` for a visual breakdown of predicted Fair Value vs Forward Prices and generated signal conviction).*

## 4. Programmatic AI/LLM Integration
To transition from pure quantitative modeling to fundamental analysis, two specific AI/LLM components were built to reduce manual analyst workload:

1. **REMIT UMM Parsing (`remit_parser.py`):** Electricity markets are heavily driven by supply outages. This module programmatically queries an LLM to read subjective unformatted text about Planned Maintenance or Extreme Weather alerts and parses them into structured relevance scores. Crucially, the LLM determines if an event is severe enough (e.g., "Generation Outage") to **invalidate statistical trading signals** that otherwise appear profitable.
2. **QA Health Reporting (`qa_health_report.py`):** A daily automated system uses an LLM to read through execution logs and synthetically generate a human-readable summary of data quality, model metrics, and active trading signals, mimicking what a desk quant might deliver in morning meetings.

---

### Artifacts Submitted:
- `docs/report.md` (This document)
- `README.md` (Environment setup and pipeline overview)
- `main.py` (Unified Orchestrator) & `requirements.txt`
- `src/` (All documented pipeline code)
- `logs/` (AI logs and curve translations)
- `docs/figures/curve_view.png` (Visualization)
