# European Power Fair Value
## Forecasting Day-Ahead Prices & Translating to Prompt Curve Views

A quantitative prototype that forecasts hourly Day-Ahead electricity prices for the German (DE) bidding zone using publicly available ENTSO-E data, and translates those forecasts into tradable base/peak curve views with dynamic, volatility-aware trading signals.

---

## Project Structure

```
├── data/
│   ├── raw/              # Unprocessed ENTSO-E API pulls (.gitignored)
│   └── processed/        # Cleaned, feature-engineered datasets (.gitignored)
├── src/
│   ├── ingestion/        # ENTSO-E API data pull scripts
│   ├── qa/               # Quality assurance & feature engineering
│   ├── models/           # Baseline, LightGBM, XGBoost, ensemble, curve translator
│   └── llm/              # LLM-based REMIT UMM parser
├── logs/                 # Execution logs, LLM outputs, metrics
├── docs/
│   ├── figures/          # Charts and plots
│   └── report.md         # Final submission document
├── .env.example          # Template for API keys
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd Forecasting_European_Power_Fair_Value

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys
```bash
# Copy the example env file and add your keys
cp .env.example .env
# Edit .env with your ENTSO-E and OpenAI API keys
```

### 3. Run the Pipeline (in order)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `python -m src.ingestion.data_ingest` | Pull raw data (ENTSO-E → Energy-Charts → cache fallback) |
| 2 | `python -m src.qa.cleaner` | QA, DST-safe UTC conversion, anomaly detection, feature engineering |
| 3 | `python -m src.models.baseline` | Train baseline models (Naive Persistence, Ridge ARX) |
| 4 | `python -m src.models.lgbm_model` | Train LightGBM with 60-day rolling window |
| 5 | `python -m src.models.xgb_model` | Train XGBoost with 60-day rolling window |
| 6 | `python -m src.models.ensemble` | RMSE-weighted ensemble + model comparison |
| 7 | `python -m src.models.curve_translator` | Generate base/peak views & trading signals |
| 8 | `python -m src.llm.remit_parser` | LLM-based REMIT UMM parsing & signal invalidation |
| 9 | `python -m src.llm.qa_health_report` | LLM-generated daily QA health report |

## Data Sources

All data is sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) under EU Regulation 543/2013:

- **Day-Ahead Prices** (Art. 12.1.D) — target variable
- **Actual Total Load** (Art. 6.1.A) — demand fundamental
- **Wind & Solar Generation Forecasts** (Art. 14.1.D) — supply fundamentals

**Market:** DE_LU (Germany-Luxembourg) bidding zone  
**Timeframe:** January 1, 2024 → yesterday (continuously updated)  
**Resolution:** Hourly (15-min load/gen resampled to hourly)

## Models

| Model | Type | Training Strategy |
|-------|------|-------------------|
| Naive Persistence | Baseline | Same-hour yesterday |
| Ridge Regression (ARX) | Baseline | Full training set |
| LightGBM | Improved | 60-day rolling window |
| XGBoost | Improved | 60-day rolling window |
| Weighted Ensemble | Improved | RMSE-weighted blend |

All improved models produce probabilistic outputs (10th, 50th, 90th quantiles).

## Evaluation Metrics

- **sMAPE** — Symmetric Mean Absolute Percentage Error (handles near-zero prices)
- **MAE** — Mean Absolute Error in €/MWh
- **Directional Accuracy** — % of hours where predicted price direction (up/down) matches actual
- **Pinball Loss** — Quantile-specific calibration (q10, q50, q90)
- **Quantile Crossing Rate** — Ensures q10 ≤ q50 ≤ q90

## QA Approach

- DST-safe UTC conversion (handles CET/CEST transitions)
- Automated anomaly detection: missing hours, duplicate timestamps, monotonicity checks
- Negative prices preserved (real market events from renewable oversupply)
- Outliers flagged via IQR but **never deleted** — extreme spikes are genuine
- Source provenance tracking (`source_quality` flag: PRIMARY/FALLBACK/CACHED)

## License

This project is a prototype case study submission.
