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
| 1 | `python src/ingestion/entsoe_pull.py` | Pull raw data from ENTSO-E API |
| 2 | `python src/qa/cleaner.py` | QA, UTC conversion, feature engineering |
| 3 | `python src/models/baseline.py` | Train baseline models (Naive, Ridge) |
| 4 | `python src/models/lgbm_model.py` | Train LightGBM with rolling window |
| 5 | `python src/models/xgb_model.py` | Train XGBoost with rolling window |
| 6 | `python src/models/ensemble.py` | Weighted ensemble of LightGBM + XGBoost |
| 7 | `python src/models/curve_translator.py` | Generate base/peak views & trading signals |
| 8 | `python src/llm/remit_parser.py` | LLM-based REMIT UMM parsing |

## Data Sources

All data is sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) under EU Regulation 543/2013:

- **Day-Ahead Prices** (Art. 12.1.D) — target variable
- **Actual Total Load** (Art. 6.1.A) — demand fundamental
- **Wind & Solar Generation Forecasts** (Art. 14.1.D) — supply fundamentals

**Market:** DE_LU (Germany-Luxembourg) bidding zone  
**Timeframe:** January 1 – December 31, 2024

## Models

| Model | Type | Training Strategy |
|-------|------|-------------------|
| Naive Persistence | Baseline | Same-hour yesterday |
| Ridge Regression (ARX) | Baseline | Full training set |
| LightGBM | Improved | 60-day rolling window |
| XGBoost | Improved | 60-day rolling window |
| Weighted Ensemble | Improved | RMSE-weighted blend |

All improved models produce probabilistic outputs (10th, 50th, 90th quantiles).

## License

This project is a prototype case study submission.
