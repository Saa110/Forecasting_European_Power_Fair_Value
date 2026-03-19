# European Power Fair Value
## Forecasting Day-Ahead Prices & Translating to Prompt Curve Views

A quantitative prototype that forecasts hourly Day-Ahead electricity prices for the German (DE_LU) bidding zone using publicly available ENTSO-E data, and translates those forecasts into tradable base/peak curve views with dynamic, volatility-aware trading signals.

---

## Project Structure

```text
├── data/
│   ├── raw/              # Unprocessed ENTSO-E API pulls (.gitignored)
│   └── processed/        # Cleaned, feature-engineered datasets (.gitignored)
├── src/
│   ├── ingestion/        # ENTSO-E API data pull scripts
│   ├── qa/               # Quality assurance & feature engineering
│   ├── models/           # Baseline & advanced models, ensemble, curve translator
│   └── llm/              # LLM-based REMIT UMM parser & QA reports
├── logs/                 # Execution logs, LLM outputs, metrics
├── docs/
│   ├── figures/          # Charts and plots
│   └── report/           # Final submission document (report.tex & report.pdf)
├── .env.example          # Template for API keys
├── requirements.txt      # Python dependencies
├── main.py               # Unified pipeline orchestrator
└── README.md             # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Saa110/Forecasting_European_Power_Fair_Value.git
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

### 3. Run the Pipeline
The entire end-to-end pipeline is automated via `main.py`.

```bash
# Run the full pipeline (Ingestion -> QA -> Modeling -> Translation -> AI Reports)
python main.py
```

**Advanced Usage:**
```bash
python main.py --steps ingest clean     # Run only ingestion + QA
python main.py --group models           # Train all models and the ensemble
python main.py --from curve             # Run from curve_translator onwards
python main.py --group tuning           # Run hyperparameter tuning
python main.py --group ablation         # Run feature ablation study
python main.py --dry-run                # Print the execution plan without running
```
Run `python main.py --help` for full options.

## Data Sources

All data is sourced from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) under EU Regulation 543/2013 (with automatic failover to Energy-Charts):

- **Day-Ahead Prices** (Art. 12.1.D) — target variable
- **Actual Total Load** (Art. 6.1.A) — demand fundamental
- **Wind & Solar Generation Forecasts** (Art. 14.1.D) — supply fundamentals

**Market:** DE_LU (Germany-Luxembourg) bidding zone  
**Timeframe:** January 1, 2024 → yesterday (continuously updated)  
**Resolution:** Hourly (15-min load/gen resampled to hourly)

## Models

We train models on an expanding 60-day rolling window to prevent lookahead bias.

| Model | Type | Strategy / Output |
|-------|------|-------------------|
| Naive Persistence | Baseline | Same-hour yesterday |
| Ridge Regression (ARX) | Baseline | L2 Regularized Regression |
| Linear Regression (OLS) | Structural | Ordinary Least Squares |
| MLP (Neural Network) | Advanced | Multi-layer Perceptron |
| LightGBM | Advanced | Probabilistic (q10, q50, q90) |
| XGBoost | Advanced | Probabilistic (q10, q50, q90) |
| CatBoost | Advanced | Probabilistic (q10, q50, q90) |
| **Weighted Ensemble** | **Production** | RMSE-weighted blend of Top 3 Gradient Boosted Trees |

## Evaluation Metrics

- **sMAPE** — Symmetric Mean Absolute Percentage Error (natively handles zero & negative prices)
- **MAE** — Mean Absolute Error in €/MWh
- **Directional Accuracy** — % of hours where predicted price direction (up/down) matches actual
- **Pinball Loss** — Quantile-specific calibration (q10, q50, q90)

## QA & Feature Engineering

- DST-safe UTC conversion (handles CET/CEST 23h/25h transitions)
- Automated anomaly detection: missing hours, duplicate timestamps, monotonicity checks
- Negative prices explicitly preserved (real market events from renewable oversupply)
- Outliers flagged via IQR but **never deleted**
- **14 Engineered Features**: Including autoregressive price/load lags, temporal signals, net load, and renewable penetration indices.

## License

This project is a prototype case study submission.
