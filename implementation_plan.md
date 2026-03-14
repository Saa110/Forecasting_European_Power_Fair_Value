# European Power Fair Value — Implementation Plan & Deliverables Checklist

## Goal

Build a **working prototype** that:
1. Ingests publicly available German (DE) day-ahead power market data
2. Applies QA and feature engineering
3. Forecasts next-day hourly prices with baseline + improved models
4. Translates forecasts into a tradable DA-to-curve view (base/peak)
5. Integrates a programmatic LLM component (REMIT parser)
6. Is packaged as a documented, reproducible repository

The plan below maps the 7-day schedule from [plan.md](file:///Users/syedasad/GIthub/Forecasting_European_Power_Fair_Value/plan.md) onto the evaluation rubric in [problemstatement.md](file:///Users/syedasad/GIthub/Forecasting_European_Power_Fair_Value/problemstatement.md), with a concrete **deliverables checklist** for every stage.

---

## Stage 1 — Day 1: Data Ingestion & Environment Setup

**Objective:** Reproducible environment + raw DE market data secured with **resilient multi-source fallback**.

### Deliverables Checklist

- [ ] **Git repository** initialized with directory skeleton:
  - `/data/raw/`, `/data/processed/`, `/data/cache/` (`.gitignore`d)
  - `/src/ingestion/`, `/src/qa/`, `/src/models/`, `/src/llm/`
  - `/logs/`, `/docs/`
- [ ] **Python virtual environment** (`.venv`) created and activated
- [ ] **`requirements.txt`** pinning: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `entsoe-py`, `openai`, `matplotlib`, `shap`, `requests`
- [ ] **ENTSO-E API token** obtained and stored in `.env` (`.gitignore`d)
- [ ] **`/src/ingestion/entsoe_pull.py`** — primary ingestion script (DE bidding zone, 2024-01-01 → present):
  - Day-Ahead Prices (target variable)
  - Actual Total Load
  - Day-Ahead Wind generation forecast
  - Day-Ahead Solar generation forecast
- [ ] **`/src/ingestion/energy_charts_pull.py`** — fallback ingestion for DE/FR/NL
- [ ] **`/src/ingestion/data_ingest.py`** — orchestrator with try/except fallback logic
- [ ] Raw CSV files saved to `/data/raw/` with `source_quality` column (`PRIMARY` / `FALLBACK`)
- [ ] **Local CSV cache** in `/data/cache/` — last known good data persisted for resilience
- [ ] **README stub** with instructions for reproducing the data pull

> [!IMPORTANT]
> The ENTSO-E API token is free but requires registration; allow up to 24 h for approval. Register **first thing**.

### Data Source Fallback Architecture

The ingestion layer uses a **tiered fallback** strategy to guarantee data availability:

```
┌─────────────────────────────────────────────────────┐
│                  data_ingest.py                     │
│              (Orchestrator Script)                  │
├─────────────────────────────────────────────────────┤
│  1. ENTSO-E API (Primary)                          │
│     └─ 3 retries with exponential backoff          │
│  2. Energy-Charts API (Fallback for DE/FR/NL)      │
│     └─ Stable JSON API, same underlying data       │
│  3. Local CSV Cache (Last Resort)                  │
│     └─ Serves last known good data + flags it      │
└─────────────────────────────────────────────────────┘
```

| Market | Primary | Fallback | Last Resort |
|--------|---------|----------|-------------|
| **DE, FR, NL** | ENTSO-E API | Energy-Charts API | Local CSV cache |
| **GB** | ENTSO-E API | Elexon API (future) | Local CSV cache |

**Key design choices:**
- Every record carries a `source_quality` flag: `PRIMARY`, `FALLBACK`, or `CACHED`
- The QA pipeline (Stage 2) uses this flag to weight data confidence
- Fallback data is structurally identical to primary data — same columns, same UTC index

---

## Stage 2 — Day 2: Quality Assurance & Feature Engineering

**Objective:** Clean, merged dataset ready for modeling.

### Deliverables Checklist

- [ ] **`/src/qa/cleaner.py`** implementing:
  - UTC timezone conversion for all timestamps (eliminates DST gaps/duplicates)
  - Rolling-median / forward-fill for short NaN sequences
  - Statistical outlier **flagging** (3σ / IQR) — do **not** delete spikes
  - Boundary checks (e.g., reject load < 0 MW)
- [ ] **QA report** (`/logs/qa_report.txt` or `.csv`) listing:
  - Number of missing values per column before/after imputation
  - Number of outliers flagged
  - Timestamp continuity check (strictly monotonic after conversion)
- [ ] **Feature engineering** added to cleaner or separate script:
  - Temporal: `hour_of_day`, `day_of_week`, `month`, `is_weekend`
  - Lag features: price lag-24 h, price lag-168 h (1 week)
  - Renewable penetration ratio: `(wind + solar) / total_load`
- [ ] **`/data/processed/model_ready.csv`** — single merged, cleaned dataset
- [ ] At least one **exploratory figure** (price vs. time, or price distribution) saved to `/docs/figures/`

---

## Stage 3 — Day 3: Baseline Modeling & Validation

**Objective:** Establish rigorous benchmark performance.

### Deliverables Checklist

- [ ] **`/src/models/baseline.py`** containing:
  - **Chronological train/test split** — first 10 months train, last 2 months test (no random shuffle)
  - **Naïve Persistence** baseline — forecast = same hour yesterday
  - **ARX / Ridge Regression** baseline — inputs: lagged prices, load, wind, solar
- [ ] **Evaluation metrics** computed on test set:
  - sMAPE for Naïve Persistence
  - sMAPE for Ridge/ARX
  - MAE for both
- [ ] Metrics **logged** to `/logs/baseline_metrics.json`
- [ ] **Forecast-vs-actual plot** for at least one test week, saved to `/docs/figures/`

---

## Stage 4 — Day 4: Improved Model (LightGBM)

**Objective:** Non-linear model that demonstrably outperforms baselines + probabilistic outputs.

### Deliverables Checklist

- [ ] **`/src/models/lgbm_model.py`** implementing:
  - LightGBM regressor with 60-day rolling-window training loop
  - Quantile regression outputs: 10th, 50th, 90th percentiles
- [ ] **Validation metrics** on test set:
  - sMAPE of 50th percentile (median) forecast
  - Pinball Loss for q10, q50, q90
  - Comparison table: Naïve vs. Ridge vs. LightGBM
- [ ] **Quantile crossing check** — verify q10 ≤ q50 ≤ q90 across all hours
- [ ] Metrics **logged** to `/logs/lgbm_metrics.json`
- [ ] **Fan chart** (10th–90th interval) for one representative test week, saved to `/docs/figures/`
- [ ] **SHAP summary plot** (top-10 features) saved to `/docs/figures/` (optional, stretch goal)

> [!TIP]
> If LightGBM does **not** outperform baselines, investigate feature importance and window length before moving on; a broken model derails the entire curve translation stage.

---

## Stage 5 — Day 5: Prompt Curve Translation

**Objective:** Convert hourly forecasts into tradable base/peak views and generate trading signals.

### Deliverables Checklist

- [ ] **`/src/models/curve_translator.py`** implementing:
  - **Baseload** calculation: arithmetic mean of all 24 forecasted hours
  - **Peakload** calculation: arithmetic mean of hours 09–20 on Mon–Fri
  - Aggregation over a simulated "Front-Week" horizon
- [ ] **Risk premium calculation:**
  - Accept a mock/real traded futures price as input
  - Compute Ex-Ante Risk Premium: `RP = F_traded − E[Spot]`
- [ ] **Probabilistic trading signals:**
  - `Forward < 10th percentile forecast` → **"Strong Buy (Undervalued)"**
  - `Forward > 90th percentile forecast` → **"Strong Sell (Overvalued)"**
  - Otherwise → **"Hold (Within Confidence Interval)"**
  - Classification of Contango (RP > 0) vs. Backwardation (RP < 0)
- [ ] **Output table** or printed summary showing for a test week:
  - Modeled Base, Modeled Peak, Forward Price, Risk Premium, Signal
- [ ] Brief **text explanation** (can be inline comments or a markdown cell) of when/why to invalidate the signal

---

## Stage 6 — Day 6: Programmatic AI/LLM Integration

**Objective:** Automate REMIT UMM parsing with an LLM to support signal invalidation.

### Deliverables Checklist

- [ ] **`/src/llm/remit_parser.py`** containing:
  - OpenAI client initialization (API key from `.env`)
  - 3 hardcoded mock REMIT UMM text samples:
    1. Catastrophic generator failure (~1 GW forced outage)
    2. Routine valve maintenance (~50 MW, planned)
    3. Extreme weather warning (Dunkelflaute)
- [ ] **Zero-shot prompt template** that instructs the LLM to return **strict JSON** with:
  - `relevance_level` (1 = Major, 2 = Moderate, 3 = Routine)
  - `root_cause` (e.g., "Generation Outage", "Extreme Weather", "Planned Maintenance")
  - `justification` (one-sentence rationale)
- [ ] **Invalidation logic:** if `relevance_level == 1` AND `root_cause == "Generation Outage"` → set `invalidate = True`
- [ ] **Logged outputs** in `/logs/llm_outputs.log`:
  - Timestamp, raw prompt sent, raw JSON response for each of the 3 mock messages
- [ ] Brief inline documentation explaining the purpose and how it integrates with the curve view

---

## Stage 7 — Day 7: Documentation & Packaging

**Objective:** Ship a polished, reviewable submission.

### Deliverables Checklist

- [ ] **Final document** (1–3 pages, PDF or Markdown) in `/docs/report.md` or `/docs/report.pdf`:
  - Section 1 — Data & QA: sources, timeframe, UTC handling, imputation
  - Section 2 — Forecasting Rigor: metrics table (Naïve vs. Ridge vs. LightGBM), rolling window, Pinball Loss
  - Section 3 — Trading Relevance: Base/Peak aggregation, probabilistic buy/sell logic, invalidation criteria
  - Section 4 — AI Integration: REMIT parser description, pointer to JSON logs
- [ ] **`README.md`** with:
  - Project overview
  - Setup instructions (`pip install -r requirements.txt`, `.env` setup)
  - Step-by-step run order (Day 1 → Day 6 scripts)
  - Directory map
- [ ] **Repository hygiene:**
  - `.gitignore` excludes `/data/raw/`, `.env`
  - No large CSV files committed
  - Code is commented and PEP-8 styled
- [ ] **(Optional)** `submission.csv` — out-of-sample predictions: columns `id`, `y_pred`
- [ ] All figures/tables referenced in the report are present in `/docs/figures/`

---

## Verification Plan

Since this is a data science prototype (not a traditional web app), verification focuses on:

### Automated / Script-Based Checks
1. **Data pipeline smoke test:** `python src/ingestion/data_ingest.py` completes and produces CSVs in `/data/raw/` with `source_quality` column
2. **Fallback test:** Simulate ENTSO-E failure → verify Energy-Charts fallback activates and produces identical schema
3. **Cache test:** Simulate all-API failure → verify local cache serves last known good data flagged as `CACHED`
4. **QA pipeline test:** `python src/qa/cleaner.py` → verify `/data/processed/model_ready.csv` exists, has no NaN rows, timestamps are monotonic
5. **Baseline model test:** `python src/models/baseline.py` → outputs sMAPE metrics to stdout and `/logs/`
6. **LightGBM model test:** `python src/models/lgbm_model.py` → outputs sMAPE + Pinball Loss; verify LightGBM sMAPE < Baseline sMAPE
7. **Curve translator test:** `python src/models/curve_translator.py` → prints base/peak/signal summary
8. **LLM integration test:** `python src/llm/remit_parser.py` → writes 3 JSON responses to `/logs/llm_outputs.log`

### Manual Verification
- Review the final report for completeness against the rubric
- Visually inspect forecast-vs-actual plots and fan charts for sanity
- Confirm `README.md` instructions are followable end-to-end on a clean clone

---

## Key Design Decisions & Risks

| Decision | Rationale |
|---|---|
| **Target market: DE** | Most liquid European market with high wind/solar penetration — ideal for demonstrating price cannibalization effects |
| **1-year data window (2024)** | Sufficient for a 60-day rolling window strategy; avoids zone-split legacy issues |
| **LightGBM over LSTM** | Short training windows (60 days) favor gradient boosting; LSTMs need multi-year data to shine |
| **Multi-source fallback** | ENTSO-E → Energy-Charts → Local cache ensures pipeline never breaks; `source_quality` flag preserves data provenance |
| **Mock futures prices** | Live EEX API requires paid access; mock values are acceptable for a prototype |
| **Mock REMIT UMMs** | Live REMIT RSS parsing adds infrastructure complexity beyond one-week scope |

> [!WARNING]
> The ENTSO-E API can be slow and occasionally returns `503`. The fallback architecture handles this automatically via Energy-Charts.
