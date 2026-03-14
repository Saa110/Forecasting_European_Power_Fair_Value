### Case Study Theme
European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

### Task: Build a prototype producing a daily fair-value view for a European power market and show how it informs prompt curve positioning.

### Requirements:

#### Data Ingestion & QA
Task: Collect publicly available data for one market (DE, FR, NL, GB).
Deliverable: Dataset including hourly Day-Ahead prices + at least two fundamental drivers; document sources and implement QA checks.

#### Forecasting & Validation
Task: Forecast either next-day hourly prices (Option A, recommended) or front-week / front-month price averages (Option B).
Deliverable: At least one baseline and one improved model with validation metrics.

#### Prompt Curve Translation
Task: Translate your forecast into a tradable DA-to-curve view.
Deliverable: Short guidance on how the forecasted values would be used or invalidated.

#### AI/LLM Integration
Task: Implement one programmatic AI/LLM component to reduce manual work in your pipeline.
Deliverable: Working code calling the AI/LLM, logged prompts and outputs, and a brief explanation of its purpose.

### Submission:

Document: 1–3 pages (PDF or Markdown) with name and email.
Repo or zipped folder including pipeline code, README, requirements, QA output, figures/tables, AI component.
Optional: submission.csv with out-of-sample predictions (id, y_pred).

### Evaluation:

Dataset correctness and QA
Forecasting rigor
Trading relevance
Engineering quality and reproducibility
Programmatic AI/LLM use

### Deadline: Please submit your case study within one week from today.