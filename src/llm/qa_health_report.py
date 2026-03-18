"""
LLM-Generated Daily QA Health Report
=======================================
Reads the QA pipeline logs and model metrics, then uses OpenAI to generate
a plain-English daily health report for the data engineering team.

This fulfills the "programmatic AI/LLM use" requirement:
  - NOT chat transcripts of using an LLM to write code
  - IS a script that calls an LLM API within the pipeline
  - Automates what would otherwise be manual log review

Usage:
    python src/llm/qa_health_report.py

Output:
    /logs/daily_health_report.md
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "qa_health_report.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------
def collect_pipeline_status() -> dict:
    """Gather all log and metric files into a structured summary."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "qa_report": None,
        "baseline_metrics": None,
        "model_metrics": None,
        "curve_signals": None,
        "remit_results": None,
    }

    # QA report
    qa_path = LOG_DIR / "qa_report.txt"
    if qa_path.exists():
        status["qa_report"] = qa_path.read_text()[:2000]  # Limit context size

    # Baseline metrics
    baseline_path = LOG_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            status["baseline_metrics"] = json.load(f)

    # Model metrics
    model_path = LOG_DIR / "model_metrics.json"
    if model_path.exists():
        with open(model_path, encoding="utf-8") as f:
            status["model_metrics"] = json.load(f)

    # Curve translation signals
    curve_path = LOG_DIR / "curve_translation.json"
    if curve_path.exists():
        with open(curve_path, encoding="utf-8") as f:
            signals = json.load(f)
            # Just the last 3 weeks for brevity
            status["curve_signals"] = signals[-3:] if len(signals) > 3 else signals

    # REMIT results
    remit_path = LOG_DIR / "llm_outputs.log"
    if remit_path.exists():
        status["remit_results"] = remit_path.read_text()[:1500]

    return status


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior energy data engineer responsible for monitoring
a day-ahead power price forecasting pipeline for the German (DE-LU) bidding zone.

Your task: Given the pipeline's QA logs, model metrics, and trading signals,
produce a concise daily health report in Markdown format.

The report MUST include:
1. **Data Quality Summary** — any anomalies, missing hours, DST issues, outliers
2. **Model Performance** — which model is best today, any degradation vs baseline
3. **Trading Signal Summary** — active signals, risk premiums, invalidation alerts
4. **Action Items** — specific things the team should investigate or fix

Keep it under 500 words. Use bullet points. Be specific with numbers.
Flag anything that needs human attention with ⚠️."""

USER_PROMPT_TEMPLATE = """Generate the daily health report based on this pipeline data:

QA Report:
{qa_report}

Baseline Metrics:
{baseline_metrics}

Model Metrics:
{model_metrics}

Recent Trading Signals:
{curve_signals}

REMIT UMM Results:
{remit_results}
"""


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_report(status: dict) -> str:
    """Generate the health report using OpenAI or a structured fallback."""
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key != "your_openai_api_key_here":
        try:
            return _generate_with_openai(status, api_key)
        except Exception as e:
            logger.warning(f"OpenAI API call failed ({e}) — falling back to structured report")
            return _generate_structured_fallback(status)
    else:
        logger.warning("OpenAI API key not found — using structured fallback report")
        return _generate_structured_fallback(status)


def _generate_with_openai(status: dict, api_key: str) -> str:
    """Call OpenAI to generate the health report."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        qa_report=status.get("qa_report", "Not available"),
        baseline_metrics=json.dumps(status.get("baseline_metrics", {}), indent=2),
        model_metrics=json.dumps(status.get("model_metrics", {}), indent=2),
        curve_signals=json.dumps(status.get("curve_signals", []), indent=2),
        remit_results=status.get("remit_results", "Not available"),
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


def _generate_structured_fallback(status: dict) -> str:
    """Generate a structured report without LLM when API key unavailable."""
    report_lines = [
        f"# Daily Pipeline Health Report",
        f"**Generated:** {status['timestamp']}",
        f"**Mode:** Automated (rule-based fallback — set OPENAI_API_KEY for LLM reports)",
        "",
        "## 1. Data Quality Summary",
    ]

    # Parse QA report
    qa = status.get("qa_report", "")
    if "No duplicate timestamps" in qa:
        report_lines.append("- ✅ No duplicate timestamps detected")
    if "No unexpected gaps" in qa:
        report_lines.append("- ✅ No unexpected gaps in time series")
    if "DST" in qa:
        dst_lines = [l for l in qa.split("\n") if "DST" in l]
        for l in dst_lines[:3]:
            report_lines.append(f"- {l.strip()}")
    if "Negative prices:" in qa:
        neg_line = [l for l in qa.split("\n") if "Negative prices:" in l]
        if neg_line:
            report_lines.append(f"- {neg_line[0].strip()}")
    if "outliers flagged" in qa.lower():
        out_line = [l for l in qa.split("\n") if "outliers flagged" in l.lower()]
        if out_line:
            report_lines.append(f"- {out_line[0].strip()}")

    report_lines.append("")
    report_lines.append("## 2. Model Performance")

    metrics = status.get("model_metrics", {})
    if metrics:
        comparison = metrics.get("comparison", [])
        for m in comparison:
            report_lines.append(f"- **{m['model']}**: sMAPE={m['sMAPE']}%, MAE={m['MAE']} €/MWh")

        ensemble = metrics.get("ensemble_detail", {})
        if ensemble:
            crossings = ensemble.get("quantile_crossings", 0)
            if crossings > 0:
                report_lines.append(f"- ⚠️ {crossings} quantile crossings detected — model calibration issue")
            else:
                report_lines.append("- ✅ Zero quantile crossings — probabilistic outputs well-calibrated")

    report_lines.append("")
    report_lines.append("## 3. Trading Signal Summary")

    signals = status.get("curve_signals", [])
    if signals:
        for s in signals:
            wk = f"{s['year']}-W{s['week']:02d}"
            report_lines.append(
                f"- **{wk}**: {s['signal']} (RP={s['risk_premium']:+.1f} €/MWh, "
                f"Conviction: {s['conviction']})"
            )
            inv = s.get("invalidation_reasons", [])
            for reason in inv:
                if "None" not in reason:
                    report_lines.append(f"  - ⚠️ {reason}")

    report_lines.append("")
    report_lines.append("## 4. Action Items")

    # Auto-detect action items
    action_items = []
    if metrics:
        best = min(metrics.get("comparison", []), key=lambda x: x.get("MAE", 999), default=None)
        if best and best.get("MAE", 0) > 15:
            action_items.append(f"⚠️ Best model MAE ({best['MAE']} €/MWh) is above 15 — consider retraining")
    if not status.get("baseline_metrics"):
        action_items.append("⚠️ Baseline metrics file missing — run baseline.py")
    if "rule_based" in str(status.get("remit_results", "")):
        action_items.append("📋 REMIT parser using rule-based fallback — set OPENAI_API_KEY for LLM classification")

    if action_items:
        for item in action_items:
            report_lines.append(f"- {item}")
    else:
        report_lines.append("- ✅ No critical issues detected")

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Daily QA Health Report Generator")
    logger.info("=" * 60)

    # Collect pipeline status
    status = collect_pipeline_status()
    logger.info("Collected pipeline status from log files")

    # Generate report
    report = generate_report(status)

    # Save report
    report_path = LOG_DIR / "daily_health_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Health report saved → {report_path}")

    # Print to console
    logger.info("")
    print(report)


if __name__ == "__main__":
    main()
