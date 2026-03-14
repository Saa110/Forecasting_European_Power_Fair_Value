"""
REMIT UMM Parser — LLM-Based Market Message Classification
============================================================
Uses OpenAI to parse REMIT Urgent Market Messages (UMMs) and
determine whether they should invalidate a trading signal.

Three mock UMMs are used for demonstration:
  1. Catastrophic generator failure (~1 GW forced outage)
  2. Routine valve maintenance (~50 MW, planned)
  3. Extreme weather warning (Dunkelflaute)

The LLM extracts:
  - relevance_level: 1 (Major), 2 (Moderate), 3 (Routine)
  - root_cause: category string
  - justification: one-sentence explanation

Invalidation logic:
  If relevance_level == 1 AND root_cause == "Generation Outage"
  → invalidate = True (signal should be overridden)

Usage:
    python src/llm/remit_parser.py

Output:
    /logs/llm_outputs.log
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "remit_parser.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock REMIT UMMs
# ---------------------------------------------------------------------------
MOCK_UMMS = [
    {
        "id": "UMM-2026-001",
        "title": "Unplanned Outage — Jänschwalde Unit B (900 MW)",
        "body": (
            "LEAG reports an unplanned full outage of Jänschwalde lignite power plant "
            "Unit B (910 MW) effective immediately due to a catastrophic boiler tube "
            "failure. The unit had been operating at full capacity. Estimated return "
            "to service: not before 14 days. The outage removes approximately 910 MW "
            "of baseload capacity from the German grid. Market participants should "
            "expect upward pressure on day-ahead prices, particularly during peak hours."
        ),
    },
    {
        "id": "UMM-2026-002",
        "title": "Planned Maintenance — Isar 2 Valve Replacement (50 MW derate)",
        "body": (
            "PreussenElektra GmbH announces a planned partial derate of Isar Unit 2 "
            "for routine safety valve replacement. The derate reduces output by "
            "approximately 50 MW for a 72-hour window starting 2026-03-20 at 06:00 CET. "
            "This maintenance was scheduled in the annual maintenance plan and has no "
            "material impact on system adequacy. Normal operations expected to resume "
            "by 2026-03-23."
        ),
    },
    {
        "id": "UMM-2026-003",
        "title": "Extreme Weather Alert — Dunkelflaute Expected",
        "body": (
            "DWD (German Weather Service) issues an extreme weather alert for the period "
            "2026-03-25 to 2026-03-31. A persistent high-pressure system combined with "
            "thick cloud cover is forecast to produce Dunkelflaute conditions across "
            "Northern Germany. Expected impact: wind generation dropping below 2 GW "
            "(typical: 15-25 GW) and solar generation near zero for consecutive days. "
            "Combined with high heating demand, this scenario may trigger price spikes "
            "exceeding 200 EUR/MWh. Gas-fired peaker plants expected to set marginal prices."
        ),
    },
]


# ---------------------------------------------------------------------------
# Zero-Shot Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert energy market analyst specializing in REMIT 
(Regulation on Energy Market Integrity and Transparency) compliance. Your task is 
to classify Urgent Market Messages (UMMs) for their relevance to power trading decisions.

For each UMM, return STRICT JSON (no markdown, no explanation outside JSON) with:
{
  "relevance_level": <int 1-3>,
  "root_cause": "<string>",
  "justification": "<one sentence>"
}

Relevance levels:
  1 = Major: Significant impact on supply/demand balance (>500 MW outage, extreme weather)
  2 = Moderate: Noticeable but manageable impact (100-500 MW, planned maintenance)
  3 = Routine: Minimal impact (<100 MW, scheduled, no market effect)

Root cause categories:
  "Generation Outage", "Planned Maintenance", "Extreme Weather", 
  "Transmission Constraint", "Demand Shift", "Other"
"""

USER_PROMPT_TEMPLATE = """Classify this REMIT UMM:

Title: {title}
Body: {body}

Return STRICT JSON only."""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
def parse_umm_with_llm(umm: dict) -> dict:
    """
    Send a REMIT UMM to OpenAI for classification.
    Falls back to rule-based parsing if API key is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key != "your_openai_api_key_here":
        return _parse_with_openai(umm, api_key)
    else:
        logger.warning("  OpenAI API key not found — using rule-based fallback")
        return _parse_rule_based(umm)


def _parse_with_openai(umm: dict, api_key: str) -> dict:
    """Call OpenAI API for UMM classification."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=umm["title"],
        body=umm["body"],
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=200,
    )

    raw_response = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```" in raw_response:
            json_str = raw_response.split("```")[1].strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            parsed = json.loads(json_str)
        else:
            parsed = {"error": "Failed to parse", "raw": raw_response}

    return {"raw_response": raw_response, "parsed": parsed, "source": "openai"}


def _parse_rule_based(umm: dict) -> dict:
    """
    Rule-based fallback UMM classifier.
    Uses keyword matching when OpenAI API is unavailable.
    """
    text = (umm["title"] + " " + umm["body"]).lower()

    # Determine relevance and root cause from keywords
    if any(w in text for w in ["unplanned", "catastrophic", "forced", "emergency"]):
        relevance = 1
        root_cause = "Generation Outage"
        justification = "Unplanned outage with significant capacity loss detected."
    elif any(w in text for w in ["dunkelflaute", "extreme weather", "price spike"]):
        relevance = 1
        root_cause = "Extreme Weather"
        justification = "Extreme weather event with high potential for supply shortfall."
    elif any(w in text for w in ["planned maintenance", "routine", "scheduled"]):
        relevance = 3
        root_cause = "Planned Maintenance"
        justification = "Routine planned maintenance with minimal market impact."
    elif any(w in text for w in ["derate", "partial"]):
        relevance = 2
        root_cause = "Planned Maintenance"
        justification = "Partial derate with moderate but manageable impact."
    else:
        relevance = 2
        root_cause = "Other"
        justification = "Message requires manual review for relevance determination."

    parsed = {
        "relevance_level": relevance,
        "root_cause": root_cause,
        "justification": justification,
    }

    return {"raw_response": json.dumps(parsed), "parsed": parsed, "source": "rule_based"}


# ---------------------------------------------------------------------------
# Invalidation Logic
# ---------------------------------------------------------------------------
def should_invalidate(parsed: dict) -> bool:
    """
    Determine if UMM should invalidate the current trading signal.
    Rule: relevance_level == 1 AND root_cause == "Generation Outage" → True
    """
    return (
        parsed.get("relevance_level") == 1
        and parsed.get("root_cause") == "Generation Outage"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("REMIT UMM Parser — LLM Integration")
    logger.info("=" * 60)

    llm_log_path = LOG_DIR / "llm_outputs.log"
    results = []

    for umm in MOCK_UMMS:
        logger.info(f"\nProcessing: {umm['id']} — {umm['title']}")

        result = parse_umm_with_llm(umm)
        parsed = result["parsed"]
        invalidate = should_invalidate(parsed)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "umm_id": umm["id"],
            "umm_title": umm["title"],
            "source": result["source"],
            "raw_prompt": USER_PROMPT_TEMPLATE.format(title=umm["title"], body=umm["body"]),
            "raw_response": result["raw_response"],
            "parsed": parsed,
            "invalidate_signal": invalidate,
        }
        results.append(log_entry)

        logger.info(f"  Source:      {result['source']}")
        logger.info(f"  Relevance:   {parsed.get('relevance_level', 'N/A')}")
        logger.info(f"  Root Cause:  {parsed.get('root_cause', 'N/A')}")
        logger.info(f"  Reason:      {parsed.get('justification', 'N/A')}")
        logger.info(f"  Invalidate:  {'⚠️  YES' if invalidate else '✅ NO'}")

    # Write structured log
    with open(llm_log_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, indent=2) + "\n\n")

    logger.info(f"\nLLM outputs logged → {llm_log_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  {'UMM ID':<16} {'Relevance':>10} {'Root Cause':<20} {'Invalidate':>10}")
    logger.info(f"  {'─'*16} {'─'*10} {'─'*20} {'─'*10}")
    for r in results:
        p = r["parsed"]
        logger.info(
            f"  {r['umm_id']:<16} {p.get('relevance_level','?'):>10} "
            f"{p.get('root_cause','?'):<20} {'YES ⚠️' if r['invalidate_signal'] else 'NO ✅':>10}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
