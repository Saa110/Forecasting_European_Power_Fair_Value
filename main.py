"""
Pipeline Orchestrator
======================
Single entry point to run the entire forecasting pipeline, or
selectively trigger individual stages.

Usage:
    python main.py                          # Run everything
    python main.py --steps ingest clean     # Only ingestion + QA
    python main.py --group models           # Steps 3-6 (baseline → ensemble)
    python main.py --from curve             # Run from curve_translator onwards
    python main.py --dry-run                # Print plan without executing
    python main.py --force-fallback         # Forwarded to data_ingest

Run `python main.py --help` for full options.
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Pipeline Definition (ordered)
# ---------------------------------------------------------------------------
STEPS = [
    {
        "name": "ingest",
        "label": "Data Ingestion",
        "module": "src.ingestion.data_ingest",
        "groups": ["ingestion", "all"],
    },
    {
        "name": "clean",
        "label": "QA & Feature Engineering",
        "module": "src.qa.cleaner",
        "groups": ["ingestion", "all"],
    },
    {
        "name": "baseline",
        "label": "Baseline Models (Naive + Ridge)",
        "module": "src.models.baseline",
        "groups": ["models", "all"],
    },
    {
        "name": "lgbm",
        "label": "LightGBM (60-day rolling)",
        "module": "src.models.lgbm_model",
        "groups": ["models", "all"],
    },
    {
        "name": "xgb",
        "label": "XGBoost (60-day rolling)",
        "module": "src.models.xgb_model",
        "groups": ["models", "all"],
    },
    {
        "name": "catboost",
        "label": "CatBoost (60-day rolling)",
        "module": "src.models.catboost_model",
        "groups": ["models", "all"],
    },
    {
        "name": "linear",
        "label": "Linear Regression (OLS)",
        "module": "src.models.linear_model",
        "groups": ["models", "all"],
    },
    {
        "name": "mlp",
        "label": "MLP (Neural Network)",
        "module": "src.models.mlp_model",
        "groups": ["models", "all"],
    },
    {
        "name": "ensemble",
        "label": "Weighted Ensemble",
        "module": "src.models.ensemble",
        "groups": ["models", "all"],
    },
    {
        "name": "curve",
        "label": "Curve Translation & Trading Signals",
        "module": "src.models.curve_translator",
        "groups": ["trading", "all"],
    },

    {
        "name": "remit",
        "label": "REMIT UMM Parser (LLM)",
        "module": "src.llm.remit_parser",
        "groups": ["trading", "llm", "all"],
    },
    {
        "name": "health",
        "label": "QA Health Report (LLM)",
        "module": "src.llm.qa_health_report",
        "groups": ["llm", "all"],
    },
    {
        "name": "tune",
        "label": "Hyperparameter Tuning",
        "module": "src.models.tune_hyperparams",
        "groups": ["tuning"],
    },
    {
        "name": "ablation",
        "label": "Feature Ablation Study",
        "module": "src.models.ablation_study",
        "groups": ["ablation"],
    },
]

STEP_NAMES = [s["name"] for s in STEPS]
GROUP_NAMES = sorted({g for s in STEPS for g in s["groups"]})


# ---------------------------------------------------------------------------
# Step Resolution
# ---------------------------------------------------------------------------
def resolve_steps(args) -> list:
    """Determine which steps to run based on CLI arguments."""

    # --steps: explicit list of step names
    if args.steps:
        selected = []
        for name in args.steps:
            if name not in STEP_NAMES:
                logger.error(f"Unknown step '{name}'. Choose from: {STEP_NAMES}")
                sys.exit(1)
            selected.append(name)
        return [s for s in STEPS if s["name"] in selected]

    # --group: run all steps in a named group
    if args.group:
        if args.group not in GROUP_NAMES:
            logger.error(f"Unknown group '{args.group}'. Choose from: {GROUP_NAMES}")
            sys.exit(1)
        return [s for s in STEPS if args.group in s["groups"]]

    # --from: run from a given step to the end
    if args.from_step:
        if args.from_step not in STEP_NAMES:
            logger.error(f"Unknown step '{args.from_step}'. Choose from: {STEP_NAMES}")
            sys.exit(1)
        idx = STEP_NAMES.index(args.from_step)
        return STEPS[idx:]

    # Default: run everything
    return list(STEPS)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_step(step: dict, step_num: int, total: int) -> bool:
    """Import and execute a single pipeline step. Returns True on success."""
    from importlib import import_module

    name = step["name"]
    label = step["label"]
    module_path = step["module"]

    logger.info("")
    logger.info(f"{'─' * 60}")
    logger.info(f"  Step {step_num}/{total}: {label}")
    logger.info(f"  Module: {module_path}")
    logger.info(f"{'─' * 60}")

    t0 = time.time()
    try:
        mod = import_module(module_path)
        mod.main()
        elapsed = time.time() - t0
        logger.info(f"  ✅ {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ❌ {name} FAILED after {elapsed:.1f}s: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the European Power Fair Value forecasting pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Step names:   {', '.join(STEP_NAMES)}
Group names:  {', '.join(GROUP_NAMES)}

Examples:
  python main.py                          Run full pipeline (steps 1-9)
  python main.py --steps ingest clean     Run only ingestion + QA
  python main.py --group models           Run baseline, lgbm, xgb, catboost, linear, mlp, ensemble
  python main.py --group tuning           Run hyperparameter tuning
  python main.py --group ablation         Run feature ablation study
  python main.py --from ensemble          Run from ensemble to end
  python main.py --dry-run                Show plan without executing
  python main.py --force-fallback         Use Energy-Charts instead of ENTSO-E
""",
    )

    select = parser.add_mutually_exclusive_group()
    select.add_argument(
        "--steps", nargs="+", metavar="STEP",
        help="Run only these steps (by name)",
    )
    select.add_argument(
        "--group", metavar="GROUP",
        help=f"Run a predefined group: {', '.join(GROUP_NAMES)}",
    )
    select.add_argument(
        "--from", dest="from_step", metavar="STEP",
        help="Run from this step to the end of the pipeline",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan without running anything",
    )

    # Pass-through flags for data_ingest
    parser.add_argument(
        "--force-fallback", action="store_true",
        help="Skip ENTSO-E, use Energy-Charts (forwarded to data_ingest)",
    )
    parser.add_argument(
        "--force-cache", action="store_true",
        help="Skip all APIs, use cached data (forwarded to data_ingest)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    selected = resolve_steps(args)

    logger.info("=" * 60)
    logger.info("  PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)
    logger.info(f"  Steps to run ({len(selected)}):")
    for i, s in enumerate(selected, 1):
        logger.info(f"    {i}. [{s['name']}] {s['label']}")
    logger.info("")

    if args.dry_run:
        logger.info("  --dry-run: nothing will be executed.")
        return

    # Inject pass-through flags into sys.argv for data_ingest
    # (data_ingest uses argparse internally, so we clear sys.argv)
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Reset so child argparsers don't see our flags
    if args.force_fallback:
        sys.argv.append("--force-fallback")
    if args.force_cache:
        sys.argv.append("--force-cache")

    # Run selected steps
    t_total = time.time()
    results = []

    for i, step in enumerate(selected, 1):
        success = run_step(step, i, len(selected))
        results.append((step["name"], success))
        if not success:
            logger.error(f"\n  Pipeline aborted at step '{step['name']}'.")
            logger.error("  Fix the issue and re-run with: "
                         f"python main.py --from {step['name']}")
            break

    # Restore argv
    sys.argv = original_argv

    # Summary
    total_time = time.time() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    for name, ok in results:
        status = "✅" if ok else "❌"
        logger.info(f"    {status} {name}")
    logger.info(f"\n  Total time: {total_time:.1f}s")

    failed = [name for name, ok in results if not ok]
    if failed:
        logger.info(f"  ⚠️  Failed steps: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info("  All steps completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
