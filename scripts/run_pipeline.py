"""Master orchestrator to run the complete DocVane MLOps pipeline."""

import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import validate_dataset
from src.models.evaluate import evaluate_saved_model
from src.models.train import train_and_track
from src.monitoring.drift import run_drift_check
from src.models.retrain import run_retraining_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DocVane-Pipeline")


def run_full_pipeline():
    logger.info("========== STEP 1: DATASET GENERATION & INGESTION ==========")
    dataset_file = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
    if not dataset_file.exists():
        from scripts.generate_dataset import generate_dataset
        generate_dataset()
    else:
        logger.info(f"Dataset already exists at {dataset_file}")

    logger.info("========== STEP 2: DATA VALIDATION ==========")
    val_summary = validate_dataset(dataset_file)
    logger.info(f"Dataset validated successfully: {val_summary['total_rows']} rows.")

    logger.info("========== STEP 3: MODEL TRAINING & MLFLOW TRACKING ==========")
    train_summary = train_and_track()
    logger.info(f"Best model trained: {train_summary['model_name']} with Macro F1: {train_summary['macro_f1']:.4f}")

    logger.info("========== STEP 4: MODEL EVALUATION & THRESHOLD CHECK ==========")
    eval_summary = evaluate_saved_model()
    logger.info(f"Evaluation passed. Macro F1: {eval_summary['macro_f1']:.4f} (Threshold met: {eval_summary['threshold_met']})")

    logger.info("========== STEP 5: PYTEST TEST SUITE ==========")
    result = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError("Pytest suite failed!")
    logger.info("All 22 pytest unit and integration tests passed.")

    logger.info("========== STEP 6: MONITORING & DRIFT ANALYSIS ==========")
    drift_res = run_drift_check()
    logger.info(f"Drift check completed. PSI Score: {drift_res.get('psi_score')}")

    logger.info("========== STEP 7: NEW-DATA QUALITY GATE & RETRAINING ==========")
    quality_decision = run_retraining_workflow(
        incoming_path=PROJECT_ROOT / "data" / "incoming" / "new_batch.csv",
        promote=False,
    )
    logger.info(
        "Candidate eligible: %s | Action: %s",
        quality_decision["eligible_for_promotion"],
        quality_decision["promotion_action"],
    )

    logger.info("=============================================================")
    logger.info("🎉 SUCCESS! Full DocVane MLOps pipeline executed successfully.")
    logger.info("To start the FastAPI REST API, run:")
    logger.info("  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("To launch the MLflow UI, run:")
    logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("=============================================================")


if __name__ == "__main__":
    run_full_pipeline()
