"""New-data evaluation, candidate retraining, and promotion-quality gate."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) in sys.path:
    pass
else:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from src.features.preprocessing import create_tfidf_vectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: Path = PROJECT_ROOT / "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def validate_incoming_batch(
    incoming_path: Path,
    expected_categories=None,
) -> Dict[str, Any]:
    if not incoming_path.exists():
        raise FileNotFoundError(f"Incoming batch not found at {incoming_path}")

    df = pd.read_csv(incoming_path)
    required_columns = {"record_id", "document_text", "label"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Incoming batch is empty")

    null_counts = df[list(required_columns)].isnull().sum().to_dict()
    if any(null_counts.values()):
        raise ValueError(f"Found missing values in incoming batch: {null_counts}")

    empty_text = int(df["document_text"].astype(str).str.strip().eq("").sum())
    if empty_text:
        raise ValueError(f"Found {empty_text} rows with empty document_text")

    duplicate_ids = int(df["record_id"].duplicated().sum())
    duplicate_records = int(df.duplicated(["document_text", "label"]).sum())
    if duplicate_ids:
        raise ValueError(f"Found {duplicate_ids} duplicate record IDs")
    if duplicate_records:
        raise ValueError(f"Found {duplicate_records} duplicate records")

    categories = expected_categories or [
        "invoice",
        "resume",
        "certificate",
        "bank_statement",
    ]
    distribution = df["label"].value_counts().to_dict()
    missing_categories = [category for category in categories if category not in distribution]
    if missing_categories:
        raise ValueError(f"Incoming batch missing categories: {missing_categories}")

    summary = {
        "valid": True,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "null_counts": null_counts,
        "duplicate_record_ids": duplicate_ids,
        "duplicate_records": duplicate_records,
        "target_distribution": distribution,
    }
    logger.info("Incoming batch validated: %s", json.dumps(summary))
    return summary


def run_retraining_workflow(
    incoming_path: Path = PROJECT_ROOT / "data" / "incoming" / "new_batch.csv",
    champion_model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib",
    champion_vectorizer_path: Path = PROJECT_ROOT / "artifacts" / "vectorizer.joblib",
    existing_dataset_path: Path = PROJECT_ROOT / "data" / "processed" / "dataset.csv",
    candidate_model_path: Path = PROJECT_ROOT / "artifacts" / "candidate_model.joblib",
    candidate_vectorizer_path: Path = PROJECT_ROOT / "artifacts" / "candidate_vectorizer.joblib",
    report_path: Path = PROJECT_ROOT / "reports" / "quality_gate_decision.json",
    promote: bool = False,
) -> Dict[str, Any]:
    params = load_params()
    random_state = params.get("base", {}).get("random_state", 42)
    feature_cfg = params.get("features", {})
    gate_cfg = params.get("monitoring", {}).get("quality_gate", {})
    minimum_f1 = float(
        gate_cfg.get(
            "minimum_macro_f1",
            params.get("train", {}).get("min_macro_f1_threshold", 0.80),
        )
    )

    incoming_path = Path(incoming_path)
    existing_dataset_path = Path(existing_dataset_path)
    validation = validate_incoming_batch(incoming_path)
    incoming = pd.read_csv(incoming_path)
    existing = pd.read_csv(existing_dataset_path)

    if not champion_model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {champion_model_path}")
    if not champion_vectorizer_path.exists():
        raise FileNotFoundError(f"Champion vectorizer not found at {champion_vectorizer_path}")

    champion_model = joblib.load(champion_model_path)
    champion_vectorizer = joblib.load(champion_vectorizer_path)
    incoming_text = incoming["document_text"].astype(str).to_numpy()
    incoming_labels = incoming["label"].astype(str).to_numpy()

    champion_predictions = champion_model.predict(
        champion_vectorizer.transform(incoming_text)
    )
    champion_metrics = classification_metrics(incoming_labels, champion_predictions)

    combined = pd.concat(
        [
            existing[["document_text", "label"]],
            incoming[["document_text", "label"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["document_text", "label"])

    candidate_vectorizer = create_tfidf_vectorizer(
        max_features=feature_cfg.get("max_features", 3000),
        ngram_range=tuple(feature_cfg.get("ngram_range", [1, 2])),
        sublinear_tf=feature_cfg.get("sublinear_tf", True),
        min_df=feature_cfg.get("min_df", 2),
    )
    combined_text = combined["document_text"].astype(str).to_numpy()
    combined_labels = combined["label"].astype(str).to_numpy()
    combined_features = candidate_vectorizer.fit_transform(combined_text)

    base_estimator = LinearSVC(
        random_state=random_state,
        C=1.0,
        max_iter=2000,
    )
    candidate_model = CalibratedClassifierCV(estimator=base_estimator, cv=3)
    candidate_model.fit(combined_features, combined_labels)

    candidate_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, candidate_model_path)
    joblib.dump(candidate_vectorizer, candidate_vectorizer_path)

    candidate_predictions = candidate_model.predict(
        candidate_vectorizer.transform(incoming_text)
    )
    candidate_metrics = classification_metrics(incoming_labels, candidate_predictions)

    champion_report = classification_report(
        incoming_labels,
        champion_predictions,
        output_dict=True,
        zero_division=0,
    )
    candidate_report = classification_report(
        incoming_labels,
        candidate_predictions,
        output_dict=True,
        zero_division=0,
    )
    evaluation_report_path = report_path.parent / "new_batch_evaluation.json"
    evaluation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incoming_batch": str(incoming_path),
        "validation": validation,
        "champion_metrics": champion_metrics,
        "champion_classification_report": champion_report,
        "candidate_metrics": candidate_metrics,
        "candidate_classification_report": candidate_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evaluation_report_path, "w", encoding="utf-8") as file:
        json.dump(evaluation_report, file, indent=2)

    classification_report_path = report_path.parent / "new_batch_classification_report.txt"
    with open(classification_report_path, "w", encoding="utf-8") as file:
        file.write("CHAMPION MODEL\n")
        file.write(
            classification_report(
                incoming_labels,
                champion_predictions,
                zero_division=0,
            )
        )
        file.write("\n\nCANDIDATE MODEL\n")
        file.write(
            classification_report(
                incoming_labels,
                candidate_predictions,
                zero_division=0,
            )
        )

    eligible = bool(
        candidate_metrics["macro_f1"] >= champion_metrics["macro_f1"]
        and candidate_metrics["macro_f1"] >= minimum_f1
    )
    promotion_action = "retained_champion"

    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "docvane-classification"))

    run_id = None
    with mlflow.start_run(run_name="candidate_retraining") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", "Calibrated_Linear_SVM")
        mlflow.log_param("existing_dataset_rows", int(len(existing)))
        mlflow.log_param("incoming_batch_rows", int(len(incoming)))
        mlflow.log_param("combined_training_rows", int(len(combined)))
        mlflow.log_param("quality_metric", "macro_f1")
        mlflow.log_param("minimum_macro_f1", minimum_f1)
        mlflow.log_param("candidate_eligible", eligible)
        for metric_name, value in candidate_metrics.items():
            mlflow.log_metric(f"candidate_{metric_name}", value)
        for metric_name, value in champion_metrics.items():
            mlflow.log_metric(f"champion_{metric_name}", value)
        mlflow.log_artifact(str(candidate_model_path), artifact_path="candidate")
        mlflow.log_artifact(str(candidate_vectorizer_path), artifact_path="candidate")

    if eligible and promote:
        joblib.dump(candidate_model, champion_model_path)
        joblib.dump(candidate_vectorizer, champion_vectorizer_path)
        promotion_action = "promoted_candidate"

    timestamp = datetime.now(timezone.utc).isoformat()
    decision = {
        "timestamp": timestamp,
        "incoming_batch": str(incoming_path),
        "validation": validation,
        "existing_training_rows": int(len(existing)),
        "combined_training_rows": int(len(combined)),
        "quality_metric": "macro_f1",
        "minimum_macro_f1": minimum_f1,
        "champion": {
            "model_path": str(champion_model_path),
            "metrics": champion_metrics,
        },
        "candidate": {
            "model_path": str(candidate_model_path),
            "vectorizer_path": str(candidate_vectorizer_path),
            "metrics": candidate_metrics,
        },
        "candidate_minus_champion_macro_f1": float(
            candidate_metrics["macro_f1"] - champion_metrics["macro_f1"]
        ),
        "eligible_for_promotion": eligible,
        "promotion_action": promotion_action,
        "explicit_promotion_requested": bool(promote),
        "mlflow_run_id": run_id,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)

    logger.info("Champion macro F1: %.4f", champion_metrics["macro_f1"])
    logger.info("Candidate macro F1: %.4f", candidate_metrics["macro_f1"])
    logger.info("Decision: %s", promotion_action)
    logger.info("Quality-gate report saved to %s", report_path)
    return decision


if __name__ == "__main__":
    run_retraining_workflow()
