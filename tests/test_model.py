"""Tests for feature preprocessing, model inference, and evaluation."""

import math
from pathlib import Path
import pytest
import numpy as np

from src.features.preprocessing import clean_text, create_tfidf_vectorizer
from src.models.predict import DocumentClassifier
from src.models.evaluate import evaluate_saved_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_clean_text():
    """Verify text cleaning normalizes URLs, emails, currencies, and casing."""
    raw = "Check https://example.com for Invoice $500! Contact: test@work.com."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "currency" in cleaned
    assert "emailaddr" in cleaned
    assert cleaned.islower()


def test_tfidf_vectorizer():
    """Verify vectorizer transforms texts to expected dimensions."""
    docs = [
        "invoice total amount 500 dollars due date",
        "resume software engineer python docker fastapi",
        "certificate awarded completion machine learning",
        "bank statement account balance opening closing",
    ]
    vec = create_tfidf_vectorizer(max_features=50, min_df=1)
    matrix = vec.fit_transform(docs)
    assert matrix.shape[0] == 4
    assert matrix.shape[1] > 0


def test_classifier_inference_and_probabilities():
    """Test inference output structure, confidence calibration, and probability sum."""
    classifier = DocumentClassifier()
    sample_text = (
        "INVOICE #INV-4029\n"
        "Total Amount Due: $3,200.00\n"
        "Client: Acme Global Solutions\n"
        "Services rendered: Cloud hosting and database maintenance."
    )
    result = classifier.predict_text(sample_text)
    assert "category" in result
    assert result["category"] == "invoice"
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    probs = result["probabilities"]
    assert set(probs.keys()) == set(classifier.classes)
    total_prob = sum(probs.values())
    assert math.isclose(total_prob, 1.0, abs_tol=0.05)


def test_classifier_all_classes():
    """Verify correct classification across all 4 target classes."""
    classifier = DocumentClassifier()

    samples = {
        "invoice": "Tax invoice INV-1102 total due $450 items ordered",
        "resume": "Software engineer with 5 years experience in Python, FastAPI, Docker, and MLflow.",
        "certificate": "Certificate of completion awarded to John Doe for completing Data Science Course.",
        "bank_statement": "National Bank Statement of Account Account Number 1029384756 Closing Balance $5,400.00",
    }

    for expected_cat, text in samples.items():
        res = classifier.predict_text(text)
        assert res["category"] == expected_cat, f"Expected {expected_cat}, got {res['category']}"


def test_evaluate_threshold():
    """Verify saved model meets the required macro F1 threshold."""
    metrics = evaluate_saved_model()
    assert metrics["threshold_met"] is True
    assert metrics["macro_f1"] >= 0.80


def test_retraining_workflow_creates_candidate_and_decision(tmp_path):
    """Verify the incoming batch workflow creates a candidate and quality report."""
    from scripts.generate_incoming_batch import generate_incoming_batch
    from src.models.retrain import run_retraining_workflow

    incoming = generate_incoming_batch(tmp_path / "incoming.csv", samples_per_class=2)
    candidate_model = tmp_path / "candidate_model.joblib"
    candidate_vectorizer = tmp_path / "candidate_vectorizer.joblib"
    decision_report = tmp_path / "quality_gate.json"

    decision = run_retraining_workflow(
        incoming_path=incoming,
        candidate_model_path=candidate_model,
        candidate_vectorizer_path=candidate_vectorizer,
        report_path=decision_report,
        promote=False,
    )

    assert candidate_model.exists()
    assert candidate_vectorizer.exists()
    assert decision_report.exists()
    assert "champion" in decision
    assert "candidate" in decision
    assert decision["quality_metric"] == "macro_f1"
    assert isinstance(decision["eligible_for_promotion"], bool)
