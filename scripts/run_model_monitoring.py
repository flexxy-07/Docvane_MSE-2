"""Demonstrate model monitoring and candidate retraining for the DocVane project."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_incoming_batch import generate_incoming_batch
from src.models.retrain import run_retraining_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote an eligible candidate over the champion.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Use an existing data/incoming/new_batch.csv file.",
    )
    args = parser.parse_args()

    incoming_path = PROJECT_ROOT / "data" / "incoming" / "new_batch.csv"
    if not args.skip_generation:
        generate_incoming_batch(incoming_path)

    decision = run_retraining_workflow(
        incoming_path=incoming_path,
        promote=args.promote,
    )

    print(json.dumps(decision, indent=2))
    print("\nCandidate artifact:", PROJECT_ROOT / "artifacts" / "candidate_model.joblib")
    print("Decision report:", PROJECT_ROOT / "reports" / "quality_gate_decision.json")


if __name__ == "__main__":
    main()
