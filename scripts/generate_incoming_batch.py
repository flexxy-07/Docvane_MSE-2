"""Generate a deterministic labelled incoming batch for model-quality monitoring."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def generate_incoming_batch(
    output_path: Path = PROJECT_ROOT / "data" / "incoming" / "new_batch.csv",
    samples_per_class: int = 8,
) -> Path:
    random.seed(2026)
    records = []

    names = ["Priya Shah", "Noah Williams", "Olivia Brown", "Ethan Wilson"]
    companies = ["Northstar Analytics", "Orbit Cloud Systems", "Vertex Research"]
    courses = ["Production ML Systems", "Advanced NLP Engineering", "Cloud MLOps"]
    institutions = ["Professional Institute", "Applied AI Academy", "Tech Council"]

    for i in range(1, samples_per_class + 1):
        records.append({
            "record_id": f"INV-NEW-{i:03d}",
            "document_text": (
                f"NEW INVOICE\nVendor: {random.choice(companies)}\n"
                f"Invoice Number: INV-NEW-{i:03d}\nBilling Date: 20/11/2026\n"
                f"Customer: {random.choice(names)}\n"
                f"Cloud infrastructure and engineering services\n"
                f"Subtotal: ${1000 + i * 75:.2f}\n"
                f"Tax: ${100 + i * 7:.2f}\n"
                f"Total Amount Due: ${1100 + i * 82:.2f}"
            ),
            "label": "invoice",
        })

        records.append({
            "record_id": f"RES-NEW-{i:03d}",
            "document_text": (
                f"{random.choice(names)}\nNEW CURRICULUM VITAE\n"
                f"Professional Summary: Machine learning engineer and MLOps specialist.\n"
                f"Technical Skills: Python, Scikit-Learn, MLflow, Docker, FastAPI, Kubernetes.\n"
                f"Experience: Built document classification and model monitoring pipelines.\n"
                f"Education: Degree in Computer Science.\nResume Reference: RES-NEW-{i:03d}"
            ),
            "label": "resume",
        })

        records.append({
            "record_id": f"CERT-NEW-{i:03d}",
            "document_text": (
                f"CERTIFICATE OF COMPLETION\nPresented to: {random.choice(names)}\n"
                f"Successfully completed: {random.choice(courses)}\n"
                f"Issued by: {random.choice(institutions)}\n"
                f"Certificate ID: CERT-NEW-{i:03d}\nIssue Date: 20/11/2026"
            ),
            "label": "certificate",
        })

        records.append({
            "record_id": f"BANK-NEW-{i:03d}",
            "document_text": (
                f"NATIONAL BANK STATEMENT\nAccount Holder: {random.choice(names)}\n"
                f"Account Number: 2026NEW{i:06d}\nStatement Period: November 2026\n"
                f"Opening Balance: ${5000 + i * 100:.2f}\n"
                f"Credits: ${800 + i * 25:.2f}\n"
                f"Debits: ${300 + i * 15:.2f}\n"
                f"Closing Balance: ${5500 + i * 110:.2f}\n"
                f"Reference: BANK-NEW-{i:03d}"
            ),
            "label": "bank_statement",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Generated incoming batch with {len(records)} rows at {output_path}")
    return output_path


if __name__ == "__main__":
    generate_incoming_batch()
