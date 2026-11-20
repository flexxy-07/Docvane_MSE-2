"""Tests for data ingestion, extraction, and validation."""

from pathlib import Path
import pytest
import pandas as pd

from src.data.extraction import (
    extract_regex_fields,
    extract_text,
    extract_text_from_txt,
)
from src.data.validation import validate_dataset
from src.models.retrain import validate_incoming_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_validate_dataset_success():
    """Verify that the generated dataset passes validation."""
    dataset_csv = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
    assert dataset_csv.exists(), "Dataset CSV must exist before running tests"
    res = validate_dataset(dataset_csv)
    assert res["valid"] is True
    assert res["total_rows"] >= 40
    for cat in ["invoice", "resume", "certificate", "bank_statement"]:
        assert cat in res["class_distribution"]


def test_validate_dataset_missing_column(tmp_path):
    """Test validation failure when required column is missing."""
    invalid_csv = tmp_path / "bad.csv"
    df = pd.DataFrame({"text": ["sample"], "label": ["invoice"]})
    df.to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(invalid_csv)


def test_validate_dataset_missing_category(tmp_path):
    """Test validation failure when a required category is missing."""
    invalid_csv = tmp_path / "missing_cat.csv"
    df = pd.DataFrame({
        "document_text": ["text1", "text2"],
        "label": ["invoice", "resume"]
    })
    df.to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="missing required document categories"):
        validate_dataset(invalid_csv)


def test_validate_incoming_batch_success(tmp_path):
    incoming_csv = tmp_path / "incoming.csv"
    pd.DataFrame({
        "record_id": [f"row-{index}" for index in range(4)],
        "document_text": ["invoice text", "resume text", "certificate text", "bank text"],
        "label": ["invoice", "resume", "certificate", "bank_statement"],
    }).to_csv(incoming_csv, index=False)

    summary = validate_incoming_batch(incoming_csv)
    assert summary["valid"] is True
    assert summary["rows"] == 4
    assert summary["duplicate_record_ids"] == 0
    assert summary["duplicate_records"] == 0
    assert set(summary["target_distribution"]) == {
        "invoice",
        "resume",
        "certificate",
        "bank_statement",
    }


def test_validate_incoming_batch_missing_column(tmp_path):
    incoming_csv = tmp_path / "incoming.csv"
    pd.DataFrame({
        "document_text": ["invoice text"],
        "label": ["invoice"],
    }).to_csv(incoming_csv, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_incoming_batch(incoming_csv)


def test_validate_incoming_batch_duplicate_id(tmp_path):
    incoming_csv = tmp_path / "incoming.csv"
    pd.DataFrame({
        "record_id": ["duplicate", "duplicate"],
        "document_text": ["first invoice", "second resume"],
        "label": ["invoice", "resume"],
    }).to_csv(incoming_csv, index=False)

    with pytest.raises(ValueError, match="duplicate record IDs"):
        validate_incoming_batch(incoming_csv)


def test_extract_text_from_txt(tmp_path):
    """Test text extraction from TXT file."""
    txt_file = tmp_path / "test.txt"
    content = "Hello DocVane document test."
    txt_file.write_text(content, encoding="utf-8")
    extracted = extract_text_from_txt(txt_file)
    assert extracted == content


def test_extract_text_from_pdf():
    """Test PDF text extraction on a sample PDF from data/raw/."""
    pdf_sample = PROJECT_ROOT / "data" / "raw" / "invoice" / "invoice_001.pdf"
    if pdf_sample.exists():
        extracted = extract_text(pdf_sample, filename="invoice_001.pdf")
        assert len(extracted) > 50
        assert "invoice" in extracted.lower() or "inv" in extracted.lower()


def test_regex_extraction_invoice():
    """Test regex field extraction on an invoice snippet."""
    sample = "Tax Invoice: INV-9902 | Date: 12/10/2026 | Total Amount: $4,500.00"
    fields = extract_regex_fields(sample, category="invoice")
    assert fields.get("invoice_number") == "INV-9902"
    assert fields.get("total_amount") == "4,500.00"


def test_regex_extraction_resume():
    """Test regex field extraction on a resume snippet."""
    sample = "John Doe\nEmail: john.doe@example.com\nPhone: +1 (555) 123-4567\nSkills: Python, Docker, Pytest"
    fields = extract_regex_fields(sample, category="resume")
    assert fields.get("email") == "john.doe@example.com"
    assert fields.get("phone") == "+1 (555) 123-4567"
    assert "Python" in fields.get("skills", [])


def test_regex_extraction_bank_statement():
    """Test regex field extraction on a bank statement snippet."""
    sample = "First National Bank\nAccount Number: 987654321012\nClosing Balance: $12,450.75"
    fields = extract_regex_fields(sample, category="bank_statement")
    assert fields.get("account_number") == "987654321012"
    assert fields.get("closing_balance") == "12,450.75"


def test_regex_extraction_certificate():
    """Test regex field extraction on a certificate snippet."""
    sample = "Certificate of Completion\nProudly presented to Alice Walker\nfor completion of: Machine Learning Specialization\nCertificate ID: CERT-8821"
    fields = extract_regex_fields(sample, category="certificate")
    assert fields.get("recipient_name") == "Alice Walker"
    assert "Machine Learning" in fields.get("course_name", "")
    assert fields.get("certificate_id") == "CERT-8821"


def test_extract_text_scanned_pdf_ocr_fallback(tmp_path):
    """Test OCR fallback for scanned/image-only PDFs where digital text is absent."""
    import io
    import pymupdf as fitz
    from PIL import Image, ImageDraw
    from src.data.extraction import extract_text_from_pdf, find_tesseract_binary

    # 1. Create a pure-image PDF (simulating a scanned document)
    img = Image.new("RGB", (800, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "Tax Invoice: INV-7788 | Total Amount: $2,500.00", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=800, height=300)
    page.insert_image(page.rect, stream=buf.getvalue())
    scanned_pdf = tmp_path / "scanned_invoice.pdf"
    doc.save(str(scanned_pdf))
    doc.close()

    # 2. Without OCR fallback: standard text extraction yields empty string
    no_ocr_text = extract_text_from_pdf(scanned_pdf, ocr_fallback=False)
    assert no_ocr_text == ""

    # 3. With OCR fallback: if Tesseract is available, it should extract the invoice text
    if find_tesseract_binary():
        ocr_text = extract_text_from_pdf(scanned_pdf, ocr_fallback=True, ocr_engine="auto")
        assert len(ocr_text) > 0
        assert "INV-7788" in ocr_text or "Invoice" in ocr_text or "Total" in ocr_text


def test_extract_text_from_image_file(tmp_path):
    """Test text extraction from a standalone image file using OCR."""
    import io
    from PIL import Image, ImageDraw
    from src.data.extraction import extract_text, find_tesseract_binary

    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 50), "Certificate ID: CERT-9900", fill=(0, 0, 0))
    img_path = tmp_path / "sample_cert.png"
    img.save(img_path)

    if find_tesseract_binary():
        extracted = extract_text(img_path, filename="sample_cert.png")
        assert len(extracted) > 0
        assert "CERT" in extracted and "9900" in extracted
