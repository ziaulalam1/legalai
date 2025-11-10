import pytest
from pydantic import ValidationError

from ldc.schema import DocResult, EmailReport, Label


def test_doc_result_valid(sample_doc_result):
    r = DocResult(**sample_doc_result)
    assert r.label == "motion"
    assert r.confidence == 0.91
    assert r.filename == "sample.pdf"


def test_doc_result_meta_defaults_empty():
    r = DocResult(doc_id="d-001", label="brief", confidence=0.75)
    assert r.meta == {}


def test_doc_result_invalid_label():
    with pytest.raises(ValidationError):
        DocResult(doc_id="d-002", label="invalid_label", confidence=0.5)


def test_doc_result_filename_optional():
    r = DocResult(doc_id="d-003", label="order", confidence=0.88)
    assert r.filename is None


def test_email_report_valid():
    r = EmailReport(
        message_id="msg-001",
        subject="Discovery docs",
        from_addr="clerk@court.gov",
        to_addrs=["attorney@firm.com"],
        results=[
            DocResult(doc_id="d-001", label="exhibit", confidence=0.82)
        ],
    )
    assert len(r.results) == 1
    assert r.results[0].label == "exhibit"


def test_email_report_empty_results():
    r = EmailReport(
        message_id="msg-002",
        subject="No attachments",
        from_addr="a@b.com",
        to_addrs=[],
        results=[],
    )
    assert r.results == []


def test_label_values():
    valid = {"motion", "brief", "deposition", "order", "exhibit"}
    assert set(Label.__args__) == valid
