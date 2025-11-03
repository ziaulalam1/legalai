import pytest


@pytest.fixture
def sample_doc_result():
    return {
        "doc_id": "doc-001",
        "filename": "sample.pdf",
        "label": "motion",
        "confidence": 0.91,
    }
