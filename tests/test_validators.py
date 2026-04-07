"""Invariant tests for the validation layer.

Each test verifies an enforced property, not just a feature:
- Confidence gate must reject uncertain classifications
- Confusion pairs must be detected when centroids are close
- Short inputs must trigger degradation warnings
- High-confidence labels must survive perturbation
- Audit records must be complete for every classification
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ldc.validators import (
    AuditRecord,
    ValidationReport,
    check_confidence_gate,
    check_confusion_pair,
    check_perturbation_stability,
    check_short_input,
    create_audit_record,
    validate,
)


# ── Confidence gate ───────────────────────────────────────────────


def test_confidence_gate_rejects_low():
    """Labels below 30% confidence MUST be flagged as uncertain.

    Threshold calibrated to model behavior: mean confidence is 0.45,
    the known misclassification scored 0.28. Below 0.30 is near-random
    for a 5-class model.
    """
    flag = check_confidence_gate(0.25)
    assert flag is not None
    assert flag.severity == "error"
    assert flag.check == "confidence_gate"


def test_confidence_gate_passes_high():
    """Labels above 30% confidence MUST NOT be flagged."""
    flag = check_confidence_gate(0.45)
    assert flag is None


# ── Confusion pair detection ──────────────────────────────────────


def test_confusion_pair_detected():
    """When two class centroids are within margin, the pair MUST be flagged."""
    # Simulate centroids where motion and order are very close
    dim = 384
    rng = np.random.RandomState(42)
    base = rng.randn(dim).astype(np.float32)
    base /= np.linalg.norm(base)

    centroids = {
        "motion": base,
        "order": base + rng.randn(dim).astype(np.float32) * 0.01,  # very close
        "brief": rng.randn(dim).astype(np.float32),
        "deposition": rng.randn(dim).astype(np.float32),
        "exhibit": rng.randn(dim).astype(np.float32),
    }
    # Normalize
    for k in centroids:
        centroids[k] = centroids[k] / np.linalg.norm(centroids[k])

    # Input embedding near the motion/order cluster
    embedding = base / np.linalg.norm(base)

    flag, top_classes = check_confusion_pair(embedding, centroids, margin=0.15)
    assert flag is not None
    assert flag.check == "confusion_pair"
    assert "motion" in flag.message or "order" in flag.message
    assert len(top_classes) == 5


# ── Short-input warning ──────────────────────────────────────────


def test_short_input_warning():
    """Inputs under 10 tokens MUST trigger a degradation warning.

    Calibrated to training data (13-28 tokens per sample). Below 10
    tokens provides too little context for the sentence embedding model.
    """
    short_text = "Motion to dismiss."
    flag, token_count = check_short_input(short_text)
    assert flag is not None
    assert flag.check == "short_input"
    assert token_count < 10

    normal_text = "COMES NOW the plaintiff and respectfully moves this Court for summary judgment."
    flag_normal, count_normal = check_short_input(normal_text)
    assert flag_normal is None
    assert count_normal >= 10


# ── Perturbation stability ───────────────────────────────────────


def test_perturbation_stability():
    """High-confidence labels that flip under 20% truncation MUST be flagged."""
    # Mock predict to return different labels for full vs truncated text
    call_count = {"n": 0}

    def mock_predict(text, model_path=None):
        call_count["n"] += 1
        # First call (original) returns motion, second call (truncated) returns order
        if call_count["n"] == 1:
            return "order", 0.7  # truncated version returns different label
        return "order", 0.7

    # Case 1: label flips — should flag
    call_count["n"] = 0
    with patch("ldc.validators.predict") as mp:
        mp.return_value = ("order", 0.7)  # truncated returns "order" instead of "motion"
        flag = check_perturbation_stability(
            " ".join(["word"] * 100), "motion", 0.9, Path("dummy")
        )
    assert flag is not None
    assert flag.check == "perturbation_stability"

    # Case 2: label stable — should NOT flag
    with patch("ldc.validators.predict") as mp:
        mp.return_value = ("motion", 0.88)  # truncated returns same label
        flag_stable = check_perturbation_stability(
            " ".join(["word"] * 100), "motion", 0.9, Path("dummy")
        )
    assert flag_stable is None

    # Case 3: low confidence — should skip check entirely
    flag_skip = check_perturbation_stability(
        " ".join(["word"] * 100), "motion", 0.5, Path("dummy")
    )
    assert flag_skip is None


# ── Audit record completeness ────────────────────────────────────


def test_audit_record_fields_complete():
    """Every audit record MUST have all fields populated and non-null."""
    report = ValidationReport(
        flags=[],
        token_count=120,
        top_classes=[{"label": "motion", "centroid_distance": 0.05}],
        stable=True,
    )
    record = create_audit_record(
        text="COMES NOW the plaintiff...",
        label="motion",
        confidence=0.92,
        report=report,
    )

    assert isinstance(record, AuditRecord)
    assert record.input_hash is not None and len(record.input_hash) == 16
    assert record.label == "motion"
    assert record.confidence == 0.92
    assert record.status == "clear"
    assert isinstance(record.flags, list)
    assert record.token_count == 120
    assert record.timestamp is not None

    # Same input must produce same hash (deterministic)
    record2 = create_audit_record("COMES NOW the plaintiff...", "motion", 0.92, report)
    assert record.input_hash == record2.input_hash
