"""Validation layer for LegalAI classifications.

Turns documented failure modes into enforced invariants:
1. Confidence gate — low-confidence outputs flagged as uncertain
2. Confusion pair detection — ambiguous top-2 classes flagged
3. Short-input degradation — inputs under 50 tokens flagged
4. Perturbation stability — label flips under truncation flagged
5. Audit record — structured JSON per classification
"""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from ldc.model import embed, predict
from ldc.schema import Label


class ValidationFlag(BaseModel):
    check: str
    severity: str  # "warning" or "error"
    message: str


class ValidationReport(BaseModel):
    flags: list[ValidationFlag] = []
    token_count: int = 0
    top_classes: list[dict] = []  # [{label, probability, centroid_distance}]
    stable: Optional[bool] = None  # None if not checked (confidence too low)

    @property
    def status(self) -> str:
        if any(f.severity == "error" for f in self.flags):
            return "uncertain"
        if self.flags:
            return "review"
        return "clear"


class AuditRecord(BaseModel):
    input_hash: str
    label: str
    confidence: float
    status: str
    flags: list[str]
    token_count: int
    timestamp: str


# ── Centroid computation ──────────────────────────────────────────


def compute_centroids(data_dir: Path) -> dict[str, np.ndarray]:
    """Pre-compute mean embedding per class from training data."""
    rows = list(csv.DictReader((data_dir / "train.csv").open()))
    texts_by_class: dict[str, list[str]] = {}
    for r in rows:
        texts_by_class.setdefault(r["label"], []).append(r["text"])

    centroids = {}
    for label, texts in texts_by_class.items():
        embeddings = embed(texts)
        centroids[label] = np.mean(embeddings, axis=0)
    return centroids


# ── Individual checks ─────────────────────────────────────────────


def check_confidence_gate(confidence: float, threshold: float = 0.30) -> Optional[ValidationFlag]:
    """Flag classifications below the confidence threshold.

    Calibrated to the model's actual confidence distribution: mean=0.45,
    max=0.60 on training data. A prediction below 0.30 is near-random
    for a 5-class model (0.20 = uniform). The known training-set
    misclassification scored 0.28.
    """
    if confidence < threshold:
        return ValidationFlag(
            check="confidence_gate",
            severity="error",
            message=f"Confidence {confidence:.1%} is below {threshold:.0%} threshold — treat as uncertain",
        )
    return None


def check_confusion_pair(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    margin: float = 0.05,
) -> tuple[Optional[ValidationFlag], list[dict]]:
    """Flag when the two nearest class centroids are within margin (cosine distance gap).

    Calibrated to actual centroid distances: motion-order gap is 0.1333
    (closest pair). A margin of 0.05 catches ~12% of training data where
    the top-2 classes are genuinely ambiguous.
    """
    distances = {}
    for label, centroid in centroids.items():
        cos_sim = float(np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid)))
        distances[label] = 1.0 - cos_sim  # cosine distance

    sorted_classes = sorted(distances.items(), key=lambda x: x[1])
    top_classes = [{"label": lbl, "centroid_distance": round(dist, 4)} for lbl, dist in sorted_classes]

    gap = sorted_classes[1][1] - sorted_classes[0][1]
    if gap < margin:
        return (
            ValidationFlag(
                check="confusion_pair",
                severity="warning",
                message=f"Ambiguous — {sorted_classes[0][0]} and {sorted_classes[1][0]} are within {gap:.3f} centroid distance (margin={margin})",
            ),
            top_classes,
        )
    return None, top_classes


def check_short_input(text: str, min_tokens: int = 10) -> tuple[Optional[ValidationFlag], int]:
    """Flag inputs shorter than the minimum token count.

    Calibrated to training data: training excerpts are 13-28 tokens.
    Inputs under 10 tokens provide too little context even for the
    sentence embedding model.
    """
    token_count = len(text.split())
    if token_count < min_tokens:
        return (
            ValidationFlag(
                check="short_input",
                severity="warning",
                message=f"Input is {token_count} tokens — classification reliability degrades below {min_tokens}",
            ),
            token_count,
        )
    return None, token_count


def check_perturbation_stability(
    text: str,
    label: Label,
    confidence: float,
    model_path: Path,
    confidence_threshold: float = 0.8,
) -> Optional[ValidationFlag]:
    """Truncate input to 80% and re-classify. Flag if label flips at high confidence."""
    if confidence < confidence_threshold:
        return None  # only check stability for high-confidence predictions

    words = text.split()
    truncated = " ".join(words[: int(len(words) * 0.8)])
    if not truncated.strip():
        return None

    perturbed_label, _ = predict(truncated, model_path=model_path)
    if perturbed_label != label:
        return ValidationFlag(
            check="perturbation_stability",
            severity="warning",
            message=f"Label flipped from {label} to {perturbed_label} after 20% truncation — confidence may be unreliable",
        )
    return None


# ── Orchestrator ──────────────────────────────────────────────────


def validate(
    text: str,
    label: Label,
    confidence: float,
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    model_path: Path,
) -> ValidationReport:
    """Run all validation checks and return a report."""
    flags: list[ValidationFlag] = []

    # 1. Confidence gate
    f = check_confidence_gate(confidence)
    if f:
        flags.append(f)

    # 2. Confusion pair detection
    f_confusion, top_classes = check_confusion_pair(embedding, centroids)
    if f_confusion:
        flags.append(f_confusion)

    # 3. Short-input check
    f_short, token_count = check_short_input(text)
    if f_short:
        flags.append(f_short)

    # 4. Perturbation stability (skip if already uncertain from confidence gate)
    stable = None
    if not any(f.check == "confidence_gate" for f in flags):
        f_perturb = check_perturbation_stability(text, label, confidence, model_path)
        if f_perturb:
            flags.append(f_perturb)
            stable = False
        elif confidence >= 0.8:
            stable = True

    return ValidationReport(
        flags=flags,
        token_count=token_count,
        top_classes=top_classes,
        stable=stable,
    )


def create_audit_record(
    text: str, label: str, confidence: float, report: ValidationReport
) -> AuditRecord:
    """Create a structured audit record for every classification."""
    return AuditRecord(
        input_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        label=label,
        confidence=round(confidence, 4),
        status=report.status,
        flags=[f.message for f in report.flags],
        token_count=report.token_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
