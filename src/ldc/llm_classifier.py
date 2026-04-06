"""LLM-based legal document classifier using the Anthropic Messages API.

Drop-in alternative to model.predict() — same interface, no trained weights needed.
Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

from typing import Tuple

import anthropic

from ldc.schema import Label

_LABELS = ("motion", "brief", "deposition", "order", "exhibit")

_SYSTEM = (
    "You are a legal document classifier. "
    "Given a legal document excerpt, output exactly one label and nothing else. "
    "Valid labels: motion, brief, deposition, order, exhibit."
)

# Five-shot examples as alternating user/assistant turns
_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "content": "COMES NOW Plaintiff and respectfully moves this Court to exclude the expert testimony of Dr. Smith on grounds of lack of foundation.",
    },
    {"role": "assistant", "content": "motion"},
    {
        "role": "user",
        "content": "The central question before this Court is whether the implied covenant of good faith bars Defendant from invoking the termination clause.",
    },
    {"role": "assistant", "content": "brief"},
    {
        "role": "user",
        "content": "Q: Please state your full name for the record. A: My name is Maria Johnson. Q: And what is your occupation? A: I am a licensed civil engineer.",
    },
    {"role": "assistant", "content": "deposition"},
    {
        "role": "user",
        "content": "IT IS HEREBY ORDERED that Defendant shall produce all responsive documents within fourteen (14) days of the entry of this Order.",
    },
    {"role": "assistant", "content": "order"},
    {
        "role": "user",
        "content": "EXHIBIT B — PURCHASE AGREEMENT dated March 15, 2023 between Alpha Corp. (\"Buyer\") and Beta LLC (\"Seller\") for the sale of the Property at 142 Main Street.",
    },
    {"role": "assistant", "content": "exhibit"},
]

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def predict(text: str) -> Tuple[Label, float]:
    """Classify a legal document excerpt via Claude few-shot prompting.

    Args:
        text: Raw document text. First 500 chars are sent to keep token cost low.

    Returns:
        (label, confidence) where confidence is always 1.0 (discrete LLM output).

    Raises:
        anthropic.APIError: on network or API-key failures.
    """
    messages = list(_EXAMPLES) + [{"role": "user", "content": text[:500]}]

    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8,
        system=_SYSTEM,
        messages=messages,
    )

    raw = response.content[0].text.strip().lower()
    label: Label = raw if raw in _LABELS else _closest(raw)
    return label, 1.0


def _closest(raw: str) -> Label:
    """Return the first valid label found as a substring; fall back to 'motion'."""
    for lbl in _LABELS:
        if lbl in raw:
            return lbl  # type: ignore[return-value]
    return "motion"  # type: ignore[return-value]
