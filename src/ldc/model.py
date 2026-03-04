from pathlib import Path
from typing import Tuple
import csv

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from ldc.schema import Label

_ST_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = Path("/app/models/classifier.pkl")

# was recreating the encoder on every call — noticed during eval it was loading 75 times
_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_ST_MODEL)
    return _encoder


def embed(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def train(data_dir: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    rows = list(csv.DictReader((data_dir / "train.csv").open()))
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    X = embed(texts)
    # LR works well here — small labeled set, embeddings carry most of the signal
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"trained on {len(texts)} samples → {model_path}")


def predict(text: str, model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[Label, float]:
    clf: LogisticRegression = joblib.load(model_path)
    X = embed([text])
    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label: Label = clf.classes_[idx]
    return label, float(proba[idx])
