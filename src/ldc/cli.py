import argparse
import csv
from pathlib import Path

from ldc.purge import purge_path
from ldc.schema import DocResult, EmailReport


def _train(args: argparse.Namespace) -> None:
    from ldc.model import train
    train(Path(args.data_dir), Path(args.model))


def _eval(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from ldc.model import embed

    rows = list(csv.DictReader((Path(args.data_dir) / "train.csv").open()))
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    X = embed(texts)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print(classification_report(y_te, y_pred, zero_division=0))
    classes = sorted(set(labels))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("            " + "  ".join(f"{c:>11}" for c in classes))
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    for row_label, row in zip(classes, cm):
        print(f"{row_label:>12}  " + "  ".join(f"{v:>11}" for v in row))


def _infer_eml(args: argparse.Namespace) -> None:
    from ldc.eml_intake import extract_docs, parse_eml
    from ldc.model import predict

    eml_path = Path(args.eml)
    out_path = Path(args.out)
    model_path = Path(args.model)

    info = parse_eml(eml_path)
    # sanitize message-id — it can contain slashes and @ which break the path
    safe_id = (
        info["message_id"]
        .replace("/", "_")
        .replace("@", "_at_")
        .replace(" ", "_")
        or "unknown"
    )
    work_dir = Path("/runtime/work") / safe_id

    results: list[DocResult] = []
    try:
        docs = extract_docs(eml_path, work_dir)
        for i, (attachment_path, doc_text) in enumerate(docs):
            if not doc_text.strip():
                continue
            label, confidence = predict(doc_text, model_path)
            results.append(
                DocResult(
                    doc_id=f"{safe_id}_{i}",
                    filename=attachment_path.name,
                    label=label,
                    confidence=confidence,
                )
            )
    finally:
        # compliance requirement: extracted text must not persist after inference
        purge_path(work_dir)

    report = EmailReport(
        message_id=info["message_id"],
        subject=info["subject"],
        from_addr=info["from"],
        to_addrs=info["to"],
        results=results,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2))
    print(f"report written to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ldc")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train classifier on data/train.csv")
    tr.add_argument("--data-dir", default="/app/data")
    tr.add_argument("--model", default="/app/models/classifier.pkl")

    ev = sub.add_parser("eval", help="eval with 80/20 split, prints precision/recall + confusion matrix")
    ev.add_argument("--data-dir", default="/app/data")

    infer = sub.add_parser("infer-eml", help="classify docs found in an .eml file")
    infer.add_argument("--eml", required=True)
    infer.add_argument("--out", required=True)
    infer.add_argument("--model", default="/app/models/classifier.pkl")

    args = p.parse_args()
    {"train": _train, "eval": _eval, "infer-eml": _infer_eml}[args.cmd](args)
