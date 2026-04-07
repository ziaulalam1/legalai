"""Benchmark: validation overhead vs raw classification.

Runs raw and validated classification on all 75 training documents,
measures throughput difference, and reports which documents trigger flags.

Outputs:
  reports/validation_benchmark.json  — structured results
  reports/validation_benchmark.png   — flag rate by class + overhead chart
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from ldc.model import embed, predict
from ldc.validators import compute_centroids, create_audit_record, validate

DATA_DIR = Path("data")
MODEL_PATH = Path("models/classifier.pkl")
REPORTS_DIR = Path("reports")


def run_benchmark():
    REPORTS_DIR.mkdir(exist_ok=True)

    rows = list(csv.DictReader((DATA_DIR / "train.csv").open()))
    texts = [r["text"] for r in rows]
    true_labels = [r["label"] for r in rows]

    print(f"Benchmarking {len(texts)} documents...")

    # Pre-compute centroids (one-time cost)
    t0 = time.perf_counter()
    centroids = compute_centroids(DATA_DIR)
    centroid_time = time.perf_counter() - t0
    print(f"Centroid computation: {centroid_time:.3f}s")

    # Phase 1: raw classification
    t0 = time.perf_counter()
    raw_results = []
    for text in texts:
        label, conf = predict(text, model_path=MODEL_PATH)
        raw_results.append((label, conf))
    raw_time = time.perf_counter() - t0
    raw_throughput = len(texts) / raw_time

    # Phase 2: validated classification
    t0 = time.perf_counter()
    validated_results = []
    for text in texts:
        label, conf = predict(text, model_path=MODEL_PATH)
        emb = embed([text])[0]
        report = validate(text, label, conf, emb, centroids, MODEL_PATH)
        audit = create_audit_record(text, label, conf, report)
        validated_results.append({
            "label": label,
            "confidence": conf,
            "status": report.status,
            "flags": [f.message for f in report.flags],
            "audit": audit.model_dump(),
        })
    validated_time = time.perf_counter() - t0
    validated_throughput = len(texts) / validated_time

    overhead_pct = ((validated_time - raw_time) / raw_time) * 100

    # Analysis
    flag_count = sum(1 for r in validated_results if r["flags"])
    status_counts = {"clear": 0, "review": 0, "uncertain": 0}
    flags_by_class = {}
    for i, r in enumerate(validated_results):
        status_counts[r["status"]] += 1
        cls = true_labels[i]
        if r["flags"]:
            flags_by_class.setdefault(cls, []).append(r["flags"])

    # Print summary
    print(f"\n{'='*50}")
    print(f"RAW CLASSIFICATION")
    print(f"  Time: {raw_time:.3f}s | Throughput: {raw_throughput:.1f} docs/s")
    print(f"\nVALIDATED CLASSIFICATION")
    print(f"  Time: {validated_time:.3f}s | Throughput: {validated_throughput:.1f} docs/s")
    print(f"  Overhead: +{overhead_pct:.1f}%")
    print(f"\nSTATUS BREAKDOWN")
    for status, count in status_counts.items():
        print(f"  {status}: {count}/{len(texts)} ({count/len(texts):.0%})")
    print(f"\nFLAGS BY CLASS")
    for cls in ["motion", "brief", "deposition", "order", "exhibit"]:
        n = len(flags_by_class.get(cls, []))
        print(f"  {cls}: {n} flagged")
    print(f"{'='*50}")

    # Save JSON report
    report_data = {
        "total_documents": len(texts),
        "raw_time_s": round(raw_time, 4),
        "raw_throughput_docs_per_s": round(raw_throughput, 1),
        "validated_time_s": round(validated_time, 4),
        "validated_throughput_docs_per_s": round(validated_throughput, 1),
        "overhead_pct": round(overhead_pct, 1),
        "centroid_computation_s": round(centroid_time, 4),
        "status_breakdown": status_counts,
        "flags_by_class": {cls: len(flags_by_class.get(cls, [])) for cls in ["motion", "brief", "deposition", "order", "exhibit"]},
        "flagged_documents": flag_count,
        "unflagged_documents": len(texts) - flag_count,
    }
    with open(REPORTS_DIR / "validation_benchmark.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate chart
    _generate_chart(report_data)

    print(f"\nReports saved to {REPORTS_DIR}/")
    return report_data


def _generate_chart(data: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: flag rate by class
    classes = ["motion", "brief", "deposition", "order", "exhibit"]
    flag_counts = [data["flags_by_class"][c] for c in classes]
    total_per_class = data["total_documents"] // 5  # 15 each
    flag_rates = [n / total_per_class for n in flag_counts]

    colors = ["#5b8dee" if r == 0 else "#f59e0b" if r < 0.3 else "#ef4444" for r in flag_rates]
    bars = ax1.bar(classes, flag_rates, color=colors, edgecolor="#333", linewidth=0.5)
    ax1.set_ylabel("Flag rate", fontsize=10)
    ax1.set_title("Validation flag rate by document class", fontsize=11)
    ax1.set_ylim(0, 1.0)
    for bar, count in zip(bars, flag_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{count}/{total_per_class}", ha="center", fontsize=8)

    # Right: throughput comparison
    labels = ["Raw\nclassification", "Validated\nclassification"]
    throughputs = [data["raw_throughput_docs_per_s"], data["validated_throughput_docs_per_s"]]
    colors2 = ["#5b8dee", "#34d399"]
    bars2 = ax2.bar(labels, throughputs, color=colors2, edgecolor="#333", linewidth=0.5)
    ax2.set_ylabel("Documents / second", fontsize=10)
    ax2.set_title(f"Throughput comparison (+{data['overhead_pct']:.0f}% overhead)", fontsize=11)
    for bar, val in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "validation_benchmark.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_benchmark()
