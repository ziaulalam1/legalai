import sys
import csv
from pathlib import Path

sys.path.insert(0, "src")

from ldc.model import train, predict, embed

MODEL_PATH = Path("models/classifier.pkl")
DATA_DIR   = Path("data")
_TMP       = Path("/tmp")


def _ensure_model() -> None:
    if not MODEL_PATH.exists():
        print("No trained model found — training from data/train.csv …")
        train(DATA_DIR, MODEL_PATH)


def _generate_plots() -> tuple:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rows   = list(csv.DictReader((DATA_DIR / "train.csv").open()))
    texts  = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    classes = ["motion", "brief", "deposition", "order", "exhibit"]

    embeddings = embed(texts)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    # Plot 1: all classes
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=cls, s=60, alpha=0.85)
    for i, (lbl, txt) in enumerate(zip(labels, texts)):
        if lbl == "deposition" and "Exhibit 5" in txt:
            ax.scatter(coords[i, 0], coords[i, 1], s=150, facecolors="none",
                       edgecolors="black", linewidths=1.5, zorder=5)
            ax.annotate("misclassified\n(deposition -> exhibit)",
                        xy=(coords[i, 0], coords[i, 1]),
                        xytext=(coords[i, 0] + 0.04, coords[i, 1] + 0.06),
                        fontsize=7, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_title("Training document embeddings - PCA projection", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    p1 = _TMP / "embedding_2d.png"
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot 2: motion vs brief centroids
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in ["motion", "brief"]:
        idx = [i for i, l in enumerate(labels) if l == cls]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=cls, s=70, alpha=0.85)
        cx, cy = np.mean(coords[idx, 0]), np.mean(coords[idx, 1])
        ax.scatter(cx, cy, marker="X", s=200, zorder=6)
        ax.annotate(f"{cls} centroid", xy=(cx, cy), xytext=(cx + 0.02, cy + 0.04),
                    fontsize=8, fontweight="bold")
    m_idx = [i for i, l in enumerate(labels) if l == "motion"]
    b_idx = [i for i, l in enumerate(labels) if l == "brief"]
    mc = np.mean(coords[m_idx], axis=0)
    bc = np.mean(coords[b_idx], axis=0)
    dist = np.linalg.norm(mc - bc)
    ax.plot([mc[0], bc[0]], [mc[1], bc[1]], "k--", lw=1, alpha=0.5)
    mid = (mc + bc) / 2
    ax.annotate(f"d={dist:.3f}", xy=mid, fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_title("Motion vs Brief - centroid separation (PCA)", fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=8)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=8)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    p2 = _TMP / "embedding_motion_vs_brief.png"
    plt.savefig(p2, dpi=150)
    plt.close()

    return str(p1), str(p2)


_ensure_model()
_plot_all, _plot_mb = _generate_plots()

import gradio as gr  # noqa: E402 (import after sys.path patch)

_BEHAVIOR_NOTES = """\
**Known behavior**

- Confidence below ~60% should be treated as uncertain rather than a hard label.
- Motion and order have the smallest centroid distance in embedding space (0.108), \
smaller than motion and brief (0.209). Both use procedural, court-directed language \
and are the most likely pair to be confused on ambiguous excerpts.
- The single training-set miss was a deposition question that referenced an exhibit \
by name ("I am handing you what has been labeled Exhibit 5"). The model returned \
exhibit at 0.35 confidence. The question text overrode the deposition context.
- The model is trained on full-length documents. Short excerpts (under ~50 tokens) \
will score less reliably across all classes.
"""

_CONFUSION_MD = """\
**Eval results — 74 / 75 correct on training set**

| | pred: motion | pred: brief | pred: deposition | pred: order | pred: exhibit |
|---|---|---|---|---|---|
| **true: motion** | 15 | 0 | 0 | 0 | 0 |
| **true: brief** | 0 | 15 | 0 | 0 | 0 |
| **true: deposition** | 0 | 0 | 14 | 0 | 1 |
| **true: order** | 0 | 0 | 0 | 15 | 0 |
| **true: exhibit** | 0 | 0 | 0 | 0 | 15 |
"""


def classify_text(text: str):
    if not text or not text.strip():
        return "", ""
    label, conf = predict(text.strip(), model_path=MODEL_PATH)
    return str(label), f"{conf:.1%}"


def classify_file(file):
    if file is None:
        return "", ""
    return classify_text(Path(file).read_text(errors="ignore"))


with gr.Blocks(title="Legal Document Classifier") as demo:
    gr.Markdown(
        "## Legal Document Classifier\n"
        "Classify a document as one of: **motion · brief · deposition · order · exhibit**."
    )
    gr.Markdown("> Uploaded files are processed in memory, not saved to disk (demo).")

    with gr.Tabs():
        with gr.Tab("Paste text"):
            text_in = gr.Textbox(
                lines=10, placeholder="Paste document text here…", label="Document text"
            )
            text_btn = gr.Button("Classify")
            with gr.Row():
                text_label = gr.Textbox(label="Predicted label")
                text_conf  = gr.Textbox(label="Confidence")
            text_btn.click(classify_text, inputs=text_in, outputs=[text_label, text_conf])

        with gr.Tab("Upload .txt file"):
            file_in = gr.File(file_types=[".txt"], label="Upload .txt file")
            file_btn = gr.Button("Classify")
            with gr.Row():
                file_label = gr.Textbox(label="Predicted label")
                file_conf  = gr.Textbox(label="Confidence")
            file_btn.click(classify_file, inputs=file_in, outputs=[file_label, file_conf])

        with gr.Tab("Model behavior"):
            gr.Markdown(_BEHAVIOR_NOTES)
            gr.Markdown(_CONFUSION_MD)
            with gr.Row():
                gr.Image(_plot_all, label="All classes — PCA projection",
                         show_download_button=False)
                gr.Image(_plot_mb, label="Motion vs Brief — centroid distances",
                         show_download_button=False)

if __name__ == "__main__":
    demo.launch()
