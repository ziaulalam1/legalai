import sys
from pathlib import Path

sys.path.insert(0, "src")

from ldc.model import train, predict

MODEL_PATH = Path("models/classifier.pkl")
DATA_DIR = Path("data")


def _ensure_model() -> None:
    if not MODEL_PATH.exists():
        print("No trained model found — training from data/train.csv …")
        train(DATA_DIR, MODEL_PATH)


_ensure_model()


def classify_text(text: str):
    if not text or not text.strip():
        return "", ""
    label, conf = predict(text.strip(), model_path=MODEL_PATH)
    return str(label), f"{conf:.1%}"


def classify_file(file):
    if file is None:
        return "", ""
    return classify_text(Path(file).read_text(errors="ignore"))


import gradio as gr  # noqa: E402 (import after sys.path patch)

# ── prose blocks (TO BE HUMANIZED before push) ──────────────────────────────
_HEADER_DESC = (
    "Classify a document as one of: **motion · brief · deposition · order · exhibit**."
)

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
# ────────────────────────────────────────────────────────────────────────────

_EMBEDDING_ALL   = Path("reports/embedding_2d.png")
_EMBEDDING_MB    = Path("reports/embedding_motion_vs_brief.png")

with gr.Blocks(title="Legal Document Classifier") as demo:
    gr.Markdown(f"## Legal Document Classifier\n{_HEADER_DESC}")
    gr.Markdown("> Uploaded files are processed in memory, not saved to disk (demo).")

    with gr.Tabs():
        with gr.Tab("Paste text"):
            text_in = gr.Textbox(
                lines=10, placeholder="Paste document text here…", label="Document text"
            )
            text_btn = gr.Button("Classify")
            with gr.Row():
                text_label = gr.Textbox(label="Predicted label")
                text_conf = gr.Textbox(label="Confidence")
            text_btn.click(classify_text, inputs=text_in, outputs=[text_label, text_conf])

        with gr.Tab("Upload .txt file"):
            file_in = gr.File(file_types=[".txt"], label="Upload .txt file")
            file_btn = gr.Button("Classify")
            with gr.Row():
                file_label = gr.Textbox(label="Predicted label")
                file_conf = gr.Textbox(label="Confidence")
            file_btn.click(classify_file, inputs=file_in, outputs=[file_label, file_conf])

        with gr.Tab("Model behavior"):
            gr.Markdown(_BEHAVIOR_NOTES)
            gr.Markdown(_CONFUSION_MD)
            with gr.Row():
                if _EMBEDDING_ALL.exists():
                    gr.Image(str(_EMBEDDING_ALL), label="All classes — PCA projection",
                             show_download_button=False)
                if _EMBEDDING_MB.exists():
                    gr.Image(str(_EMBEDDING_MB), label="Motion vs Order/Brief — centroid distances",
                             show_download_button=False)

if __name__ == "__main__":
    demo.launch()
