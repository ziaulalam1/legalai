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

with gr.Blocks(title="Legal Document Classifier") as demo:
    gr.Markdown(
        "## Legal Document Classifier\n"
        "Classifies a document into one of: **motion · brief · deposition · order · exhibit**."
    )
    gr.Markdown(
        "> Uploaded files are processed in memory and not persisted (best-effort; this is a demo)."
    )

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

if __name__ == "__main__":
    demo.launch()
