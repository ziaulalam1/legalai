---
title: Legal Document Classifier
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.3"
app_file: app.py
pinned: false
---

# ldc - Legal Document Classifier

Most of the classification examples used in Professor Kewei Li's AI 681 at LIU Brooklyn were clean -- the classes were balanced, the text was formatted properly, the structure was predictable. Legal documents are none of these things. Depositions and briefs can appear very similar at the sentence level. Court filings are received as emails containing PDF attachments; some are scanned and others contain multi-column layouts which render most text extraction tools useless. I wanted to develop a tool that could operate on actual court documents, regardless of how they looked in a dataset. The classifier uses sentence embeddings from a pre-trained model that was fine-tuned to the specific language and terminology of law along with a logistic regression layer on top -- given the limited amount of examples (15 total per class), fine-tuning the entire transformer head would simply allow it to memorize the training data.

Live demo: [huggingface.co/spaces/ziaulalam1/ldc](https://huggingface.co/spaces/ziaulalam1/ldc)

## Architecture

| Layer | Component | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) | Strong semantic representations without using a GPU or fine-tuning |
| Classifier | `LogisticRegression` (scikit-learn) | 75-sample training set, LR performs better than a fine-tuned head at this data size |
| Intake | stdlib `email` + `pdftotext` | Not dependent on heavy OCR, works with body text, plain attachments, and PDFs |
| Purge | `shutil.rmtree` in a `finally` block | Extracted files are always removed, regardless of exceptions |

The encoder is a module-level singleton. During eval I noticed inference was much slower than expected and traced it to the SentenceTransformer loading from disk on every `predict()` call. 75 loads for a 75-row eval set. One load per process cut eval time roughly 4x.

## Scope

In:
- 5 label classes: `motion`, `brief`, `deposition`, `order`, `exhibit`
- `.eml` intake with PDF and plain-text attachment support
- Compliance purge: no extracted plaintext survives inference
- CLI (`ldc train`, `ldc eval`, `ldc infer-eml`) + Gradio web UI
- Fully containerised, read-only root filesystem (Docker)

Out:
- No database, no auth, no web framework
- No OCR for scanned PDFs
- No retraining from the UI

## Quickstart (Docker)

```bash
# Build
docker compose build

# Train
docker compose run --rm legalai ldc train

# Classify an email
docker compose run --rm legalai ldc infer-eml \
  --eml /data/discovery.eml --out /runtime/results.json

# Run tests
docker compose run --rm legalai python -m pytest -q
```

## Tests

| Test | What it covers |
|---|---|
| `test_purge_file` / `test_purge_dir` | `purge_path` deletes files and directories |
| `test_infer_eml_purge` | Full extraction from a real `.eml`, asserts docs extracted, asserts work dir deleted |

The purge tests exist because the compliance guarantee is the hardest property to verify by inspection alone. The test runs the full extraction pipeline against `samples/sample.eml`, confirms documents were extracted, then confirms the work directory no longer exists after `purge_path()`.

## Troubleshooting

### Cold start is slow (~30 s)
The Space trains the classifier from `data/train.csv` on first launch. No pre-built model is committed to the repo. Expected behaviour.

### `pdftotext` not found
Requires `poppler-utils`. The Dockerfile installs it. Outside Docker: `apt install poppler-utils` or `brew install poppler`.

### Permission denied writing to `models/`
The container runs read-only. Only `/app`, `/runtime`, and `/tmp` are writable (see `docker-compose.yml`). Model path must be under one of these.

### Low confidence on short documents
The classifier was trained on full-length documents. Inputs under ~50 tokens will show lower confidence. Expected, not a bug.

## Tradeoffs and Limitations

### Why logistic regression, not fine-tuning?
With 15 examples per class, fine-tuning a transformer head can lead to simply memorizing the training set. However, all-MiniLM-L6-v2 already performs well on encoding legal language, therefore LR on top of those embeddings will generalize cleanly at this data size, and will train in less than one second on CPU.

### Why `pdftotext` over a pure-Python PDF library?
Legal PDFs from court filings usually contain two-column layouts, embedded form fields and boilerplate running headers/footers. `pdfminer` and `pypdf` return that structure verbatim, thus the classifier has to deal with column-interleaved text and footer noise mixed into the document body which hurts accuracy. `pdftotext` (Poppler) uses layout heuristics to reconstruct the reading order and remove the noise. It adds a system dependency, but the quality of the text directly influences the quality of the classification, so correctness won. The Dockerfile takes care of the installation.

### Limitations
- There are only 15 examples per class. Precision decreases on ambiguous documents, e.g., a brief that includes exhibit attachments.
- The intake assumes either UTF-8 or latin-1 encoding. Non-Latin attachments are skipped silently.
- Only plain text is accepted by the Gradio UI. The complete .eml intake pipeline is not available in the web demo.

## What I'd do next

### Grow training data via active learning
Run inference on unlabeled documents; surface low-confidence predictions for human review; retrain. The pipeline is already implemented. The bottleneck is labeled data.

### Add a confidence threshold
Below ~60% return `"uncertain"` instead of a potentially wrong label. A wrong label during discovery is worse than no label.

### Async purge
The `finally` block currently locks the response. In a higher-throughput environment, purge should be fire-and-forget with a watchdog to catch failures.

### Structured logging
Currently, output is `print()`. JSON logs to stdout would make the compliance audit trail machine-readable.
