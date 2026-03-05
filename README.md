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

A CLI + web tool that uses sentence embeddings and logistic regression to classify legal documents into five categories: motion, brief, deposition, order, and exhibit. The application is designed specifically for e-discovery processes in which each `.eml` email is parsed, the attachments are extracted from each document, each document is classified based on its content, and all of the plaintext from each document is immediately deleted after it has been used for classification to ensure compliance with discovery privilege rules.

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
With 15 examples per class, fine-tuning a transformer head risks memorising the training set. `all-MiniLM-L6-v2` already encodes legal language well. LR on top of those embeddings generalises cleanly at this data size and trains in under a second on CPU.

### Why `pdftotext` over a pure-Python PDF library?
Legal PDFs from court filings often have two-column layouts, embedded form fields, and running header/footer boilerplate. `pdfminer` and `pypdf` return that structure verbatim, so the classifier ends up seeing column-interleaved text and footer noise mixed into the document body, which tanks accuracy. `pdftotext` (Poppler) applies layout heuristics to reconstruct reading order and strips the noise. It adds a system dependency, but text quality directly determines classification quality, so correctness won. The Dockerfile handles the install.

### Limitations
- 15 examples per class is thin. Precision degrades on ambiguous documents, for example a brief that embeds exhibit attachments.
- Intake assumes UTF-8 or latin-1 encoding. Non-Latin attachments are skipped silently.
- The Gradio UI accepts plain text only. The full `.eml` intake pipeline is not exposed in the web demo.

## What I'd do next

### Grow training data via active learning
Run inference on unlabelled documents, surface low-confidence predictions for human review, retrain. The pipeline is already there. The bottleneck is labelled data.

### Add a confidence threshold
Below ~60%, return `"uncertain"` rather than a potentially wrong label. A wrong label in discovery is worse than no label.

### Async purge
The `finally` block currently blocks the response. In a higher-throughput setting, purge should be fire-and-forget with a watchdog to catch failures.

### Structured logging
Output is currently `print()`. JSON logs to stdout would make the compliance audit trail machine-readable.
