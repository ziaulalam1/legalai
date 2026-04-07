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

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space (SDK: **Gradio**, hardware: **CPU Free**)
2. Push this repo to the Space:
   ```
   git remote add space https://huggingface.co/spaces/ziaulalam1/ldc
   git push space main
   ```
3. The Space installs `requirements.txt`, trains the model from `data/train.csv` on first cold start (~30 s), then serves the Gradio UI.

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

## LLM Classifier Comparison

`src/ldc/llm_classifier.py` adds a second classification approach using the Anthropic Claude API. Same `predict(text)` interface as the embedding model — no weights, no training data required.

| Approach | Method | Weights | Speed |
|---|---|---|---|
| Embedding + LR | `all-MiniLM-L6-v2` + logistic regression | Trained on 75 samples | Fast (local) |
| LLM few-shot | Claude Haiku (5-shot prompting) | None | API latency |

To run the LLM classifier:

```python
from ldc.llm_classifier import predict  # requires ANTHROPIC_API_KEY in env
label, confidence = predict("COMES NOW Plaintiff and respectfully moves...")
```

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

## Validation Layer

The classifier alone tells you what a document is. The validation layer tells you when to trust that answer.

Every classification runs through 5 invariant checks, each derived from a documented failure mode:

| Check | What it catches | Threshold | Calibration source |
|-------|----------------|-----------|-------------------|
| Confidence gate | Near-random predictions | < 30% | Known misclassification scored 0.28; model mean is 0.45 |
| Confusion pair | Ambiguous top-2 classes | Gap < 0.05 | Motion/order centroids are 0.133 apart (closest pair) |
| Short input | Insufficient context | < 10 tokens | Training excerpts are 13-28 tokens |
| Perturbation stability | Fragile high-confidence labels | Label flips on 20% truncation | Only checked when confidence > 80% |
| Audit record | Completeness | All fields non-null | SHA-256 input hash, timestamp, flags |

### Benchmark results (75 training documents)

| Metric | Value |
|--------|-------|
| Clear (all checks passed) | 66/75 (88%) |
| Review (ambiguous, needs human check) | 7/75 (9%) |
| Uncertain (confidence too low) | 2/75 (3%) |
| Raw throughput | 102.9 docs/s |
| Validated throughput | 35.5 docs/s |
| Overhead | +189% |

The order class has the highest flag rate (6/15) because motion and order share procedural, court-directed language and have the smallest centroid separation (0.133). This is exactly the confusion pair the model is most likely to get wrong, and the validation layer catches it.

### Why calibrate?

The original README noted "confidence below ~60% should be treated as uncertain." On inspection, the model never exceeds 60% confidence -- even on training data. Setting a 60% threshold would flag every single prediction. The actual thresholds were derived from the model's empirical behavior: confidence distribution, centroid distances, and training-set token lengths. Validation that isn't calibrated to the model it validates is noise.

### Running the validation benchmark

```bash
PYTHONPATH=src python benchmark_validators.py
# outputs: reports/validation_benchmark.json, reports/validation_benchmark.png
```

---

## What I'd do next

### Grow training data via active learning
Run inference on unlabeled documents; surface low-confidence predictions for human review; retrain. The pipeline is already implemented. The bottleneck is labeled data.

### Cross-validation on the validation layer itself
The current thresholds were calibrated on training data. A held-out validation set would test whether the thresholds generalize -- specifically, whether the confusion pair margin and confidence gate catch the same types of errors on unseen documents.

### Async purge
The `finally` block currently locks the response. In a higher-throughput environment, purge should be fire-and-forget with a watchdog to catch failures.
