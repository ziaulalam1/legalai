Acceptance Criteria (non-negotiable)
1) Classify legal documents into exactly 5 labels:
   - motion, brief, deposition, order, exhibit
2) Uses transformer embeddings + classifier head.
3) `.eml` intake:
   - parse headers
   - extract body text and/or attachments (pdf/txt/html)
   - run inference for each document
   - output JSON report per email
4) Compliance purge:
   - any extracted plaintext + any extracted attachment files are deleted immediately after inference
   - only the JSON output may remain
5) Evaluation:
   - precision/recall per class + confusion matrix on held-out split

Minimalism constraints
- CLI-only, local-only, no DB, no web UI.
