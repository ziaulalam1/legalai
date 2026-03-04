#!/usr/bin/env bash
set -euo pipefail
cd /app || true

echo "== tests =="
python -m pytest -q

echo "== train =="
ldc train

echo "== eval =="
ldc eval

echo "== infer =="
rm -f /runtime/outbox/sample.json || true
ldc infer-eml --eml /app/samples/sample.eml --out /runtime/outbox/sample.json
cat /runtime/outbox/sample.json
