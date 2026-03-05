#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

# non-Docker path: schema + unit tests only
if [ "${1:-}" = "--local" ]; then
  echo "== local tests (no Docker required) =="

  if [ ! -x ".venv/bin/python3" ]; then
    echo "ERROR: .venv not found. Bootstrap with:"
    echo "  python3 -m venv .venv && .venv/bin/pip install pytest pydantic"
    exit 1
  fi

  PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_smoke.py tests/test_purge.py -v
  echo "done"
  exit 0
fi

# Docker path: full integration suite
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running."
  echo "To run unit tests without Docker: bash scripts/verify.sh --local"
  exit 1
fi

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
