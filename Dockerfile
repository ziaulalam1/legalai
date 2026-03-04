FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates \
  poppler-utils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY tests /app/tests

RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e . \
  && python -c "import ldc.cli"

COPY . /app

CMD ["ldc", "--help"]
