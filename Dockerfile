# =============================================================================
# Local LLM Document Q&A (RAG)
#
# Two stages: a builder that resolves dependencies into a virtualenv, and a
# slim runtime that carries only the venv, the source and the web UI.
#
# Model weights are deliberately NOT baked into the image.  They are downloaded
# on first use into HF_HOME, which the compose file mounts as a named volume so
# an image rebuild does not re-download several hundred megabytes.
# =============================================================================

# --- Stage 1: build the virtualenv -------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Some wheels (hnswlib, tokenizers) may need to compile if no binary matches.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt


# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Model cache lives on a mounted volume so rebuilds stay cheap.
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    # In-container defaults; override in compose or at run time.
    CHROMA_DIR=/app/data/vectorstore \
    RAGOBSERVE_DB_PATH=/app/.ragobserve/ragobserve.db \
    OLLAMA_BASE_URL="" \
    OLLAMA_MODEL=""

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser evaluation ./evaluation
COPY --chown=appuser:appuser scripts ./scripts

# The test suite and pytest.ini are intentionally not copied: `.dockerignore`
# excludes them, and a production image has no use for them. Build a test image
# with `docker build --target builder` if you need to run the suite in Docker.

# Writable locations for the index, documents and traces.
RUN mkdir -p /app/data/vectorstore /app/data/documents /app/.ragobserve /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
