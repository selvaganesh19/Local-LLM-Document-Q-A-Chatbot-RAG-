# Local LLM Document Q&A (RAG)

Ask questions about your own documents and get answers with every claim traced
back to the passage it came from. Retrieval is hybrid (BM25 + dense vectors,
fused with reciprocal rank fusion), candidates are reordered by a cross-encoder,
and generation is grounded in the retrieved passages.

> ### Where the model actually runs
>
> This implementation was developed and verified against a **Modal-hosted vLLM
> server**, not a local Ollama install. The development machine has no dedicated
> GPU, so the generation model runs on a rented GPU and the service reaches it
> over HTTPS. **Ollama is fully supported** and is the better choice if you have
> a capable GPU locally - the client speaks both protocols and picks one from
> the URL.
>
> See [Choose a generation backend](#2-choose-a-generation-backend) for setup
> instructions for both, and for what this means for privacy.

---

## Demo

A recorded walkthrough - ingesting documents, asking questions, and inspecting
the citations returned with each answer:

**[`videos/local_llm_rag.mp4`](videos/local_llm_rag.mp4)** (70 MB)

GitHub renders neither form of embed for a repository-hosted MP4: its Markdown
sanitiser strips `<video>` tags, and the blob viewer downloads rather than
streams. For an inline player in the rendered README, drag the file into a
GitHub issue or comment box and paste the resulting
`https://github.com/user-attachments/assets/...` URL here - that is the only
form GitHub plays. A local Markdown previewer will play it from the repository
directly, either from the link above or via:

```html
<video src="videos/local_llm_rag.mp4" controls muted playsinline width="100%"></video>
```

---

## Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Web interface](#web-interface)
- [API](#api)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Observability](#observability)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)

---

## Features

**Core**

| Requirement | Implementation |
| --- | --- |
| Local LLM, any server | `app/generation/ollama_client.py` - async `httpx` client speaking either Ollama's native `/api/chat` or the OpenAI-compatible `/v1/chat/completions`, with token streaming and cold-start replay |
| Document ingestion | `app/ingestion/` - PDF, DOCX, Markdown, text and HTML loaders; sentence-aware chunking with overlap; batched embeddings; persistence to a vector store |
| Semantic retrieval | `app/retrieval/` - ChromaDB HNSW index queried with cosine similarity |
| Grounded generation | `app/generation/generator.py` - numbered passages, a strict grounding prompt, and an explicit refusal path |
| Source citations | Inline `[n]` markers parsed back into structured citations with document, page, snippet and score |
| Backend API | FastAPI (`app/api/`), OpenAPI docs at `/docs` |
| Web chat interface | Responsive vanilla-JS UI served from `/` (`static/`) |

**Bonus**

| Bonus item | Implementation |
| --- | --- |
| BM25 + semantic hybrid | `app/retrieval/bm25_index.py` + reciprocal rank fusion in `app/retrieval/hybrid.py` |
| Reranking | `app/retrieval/reranker.py` - local cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| HNSW | Chroma collection created with explicit `hnsw:M`, `construction_ef`, `search_ef` |
| Semantic caching | `app/cache/semantic_cache.py` - cosine-matched, TTL-bounded, LRU-evicting |
| Rate limiting | `app/middleware/rate_limit.py` - slowapi, per-route limits, structured 429 |
| RAGObserve | `app/observability/tracing.py` - every pipeline stage traced to a local SQLite store |
| Evaluation | `evaluation/` - precision@k, recall@k, nDCG@k, MRR, MAP, faithfulness, refusal probes |
| Docker | `Dockerfile` + `docker-compose.yml`, non-root, healthcheck, model-cache volume |
| Automated testing | `tests/` - 226 tests, hermetic (no downloads, no network), plus 8 real-model tests behind the `slow` marker |
| Logging | `app/logging_config.py` - request-correlated text or JSON logs |
| Deployment | Uvicorn, Compose with an optional observability profile |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   documents ──────▶│ INGESTION                                    │
   (.pdf .docx      │  loaders ─▶ chunker ─▶ embedder ─┐           │
    .md .txt .html) │                                  │           │
                    └──────────────────────────────────┼───────────┘
                                                       │
                            ┌──────────────────────────┴───────────┐
                            ▼                                      ▼
                   ┌─────────────────┐                  ┌──────────────────┐
                   │ ChromaDB (HNSW) │                  │ BM25 index       │
                   │ cosine vectors  │                  │ lexical tokens   │
                   └────────┬────────┘                  └────────┬─────────┘
                            │                                    │
   question ────────────────┼────────────────────────────────────┼──────────
                            ▼                                    ▼
                   ┌─────────────────┐                  ┌──────────────────┐
                   │ dense recall    │                  │ lexical recall   │
                   └────────┬────────┘                  └────────┬─────────┘
                            └──────────────┬─────────────────────┘
                                           ▼
                              ┌────────────────────────┐
                              │ reciprocal rank fusion │
                              └───────────┬────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ cross-encoder rerank   │
                              └───────────┬────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ numbered context + LLM │──▶ answer with [n]
                              └───────────┬────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ citation parser        │──▶ sources + scores
                              └────────────────────────┘

   Every stage reports to RAGObserve (local SQLite) and is timed independently.
```

**Why hybrid?** Dense embeddings handle paraphrase well but miss exact tokens -
identifiers, error codes, product names. BM25 is the opposite. Fusing their
*ranks* (rather than their scores, which live on incomparable scales) is what
makes both usable at once. The cross-encoder then runs only over the fused
top-N, because it costs one forward pass per candidate.

---

## Quick start

### 0. Get the code

The demo video is stored with **Git LFS**, so install it before cloning or you
will get a 130-byte pointer file instead of the MP4.

```bash
git lfs install                       # once per machine
git clone https://github.com/selvaganesh19/Local-LLM-Document-Q-A-Chatbot-RAG-.git
cd Local-LLM-Document-Q-A-Chatbot-RAG-
```

If you already cloned without LFS, fetch the video afterwards:

```bash
git lfs pull
```

The application does not need the video - it is documentation only. A clone
without it works fine.

### 1. Prerequisites

- **Python 3.11+** (developed and tested on 3.13)
- **A reachable LLM server** - either a hosted OpenAI-compatible endpoint (see
  [below](#2-choose-a-generation-backend)) or a local Ollama install
- ~2 GB of disk for the embedding and reranker models

No GPU is required on the machine that runs this service. Embedding and
reranking run on CPU; generation happens wherever your LLM server runs.

### 2. Choose a generation backend

> **This project was developed and verified against a Modal-hosted vLLM
> server, not a local Ollama install.** The development laptop has no dedicated
> GPU, and running a capable instruction-following model on CPU is not
> practical - a 7B model in Q4 needs roughly 5 GB of VRAM to be usable, and CPU
> inference would make a single grounded answer take minutes. The generation
> model therefore runs on a rented GPU via [Modal](https://modal.com), and the
> service reaches it over HTTPS.
>
> Ollama is fully supported and is the better choice if you do have a capable
> GPU locally. It is the second option below.

#### Option A - Modal (what this project uses)

Modal runs a serverless GPU container that scales to zero when idle. You deploy
a vLLM server once, and Modal gives you an HTTPS endpoint speaking the
OpenAI-compatible API.

```bash
pip install modal
modal setup                       # opens a browser to authenticate
```

Create `modal_serve.py`:

```python
"""Serve an instruct model with vLLM on Modal.

Deploy with:  modal deploy modal_serve.py
"""

import modal

MODEL = "Qwen/Qwen3-8B-AWQ"

app = modal.App("qwen-serve")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.25.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=image,
    gpu="A10G",                    # 24 GB - enough for a 7-8B model in 4-bit
    timeout=600,
    scaledown_window=300,          # idle timeout before the container stops
    allow_concurrent_inputs=16,
)
@modal.web_server(port=8000, startup_timeout=900)
def serve() -> None:
    """Start the vLLM OpenAI-compatible server."""
    import subprocess

    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--quantization", "awq",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.90",
        ]
    )
```

```bash
modal deploy modal_serve.py
# Modal prints the endpoint, e.g.
#   https://<workspace>--qwen-serve-serve.modal.run
```

Then point the service at it. **The `/v1` suffix is what selects the
OpenAI-compatible wire protocol** - do not omit it:

```ini
OLLAMA_BASE_URL=https://<your-endpoint>.modal.run/v1
OLLAMA_MODEL=<the model id the server reports>
LLM_API_KEY=
```

Confirm the model id before starting the app:

```bash
curl -s https://<your-endpoint>.modal.run/v1/models
# {"object":"list","data":[{"id":"Qwen/Qwen3-8B-AWQ", ...}]}
```

Two consequences of a scale-to-zero backend are handled automatically and
documented under [Backend selection](#backend-selection): the first request
after an idle period pays a cold start, and Modal opens it with a redirect that
a naive HTTP client mishandles.

Cost note: a scale-to-zero GPU container bills only while it is running. The
container stays up for `scaledown_window` seconds after the last request, so
leaving the app open and asking questions sporadically can keep a GPU warm and
billing. Raise `scaledown_window` if cold starts are annoying, lower it if cost
matters more - the trade-off is cold-start latency against idle billing.

#### Option B - local Ollama

```bash
# https://ollama.com/download
ollama pull qwen3:8b          # or llama3.1:8b, mistral:7b, phi3:mini, ...
ollama serve                  # usually already running as a service
ollama list                   # confirm the tag you pulled
```

```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

Any instruction-following model works. Larger models give better grounding
behaviour; anything from 3B upward is usable. The model is chosen entirely by
configuration - nothing is hard-coded.

### 3. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` and set the two mandatory values - using whichever backend you chose
in step 2. Everything else has a sensible default.

```ini
# Option A - Modal
OLLAMA_BASE_URL=https://<your-endpoint>.modal.run/v1
OLLAMA_MODEL=Qwen/Qwen3-8B-AWQ

# Option B - local Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

The application starts even when these are blank - it serves the UI, reports the
problem on `/api/health`, and returns an actionable `503` from `/api/chat` - but
it cannot answer questions.

### 5. Run

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. On first use the embedding model (~130 MB) and
reranker (~90 MB) download into the Hugging Face cache.

Check the backend was picked up correctly:

```bash
curl -s http://localhost:8000/api/health
# "flavor": "openai" for Modal, "ollama" for a local server
# "status":   "ok"     when the model is reachable and available
```

### 6. Add documents

Either drop files into the left-hand panel of the UI, or index the bundled
sample corpus from the command line:

```bash
python -m scripts.ingest_cli --path data/documents
python -m scripts.ingest_cli --list
```

Then ask something like *"What is the notice period for a Head of Department?"*
and check the cited passages under the answer.

---

## Configuration

All settings are environment variables, read from `.env` or the process
environment. See `.env.example` for the annotated full list.

### Required

| Variable | Example | Notes |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Blank by default, by design |
| `OLLAMA_MODEL` | `qwen3:8b` | Must match a tag from `ollama list` |

These two names are historical - they point at whichever LLM server you run,
Ollama or not. The client speaks both wire protocols and picks one from the URL.

### Backend selection

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_API_FLAVOR` | `auto` | `auto`, `ollama` or `openai`. `auto` treats a URL ending in `/v1` as OpenAI-compatible |
| `LLM_API_KEY` | *(blank)* | Sent as `Authorization: Bearer …` when set |
| `LLM_DISABLE_THINKING` | `true` | Asks a reasoning model to skip its chain of thought (see below) |

| Backend | `OLLAMA_BASE_URL` | Wire protocol |
| --- | --- | --- |
| Ollama (`ollama serve`) | `http://localhost:11434` | `POST /api/chat` |
| Ollama with prefix | `http://localhost:11434/v1` | `POST /v1/chat/completions` |
| vLLM / LM Studio / llama.cpp | `http://localhost:8000/v1` | `POST /v1/chat/completions` |
| Hosted GPU endpoint | `https://<host>/v1` | `POST /v1/chat/completions` |

Two behaviours are worth knowing about on the OpenAI-compatible path:

* **Cold starts.** Backends that scale to zero answer the first request with a
  3xx to a tokenized copy of the same URL. The client replays the request -
  body intact, since a 303 would otherwise be downgraded to a GET - up to three
  times. This is what makes a sleeping Modal or RunPod deployment work.
* **Reasoning models.** Qwen3-style models emit `<think> … </think>` around
  their deliberation and some servers leave it in `content`, where it would
  pollute the answer and confuse the citation parser. The client asks the
  server to disable thinking via `chat_template_kwargs` (dropping the option
  for the rest of the session if the server rejects it with a 400), strips any
  leaked block from non-streamed answers, and withholds reasoning tokens from
  a stream until the closing tag proves the answer has started. Set
  `LLM_DISABLE_THINKING=false` to keep reasoning on.

### Retrieval

| Variable | Default | Notes |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim, CPU-friendly |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Set `RERANK_ENABLED=false` to skip |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K_DENSE` / `TOP_K_SPARSE` | `20` / `20` | Candidates per retrieval leg |
| `TOP_K_FUSED` | `20` | Candidates entering the reranker |
| `TOP_K_FINAL` | `5` | Passages given to the model |
| `RRF_K` | `60` | RRF damping constant |
| `HNSW_M` / `HNSW_CONSTRUCTION_EF` / `HNSW_SEARCH_EF` | `32` / `200` / `100` | Index build/query trade-offs |

### Behaviour

| Variable | Default | Notes |
| --- | --- | --- |
| `CACHE_ENABLED` | `true` | Semantic response cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Deliberately conservative |
| `RATE_LIMIT_CHAT` | `20/minute` | Per client address |
| `RAGOBSERVE_ENABLED` | `true` | Local SQLite tracing |
| `ALLOW_INGEST_ANY_PATH` | `false` | Lift the `data/documents` confinement |
| `DEV_FAKE_LLM` | `false` | Offline mode: hashing embeddings + a fake LLM |

**Switching models:** change `OLLAMA_MODEL`, restart, done. **Switching
embedding models** changes the vector width, so reset the index and re-ingest:

```bash
python -m scripts.ingest_cli --reset --path data/documents
```

---

## Web interface

`static/` is a dependency-free single-page app - no build step, no npm.

- **Chat** with streamed token-by-token answers.
- **Citations** rendered as clickable `[n]` chips that scroll to and expand the
  matching source card.
- **Source cards** for every passage considered, marked as cited or not, with a
  relevance bar, the fusion/rerank score, a snippet, and the full text.
- **Grounding badges** - *grounded in N passages*, *uncited - verify before
  trusting*, or *no supporting passages found*.
- **Diagnostics row** under each answer: total / retrieval / generation latency,
  token counts and throughput, and the RAGObserve trace id.
- **Ingestion panel**: drag-and-drop upload, paste-text, or server-side path.
- **Document list** with per-document chunk counts and deletion.
- Light/dark theme, keyboard shortcuts, responsive down to phone width, and a
  health banner that tells you exactly what to configure when Ollama is missing.

---

## API

Interactive docs: <http://localhost:8000/docs>.

### Chat

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the notice period for senior staff?"}'
```

```json
{
  "question": "What is the notice period for senior staff?",
  "answer": "Senior staff must give 90 calendar days of notice [1]. Notice must be given in writing to the line manager and the People team [1].",
  "citations": [
    {
      "index": 1,
      "chunk_id": "6f1c…",
      "source": "acme_handbook.txt",
      "page": null,
      "label": "acme_handbook.txt",
      "snippet": "Notice periods: employees in grades 1 to 4 give 30 calendar days…",
      "relevance": 0.94,
      "text": "…full passage text…",
      "metadata": { "source": "acme_handbook.txt", "chunk_index": 1 }
    }
  ],
  "retrieved": [ "…every passage considered, with dense/sparse/fusion/rerank scores…" ],
  "grounded": true,
  "uncited": false,
  "insufficient_context": false,
  "cached": false,
  "model": "qwen3:8b",
  "trace_id": "9f2c4a1b8e7d0c31",
  "timings": { "embedding_ms": 8.1, "dense_ms": 3.4, "sparse_ms": 1.2,
               "fusion_ms": 0.3, "rerank_ms": 41.7, "generation_ms": 2140.5,
               "total_ms": 2195.2 },
  "stats": { "prompt_tokens": 1187, "completion_tokens": 42, "tokens_per_second": 27.4 }
}
```

Request fields: `question` (required), `top_k`, `source_filter`, `use_cache`.

### Streaming

```bash
curl -N http://localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "How are security incidents reported?"}'
```

Server-sent events, in this order:

| Frame | Payload |
| --- | --- |
| `timings` | retrieval timings |
| `trace` | `trace_id` |
| `sources` | every passage that will be offered to the model |
| `token` | streamed answer fragments (repeated) |
| `done` | the full response body, including parsed citations |
| `error` | failure detail and status (on the error path only) |

A terminal `data: [DONE]` closes the stream on every path, including errors, so
a client can distinguish a clean end from a dropped socket.

### Ingestion and documents

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/ingest/text` | Index a block of text |
| `POST` | `/api/ingest/upload` | Index an uploaded file |
| `POST` | `/api/ingest/path` | Index a server-side file or directory |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{source}` | Remove one document |
| `POST` | `/api/documents/reset` | Drop the entire index |

### System

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | `ok` / `degraded` / `unconfigured`, plus model availability |
| `GET` | `/api/stats` | Index sizes and cache counters |
| `GET` | `/api/config` | Non-secret settings for the UI |
| `GET` | `/docs` | OpenAPI documentation |

---

## Evaluation

A golden set of question/answer pairs with labelled relevant documents drives
the metrics. The bundled set (`evaluation/golden_set.json`) targets the sample
corpus in `data/documents/` and covers all three sources, plus two **refusal
probes** - questions the corpus deliberately cannot answer.

```bash
# Full run: uses your configured models for generation and judge-based faithfulness
python -m evaluation.run_eval --dataset data/documents

# Retrieval only - no generation, much faster
python -m evaluation.run_eval --dataset data/documents --retrieval-only

# Offline smoke test: hashing embeddings and a fake LLM, no downloads
python -m evaluation.run_eval --fake

# Compare reranking on and off
RERANK_ENABLED=false python -m evaluation.run_eval --dataset data/documents
```

Reports land in `evaluation/reports/` as JSON (full detail) and Markdown (a
scannable summary).

### Metrics

| Metric | Question it answers |
| --- | --- |
| `precision@k` | Of the top *k* passages, how many are relevant? |
| `recall@k` | Of all relevant passages, how many made the top *k*? |
| `nDCG@k` | Are relevant passages ranked *early*, not just present? |
| `MRR` | How high does the first relevant passage rank? |
| `MAP` | Mean precision at each relevant rank |
| `hit_rate@k` | Did *any* relevant passage make the top *k*? |
| `faithfulness` | Is every claim supported by the retrieved passages? |
| `citation_coverage` | What share of retrieved passages did the answer cite? |

Relevance is labelled at the **document** level and resolved to chunk ids
against the live index at evaluation time, so the fixture survives chunking
changes. Faithfulness is scored by the local model acting as a judge with a
strict "supported only if verifiable from the passages" instruction; when no
model is available the run falls back to a lexical coverage proxy and records
which method produced each number (`faithfulness detail` in the report).

Refusal probes are excluded from the ranking metrics - there is no relevant
passage to find - and scored separately as `refusal accuracy`.

### Adding your own golden set

```json
{
  "items": [
    {
      "question": "What is the notice period for senior staff?",
      "relevant_sources": ["acme_handbook.txt"],
      "reference_answer": "90 calendar days, in writing to the line manager."
    },
    {
      "question": "What is the capital of Portugal?",
      "relevant_sources": [],
      "expect_refusal": true
    }
  ]
}
```

Use `relevant_chunk_ids` instead of `relevant_sources` when you need to pin an
exact passage.

---

## Testing

```bash
pip install -r requirements.txt
pytest                       # 226 tests, hermetic, no downloads
pytest -m slow               # the 8 tests that load the real models
pytest --cov=app --cov=evaluation
```

The default suite is **hermetic**: no test downloads a model or contacts an
LLM server. A deterministic hashing embedder, a pass-through reranker and an
offline LLM client are injected through the container, so the real pipeline -
chunking, indexing, fusion, ranking, citation parsing, caching, rate limiting,
HTTP - is exercised end to end.

`tests/test_models_slow.py` is marked `slow` and excluded from the default run
because it downloads ~220 MB of weights. Run it once after a fresh setup to
confirm the embedding and reranking models named in your configuration actually
resolve. Once the weights are cached the tier takes about 34 seconds; the first
run takes as long as the download does, which on a slow link is tens of minutes
with no output in between - see [Troubleshooting](#troubleshooting).

| File | Covers |
| --- | --- |
| `test_chunker.py` | sentence alignment, overlap, determinism, invalid parameters |
| `test_ingestion.py` | loaders, error handling, directory scanning, pipeline state |
| `test_bm25.py` | tokenisation, scoring, idempotent adds, persistence |
| `test_vectorstore.py` | upserts, HNSW search, filters, deletion, metadata coercion |
| `test_hybrid.py` | RRF arithmetic, fusion ordering, tie-breaking, no-op reranker |
| `test_retrieval.py` | end-to-end retrieval, service layer, cache invalidation |
| `test_cache.py` | hits, misses, TTL expiry, LRU eviction, statistics |
| `test_generation.py` | prompts, citation parsing, refusals, streaming, client errors |
| `test_metrics.py` | every metric against hand-computed values |
| `test_tracing.py` | real RAGObserve initialisation and graceful degradation |
| `test_api.py` | every endpoint, error paths, SSE protocol, rate limiting, static UI |
| `test_models_slow.py` | real bge-small embeddings and MiniLM cross-encoder (`-m slow`) |

---

## Docker

Ollama runs on the **host**, not in the container - it wants direct GPU access
and a persistent model store. The container reaches it over
`host.docker.internal`.

```bash
# 1. On the host: let Ollama accept connections from outside localhost
OLLAMA_HOST=0.0.0.0 ollama serve
ollama pull qwen3:8b

# 2. Start the application
OLLAMA_MODEL=qwen3:8b docker compose up --build

# 3. Open http://localhost:8000
```

The image is multi-stage, runs as a non-root user, ships a healthcheck, and
keeps model weights out of the image (they land in the `model-cache` volume so
rebuilds stay cheap).

To run the RAGObserve dashboard alongside it:

```bash
docker compose --profile observability up
# dashboard at http://localhost:5601
```

Set `OLLAMA_BASE_URL` explicitly if Ollama listens somewhere other than the
host. On Linux the compose file already maps `host.docker.internal` via
`extra_hosts`.

---

## Observability

Every query is traced to a local SQLite database via
[RAGObserve](https://pypi.org/project/ragobserve/) - ingestion, chunking,
embedding, both retrieval legs, fusion, reranking, context assembly and
generation, each with timings, scores and the exact prompt.

```bash
ragobserve ui                  # dashboard at http://127.0.0.1:5601
ragobserve export --project local-llm-rag --output traces.ndjson
```

Tracing is best-effort by design: if RAGObserve cannot initialise, the
application logs a warning and carries on. The trace id is returned on every
chat response so a user-visible answer can be matched to its trace.

Logs are request-correlated (`X-Request-ID`, generated if absent and echoed
back) and available as either aligned text or single-line JSON:

```ini
LOG_LEVEL=INFO
LOG_JSON=true      # for log shippers
```

---

## Project structure

```
app/
  main.py                  FastAPI factory: middleware order, routers, static mount
  config.py                pydantic-settings configuration with validation
  container.py             dependency container (lazily built, overridable for tests)
  logging_config.py        text/JSON formatters, request-id context
  api/
    routes_chat.py         /api/chat, /api/chat/stream
    routes_ingest.py       ingestion and document management
    routes_health.py       /api/health, /api/stats, /api/config
    schemas.py             request/response models
  ingestion/
    loaders.py             PDF / DOCX / Markdown / text / HTML
    chunker.py             sentence-aware overlapping windows, content-addressed ids
    embedder.py            sentence-transformers wrapper + offline hashing stub
    pipeline.py            load -> chunk -> embed -> index -> persist
  retrieval/
    vectorstore.py         ChromaDB collection with explicit HNSW settings
    bm25_index.py          BM25Plus lexical index with persistence
    hybrid.py              reciprocal rank fusion
    reranker.py            cross-encoder with a no-op fallback
    retriever.py           orchestrates the four retrieval stages
  generation/
    ollama_client.py       async Ollama client, streaming, token accounting
    prompts.py             grounding prompt and numbered context assembly
    generator.py           citation parsing and grounded answers
  cache/semantic_cache.py  cosine-matched, TTL-bounded, LRU response cache
  middleware/              rate limiting, request context
  observability/tracing.py RAGObserve wrappers that never break a request
  services/rag_service.py  cache policy and the shared response shape
static/                    dependency-free web UI
evaluation/                metrics, LLM judge, golden set, CLI + reports
scripts/ingest_cli.py      command-line ingestion
tests/                     hermetic pytest suite
data/documents/            sample corpus (handbook, security policy, API reference)
```

---

## Design decisions

**Chunk ids are content-addressed** (SHA-1 over source, position and text), so
re-ingesting an unchanged file is an idempotent upsert rather than a source of
duplicates - which is what makes `--reset` rarely necessary.

**BM25Plus rather than classic BM25.** Classic BM25 assigns an IDF of exactly
zero to a term appearing in half a small corpus, which silently drops the most
distinctive query term on a two-document index. BM25Plus uses
`log((N + 1) / df)`, always positive. Its additive `delta` term is set to zero,
otherwise every document scores above zero for every query and nothing is ever
"not a match".

**RRF over score normalisation.** Cosine similarity and BM25 live on
incomparable scales and shift as the corpus grows. Fusing ranks sidesteps both
problems.

**Conservative cache threshold.** The default 0.95 similarity means a semantic
hit is nearly always the same question. Serving a subtly wrong cached answer
costs more than recomputing. Cached answers are also invalidated whenever the
corpus changes, and uncited answers are never cached.

**Citations are parsed, not trusted.** A `[7]` when five passages were supplied
is discarded. An answer that cites nothing is flagged `uncited` and surfaced to
the user rather than presented as verified.

**Generation is skipped when retrieval is empty.** With no passages there is
nothing to ground an answer in, so the service short-circuits to the refusal
sentence instead of letting the model answer from parametric memory.

**Observability cannot fail a request.** Every tracing call is guarded; a broken
or absent RAGObserve degrades to a no-op.

**The web UI has no build step.** A dependency-free ES2020 page keeps the
repository runnable with `pip install` alone and avoids a second toolchain in
the Docker image.

---

## Security notes

- **`POST /api/ingest/path` is confined to `data/documents`** by default, so a
  caller cannot read arbitrary files the service account can reach. Setting
  `ALLOW_INGEST_ANY_PATH=true` lifts this - only do so on a trusted network.
- **`/api/documents/reset` is destructive and irreversible.** It drops the whole
  index; documents must be re-ingested.
- **Uploads are bounded** by `MAX_UPLOAD_MB` and restricted to known extensions.
- **The API is unauthenticated.** It is built for a local, single-user
  deployment. If you expose it, put it behind a reverse proxy with
  authentication, set explicit `CORS_ORIGINS` instead of `*`, and tighten the
  rate limits.
- **Documents are embedded and indexed locally.** No text or vector leaves the
  machine during ingestion or retrieval, but anything indexed is retrievable by
  anyone who can reach the API.
- **Generation may leave the machine, depending on your backend.** With local
  Ollama, nothing does. With Modal or any other remote endpoint, **the retrieved
  passages and your question are sent to that endpoint** - that is inherent to
  asking a remote model to answer from your documents. Choose the backend with
  that in mind; the local option exists and needs no code change.
- **Never commit your `.env`.** It holds your endpoint URL and, for a hosted
  backend, possibly an API key. It is listed in `.gitignore`; keep it there.
  `.env.example` is the shareable template and ships blank.
- **A deployed Modal endpoint is public by default.** Anyone with the URL can
  send it requests and spend your GPU quota. Treat the URL as a credential,
  keep it out of the repository and out of screenshots, and add
  [Modal proxy auth](https://modal.com/docs/guide/webhooks#authentication) or
  rotate the endpoint if it leaks. `LLM_API_KEY` is supported for endpoints that
  require a bearer token.
- **The evaluation judge is local.** The RAGObserve cloud faithfulness scorer is
  deliberately not used, because it would ship retrieved passages to a third
  party; a local model grades grounding instead.

---

## Troubleshooting

**`/api/chat` returns 503** - `OLLAMA_BASE_URL` or `OLLAMA_MODEL` is blank.
`GET /api/health` lists exactly what is missing.

**`Answer: I don't know based on the provided documents.`** - retrieval found
nothing relevant. Check that documents are indexed (`/api/documents`) and that
`GET /api/stats` reports a non-zero vector count.

**`Model 'x' is not installed`** - the tag does not match `ollama list`. Pull it
or fix the casing in `.env`.

**Docker cannot reach Ollama** - Ollama must not bind to `127.0.0.1` only. Run
`OLLAMA_HOST=0.0.0.0 ollama serve`, and use
`OLLAMA_BASE_URL=http://host.docker.internal:11434`.

**Retrieval quality is poor** - lower `CACHE_SIMILARITY_THRESHOLD` is *not* the
fix. Check `chunk_overlap` is not too small, raise `TOP_K_FUSED` so more
candidates reach the reranker, or switch to a stronger embedding model
(`BAAI/bge-base-en-v1.5`) and re-ingest.

**Out of memory while embedding** - set `EMBEDDING_DEVICE=cpu` (the default) and
reduce the batch size in `app/ingestion/embedder.py`.

**First request is very slow** - the embedding and reranker models download on
first use (~220 MB total). Subsequent runs load from cache. The reranker alone
accounts for roughly 14 seconds of that on its first use, because it loads
lazily on the request that needs it. Warm queries return in about a second.

**`pytest -m slow` appears to hang** - it is almost certainly downloading. The
tier pulls ~220 MB and prints nothing until the first test runs, so on a slow
link it is silent for tens of minutes. This is not a deadlock. Confirm progress
by watching the cache:

```bash
ls -lh ~/.cache/huggingface/hub/*/blobs/*.incomplete
```

Once the weights are cached the whole tier takes about 34 seconds.

---

## Known limitations

- The semantic cache is in-process and per-worker. Running multiple Uvicorn
  workers gives each its own cache; it is a latency optimisation, not a shared
  store.
- Rate limiting is also per-process. Behind multiple workers, use a shared
  storage backend (`slowapi` supports Redis via `storage_uri`).
- The reranker is a general-domain MS MARCO model. Domain-specific corpora would
  benefit from a fine-tuned cross-encoder.
- Faithfulness evaluation depends on the local model's judgement. A small model
  is a noisy judge - treat the score as a signal, not a verdict, and prefer
  `--no-judge` plus manual review for high-stakes evaluation.
- PDF extraction relies on `pypdf`, which does not recover text from scanned
  pages. Add OCR upstream if that matters.
- The Ollama-native wire protocol is covered by tests but has not been exercised
  against a live Ollama server; the live verification was done on the
  OpenAI-compatible path. See
  [Other observations](Documentation/03-other-observations.md#3-honest-statement-of-what-has-and-has-not-been-verified)
  for the full list of what has and has not been verified.

---

## Documentation

Supporting notes on how this was built are in [`Documentation/`](Documentation/):

| Document | Contents |
| --- | --- |
| [Questions, assumptions and difficulties](Documentation/01-questions-assumptions-difficulties.md) | Open questions, the assumptions made in their absence, and every defect hit along the way |
| [Development time](Documentation/02-development-time.md) | Time taken per task, and the basis for each figure |
| [Other observations](Documentation/03-other-observations.md) | Observations, suggested improvements, and an explicit statement of what has and has not been verified |
