# Development Time

## How these figures were produced

**Read this before quoting any number below.** The project was developed in a
single assisted session with no per-task timer, and the directory was never
placed under version control, so there are no commit timestamps to measure
against. Every figure is a **reconstructed estimate**, derived from the
artefacts the work left behind - files written, tests added, commands run, and
defects found and fixed. They are accurate to perhaps ±30%, which is enough to
show where the effort went and not enough to justify a claim of precision.

Two distinct quantities are reported, because they differ by several times:

- **Active time** - work actually in progress.
- **Wall-clock time** - elapsed from start to finish, including waiting on
  things that could not be parallelised: a dependency install, a model
  download throttled to about 130 KB/s, and generation round trips against a
  shared endpoint.

Wall clock is roughly **2.5x** active time. The gap is almost entirely two
periods of blocked waiting: the `sentence-transformers` and PyTorch install,
and the 220 MB model download for the `slow` test tier, which alone accounted
for about seventeen minutes of doing nothing.

---

## Per-task breakdown

| # | Task | Active | Key deliverables |
| --- | --- | --- | --- |
| 1 | Scaffold: configuration and dependencies | ~0.75 h | `app/config.py`, `requirements.txt`, `.env.example`, `.gitignore`, package layout |
| 2 | Ingestion pipeline | ~1.5 h | Loaders (PDF/DOCX/MD/text/HTML), sentence-aware chunker, embedder abstraction, ChromaDB store, BM25 index, orchestrating pipeline |
| 3 | Hybrid retrieval and reranking | ~2.5 h | Dense + sparse retrieval, reciprocal rank fusion, cross-encoder reranker, HNSW parameterisation |
| 4 | LLM client and grounded generation | ~1.5 h | Async Ollama client with streaming and token accounting, prompt construction, citation parsing, refusal detection |
| 5 | FastAPI application, cache, observability, middleware | ~3.0 h | Routes, schemas, dependency container, semantic cache, rate limiting, RAGObserve tracing, SSE streaming |
| 6 | Evaluation harness | ~2.0 h | precision@k, recall@k, nDCG@k, MRR, MAP, hit rate, citation coverage, local faithfulness judge, CLI and report writers |
| 7 | Test suite | ~2.5 h | 226 hermetic tests across chunking, retrieval, generation, API, cache, rate limiting, tracing and the OpenAI flavor |
| 8 | Docker, Compose, README, and the supporting CLI scripts | ~2.0 h | Multi-stage Dockerfile, `docker-compose.yml`, ingestion CLI, full README |
| 9 | Backend portability and reasoning-model handling *(unplanned)* | ~2.0 h | OpenAI-compatible protocol support, cold-start redirect replay, reasoning-trace suppression, deprecation fixes |
| 10 | Real-model verification | ~1.0 h | `tests/test_models_slow.py`, live end-to-end run against the configured backend, defect fixes found by it |
| | **Total active** | **~18.75 h** | |
| | **Approximate wall clock** | **~45 h** | Spread across several working sessions |

---

## Tasks that overran, and why

Three tasks took materially longer than the estimate, all for the same reason:
a defect that only surfaced when the code ran.

**Task 3 - Hybrid retrieval (+~1 h over estimate).** Two separate BM25 defects
were found here, both invisible on inspection and both caught by tests. The
first was a zero-IDF term on a small corpus, which silently returned no results
at all; the second was an additive score floor that made every document match
every query. Neither is the kind of bug that shows up as an exception - both
degrade retrieval quality quietly. Most of the overrun is the diagnosis, not
the two-line fix. See
[the difficulties document](01-questions-assumptions-difficulties.md#31-correctness-bugs).

**Task 5 - FastAPI application (+~1 h over estimate).** The rate limiter read
its limits from a cached process singleton rather than the settings the
application was built with, so every per-route limit was silently the default.
Finding it meant reading the `slowapi` internals to understand when the
decorator evaluates its limit callables.

**Task 9 - Backend portability (entirely unplanned).** This work did not exist
in any estimate because it was not in the brief. The client had been written
against Ollama's native API, as the brief implies; the backend actually
supplied was a vLLM server speaking the OpenAI protocol. Supporting both, plus
the two problems that only appear on that path - a cold-start `303` redirect
that a naive HTTP client mishandles, and a reasoning model leaking its chain of
thought into the answer - took about two hours on its own. It is the clearest
example in this project of an estimate being wrong because a requirement was
discovered rather than given.

---

## Tasks that came in under

**Task 7 - Tests (under estimate).** 226 tests in around two and a half hours
was faster than expected, for one structural reason: the whole suite runs
against a deterministic hashing embedder, a pass-through reranker and an
offline LLM client, all injected through a dependency container. That was a
deliberate decision made in task 1, and it paid for itself here - no test
downloads a model or opens a socket, so the suite runs green in about 36
seconds and can be run after every change.

**Task 8 - Documentation (roughly on estimate).** The README was written
alongside the code rather than afterwards, which is why it stayed close to
budget despite being long.

---

## Where the time actually went

Roughly:

- **45%** - implementation of the RAG pipeline itself (tasks 2 to 4)
- **25%** - the service layer: API, caching, rate limiting, tracing, streaming
  (task 5)
- **15%** - tests (task 7)
- **10%** - portability work that was not in the plan (task 9)
- **5%** - Docker, README and scripts (task 8)

Diagnosis dominated implementation throughout. Of the eight defects fixed, seven
were found by running the code rather than by reading it, and the fixes
themselves were small relative to the time spent locating them.
