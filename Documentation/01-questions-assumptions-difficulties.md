# Questions, Assumptions and Difficulties

This document records what was unclear before starting, what was assumed in
place of an answer, and what went wrong along the way. The difficulties section
lists only problems that actually occurred and were resolved - each one is
reproducible from the test suite or from the configuration described.

---

## 1. Questions

### 1.1 Questions asked and answered

| Question | Answer | Consequence |
| --- | --- | --- |
| Which LLM server and model should the generation step target? | Leave `OLLAMA_BASE_URL` and `OLLAMA_MODEL` blank; the values would be supplied in a personal `.env` | Both settings ship empty in the repository, and the service reports `unconfigured` rather than failing obscurely |
| Which embedding and reranking models? | Sentence Transformers, but "not high size model" | `BAAI/bge-small-en-v1.5` (384-dim) and `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Which vector database - ChromaDB or FAISS? | ChromaDB | ChromaDB's `PersistentClient`, with HNSW tuned explicitly rather than left on defaults |
| What form should the web interface take? | Static HTML/CSS/JS served by FastAPI | No build step, no npm, no bundler - the UI is three files under `static/` |
| How much of the bonus list should be attempted? | All of it | Hybrid retrieval, reranking, citations, HNSW tuning, semantic caching, rate limiting, RAGObserve, Docker, tests, evaluation metrics |

### 1.2 Questions that remained open

These were never resolved, and the assumptions in section 2 stand in for them.

- **Is the generation model expected to be genuinely local?** The brief says
  "host an open-source LLM locally via Ollama". The environment eventually
  configured was a Modal-hosted vLLM deployment reached over HTTPS - not local,
  and not Ollama. The client was written to support both rather than to pick
  one, but the question of which the assessment expects was never settled.
- **Is authentication in scope?** The API has no authentication. Nothing in the
  brief asks for it, and the deployment target is a single-user local machine,
  so it was treated as out of scope rather than overlooked.
- **What corpus is this meant to run against?** No documents were supplied. The
  system was built and tested against a small synthetic corpus (an employee
  handbook, a security policy, an API reference) sized for fast tests, and
  against arbitrary user-supplied files at runtime.
- **Are the evaluation metrics expected to be reported against a real corpus
  with a curated golden set?** The harness ships with a small golden set and
  runs end to end in `--fake` mode, but no production corpus was available to
  produce meaningful absolute numbers.

---

## 2. Assumptions

Each assumption is recorded with the reason it was made and what would break if
it were wrong.

### 2.1 Deployment and operating assumptions

| Assumption | Reason | If wrong |
| --- | --- | --- |
| Single user, single process, no authentication | The brief describes a local tool; nothing implies multi-tenancy | The rate limiter is per client address and would need a shared backing store; there is no per-user isolation of the index |
| The service account can read the documents it is asked to ingest | Ingestion is the product | `ALLOW_INGEST_ANY_PATH` defaults to `false`, confining `POST /api/ingest/path` to `data/documents`, so a misconfigured deployment fails closed rather than reading arbitrary host paths |
| Documents fit comfortably in memory | Chunking and embedding are batched, not streamed | A multi-gigabyte PDF would be loaded whole; the loader layer would need streaming |
| The vector index fits on local disk | ChromaDB persists to `./data/vectorstore` | No sharding or remote store is implemented |
| The LLM server is reachable and reasonably fast | Generation is synchronous on the request path | A cold or overloaded backend makes a query take tens of seconds; there is no queue, no background job, and no client-side deadline beyond `OLLAMA_TIMEOUT_SECONDS` |
| Answers must be grounded or refused | The brief asks for grounded generation with citations | The system prompt instructs an exact refusal sentence, and the generator flags uncited answers rather than hiding them - so an ungrounded model degrades to visible refusals, not to confident fabrication |

### 2.2 Modelling and retrieval assumptions

| Assumption | Reason | If wrong |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` is a good quality-to-size trade-off | Explicit user constraint: "not high size model" | Recall on domain-specific vocabulary would improve with a larger model; changing `EMBEDDING_MODEL` also changes the vector width, which requires resetting and re-ingesting the index |
| Cosine similarity is the right distance | Embeddings are L2-normalised on the way out of the embedder | Metadata and index would need rebuilding with a different `hnsw:space` |
| A cross-encoder is worth its latency | It runs once per candidate passage, not once per document | Reranking dominates first-query latency; `RERANK_ENABLED=false` disables it, and `TOP_K_FUSED` bounds the cost |
| Reciprocal Rank Fusion with k=60 is the right fusion | It is the standard formulation and needs no score normalisation between two scales that are not comparable | A learned or weighted fusion would likely do better on a tuned corpus; `RRF_K` is configurable |
| A semantic cache threshold of 0.95 is safe | Below that, two different questions start colliding | Too low a threshold silently serves a wrong answer. It is deliberately conservative, and uncited answers are never cached |
| English-language documents | The embedding and reranking models are English-only | Other languages would need different models; the loaders and chunker are language-agnostic |

### 2.3 Interface assumptions

| Assumption | Reason | If wrong |
| --- | --- | --- |
| `[n]` inline markers are an acceptable citation format | They are easy for the model to emit reliably and easy to parse back | A model that cites in another style produces `uncited: true`; the structured sources are still returned either way |
| Server-sent events are preferable to WebSockets | The traffic is one-directional and the client is a browser | SSE is blocked by some corporate proxies; the non-streaming `POST /api/chat` endpoint exists as a fallback |
| A `data: [DONE]` sentinel is required | Without it a client cannot distinguish "finished" from "connection dropped" | It is emitted last on every successful stream |

---

## 3. Difficulties encountered

Every item below is a real defect or obstacle from the build. The ones marked
**bug** were found by the test suite or by probing the running system, not by
inspection.

### 3.1 Correctness bugs

**BM25 returned no matches on small corpora — bug.**
The lexical leg used `BM25Okapi`, whose IDF is
`log((N - n + 0.5) / (n + 0.5))`. For a term appearing in one of two
documents that evaluates to exactly `0.0`, so the single most distinctive query
term contributed nothing and a two-document corpus returned no hits at all.
Switched to `BM25Plus`, whose IDF is `log((N + 1) / df)` and is always positive.
Two tests caught this.

**BM25 matched documents that shared no vocabulary with the query — bug.**
`BM25Plus` adds a constant `delta` to every term frequency. At the default
`delta=1.0`, every document accumulated a positive score for every query term,
so an unrelated handbook chunk scored 2.71 for a security question and the
"no match returns nothing" contract was unenforceable. Fixed by passing
`delta=0.0`, with a comment explaining why. The residual test failure turned out
to be the test's own fault - the word "days" genuinely occurred in both
documents - so the probe query was changed to vocabulary unique to one file.

**The chunker rejected any `chunk_size` below the default overlap — bug.**
`chunk_overlap` defaulted to a fixed `150`. Any caller passing a smaller
`chunk_size` - which the test suite does, to keep fixtures small - hit
`"chunk_overlap must be smaller than chunk_size"`. The parameter is now
`Optional[int] = None` and resolves to `chunk_size // 7`, so the invariant holds
by construction while an explicitly bad value is still rejected. Three tests
caught this.

**Per-route rate limits were unconfigurable at runtime — bug.**
A probe showed `x-ratelimit-limit: 20` against a test configured for
`2/minute`. The dynamic limit callables were reading the `@lru_cache`d
`get_settings()` singleton rather than the `Settings` the application had
actually been constructed with, so every test and every embedded use silently
got the defaults. Fixed with a module-level `_active_settings` set by
`configure_limiter()`, falling back to the singleton when unset. This one was
worth the effort of reading the `slowapi` internals: the decorator reads
`Limiter.enabled` at call time, and `LimitGroup.__iter__` invokes a zero-argument
callable when its signature does not mention `key`.

**The Dockerfile could not build — bug.**
`.dockerignore` excludes `tests/` and `pytest.ini`, but the Dockerfile still
contained `COPY tests ./tests` and `COPY pytest.ini ./`. The build would have
failed on a missing source path. Both lines removed, with a comment recording
that a test image needs `--target builder`.

### 3.2 Integration and environment obstacles

**Ollama's native API was the wrong protocol for the configured backend.**
The client was written against `POST /api/chat`, as the brief implies. The
endpoint actually supplied was a vLLM server exposing `POST /v1/chat/completions`.
Rather than rewrite the client, both protocols were implemented behind one
interface with the flavor inferred from the URL. This is the single largest
piece of unplanned work in the project.

**A scale-to-zero backend answers the first request with a redirect.**
Modal returns `303 See Other` to a tokenized copy of the same URL on a cold
start. `httpx` does not follow redirects by default, so the client reported the
backend as unreachable. Enabling `follow_redirects` would have been wrong: a 303
downgrades the method to `GET` and drops the request body, which no chat
endpoint accepts. The client now replays the request - same method, same body -
up to three times. This is a genuine trap, because the failure looks like a
network problem rather than a protocol one.

**The model's chain of thought leaked into its answers.**
`qwen3.8-27b` emitted `<think> ... </think>` ahead of the real answer, and the
server left it in `message.content`. Two problems followed: the user sees
deliberation instead of an answer, and the citation parser sees `[1]` mentioned
inside the reasoning before the actual claim. Three mitigations were added - ask
the server to disable thinking via `chat_template_kwargs`, strip a leaked block
from non-streamed answers, and withhold reasoning tokens from a stream until the
closing tag proves the answer has started. The third is fiddly: token-by-token
filtering cannot wait forever for a tag that may never arrive, so the buffer is
released once it exceeds 4000 characters.

**Strict OpenAI-compatible servers reject unknown body fields.**
`chat_template_kwargs` is a vLLM extension. A server that does not understand it
answers `400`. The client now detects that specific rejection, drops the field
for the rest of the session, and retries - so the option costs one extra round
trip on a strict server and nothing thereafter.

**The Hugging Face download was throttled to roughly 130 KB/s.**
The two models total about 220 MB. At the observed rate that is roughly half an
hour, and the symptom is indistinguishable from a deadlock: `pytest -m slow`
produces no output and no error for seventeen minutes. Both `huggingface.co` and
`hf-mirror.com` were throttled identically, and enabling parallel transfer made
no difference, which ruled out a per-connection limit and pointed at the link
itself. Nothing was wrong with the code.

**`sentence-transformers` 5.7 renamed a method that the code called.**
`get_sentence_embedding_dimension()` now warns and delegates to
`get_embedding_dimension()`. The embedder prefers the new name when present and
falls back to the old one, so it works on both major versions.

### 3.3 Ambiguities that had to be resolved by judgement

**Where does the SSE stream end?**
A test asserted the first frame was `sources`; the implementation sent
`timings` first. The implementation was right - retrieval timings are useful the
moment they exist, and withholding them until after the sources buys nothing -
so the route docstring was extended to state the exact frame order and the test
was corrected. The contract is now explicit: `timings`, `trace`, `sources`,
`token` repeated, `done`, then the `[DONE]` sentinel.

**Should the RAGObserve cloud faithfulness scorer be used?**
No. It requires a `GROQ_API_KEY`, which would send retrieved passages and
generated answers to a third party in a project whose premise is that nothing
leaves the machine. A local Ollama judge was written instead, falling back to a
lexical grounding score when no model is reachable. The metric is weaker; the
privacy property is preserved.
