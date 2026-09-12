# Other Observations

Observations from building and running this system, suggestions for improving
it, and the honest limits of what has been verified.

---

## 1. Observations

### 1.1 What the measurements actually showed

A live run against the configured backend, with the real embedding model, the
real cross-encoder and a two-document corpus, produced this latency profile:

| Stage | Warm query | Notes |
| --- | --- | --- |
| Query embedding | 18-26 ms | Single short query; dominated by Python overhead, not the model |
| Dense search (ChromaDB HNSW) | 3-17 ms | Two vectors; not a meaningful benchmark |
| Sparse search (BM25) | 0.1-0.3 ms | Pure Python, in-memory |
| Fusion (RRF) | 0.02-0.08 ms | Negligible at any realistic candidate count |
| Cross-encoder rerank | 24 ms warm, **13.8 s cold** | One forward pass per candidate |
| Generation | 0.8-2.3 s | Remote endpoint, 5-19 tokens/second |

Two things stand out. First, **the reranker's cold start dominated the first
query by three orders of magnitude** - 13.8 seconds to load the model, 24
milliseconds to use it. Second, on a corpus this small every retrieval stage
except reranking is effectively free; the retrieval pipeline is not the
bottleneck and measuring it on two documents tells you almost nothing. A
realistic evaluation needs a corpus in the thousands of chunks before the
dense-versus-sparse trade-off is even visible.

**Generation is the bottleneck in practice.** At 5-19 tokens per second, a
grounded answer of 60 tokens takes several seconds, and the rate limiter
(20 requests per minute) is far from the binding constraint. Retrieval
optimisation would be premature here.

### 1.2 The retrieval stack was harder to get right than the generation stack

The asymmetry was unexpected. Grounded generation - prompting, streaming,
citation parsing, refusal detection - worked close to first time. Hybrid
retrieval produced two silent correctness bugs on the same afternoon, both of
which would have degraded answer quality without ever raising an error.

The lesson is that **the failure modes of retrieval are quiet and the failure
modes of generation are loud**. A broken generator produces garbage or an
exception. A broken ranker produces a plausible answer from a slightly wrong
passage, and nothing anywhere says so. This is the strongest argument in the
project for the golden-set evaluation harness existing at all.

### 1.3 Small corpora expose bugs that large ones hide

Every retrieval bug found here was caught because the test corpus had two or
three documents in it. `BM25Okapi`'s IDF is exactly zero for a term in one of
two documents - a condition that becomes vanishingly rare as the corpus grows,
and that a large test fixture would have hidden indefinitely.

There is a real tension here. Small fixtures make tests fast and hermetic, and
they also happen to sit in the range where ranking arithmetic degenerates. Both
were needed: small corpora for the unit tests, and at least one
realistically-sized corpus for the end-to-end evaluation.

### 1.4 Grounding is only as good as the refusal path

The refusal path turned out to be the feature that makes the rest of the system
trustworthy. Asked about the capital of Portugal against a corpus of two policy
documents, the system answered:

> I don't know based on the provided documents.

and returned no citations. That single behaviour does more for the credibility
of the citations than any amount of prompt engineering, because it demonstrates
the model is not simply producing fluent text adjacent to the retrieved
passages.

The related observation is that `uncited` and `ungrounded` are **reported, not
hidden**. An answer with no citation markers is returned with
`uncited: true` rather than being silently repaired or suppressed. Surfacing the
weakness is more useful than concealing it.

### 1.5 The hardest problems were environmental, not algorithmic

Of the eight defects fixed, the retrieval ones took the most thought, but the
most *frustrating* were environmental: a wire protocol that did not match the
brief, a cold-start redirect that breaks naive HTTP clients, a model that leaks
its reasoning into its output, and a 130 KB/s download that looks exactly like a
deadlock for seventeen minutes.

None of these are RAG problems. All of them consumed real time. A project plan
that budgets for the algorithm and not for the environment will underestimate
badly.

### 1.6 The privacy property has a caveat

The README's premise is that nothing leaves the machine, and for a standard
Ollama deployment that holds. It is worth being explicit that the
OpenAI-compatible path breaks it: pointing `OLLAMA_BASE_URL` at a remote
endpoint sends both the user's question and the retrieved passages to that
endpoint. This is opt-in and never a default, but the README's opening claim was
qualified once this became clear, because a privacy claim that is true of the
default configuration and false of a supported one should say so.

The same reasoning is why the RAGObserve cloud faithfulness scorer - which
needs a `GROQ_API_KEY` and would ship retrieved passages to a third party - was
deliberately not used, in favour of a local judge.

---

## 2. Improvements and suggestions

Ordered by the ratio of value to effort.

### 2.1 Worth doing next

**Evaluate against a real corpus, with real numbers.** The single biggest gap.
The harness computes precision@k, recall@k, nDCG@k, MRR, MAP, hit rate, citation
coverage and faithfulness, and it runs end to end - but on a three-document
golden set. Running it against a few hundred documents with a curated set of
perhaps fifty question/answer pairs would turn the metric code into actual
evidence, and would show whether the cross-encoder earns its latency.

**Measure what reranking contributes.** `RERANK_ENABLED` is a switch, but no
measurement exists of what flipping it does to nDCG@5. That ablation is cheap -
the harness already computes the metric - and it would settle whether the
13.8-second cold start is worth paying.

**Batch the embedding of queries.** Query embedding is currently one call for
one query. At 18-26 ms it does not matter yet, but it is the obvious first thing
to profile under load.

**Warm the models at startup.** The container loads the embedder and the
cross-encoder lazily on first use, which keeps startup fast but pushes a
one-time cost of roughly fourteen seconds onto whichever user asks the first
question. A background warm-up task in the `lifespan` handler would move that
cost to boot, where it is invisible.

### 2.2 Worth considering

**Persist the semantic cache.** It is in-memory, so every restart loses it. A
SQLite-backed cache would survive restarts and make the first-query penalty less
frequent. It would also need the same invalidation-on-corpus-change logic that
the in-memory version already has.

**Stream the reranker's progress.** Reranking is the longest stage and the SSE
stream reports it only once, after it finishes. Since the `sources` frame is
already sent before generation begins, the UI could show retrieved-but-unranked
passages immediately and refine them.

**Replace the hand-rolled faithfulness judge with a stronger local model when
one is available.** The current judge works and falls back gracefully, but a
small local model grading grounding is a weak instrument. It is a reasonable
place to spend a larger model on.

**Add a `/api/chat` request-level deadline.** `OLLAMA_TIMEOUT_SECONDS` bounds
the whole request, but a slow backend will simply hang until it expires. An
overall budget with a partial-answer return would degrade more gracefully.

### 2.3 Deliberately not done

These were considered and rejected, and the reasoning is recorded so the
omission reads as a decision rather than an oversight.

**Authentication.** Out of scope for a single-user local tool, and adding a
half-considered auth layer is worse than having none. If this were ever exposed
beyond localhost it would need real authentication, per-user document isolation,
and a shared rate-limit store - a substantially different system.

**Replacing BM25 with a learned sparse retriever** (SPLADE or similar). Would
likely improve recall, at the cost of a model download, a heavier index, and
loss of the current property that the sparse index is explainable. The existing
hybrid already covers the failure mode that BM25 alone would have - exact term
matching on rare vocabulary - which is the reason it is there.

**Query rewriting or multi-query expansion.** Real gains are available, but each
added LLM call multiplies latency against a backend already running at 5-19
tokens per second. On a faster local model this would move up the list.

**Microservices.** The system is one process for a reason. Splitting ingestion,
retrieval and generation would add network hops and deployment complexity to
solve a scaling problem this workload does not have.

---

## 3. Honest statement of what has and has not been verified

**Verified, by running it:**

- 226 hermetic tests pass in about 36 seconds; the 8 real-model tests pass in
  about 34 seconds once the weights are cached.
- The real embedding model produces 384-dimensional, L2-normalised vectors, and
  scores a semantically relevant passage above an unrelated one.
- The real cross-encoder ranks the relevant passage first and truncates to
  `top_n` correctly.
- Ingestion, hybrid retrieval, reranking, grounded generation and citation
  parsing work end to end against a live LLM endpoint, including a correct
  refusal on an out-of-corpus question.
- The HTTP layer works: `/api/health` reports `ok` with the model available,
  `/api/config` and `/api/stats` respond, the UI is served at `/`, and
  `POST /api/chat/stream` emits the documented frame order and terminates with
  `data: [DONE]`.

**Not verified, and not claimed:**

- **No measurement on a realistic corpus.** Every latency and quality figure
  comes from a corpus of two or three short documents. Nothing here supports a
  claim about behaviour at scale.
- **No meaningful absolute evaluation metrics.** The harness runs, but a golden
  set of three documents is a smoke test, not an evaluation.
- **The Ollama-native path has never run against a real Ollama server.** It is
  covered by tests and by the frame-parsing contract, and the protocol is
  implemented as documented, but the live verification was done on the
  OpenAI-compatible path because that is what was available.
- **Docker images have not been built or run.** The Dockerfile and Compose file
  are written and the build-breaking `COPY` mismatch was fixed, but no image was
  built during this work.
- **PDF, DOCX and HTML loaders have not been exercised on real files of those
  formats** - only on synthetic input in tests.
