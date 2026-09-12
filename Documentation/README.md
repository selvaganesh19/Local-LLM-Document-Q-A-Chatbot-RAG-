# Documentation

Supplementary notes for the Local LLM Document Q&A (RAG) submission. The
[main README](../README.md) covers what the system is and how to run it; these
documents cover how it was built and what was learned doing so.

| Document | Contents |
| --- | --- |
| [Questions, assumptions and difficulties](01-questions-assumptions-difficulties.md) | Open questions, the assumptions made in their absence, and the problems hit |
| [Development time](02-development-time.md) | Time taken per task, with the basis for each figure |
| [Other observations](03-other-observations.md) | Observations, suggested improvements, and what would be done differently |

---

## A note on the numbers

The development-time figures in
[02-development-time.md](02-development-time.md) are reconstructed, not
instrumented. The project was built in a single assisted session with no
per-task timer and the directory was never under version control, so there are
no commit timestamps to measure against. Each figure is an estimate derived from
the artefacts left behind - files written, tests added, commands run - and is
labelled as such. Wall-clock elapsed time is reported separately, because it is
several times larger than productive time: a meaningful fraction of the session
went to a model download throttled to roughly 130 KB/s (see the observations
document), during which no implementation work could proceed.
