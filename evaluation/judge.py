"""LLM-as-judge evaluation helpers.

Faithfulness asks a deceptively simple question: *is every claim in the answer
supported by the retrieved passages?*  There is no cheap lexical test for that -
an answer can reuse the document's vocabulary while asserting the opposite - so
the primary scorer asks the local model to adjudicate, and a lexical proxy
stands in when no model is available.

The judge runs on the same local model as generation, so evaluation stays
offline and reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.generation.ollama_client import ChatMessage, OllamaClient, OllamaError
from app.observability import tracing
from evaluation.metrics import lexical_grounding_score

logger = logging.getLogger(__name__)

#: Instruction for the faithfulness judge.  Kept terse and JSON-only so the
#: response is machine-parseable without a second pass.
FAITHFULNESS_SYSTEM_PROMPT = """You are a strict evaluation judge for retrieval-augmented generation.

You will be given numbered context passages and a candidate answer.

A claim is SUPPORTED only if it can be verified from the passages alone.
Outside knowledge, plausible inference and correct-but-absent facts do NOT count.

Respond with a single JSON object and nothing else:
{"faithfulness": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}

Where 1.0 means every claim is supported by the passages, and 0.0 means the
answer contradicts or invents everything."""

#: Extracts the first JSON object from a model response.
_JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)

#: Fallback when the model returns prose containing a bare number.
_NUMBER_PATTERN = re.compile(r"(?<![\d.])(\d(?:\.\d+)?)(?![\d.])")


@dataclass
class FaithfulnessVerdict:
    """Outcome of a faithfulness evaluation.

    Attributes:
        score: Faithfulness in ``[0, 1]``.
        reason: Short justification.
        method: ``"llm"`` or ``"lexical"``, so a report never conflates the two.
        judged: ``True`` when the LLM judge produced the score.
    """

    score: float
    reason: str = ""
    method: str = "lexical"
    judged: bool = False

    def to_dict(self) -> dict:
        """Serialise for the JSON report."""
        return {
            "score": round(self.score, 4),
            "reason": self.reason,
            "method": self.method,
            "judged": self.judged,
        }


def _clamp(value: float) -> float:
    """Constrain a score to ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def _extract_score(text: str) -> Optional[float]:
    """Pull a ``0..1`` score out of a judge response.

    Accepts either a JSON object with a ``faithfulness`` key or a bare number,
    normalising 0-10 and 0-100 scales down to 0-1.

    Args:
        text: The raw model response.

    Returns:
        The parsed score, or ``None`` when nothing usable was found.
    """
    match = _JSON_OBJECT_PATTERN.search(text)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("faithfulness", "score", "faithfulness_score"):
                raw = payload.get(key)
                if isinstance(raw, (int, float)):
                    value = float(raw)
                    return _clamp(value / 100.0 if value > 10 else (value / 10.0 if value > 1 else value))

    numbers = _NUMBER_PATTERN.findall(text)
    if numbers:
        value = float(numbers[0])
        return _clamp(value / 100.0 if value > 10 else (value / 10.0 if value > 1 else value))
    return None


def build_judge_prompt(question: str, contexts: Sequence[str], answer: str) -> str:
    """Assemble the judge's user turn.

    Args:
        question: The question that was asked.
        contexts: Passages that were supplied to the answering model.
        answer: The answer under evaluation.

    Returns:
        The formatted prompt.
    """
    passages = "\n\n".join(
        f"[{index}] {text}" for index, text in enumerate(contexts, start=1)
    )
    return (
        f"Context passages:\n\n{passages or '(none)'}\n\n"
        f"---\nQuestion: {question}\n\n"
        f"Candidate answer: {answer}\n\n"
        "Return only the JSON object."
    )


class FaithfulnessJudge:
    """Scores how well an answer is supported by its retrieved context."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        enabled: bool = True,
    ) -> None:
        """Create a judge.

        Args:
            client: LLM client used for judging.  When ``None``, or when
                ``enabled`` is ``False``, the lexical proxy is used instead.
            enabled: Allow the LLM judge to run.
        """
        self._client = client
        self._enabled = enabled and client is not None

    @property
    def available(self) -> bool:
        """Whether the LLM judge can be used."""
        return self._enabled

    async def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> FaithfulnessVerdict:
        """Score the faithfulness of ``answer``.

        Falls back to :func:`evaluation.metrics.lexical_grounding_score` when
        the judge is unavailable or the model errors, and records which method
        produced the number.

        Args:
            question: The question that was asked.
            answer: The candidate answer.
            contexts: Passages supplied to the answering model.

        Returns:
            A :class:`FaithfulnessVerdict`.
        """
        if not answer.strip():
            return FaithfulnessVerdict(score=0.0, reason="empty answer", method="lexical")

        if not self._enabled:
            return self._lexical(answer, contexts, "judge disabled")

        prompt = build_judge_prompt(question, contexts, answer)
        messages = [
            ChatMessage(role="system", content=FAITHFULNESS_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        with tracing.Stopwatch() as timer:
            try:
                response = await self._client.chat(messages, temperature=0.0)  # type: ignore[union-attr]
            except OllamaError as exc:
                logger.warning("Faithfulness judge failed", extra={"error": str(exc)})
                return self._lexical(answer, contexts, f"judge error: {exc}")

        parsed = _extract_score(response.content)
        if parsed is None:
            logger.warning("Judge returned an unparseable score", extra={"raw": response.content[:120]})
            return self._lexical(answer, contexts, "unparseable judge response")

        reason = self._extract_reason(response.content)
        logger.debug("Judged faithfulness", extra={"score": round(parsed, 3), "latency_ms": round(timer.elapsed_ms, 1)})
        return FaithfulnessVerdict(score=parsed, reason=reason, method="llm", judged=True)

    @staticmethod
    def _extract_reason(text: str) -> str:
        """Pull the ``reason`` field out of a judge response, if present."""
        match = _JSON_OBJECT_PATTERN.search(text)
        if not match:
            return ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
        if isinstance(payload, dict) and isinstance(payload.get("reason"), str):
            return payload["reason"][:300]
        return ""

    @staticmethod
    def _lexical(answer: str, contexts: Sequence[str], reason: str) -> FaithfulnessVerdict:
        """Lexical fallback used whenever the LLM judge cannot run."""
        return FaithfulnessVerdict(
            score=lexical_grounding_score(answer, contexts),
            reason=reason,
            method="lexical",
            judged=False,
        )


def summarise_verdicts(verdicts: Sequence[FaithfulnessVerdict]) -> dict:
    """Aggregate faithfulness verdicts into a report block.

    Args:
        verdicts: Per-question verdicts.

    Returns:
        Mean score and the LLM/lexical split, so a reader can tell how much of
        the number came from an actual judge.
    """
    if not verdicts:
        return {"mean": 0.0, "judged_count": 0, "lexical_count": 0, "llm_share": 0.0}

    judged = [verdict for verdict in verdicts if verdict.judged]
    return {
        "mean": round(sum(verdict.score for verdict in verdicts) / len(verdicts), 4),
        "judged_count": len(judged),
        "lexical_count": len(verdicts) - len(judged),
        "llm_share": round(len(judged) / len(verdicts), 4),
    }
