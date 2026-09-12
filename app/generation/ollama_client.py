"""Async client for a locally hosted LLM server.

Supports two wire protocols behind one interface:

* Ollama's native REST API (``POST /api/chat``) - the default, and what a
  standard ``ollama serve`` exposes.
* Any OpenAI-compatible server (``POST /v1/chat/completions``) - vLLM,
  llama.cpp's server, LM Studio, Ollama's own OpenAI shim, or a GPU deployment
  reachable over the network.

The flavor is inferred from the base URL (a path ending in ``/v1`` means
OpenAI) and can be pinned with ``LLM_API_FLAVOR``.  The client is deliberately
small: it handles transport, streaming, and token accounting, and knows nothing
about retrieval or prompts.

The base URL and model tag are read from configuration and are intentionally
blank by default - see :mod:`app.config`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: How many cold-start redirects to replay a request through before giving up.
_MAX_REDIRECTS = 3

#: Characters of un-tagged output the stream filter will hold before deciding
#: that the model is not emitting a reasoning block at all.
_REASONING_PREFIX_LIMIT = 4000

#: Delimiters reasoning models wrap their chain of thought in.
_REASONING_TAGS = ("</think>", "<｜end▁of▁thinking｜>")


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's chain of thought from an answer.

    Qwen3-style models emit ``<think> ... </think>`` ahead of the real answer.
    Some servers strip the opening tag themselves - and with it the ability to
    match a balanced pair - so this drops everything up to and including the
    final closing tag whenever one appears before the answer text.
    """
    stripped = text.lstrip()
    for tag in _REASONING_TAGS:
        index = stripped.rfind(tag)
        if index != -1:
            return stripped[index + len(tag) :].strip()
    return text.strip()


class _ReasoningStreamFilter:
    """Withholds a streamed chain of thought from the token stream.

    Token-by-token filtering cannot wait for a closing tag that may never
    arrive, so the buffer is released once it grows past
    :data:`_REASONING_PREFIX_LIMIT` and the text is assumed to be the answer.
    """

    def __init__(self) -> None:
        """Start in the buffering state."""
        self._buffer = ""
        self._settled = False

    def feed(self, token: str) -> str:
        """Return the part of ``token`` that is safe to show, possibly empty."""
        if self._settled:
            return token
        self._buffer += token

        lowered = self._buffer.lower()
        for tag in _REASONING_TAGS:
            index = lowered.rfind(tag)
            if index != -1:
                self._settled = True
                visible = self._buffer[index + len(tag) :].lstrip()
                self._buffer = ""
                return visible

        if len(self._buffer) > _REASONING_PREFIX_LIMIT:
            self._settled = True
            visible, self._buffer = self._buffer, ""
            return visible
        return ""

    def flush(self) -> str:
        """Release anything still buffered when the stream ends."""
        visible, self._buffer = self._buffer, ""
        self._settled = True
        return visible


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class OllamaNotConfiguredError(OllamaError):
    """Raised when base URL or model tag is missing from configuration."""


@dataclass
class ChatMessage:
    """A single chat turn in Ollama's message format."""

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to Ollama's wire format."""
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationStats:
    """Token accounting and timing reported by Ollama.

    Attributes:
        model: Model tag that produced the response.
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens produced.
        total_duration_ms: Server-reported total duration.
        load_duration_ms: Time spent loading the model into memory.
        eval_duration_ms: Time spent generating tokens.
    """

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Decoding throughput, or ``0.0`` when the server did not report it."""
        if self.eval_duration_ms <= 0:
            return 0.0
        return round(self.completion_tokens / (self.eval_duration_ms / 1000.0), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API responses."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "load_duration_ms": round(self.load_duration_ms, 2),
            "eval_duration_ms": round(self.eval_duration_ms, 2),
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class ChatResponse:
    """A completed generation.

    Attributes:
        content: The model's answer text.
        stats: Token accounting and timings.
        finish_reason: Why generation stopped, as reported by Ollama.
    """

    content: str = ""
    stats: GenerationStats = field(default_factory=GenerationStats)
    finish_reason: str = ""


def _nanos_to_ms(value: Any) -> float:
    """Convert Ollama's nanosecond durations to milliseconds."""
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def _stats_from_payload(payload: Dict[str, Any], fallback_model: str) -> GenerationStats:
    """Build :class:`GenerationStats` from an Ollama response body."""
    return GenerationStats(
        model=str(payload.get("model") or fallback_model),
        prompt_tokens=int(payload.get("prompt_eval_count") or 0),
        completion_tokens=int(payload.get("eval_count") or 0),
        total_duration_ms=_nanos_to_ms(payload.get("total_duration")),
        load_duration_ms=_nanos_to_ms(payload.get("load_duration")),
        eval_duration_ms=_nanos_to_ms(payload.get("eval_duration")),
    )


def _stats_from_usage(
    payload: Dict[str, Any],
    fallback_model: str,
    elapsed_ms: float,
) -> GenerationStats:
    """Build :class:`GenerationStats` from an OpenAI-compatible response body.

    OpenAI-shaped responses report a ``usage`` block but no server-side
    timings, so the caller passes the wall-clock duration it measured; that
    keeps ``tokens_per_second`` meaningful on those servers too.
    """
    usage = payload.get("usage") or {}
    return GenerationStats(
        model=str(payload.get("model") or fallback_model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_duration_ms=elapsed_ms,
        load_duration_ms=0.0,
        eval_duration_ms=elapsed_ms,
    )


class OllamaClient:
    """Minimal async client for a local LLM chat API.

    Despite the name (kept for backwards compatibility, since Ollama is the
    default backend) the client speaks either Ollama's native protocol or the
    OpenAI-compatible one; see :attr:`flavor`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        settings: Optional[Settings] = None,
        flavor: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Override for the LLM base URL.  For the OpenAI flavor
                this must include the ``/v1`` prefix.
            model: Override for the model tag or id.
            timeout: Override for the request timeout in seconds.
            settings: Configuration override; defaults to process settings.
            flavor: ``"ollama"`` or ``"openai"``; inferred from the base URL
                when omitted.
            api_key: Bearer token; usually blank for a local server.
        """
        self._settings = settings or get_settings()
        self._base_url = (base_url if base_url is not None else self._settings.ollama_base_url).rstrip("/")
        self._model = model if model is not None else self._settings.ollama_model
        self._timeout = timeout or self._settings.ollama_timeout_seconds
        if flavor is None:
            flavor = (
                self._settings.resolved_api_flavor
                if base_url is None
                else ("openai" if self._base_url.endswith("/v1") else "ollama")
            )
        self._flavor = flavor
        self._api_key = api_key if api_key is not None else self._settings.llm_api_key
        # Cleared for the client's lifetime if the server rejects the option.
        self._thinking_off_requested = bool(self._settings.llm_disable_thinking) and self._flavor == "openai"
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        """Configured model tag."""
        return self._model

    @property
    def base_url(self) -> str:
        """Configured base URL."""
        return self._base_url

    @property
    def flavor(self) -> str:
        """Wire protocol in use: ``"ollama"`` or ``"openai"``."""
        return self._flavor

    @property
    def is_configured(self) -> bool:
        """True when both base URL and model tag are present."""
        return bool(self._base_url and self._model)

    def _ensure_configured(self) -> None:
        """Raise a helpful error when the mandatory settings are missing."""
        problems = self._settings.llm_configuration_errors()
        if not self._base_url or not self._model:
            raise OllamaNotConfiguredError(
                "The LLM backend is not configured. "
                + " ".join(problems or ["Check OLLAMA_BASE_URL and OLLAMA_MODEL."])
            )

    def _default_headers(self) -> Dict[str, str]:
        """Headers applied to every request; a bearer token when configured."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _http(self) -> httpx.AsyncClient:
        """Return the shared ``httpx`` client, creating it on first use.

        Redirects are not followed automatically: the retry below has to
        re-send the request body, which an automatic 303 would drop.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url or "http://localhost:11434",
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers=self._default_headers(),
                follow_redirects=False,
            )
        return self._client

    @staticmethod
    def _redirect_target(response: httpx.Response) -> Optional[str]:
        """Return the URL to re-issue the request to, if this is a redirect.

        Hosted backends that scale to zero answer the first request of a cold
        start with a 3xx to the same route carrying a one-shot attempt token.
        The request has to be replayed verbatim - a 303 would otherwise be
        downgraded to a bodyless GET, which no chat endpoint accepts.
        """
        if response.status_code not in (301, 302, 303, 307, 308):
            return None
        location = response.headers.get("location", "").strip()
        if not location:
            return None
        return str(httpx.URL(location))

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """Probe the server and report whether the model is available.

        Returns:
            A dict with ``reachable``, ``model``, ``model_available`` and,
            when reachable, the list of installed model tags.
        """
        report: Dict[str, Any] = {
            "reachable": False,
            "configured": self.is_configured,
            "base_url": self._base_url or "<unset>",
            "model": self._model or "<unset>",
            "flavor": self._flavor,
            "model_available": False,
            "models": [],
        }
        if not self._base_url:
            report["error"] = "OLLAMA_BASE_URL is not set"
            return report

        path = "/api/tags" if self._flavor == "ollama" else "/models"
        try:
            response = await self._http().get(path)
            for _ in range(_MAX_REDIRECTS):
                target = self._redirect_target(response)
                if target is None:
                    break
                logger.info("Following cold-start redirect for health probe")
                response = await self._http().get(target)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            report["error"] = str(exc)
            return report

        if self._flavor == "ollama":
            tags = [str(entry.get("name", "")) for entry in payload.get("models", [])]
        else:
            tags = [str(entry.get("id", "")) for entry in payload.get("data", [])]

        report["reachable"] = True
        report["models"] = tags
        report["model_available"] = self._model in tags
        if self._model and not report["model_available"]:
            hint = (
                f"Pull it with `ollama pull {self._model}`."
                if self._flavor == "ollama"
                else "Check the id reported by the server's /models endpoint."
            )
            report["error"] = (
                f"Model '{self._model}' is not available. "
                f"Available: {', '.join(tags) or 'none'}. {hint}"
            )
        return report

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> ChatResponse:
        """Run a blocking chat completion.

        Args:
            messages: Conversation turns, oldest first.
            temperature: Sampling temperature override.
            num_ctx: Context-window override (Ollama flavor only).

        Returns:
            The completed :class:`ChatResponse`.

        Raises:
            OllamaNotConfiguredError: If base URL or model is unset.
            OllamaError: If the server is unreachable or returns an error.
        """
        self._ensure_configured()
        if self._flavor == "ollama":
            path = "/api/chat"
            body = self._build_body(messages, temperature, num_ctx, stream=False)
        else:
            path = "/chat/completions"
            body = self._build_openai_body(messages, temperature, stream=False)

        started = time.perf_counter()
        try:
            response = await self._http().post(path, json=body)
            for _ in range(_MAX_REDIRECTS):
                target = self._redirect_target(response)
                if target is None:
                    break
                logger.info("Following cold-start redirect for chat completion")
                response = await self._http().post(target, json=body)

            if response.status_code == 400 and self._thinking_off_requested:
                response = await self._retry_without_thinking_option(path, body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"LLM server returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach the LLM server at {self._base_url}: {exc}. "
                "Is your local model server running?"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if self._flavor == "ollama":
            message = payload.get("message") or {}
            return ChatResponse(
                content=str(message.get("content", "")),
                stats=_stats_from_payload(payload, self._model),
                finish_reason=str(payload.get("done_reason", "") or ""),
            )

        choices = payload.get("choices") or [{}]
        choice = choices[0] or {}
        message = choice.get("message") or {}
        return ChatResponse(
            content=strip_reasoning(str(message.get("content") or "")),
            stats=_stats_from_usage(payload, self._model, elapsed_ms),
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    async def _retry_without_thinking_option(self, path: str, body: Dict[str, Any]) -> httpx.Response:
        """Re-send a rejected request with the thinking option removed.

        Strict OpenAI-compatible servers reject unknown body fields with a 400.
        Dropping the field permanently keeps later requests from paying for a
        second round trip.
        """
        logger.info("Server rejected chat_template_kwargs; disabling the thinking option")
        self._thinking_off_requested = False
        retry_body = {key: value for key, value in body.items() if key != "chat_template_kwargs"}
        return await self._http().post(path, json=retry_body)

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat completion as it is generated.

        Yields:
            Event dicts: ``{"type": "token", "content": str}`` for each delta,
            then a final ``{"type": "done", "stats": {...}, "finish_reason": str}``.

        Raises:
            OllamaNotConfiguredError: If base URL or model is unset.
            OllamaError: If the server is unreachable or returns an error.
        """
        self._ensure_configured()
        if self._flavor == "ollama":
            path = "/api/chat"
            body = self._build_body(messages, temperature, num_ctx, stream=True)
        else:
            path = "/chat/completions"
            body = self._build_openai_body(messages, temperature, stream=True)

        started = time.perf_counter()
        target = path
        # Only engaged on the OpenAI flavor when the server was not asked to
        # skip reasoning: with thinking disabled the first token is the answer.
        filter_reasoning = self._flavor == "openai" and not self._thinking_off_requested
        reasoning_filter = _ReasoningStreamFilter() if filter_reasoning else None
        last_finish_reason = ""
        try:
            for attempt in range(_MAX_REDIRECTS + 1):
                async with self._http().stream("POST", target, json=body) as response:
                    redirect = self._redirect_target(response) if attempt < _MAX_REDIRECTS else None
                    if redirect is not None:
                        logger.info("Following cold-start redirect for streamed completion")
                        target = redirect
                        continue  # leaving the block releases the unread response
                    if response.status_code == 400 and self._thinking_off_requested and attempt == 0:
                        await response.aread()  # drain before retrying on a live connection
                        logger.info("Server rejected chat_template_kwargs; disabling the thinking option")
                        self._thinking_off_requested = False
                        body = {key: value for key, value in body.items() if key != "chat_template_kwargs"}
                        reasoning_filter = _ReasoningStreamFilter()
                        continue
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if self._flavor == "ollama":
                            events, reason = self._parse_ollama_line(line)
                        else:
                            events, reason = self._parse_openai_line(line, started)
                        if reason:
                            last_finish_reason = reason
                        for event in events:
                            if event["type"] == "done" and not event.get("finish_reason"):
                                event["finish_reason"] = last_finish_reason
                            if event["type"] == "token" and reasoning_filter is not None:
                                visible = reasoning_filter.feed(event["content"])
                                if visible:
                                    yield {"type": "token", "content": visible}
                            else:
                                yield event
                    if reasoning_filter is not None:
                        pending = reasoning_filter.flush()
                        if pending:
                            yield {"type": "token", "content": pending}
                    return
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"LLM server returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not reach the LLM server at {self._base_url}: {exc}. "
                "Is your local model server running?"
            ) from exc

    @staticmethod
    def _parse_ollama_line(line: str) -> tuple[List[Dict[str, Any]], str]:
        """Translate one line of an Ollama stream into client events."""
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed stream chunk")
            return [], ""

        events: List[Dict[str, Any]] = []
        delta = (chunk.get("message") or {}).get("content", "")
        if delta:
            events.append({"type": "token", "content": delta})
        finish_reason = str(chunk.get("done_reason", "") or "")
        if chunk.get("done"):
            events.append(
                {
                    "type": "done",
                    "stats": _stats_from_payload(chunk, str(chunk.get("model", ""))).to_dict(),
                    "finish_reason": finish_reason,
                }
            )
        return events, finish_reason

    @staticmethod
    def _parse_openai_line(
        line: str, started: float
    ) -> tuple[List[Dict[str, Any]], str]:
        """Translate one line of an OpenAI-compatible SSE stream into events.

        Returns:
            The events for this line, and the ``finish_reason`` it reported (an
            empty string when it reported none).  vLLM sends the reason on the
            last content chunk and leaves the following usage-only chunk blank,
            so the caller has to remember it across lines.
        """
        if not line.startswith("data:"):
            return [], ""  # comment/keep-alive frames carry no payload
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return [], ""
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed stream chunk")
            return [], ""

        events: List[Dict[str, Any]] = []
        finish_reason = ""
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            # Reasoning models expose their chain of thought under
            # `reasoning_content`; only the user-visible answer is streamed.
            content = delta.get("content")
            if content:
                events.append({"type": "token", "content": content})
            finish_reason = str(choice.get("finish_reason") or "")

        if chunk.get("usage"):
            events.append(
                {
                    "type": "done",
                    "stats": _stats_from_usage(
                        chunk, str(chunk.get("model", "")), (time.perf_counter() - started) * 1000.0
                    ).to_dict(),
                    "finish_reason": finish_reason,
                }
            )
        return events, finish_reason

    def _build_body(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float],
        num_ctx: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        """Assemble an Ollama ``/api/chat`` request body."""
        return {
            "model": self._model,
            "messages": [message.to_dict() for message in messages],
            "stream": stream,
            "options": {
                "temperature": (
                    temperature if temperature is not None else self._settings.ollama_temperature
                ),
                "num_ctx": num_ctx or self._settings.ollama_num_ctx,
            },
        }

    def _build_openai_body(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float],
        stream: bool,
    ) -> Dict[str, Any]:
        """Assemble an OpenAI-compatible ``/chat/completions`` request body.

        Sampling options live at the top level here rather than nested under
        ``options``, and ``stream_options`` asks the server to report token
        usage in the final stream chunk so throughput is still measurable.
        """
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_dict() for message in messages],
            "stream": stream,
            "temperature": (
                temperature if temperature is not None else self._settings.ollama_temperature
            ),
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if self._thinking_off_requested:
            # vLLM/Qwen-style servers honour this in the chat template. A server
            # that does not understand the field rejects the request, and
            # `_drop_thinking_option` retries without it.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body


class FakeOllamaClient(OllamaClient):
    """Offline stand-in for tests and ``DEV_FAKE_LLM`` mode.

    Produces a deterministic, citation-shaped answer derived from the context
    passed in the prompt, so the whole pipeline can be exercised without a
    running model.
    """

    def __init__(self, model: str = "fake-model", base_url: str = "http://fake-ollama") -> None:
        """Create a fake client that never touches the network."""
        super().__init__(base_url=base_url, model=model)
        self.calls: List[List[ChatMessage]] = []

    @property
    def is_configured(self) -> bool:
        """Always configured - that is the point of the fake."""
        return True

    def _ensure_configured(self) -> None:
        """No-op: the fake is always configured."""

    async def health(self) -> Dict[str, Any]:
        """Report a healthy fake server with the fake model installed."""
        return {
            "reachable": True,
            "configured": True,
            "base_url": self.base_url,
            "model": self.model,
            "model_available": True,
            "models": [self.model],
        }

    def _fabricate(self, messages: Sequence[ChatMessage]) -> str:
        """Build a canned answer citing the first two context passages."""
        prompt = messages[-1].content if messages else ""
        citations = []
        for index in range(1, 3):
            if f"[{index}]" in prompt:
                citations.append(f"[{index}]")
        if not citations:
            return "I don't know based on the provided documents."
        joined = " ".join(citations)
        return (
            f"Fake grounded answer synthesised from the retrieved context {joined}. "
            "This response was produced by DEV_FAKE_LLM mode and contains no real reasoning."
        )

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> ChatResponse:
        """Return a canned answer with plausible token statistics."""
        self.calls.append(list(messages))
        content = self._fabricate(messages)
        prompt_tokens = sum(len(message.content.split()) for message in messages)
        return ChatResponse(
            content=content,
            stats=GenerationStats(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(content.split()),
                total_duration_ms=12.0,
                eval_duration_ms=10.0,
            ),
            finish_reason="stop",
        )

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream the canned answer one word at a time."""
        self.calls.append(list(messages))
        content = self._fabricate(messages)
        for word in content.split(" "):
            yield {"type": "token", "content": word + " "}
        yield {
            "type": "done",
            "stats": GenerationStats(
                model=self.model,
                completion_tokens=len(content.split()),
                total_duration_ms=12.0,
            ).to_dict(),
            "finish_reason": "stop",
        }
