"""
Anthropic LLM provider using the Anthropic Python SDK.

This provider enables using Claude models from Anthropic with support for:
- Structured JSON output
- Tool/function calling with proper format conversion
- Extended thinking mode
- Retry logic with exponential backoff
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """
    LLM provider using Anthropic's Claude models.

    Supports structured output, tool calling, and extended thinking mode.
    Handles format conversion between OpenAI-style messages and Anthropic's format.
    """

    #: Appended to the system prompt once, when the model answered a JSON request
    #: with prose. Real case 2026-09-16: a batch_retain task failed 4x11 attempts
    #: on identical requests because nothing changed between retries.
    JSON_ONLY_NUDGE = "Respond with the JSON object only — no prose, no code fences, no explanation."

    #: When ``stop_reason == "max_tokens"`` the text was cut before a full JSON
    #: object existed (often: empty text, ``json.loads("")``). The budget doubles
    #: per retry, capped at this factor times the initial budget so a runaway
    #: prompt costs at most 4x, never unbounded.
    MAX_TOKENS_ESCALATION_CEILING_FACTOR = 4

    #: Characters of the rejected text that go into the log line. Measured
    #: 2026-10-02: 161 invalid-JSON warnings, every head looked like valid fenced
    #: JSON — the head identifies the response, the dump file explains it.
    INVALID_JSON_TEXT_HEAD_CHARS = 200

    #: The WHOLE text wrapped in one code fence, anchored at both ends. Anchoring
    #: at the end matters: a ``` INSIDE a string value (facts about markdown,
    #: hooks, heredocs) must not end the object early — ``split("```")`` did.
    _JSON_FENCE_RE = re.compile(r"^\s*```[A-Za-z]*[ \t]*\n?(.*?)\n?[ \t]*```\s*$", re.DOTALL)

    #: Characters that may legally follow a closing quote in JSON. A quote followed
    #: by anything else did not close the string — it is text the model forgot to
    #: escape. Measured 2026-10-02 (dumps attempt1..5of11 of op 5a38eb75): Haiku
    #: closes a German „…“ quotation with a bare ASCII quote inside the value, the
    #: string ends early and every one of 11 retries fails at the same character.
    _JSON_AFTER_STRING = ",:}]"

    #: Error prefixes the parser emits when a string ended early and the next
    #: token is plain text. Anything else is not a stray quote and is not repaired.
    _STRAY_QUOTE_ERRORS = ("Expecting ',' delimiter", "Expecting ':' delimiter")

    #: Emitted when a string ended early right before a comma: ``„…", weil`` —
    #: the parser accepts the comma and then misses the quote of the next key.
    #: Measured 2026-10-02 (dumps attempt1..4of11, 16:49): 4 of 4 attempts at the
    #: same character. A real ``",`` is always followed by ``"``; this one is not.
    _STRAY_QUOTE_BEFORE_COMMA_ERROR = "Expecting property name enclosed in double quotes"

    #: Dump files kept in the dump directory; older ones are deleted after each write.
    #: Both sides measured 2026-10-02. KEPT: the worst real incident is one retain
    #: burning its 11 attempts, several retains in parallel — far below 1000, so a
    #: running incident is never truncated. DELETED: 1000 files at the measured mean
    #: of 7.6 KB (17 real dumps, 2.8-10.4 KB) is ~7.4 MB, and at the measured peak of
    #: 161 rejections/day that is ~6 days of evidence — the same order as the 20 MB /
    #: ~10 days of hindsight.log (start.sh). Fail-safe direction: the OLDEST evidence
    #: is lost, never the newest, and no retain data — only diagnostics.
    INVALID_JSON_DUMP_KEEP = 1000

    #: Matches only this class's own dump files (``<stamp>-attemptNofM-<id>.txt``).
    #: Rotation deletes, so it must never see a foreign file in a shared directory.
    _INVALID_JSON_DUMP_GLOB = "*-attempt*of*-*.txt"

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float = 300.0,
        invalid_json_dump_dir: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic LLM provider.

        Args:
            provider: Provider name (should be "anthropic").
            api_key: Anthropic API key.
            base_url: Base URL for the API (optional, uses Anthropic default if empty).
            model: Model name (e.g., "claude-sonnet-4-20250514").
            reasoning_effort: Reasoning effort level (not used by Anthropic).
            timeout: Request timeout in seconds.
            invalid_json_dump_dir: Directory for full dumps of responses that never
                parsed as JSON. None = ``HINDSIGHT_API_LLM_INVALID_JSON_DUMP_DIR``
                from the global config, resolved lazily at the first dump.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)
        self._invalid_json_dump_dir = invalid_json_dump_dir

        if not self.api_key:
            raise ValueError("API key is required for Anthropic provider")

        # Import and initialize Anthropic client
        try:
            from anthropic import AsyncAnthropic

            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if timeout:
                client_kwargs["timeout"] = timeout

            self._client = AsyncAnthropic(**client_kwargs)
            logger.info(f"Anthropic client initialized for model: {self.model}")
        except ImportError as e:
            raise RuntimeError("Anthropic SDK not installed. Run: uv add anthropic or pip install anthropic") from e

    @classmethod
    def _extract_json(cls, content: str) -> Any:
        """Parse the model's text as JSON, tolerating the wrappers seen in production.

        Candidates in order, first one that parses wins:
          1. the raw text — ``strict=False`` accepts literal control characters
             (a newline the model did not escape) inside string values;
          2. the text inside ONE surrounding code fence (``_JSON_FENCE_RE``);
          3. the slice from the first ``{`` to the last ``}`` — prose before or
             after the object, or a fence the model never closed.
        If every candidate fails, the error that got FURTHEST is raised: for a
        fenced response the raw text always fails at char 0 (the fence), the
        inner text fails at the real defect. ``error.doc`` is the candidate
        the position refers to.
        """
        candidates = [content]
        fenced = cls._JSON_FENCE_RE.match(content)
        if fenced is not None:
            candidates.append(fenced.group(1))
        first_brace, last_brace = content.find("{"), content.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(content[first_brace : last_brace + 1])

        deepest_error: json.JSONDecodeError | None = None
        for candidate in candidates:
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError as e:
                if deepest_error is None or e.pos > deepest_error.pos:
                    deepest_error = e
        if deepest_error is None:  # unreachable: candidates[0] is the raw text and did not return
            raise json.JSONDecodeError("no JSON candidate in response", content, 0)
        repaired = cls._repair_unescaped_quotes(deepest_error.doc)
        if repaired is not None:
            text, escaped = repaired
            logger.warning("Anthropic JSON parsed only after escaping %d stray quote(s) inside string values", escaped)
            return json.loads(text, strict=False)
        raise deepest_error

    @classmethod
    def _repair_unescaped_quotes(cls, text: str) -> tuple[str, int] | None:
        """Escape ASCII quotes that end a string although no structural character follows.

        Driven by the parser: the last quote before the error position closed the
        string early. Two provable cases are escaped, everything else is left alone
        and ``None`` is returned — fail-closed, the original error stands:
          * ``_STRAY_QUOTE_ERRORS``: the first non-blank character after the quote
            is not in ``_JSON_AFTER_STRING``, so the quote can not be structural;
          * ``_STRAY_QUOTE_BEFORE_COMMA_ERROR``: only a comma separates the quote
            from the error, and a real ``",`` would be followed by a quote.
        A missing comma or a key without its opening quote is neither, and stays an
        error. Bounded by the number of quotes: each round escapes one.
        Returns the repaired text and the number of escaped quotes.
        """
        repaired, escaped = text, 0
        for _ in range(text.count('"')):
            try:
                json.loads(repaired, strict=False)
                return (repaired, escaped) if escaped else None
            except json.JSONDecodeError as e:
                quote = repaired.rfind('"', 0, e.pos)
                if quote == -1:
                    return None
                if e.msg.startswith(cls._STRAY_QUOTE_ERRORS):
                    following = repaired[quote + 1 :].lstrip()
                    stray = bool(following) and following[0] not in cls._JSON_AFTER_STRING
                elif e.msg.startswith(cls._STRAY_QUOTE_BEFORE_COMMA_ERROR):
                    # e.pos is the offending token; only a comma may sit between quote and token.
                    stray = repaired[quote + 1 : e.pos].strip() == ","
                else:
                    return None
                if not stray:
                    return None
                repaired = f"{repaired[:quote]}\\{repaired[quote:]}"
                escaped += 1
        return None

    @classmethod
    def _rotate_invalid_json_dumps(cls, directory: Path) -> None:
        """Delete this class's own dump files beyond ``INVALID_JSON_DUMP_KEEP``.

        The names start with a UTC timestamp, so sorting them is chronological and
        the tail of the sorted list is the newest. Only files matching
        ``_INVALID_JSON_DUMP_GLOB`` are considered — the directory sits next to
        hindsight.log and may hold anything else.
        """
        dumps = sorted(directory.glob(cls._INVALID_JSON_DUMP_GLOB))
        for stale in dumps[: max(0, len(dumps) - cls.INVALID_JSON_DUMP_KEEP)]:
            stale.unlink(missing_ok=True)

    @staticmethod
    def _text_at_error(error: json.JSONDecodeError) -> str:
        """The failing candidate around the error position — the defect itself, not the head."""
        half = AnthropicLLM.INVALID_JSON_TEXT_HEAD_CHARS // 2
        return error.doc[max(0, error.pos - half) : error.pos + half]

    def _dump_invalid_json(
        self,
        content: str,
        *,
        error: json.JSONDecodeError,
        stop_reason: Any,
        output_tokens: int,
        attempt: int,
        max_attempts: int,
    ) -> Path | None:
        """Write the FULL rejected response to disk and return the file path.

        Called for EVERY failed attempt, not only the last: measured 2026-10-02
        after the first fix, a retain needed 5 attempts, the 4 rejected responses
        were gone with the retry and the log head (``INVALID_JSON_TEXT_HEAD_CHARS``
        chars) looked like valid JSON — the cause stayed invisible. Measured
        volume: 161 rejections/day at a mean of 7.6 KB, ~1.2 MB/day; bounded by
        ``_rotate_invalid_json_dumps`` after every write.
        Diagnostics only: an unwritable directory is logged, never raised, so it
        can not turn a parse failure into a different failure.
        """
        try:
            if self._invalid_json_dump_dir is None:
                from hindsight_api.config import get_config

                self._invalid_json_dump_dir = get_config().llm_invalid_json_dump_dir
            directory = Path(self._invalid_json_dump_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            path = directory / f"{stamp}-attempt{attempt}of{max_attempts}-{uuid.uuid4().hex[:8]}.txt"
            header = (
                f"# model={self.model}\n# stop_reason={stop_reason}\n# output_tokens={output_tokens}\n"
                f"# attempt={attempt}/{max_attempts}\n# chars={len(content)}\n# error={error}\n"
                f"# text_at_error={self._text_at_error(error)!r}\n\n"
            )
            path.write_text(header + content, encoding="utf-8")
            self._rotate_invalid_json_dumps(directory)
            return path
        except OSError as e:
            logger.error("Could not dump invalid JSON response (%d chars): %s", len(content), e)
            return None

    async def verify_connection(self) -> None:
        """
        Verify that the Anthropic provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=10,
                temperature=0.0,
                scope="verification",
                max_retries=0,
            )
            logger.info("Anthropic connection verified successfully")
        except Exception as e:
            logger.error(f"Anthropic connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify Anthropic connection: {e}") from e

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        return_usage: bool = False,
    ) -> Any:
        """
        Make an LLM API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Use strict JSON schema enforcement (not supported by Anthropic).
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        start_time = time.time()

        # Convert OpenAI-style messages to Anthropic format
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            if system_prompt:
                system_prompt += schema_msg
            else:
                system_prompt = schema_msg

        # Prepare parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_completion_tokens if max_completion_tokens is not None else 4096,
        }

        if system_prompt:
            call_params["system"] = system_prompt

        if temperature is not None:
            call_params["temperature"] = temperature

        last_exception = None
        # Recovery state for invalid-JSON retries (see JSON_ONLY_NUDGE /
        # MAX_TOKENS_ESCALATION_CEILING_FACTOR): the budget ceiling derives from
        # the budget the caller asked for, the nudge is appended at most once.
        initial_max_tokens: int = call_params["max_tokens"]
        max_tokens_ceiling = initial_max_tokens * self.MAX_TOKENS_ESCALATION_CEILING_FACTOR
        json_nudge_applied = False

        for attempt in range(max_retries + 1):
            response: Any = None
            content = ""
            try:
                response = await self._client.messages.create(**call_params)

                # Anthropic response content is a list of blocks
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                if response_format is not None:
                    json_data = self._extract_json(content)

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = content

                # Record metrics and log slow calls
                duration = time.time() - start_time
                input_tokens = response.usage.input_tokens or 0 if response.usage else 0
                output_tokens = response.usage.output_tokens or 0 if response.usage else 0
                total_tokens = input_tokens + output_tokens

                # Record LLM metrics
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                # Record trace span
                from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                finish_reason = response.stop_reason if hasattr(response, "stop_reason") else None
                span_recorder = get_span_recorder()
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    messages=messages,
                    response_content=_serialize_for_span(result),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                )

                # Log slow calls
                if duration > 10.0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                    return result, token_usage
                return result

            except json.JSONDecodeError as e:
                last_exception = e
                # `response` and `content` are bound: the parse runs after both.
                stop_reason = getattr(response, "stop_reason", None)
                usage = getattr(response, "usage", None)
                failed_output_tokens = (usage.output_tokens or 0) if usage else 0
                text_head = content[: self.INVALID_JSON_TEXT_HEAD_CHARS]
                dump_path = self._dump_invalid_json(
                    content,
                    error=e,
                    stop_reason=stop_reason,
                    output_tokens=failed_output_tokens,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                )
                logger.warning(
                    "Anthropic returned invalid JSON (attempt %d/%d, stop_reason=%s, output_tokens=%s, "
                    "max_tokens=%s, error=%s, text_at_error=%r, text_head=%r, full response: %s)",
                    attempt + 1,
                    max_retries + 1,
                    stop_reason,
                    failed_output_tokens,
                    call_params["max_tokens"],
                    e,
                    self._text_at_error(e),
                    text_head,
                    dump_path,
                )
                if attempt < max_retries:
                    if stop_reason == "max_tokens":
                        # Cut off before a complete object: more budget, bounded.
                        call_params["max_tokens"] = min(call_params["max_tokens"] * 2, max_tokens_ceiling)
                    elif content.strip() and not json_nudge_applied:
                        # Prose instead of JSON: ask once, explicitly, for JSON only.
                        existing_system = call_params.get("system")
                        call_params["system"] = (
                            f"{existing_system}\n\n{self.JSON_ONLY_NUDGE}" if existing_system else self.JSON_ONLY_NUDGE
                        )
                        json_nudge_applied = True
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    message = (
                        f"Anthropic returned invalid JSON after {max_retries + 1} attempts "
                        f"(stop_reason={stop_reason}, output_tokens={failed_output_tokens}, "
                        f"error={e}, text_head={text_head!r})"
                    )
                    if dump_path is not None:
                        message += f"; full response: {dump_path}"
                    logger.error(message)
                    raise ValueError(message) from e

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                # Fast fail on 401/403
                if isinstance(e, APIStatusError) and e.status_code in (401, 403):
                    logger.error(f"Anthropic auth error (HTTP {e.status_code}), not retrying: {str(e)}")
                    raise

                last_exception = e
                if attempt < max_retries:
                    # Check if it's a rate limit or server error
                    should_retry = isinstance(e, (APIConnectionError, RateLimitError)) or (
                        isinstance(e, APIStatusError) and e.status_code >= 500
                    )

                    if should_retry:
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        await asyncio.sleep(backoff + jitter)
                        continue

                logger.error(f"Anthropic API error after {max_retries + 1} attempts: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error during Anthropic call: {type(e).__name__}: {str(e)}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Anthropic call failed after all retries")

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """
        Make an LLM API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools - "auto", "none", "required", or specific function.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        from anthropic import APIConnectionError, APIStatusError

        start_time = time.time()

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )

        # Convert messages - handle tool results
        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = (system_prompt + "\n\n" + content) if system_prompt else content
            elif role == "tool":
                # Anthropic uses tool_result blocks
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": content}
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool calls
                tool_use_blocks = []
                for tc in msg["tool_calls"]:
                    tool_use_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": tool_use_blocks})
            else:
                anthropic_messages.append({"role": role, "content": content})

        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": max_completion_tokens or 4096,
        }
        if system_prompt:
            call_params["system"] = system_prompt

        if temperature is not None:
            call_params["temperature"] = temperature

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.messages.create(**call_params)

                # Extract content and tool calls
                content_parts = []
                tool_calls: list[LLMToolCall] = []

                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append(LLMToolCall(id=block.id, name=block.name, arguments=block.input or {}))

                content = "".join(content_parts) if content_parts else None
                finish_reason = "tool_calls" if tool_calls else "stop"

                # Extract token usage
                input_tokens = response.usage.input_tokens or 0
                output_tokens = response.usage.output_tokens or 0

                # Record metrics
                metrics = get_metrics_collector()
                duration = time.time() - start_time
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                # Record OpenTelemetry span
                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                # Convert LLMToolCall objects to dicts for span recording
                tool_calls_dict = (
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                )
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    messages=messages,
                    response_content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                    tool_calls=tool_calls_dict,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except (APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code in (401, 403):
                    raise
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Anthropic tool call failed")

    async def cleanup(self) -> None:
        """Clean up resources (close Anthropic client connections)."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
