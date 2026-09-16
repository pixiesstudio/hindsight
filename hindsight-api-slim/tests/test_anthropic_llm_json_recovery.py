"""Regression tests for AnthropicLLM.call() when the model answers with empty or
non-JSON text although ``response_format`` demands JSON.

Real incident 2026-09-16 (hindsight.log 14:16–14:43): one batch_retain task
failed four times with ``json.decoder.JSONDecodeError: Expecting value: line 1
column 1 (char 0)`` — i.e. ``json.loads("")`` — after 11 identical attempts each.
The provider retried the very same request ten times, logged neither
``stop_reason`` nor the returned text, and the batch was marked failed. These
tests pin the recovery contract: diagnose (stop_reason, output tokens, text
head in the final error), escalate ``max_tokens`` when the model ran out of
budget, and nudge for JSON-only output when it answered with prose.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM


class _Out(BaseModel):
    ok: bool


def _response(text: str, stop_reason: str, output_tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=output_tokens),
    )


def _llm(responses: list[SimpleNamespace]) -> tuple[AnthropicLLM, AsyncMock]:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-haiku-4-5-20251001",
    )
    create = AsyncMock(side_effect=responses)
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


@pytest.mark.asyncio
async def test_empty_text_with_max_tokens_stop_escalates_budget_on_retry() -> None:
    llm, create = _llm(
        [
            _response("", "max_tokens", output_tokens=4096),
            _response('{"ok": true}', "end_turn"),
        ]
    )

    result = await llm.call(
        messages=[{"role": "user", "content": "extract"}],
        response_format=_Out,
        max_retries=3,
        initial_backoff=0.0,
    )

    assert result == _Out(ok=True)
    assert create.call_count == 2
    assert create.call_args_list[0].kwargs["max_tokens"] == 4096
    assert create.call_args_list[1].kwargs["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_prose_instead_of_json_retries_with_json_only_nudge() -> None:
    llm, create = _llm(
        [
            _response("Sure! Here are the facts I found:", "end_turn"),
            _response('{"ok": true}', "end_turn"),
        ]
    )

    result = await llm.call(
        messages=[{"role": "system", "content": "base"}, {"role": "user", "content": "extract"}],
        response_format=_Out,
        max_retries=3,
        initial_backoff=0.0,
    )

    assert result == _Out(ok=True)
    assert create.call_count == 2
    first_system = create.call_args_list[0].kwargs["system"]
    second_system = create.call_args_list[1].kwargs["system"]
    assert second_system == first_system + "\n\n" + AnthropicLLM.JSON_ONLY_NUDGE
    # Prose is not a budget problem: the token budget must stay untouched.
    assert create.call_args_list[1].kwargs["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_max_tokens_escalation_is_bounded_by_ceiling_factor() -> None:
    empty = _response("", "max_tokens", output_tokens=4096)
    llm, create = _llm([empty, empty, empty, empty])

    with pytest.raises(ValueError, match=r"stop_reason=max_tokens"):
        await llm.call(
            messages=[{"role": "user", "content": "extract"}],
            response_format=_Out,
            max_retries=3,
            initial_backoff=0.0,
        )

    budgets = [c.kwargs["max_tokens"] for c in create.call_args_list]
    # 4096 -> 8192 -> 16384 -> capped at 4x the initial budget.
    assert budgets == [4096, 8192, 16384, 16384]


@pytest.mark.asyncio
async def test_exhausted_retries_raise_error_naming_stop_reason_tokens_and_text_head() -> None:
    llm, create = _llm(
        [
            _response("", "end_turn", output_tokens=0),
            _response("", "end_turn", output_tokens=0),
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"invalid JSON after 2 attempts \(stop_reason=end_turn, output_tokens=0, text_head=''\)",
    ):
        await llm.call(
            messages=[{"role": "user", "content": "extract"}],
            response_format=_Out,
            max_retries=1,
            initial_backoff=0.0,
        )

    assert create.call_count == 2
