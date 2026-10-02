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

import json
import re
from pathlib import Path
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


class _Note(BaseModel):
    ok: bool
    note: str


def _llm(responses: list[SimpleNamespace], dump_dir: Path | None = None) -> tuple[AnthropicLLM, AsyncMock]:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-haiku-4-5-20251001",
        invalid_json_dump_dir=str(dump_dir) if dump_dir is not None else None,
    )
    create = AsyncMock(side_effect=responses)
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


@pytest.mark.asyncio
async def test_empty_text_with_max_tokens_stop_escalates_budget_on_retry(tmp_path: Path) -> None:
    llm, create = _llm(
        [
            _response("", "max_tokens", output_tokens=4096),
            _response('{"ok": true}', "end_turn"),
        ],
        dump_dir=tmp_path,
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
async def test_prose_instead_of_json_retries_with_json_only_nudge(tmp_path: Path) -> None:
    llm, create = _llm(
        [
            _response("Sure! Here are the facts I found:", "end_turn"),
            _response('{"ok": true}', "end_turn"),
        ],
        dump_dir=tmp_path,
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
async def test_max_tokens_escalation_is_bounded_by_ceiling_factor(tmp_path: Path) -> None:
    empty = _response("", "max_tokens", output_tokens=4096)
    llm, create = _llm([empty, empty, empty, empty], dump_dir=tmp_path)

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
async def test_exhausted_retries_raise_error_naming_stop_reason_tokens_and_text_head(tmp_path: Path) -> None:
    llm, create = _llm(
        [
            _response("", "end_turn", output_tokens=0),
            _response("", "end_turn", output_tokens=0),
        ],
        dump_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"invalid JSON after 2 attempts \(stop_reason=end_turn, output_tokens=0, "
            r"error=Expecting value: line 1 column 1 \(char 0\), text_head=''\)"
        ),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "extract"}],
            response_format=_Out,
            max_retries=1,
            initial_backoff=0.0,
        )

    assert create.call_count == 2


# Real wrappers around a COMPLETE object. Each must parse on the first attempt —
# no retry, no JSON-only nudge — because only the dressing differs. Measured
# 2026-10-02: 161 invalid-JSON warnings in one day, every text_head starting
# with a ```json fence at stop_reason=end_turn; the old split("```") cut the
# object at the first fence INSIDE a string value and the raw fallback then
# choked on the opening fence, so such a response could never parse.
_PARSEABLE_WRAPPERS = [
    pytest.param(
        '```json\n{"ok": true, "note": "strip ```json fences before json.loads"}\n```',
        "strip ```json fences before json.loads",
        id="fence-inside-string-value",
    ),
    pytest.param(
        '{"ok": true, "note": "line one\nline two"}',
        "line one\nline two",
        id="literal-newline-inside-string",
    ),
    pytest.param(
        'Here is the extraction:\n{"ok": true, "note": "prose around"}\nLet me know if you need more.',
        "prose around",
        id="prose-around-object",
    ),
    # Measured 2026-10-02 (op 5a38eb75, 5 of 5 attempts): Haiku closes the German
    # „…“ quotation with a bare ASCII quote inside the value — the string ends early.
    pytest.param(
        '{"ok": true, "note": "wurden als „verified" gemeldet, obwohl der Fehlertext stand"}',
        'wurden als „verified" gemeldet, obwohl der Fehlertext stand',
        id="stray-quote-closing-german-quotation",
    ),
    pytest.param(
        '```json\n{"ok": true, "note": "He said "hi" today"}\n```',
        'He said "hi" today',
        id="two-stray-quotes-inside-fence",
    ),
    # Measured 2026-10-02 16:49 (4 of 4 attempts): the stray quote sits right before a
    # comma, so the parser accepts the comma and then misses the next key's quote.
    pytest.param(
        '{"ok": true, "note": "scheiterte mit „address already in use", weil die alte Instanz lief"}',
        'scheiterte mit „address already in use", weil die alte Instanz lief',
        id="stray-quote-before-comma",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "expected_note"), _PARSEABLE_WRAPPERS)
async def test_wrapped_but_complete_json_is_parsed_on_first_attempt(
    text: str, expected_note: str, tmp_path: Path
) -> None:
    # dump_dir always injected: a failing case must never write into the real dump directory.
    llm, create = _llm([_response(text, "end_turn", output_tokens=40)], dump_dir=tmp_path)

    result = await llm.call(
        messages=[{"role": "system", "content": "base"}, {"role": "user", "content": "extract"}],
        response_format=_Note,
        max_retries=3,
        initial_backoff=0.0,
    )

    assert result == _Note(ok=True, note=expected_note)
    assert create.call_count == 1
    # The schema instruction is always appended; the JSON-only nudge must not be.
    system_prompt = create.call_args_list[0].kwargs["system"]
    assert system_prompt.startswith("base\n\n")
    assert AnthropicLLM.JSON_ONLY_NUDGE not in system_prompt


@pytest.mark.asyncio
async def test_final_failure_dumps_full_response_and_names_the_file(tmp_path: Path) -> None:
    # 2,500+ chars of fenced, broken JSON: the 200-char text_head alone looks like valid JSON.
    broken = '```json\n{"ok": true, "note": "' + "x" * 2500 + '"", "dangling": }\n```'
    llm, create = _llm([_response(broken, "end_turn", output_tokens=900)], dump_dir=tmp_path)

    with pytest.raises(ValueError, match=r"text_head='```json") as excinfo:
        await llm.call(
            messages=[{"role": "user", "content": "extract"}],
            response_format=_Note,
            max_retries=0,
            initial_backoff=0.0,
        )

    dumps = sorted(tmp_path.iterdir())
    assert len(dumps) == 1
    assert re.fullmatch(r"\d{8}T\d{12}Z-attempt1of1-[0-9a-f]{8}\.txt", dumps[0].name)
    assert str(excinfo.value).endswith(f"; full response: {dumps[0]}")
    # The reported error is the one that got furthest: the fenced inner text fails at
    # the stray quote after the x's, not the raw text at char 0 (the fence).
    inner = broken.removeprefix("```json\n").removesuffix("\n```")
    with pytest.raises(json.JSONDecodeError, match=r"^Expecting ',' delimiter") as inner_err:
        json.loads(inner, strict=False)
    expected_error = inner_err.value
    window = inner[expected_error.pos - 100 : expected_error.pos + 100]
    header, _, body = dumps[0].read_text(encoding="utf-8").partition("\n\n")
    assert header == (
        "# model=claude-haiku-4-5-20251001\n# stop_reason=end_turn\n# output_tokens=900\n"
        f"# attempt=1/1\n# chars={len(broken)}\n# error={expected_error}\n# text_at_error={window!r}"
    )
    assert body == broken
    assert f"error={expected_error}, text_head=" in str(excinfo.value)
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_every_failed_attempt_is_dumped_even_when_a_retry_succeeds(tmp_path: Path) -> None:
    """Measured 2026-10-02 after the first fix: a retain needed 5 attempts, the 4
    rejected responses were gone with the retry — the cause stayed invisible."""
    broken = '```json\n{"ok": true, "note": "x"", "dangling": }\n```'
    llm, create = _llm(
        [_response(broken, "end_turn", output_tokens=20), _response('{"ok": true, "note": "fine"}', "end_turn")],
        dump_dir=tmp_path,
    )

    result = await llm.call(
        messages=[{"role": "user", "content": "extract"}],
        response_format=_Note,
        max_retries=3,
        initial_backoff=0.0,
    )

    assert result == _Note(ok=True, note="fine")
    assert create.call_count == 2
    dumps = sorted(tmp_path.iterdir())
    assert [p.name.split("-")[1] for p in dumps] == ["attempt1of4"]
    assert dumps[0].read_text(encoding="utf-8").endswith(broken)


# Defects the repair must NOT touch: escaping the quote before the error would turn
# them into a different, "valid" object. The original parser error has to surface.
_UNREPAIRABLE = [
    pytest.param(
        '{"ok": true "note": "x"}',
        r"error=Expecting ',' delimiter: line 1 column 13 \(char 12\), text_head=",
        id="missing-comma",
    ),
    pytest.param(
        '{"ok": true, note": "x"}',
        r"error=Expecting property name enclosed in double quotes: line 1 column 14 \(char 13\), text_head=",
        id="key-without-opening-quote",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "error_pattern"), _UNREPAIRABLE)
async def test_structural_defects_are_not_repaired_into_a_different_object(
    text: str, error_pattern: str, tmp_path: Path
) -> None:
    llm, create = _llm([_response(text, "end_turn")], dump_dir=tmp_path)

    with pytest.raises(ValueError, match=error_pattern):
        await llm.call(
            messages=[{"role": "user", "content": "extract"}],
            response_format=_Note,
            max_retries=0,
            initial_backoff=0.0,
        )

    assert create.call_count == 1
