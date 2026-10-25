"""
Tests for Anthropic structured outputs (``output_config.format``) per scope.

Hindsight asked for JSON only in the prompt and repaired the answer afterwards.
Measured 2026-10-23/24 on claude-haiku-5-5 (MIRA spikes hindsight-json-*): fact
extraction returned strict JSON in 5-7 of 108 answers when only asked, and in
1728 of 1728 with ``output_config.format``; consolidation 34 of 36 vs 72 of 72.
The schema text in the system prompt is dropped when the format is sent (10 %
cheaper, same facts). Scopes outside the list keep the old prompt path, because
reflect_structured builds its schema from caller input that the API may reject.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anthropic.lib._parse._transform import transform_schema
from pydantic import BaseModel

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

_ENV_KEYS = ("HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES", "HINDSIGHT_API_LLM_PROVIDER")
_SCHEMA_MARKER = "\n\nYou must respond with valid JSON matching this schema:\n"


class _Facts(BaseModel):
    facts: list[str]


_FORMAT = {"type": "json_schema", "schema": transform_schema(_Facts.model_json_schema())}
_PROMPT_SCHEMA_TEXT = f"{_SCHEMA_MARKER}{json.dumps(_Facts.model_json_schema(), indent=2)}"


@pytest.fixture(autouse=True)
def restore_env():
    """Restore the touched environment variables and the config cache."""
    from hindsight_api.config import clear_config_cache

    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    clear_config_cache()
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    clear_config_cache()


def _llm(
    scopes: tuple[str, ...] | None,
    effort: str = "high",
    text: str = '{"facts": ["a"]}',
    stop_reason: str = "end_turn",
    dump_dir: str | None = None,
) -> tuple[AnthropicLLM, AsyncMock]:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-haiku-5-5",
        thinking="disabled",
        effort=effort,
        structured_output_scopes=scopes,
        invalid_json_dump_dir=dump_dir,
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    create = AsyncMock(return_value=response)
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


_MESSAGES = [{"role": "system", "content": "Extract."}, {"role": "user", "content": "Alice lives in Paris."}]


async def test_listed_scope_sends_format_and_drops_schema_text():
    llm, create = _llm(("retain_extract_facts", "consolidation"))

    result = await llm.call(messages=_MESSAGES, response_format=_Facts, scope="retain_extract_facts", max_retries=0)

    assert result == _Facts(facts=["a"])
    assert create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {"output_config": {"effort": "high", "format": _FORMAT}}
    assert kwargs["system"] == "Extract."


async def test_unlisted_scope_keeps_prompt_schema_text():
    llm, create = _llm(("retain_extract_facts", "consolidation"))

    await llm.call(messages=_MESSAGES, response_format=_Facts, scope="reflect_structured", max_retries=0)

    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {"output_config": {"effort": "high"}}
    assert kwargs["system"] == "Extract." + _PROMPT_SCHEMA_TEXT


async def test_empty_scope_list_restores_prompt_path_everywhere():
    llm, create = _llm(())

    await llm.call(messages=_MESSAGES, response_format=_Facts, scope="consolidation", max_retries=0)

    kwargs = create.await_args.kwargs
    assert kwargs["extra_body"] == {"output_config": {"effort": "high"}}
    assert kwargs["system"] == "Extract." + _PROMPT_SCHEMA_TEXT


async def test_format_without_effort_when_effort_is_model_default():
    llm, create = _llm(("consolidation",), effort="model_default")

    await llm.call(messages=_MESSAGES, response_format=_Facts, scope="consolidation", max_retries=0)

    assert create.await_args.kwargs["extra_body"] == {"output_config": {"format": _FORMAT}}


async def test_plain_text_call_never_sends_format():
    llm, create = _llm(("retain_extract_facts",), text="plain answer")

    result = await llm.call(messages=_MESSAGES, scope="retain_extract_facts", max_retries=0)

    assert result == "plain answer"
    assert create.await_args.kwargs["extra_body"] == {"output_config": {"effort": "high"}}
    assert create.await_args.kwargs["system"] == "Extract."


async def test_format_without_system_message_sends_no_system():
    llm, create = _llm(("consolidation",))

    await llm.call(
        messages=[{"role": "user", "content": "facts?"}], response_format=_Facts, scope="consolidation", max_retries=0
    )

    assert "system" not in create.await_args.kwargs


@pytest.mark.parametrize("scope", ["retain_extract_facts", "reflect_structured"])
async def test_refusal_raises_without_paid_retry(scope, tmp_path):
    llm, create = _llm(
        ("retain_extract_facts",), text="I can't help with that.", stop_reason="refusal", dump_dir=str(tmp_path)
    )

    with pytest.raises(ValueError, match="Anthropic refused the request"):
        await llm.call(messages=_MESSAGES, response_format=_Facts, scope=scope, max_retries=3)

    assert create.await_count == 1


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, ("retain_extract_facts", "consolidation")),
        ("", ()),
        ("consolidation", ("consolidation",)),
        (" retain_extract_facts , consolidation ", ("retain_extract_facts", "consolidation")),
    ],
)
def test_config_scopes(env_value, expected):
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    if env_value is None:
        os.environ.pop("HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES", None)
    else:
        os.environ["HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES"] = env_value

    assert HindsightConfig.from_env().llm_anthropic_structured_output_scopes == expected


@pytest.mark.parametrize("bad_value", ["Consolidation", "retain extract", "consolidation;retain_extract_facts", "a-b"])
def test_config_rejects_malformed_scopes(bad_value):
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES"] = bad_value

    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES must be"):
        HindsightConfig.from_env()


async def test_unset_constructor_value_resolves_from_global_config():
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_LLM_ANTHROPIC_STRUCTURED_OUTPUT_SCOPES"] = "consolidation"
    llm, create = _llm(None)

    await llm.call(messages=_MESSAGES, response_format=_Facts, scope="consolidation", max_retries=0)

    assert create.await_args.kwargs["extra_body"]["output_config"]["format"] == _FORMAT
