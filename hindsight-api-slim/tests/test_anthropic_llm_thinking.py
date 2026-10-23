"""
Tests for the Anthropic thinking switch.

claude-haiku-5-5 thinks by default when the request omits ``thinking`` (adaptive,
effort medium), and thinking tokens are billed as output tokens. Measured
2026-10-23 in the invalid-JSON dumps: one consolidation answer used 4096 output
tokens for 0 characters of text, another 4096 tokens for 417 characters. Hindsight
ran without thinking on Haiku 4.5, so the default switches it off again.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

_ENV_KEYS = ("HINDSIGHT_API_LLM_ANTHROPIC_THINKING", "HINDSIGHT_API_LLM_PROVIDER")


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


def _llm(thinking: str | None) -> tuple[AnthropicLLM, AsyncMock]:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-haiku-5-5",
        thinking=thinking,
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    create = AsyncMock(return_value=response)
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


async def test_call_sends_thinking_disabled_when_configured():
    llm, create = _llm("disabled")

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result == "done"
    assert create.await_count == 1
    assert create.await_args.kwargs["thinking"] == {"type": "disabled"}


async def test_call_omits_thinking_for_model_default():
    llm, create = _llm("model_default")

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert create.await_count == 1
    assert "thinking" not in create.await_args.kwargs


@pytest.mark.parametrize(
    ("env_value", "expected_thinking"),
    [(None, {"type": "disabled"}), ("disabled", {"type": "disabled"}), ("model_default", None)],
)
async def test_unset_constructor_value_resolves_from_global_config(env_value, expected_thinking):
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    if env_value is None:
        os.environ.pop("HINDSIGHT_API_LLM_ANTHROPIC_THINKING", None)
    else:
        os.environ["HINDSIGHT_API_LLM_ANTHROPIC_THINKING"] = env_value
    llm, create = _llm(None)

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert create.await_args.kwargs.get("thinking") == expected_thinking


def test_config_default_is_disabled():
    from hindsight_api.config import DEFAULT_LLM_ANTHROPIC_THINKING, HindsightConfig

    os.environ.pop("HINDSIGHT_API_LLM_ANTHROPIC_THINKING", None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert DEFAULT_LLM_ANTHROPIC_THINKING == "disabled"
    assert config.llm_anthropic_thinking == "disabled"


@pytest.mark.parametrize("bad_value", ["off", "Disabled", ""])
def test_config_rejects_unknown_thinking_value(bad_value):
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_ANTHROPIC_THINKING"] = bad_value
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_ANTHROPIC_THINKING must be one of"):
        HindsightConfig.from_env()
