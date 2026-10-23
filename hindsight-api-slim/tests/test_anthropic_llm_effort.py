"""
Tests for the Anthropic effort switch.

Effort is the main control for how much claude-haiku-5-5 thinks (default medium);
"high" is documented for strict instruction following. anthropic SDK 0.75.0 has no
``output_config`` argument on ``messages.create``, so the value travels in
``extra_body``. xhigh and max are not offered: combined with thinking disabled
they return a 400.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

_ENV_KEYS = ("HINDSIGHT_API_LLM_ANTHROPIC_EFFORT", "HINDSIGHT_API_LLM_PROVIDER")


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


def _llm(effort: str | None, thinking: str = "model_default") -> tuple[AnthropicLLM, AsyncMock]:
    llm = AnthropicLLM(
        provider="anthropic",
        api_key="test-key",
        base_url="",
        model="claude-haiku-5-5",
        thinking=thinking,
        effort=effort,
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    create = AsyncMock(return_value=response)
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm, create


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_call_sends_effort_in_extra_body(effort):
    llm, create = _llm(effort)

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert result == "done"
    assert create.await_count == 1
    assert create.await_args.kwargs["extra_body"] == {"output_config": {"effort": effort}}


async def test_call_omits_effort_for_model_default():
    llm, create = _llm("model_default")

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert "extra_body" not in create.await_args.kwargs


async def test_effort_high_and_thinking_disabled_are_sent_together():
    llm, create = _llm("high", thinking="disabled")

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["extra_body"] == {"output_config": {"effort": "high"}}


@pytest.mark.parametrize(
    ("env_value", "expected_extra_body"),
    [(None, None), ("model_default", None), ("low", {"output_config": {"effort": "low"}})],
)
async def test_unset_constructor_value_resolves_from_global_config(env_value, expected_extra_body):
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    if env_value is None:
        os.environ.pop("HINDSIGHT_API_LLM_ANTHROPIC_EFFORT", None)
    else:
        os.environ["HINDSIGHT_API_LLM_ANTHROPIC_EFFORT"] = env_value
    llm, create = _llm(None)

    await llm.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)

    assert create.await_args.kwargs.get("extra_body") == expected_extra_body


def test_config_default_is_model_default():
    from hindsight_api.config import DEFAULT_LLM_ANTHROPIC_EFFORT, HindsightConfig

    os.environ.pop("HINDSIGHT_API_LLM_ANTHROPIC_EFFORT", None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert DEFAULT_LLM_ANTHROPIC_EFFORT == "model_default"
    assert config.llm_anthropic_effort == "model_default"


@pytest.mark.parametrize("bad_value", ["max", "xhigh", "High", ""])
def test_config_rejects_unsupported_effort(bad_value):
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_ANTHROPIC_EFFORT"] = bad_value
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    with pytest.raises(ValueError, match="HINDSIGHT_API_LLM_ANTHROPIC_EFFORT must be one of"):
        HindsightConfig.from_env()
