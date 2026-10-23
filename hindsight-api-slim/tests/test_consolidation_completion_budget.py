"""
Tests for the consolidation LLM output budget.

Without an explicit budget the Anthropic provider falls back to 4096 output
tokens. On claude-haiku-5-5 consolidation answers reach 6160 output tokens
(measured 2026-10-23: median 3403, p95 5486, max 6160 over 43 slow calls), so
the JSON was cut at 4096 and every such call was paid twice: once truncated,
once again with the escalated budget.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_ENV_KEYS = (
    "HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS",
    "HINDSIGHT_API_LLM_PROVIDER",
)


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


def _llm_config_returning_empty_batch() -> MagicMock:
    from hindsight_api.engine.consolidation.consolidator import _ConsolidationBatchResponse

    llm_config = MagicMock()
    llm_config.call = AsyncMock(return_value=_ConsolidationBatchResponse())
    return llm_config


_MEMORIES = [{"id": "fact-1", "text": "The consolidation budget is configurable."}]


@pytest.mark.parametrize("budget", [64000, 12000])
async def test_consolidation_call_passes_configured_budget(budget):
    from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm

    llm_config = _llm_config_returning_empty_batch()
    config = SimpleNamespace(observations_mission=None, consolidation_max_completion_tokens=budget)

    result = await _consolidate_batch_with_llm(llm_config, _MEMORIES, [], {}, config=config)

    assert llm_config.call.await_count == 1
    kwargs = llm_config.call.await_args.kwargs
    assert kwargs["max_completion_tokens"] == budget
    assert kwargs["scope"] == "consolidation"
    assert result.failed is False


async def test_consolidation_call_without_config_uses_default_budget_not_provider_fallback():
    from hindsight_api.config import DEFAULT_CONSOLIDATION_MAX_COMPLETION_TOKENS
    from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm

    llm_config = _llm_config_returning_empty_batch()

    await _consolidate_batch_with_llm(llm_config, _MEMORIES, [], {}, config=None)

    assert llm_config.call.await_args.kwargs["max_completion_tokens"] == DEFAULT_CONSOLIDATION_MAX_COMPLETION_TOKENS


def test_consolidation_budget_default_is_64000():
    from hindsight_api.config import DEFAULT_CONSOLIDATION_MAX_COMPLETION_TOKENS, HindsightConfig

    os.environ.pop("HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS", None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert DEFAULT_CONSOLIDATION_MAX_COMPLETION_TOKENS == 64000
    assert config.consolidation_max_completion_tokens == 64000


def test_consolidation_budget_env_override():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS"] = "32000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert config.consolidation_max_completion_tokens == 32000


def test_consolidation_budget_rejects_non_integer_value():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS"] = "lots"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    with pytest.raises(ValueError, match="invalid literal for int"):
        HindsightConfig.from_env()
