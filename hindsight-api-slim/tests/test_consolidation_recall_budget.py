"""
Tests for the token budget of consolidation's related-observation recall.

Consolidation looks up existing observations similar to each new fact and hands
them to the LLM. That lookup runs quiet and never appears in the log, so it was
replayed (2026-11-01) with its exact shape: 60 real facts as query, their tags
with all_strict, observations only, budget 300. At the former default of 512
tokens the result had median 476 tokens and 57 of 60 lookups were cut; without
the cap median 670 and max 995 tokens, never more than 1024, so 4096 (the
server's own recall default) cuts none of them.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_ENV_KEYS = (
    "HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS",
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


def _memory_engine_with_budget(budget: int) -> MagicMock:
    config = SimpleNamespace(
        consolidation_max_tokens=budget,
        consolidation_source_facts_max_tokens=-1,
        consolidation_source_facts_max_tokens_per_observation=256,
    )
    engine = MagicMock()
    engine._config_resolver.resolve_full_config = AsyncMock(return_value=config)
    engine.recall_async = AsyncMock(return_value="recall-result")
    return engine


@pytest.mark.parametrize("budget", [4096, 2048])
async def test_related_observation_recall_passes_configured_budget(budget):
    from hindsight_api.engine.consolidation.consolidator import _find_related_observations

    engine = _memory_engine_with_budget(budget)
    request_context = object()

    result = await _find_related_observations(engine, "mira", "a new fact", request_context, tags=["mira", "hooks"])

    assert result == "recall-result"
    engine.recall_async.assert_awaited_once()
    kwargs = engine.recall_async.await_args.kwargs
    assert kwargs["max_tokens"] == budget
    assert kwargs["fact_type"] == ["observation"]
    assert kwargs["tags"] == ["mira", "hooks"]
    assert kwargs["tags_match"] == "all_strict"
    assert kwargs["max_source_facts_tokens_per_observation"] == 256


async def test_related_observation_recall_without_tags_matches_any():
    from hindsight_api.engine.consolidation.consolidator import _find_related_observations

    engine = _memory_engine_with_budget(4096)

    await _find_related_observations(engine, "global", "untagged fact", object(), tags=None)

    kwargs = engine.recall_async.await_args.kwargs
    assert kwargs["tags_match"] == "any"
    assert kwargs["tags"] is None


def test_recall_budget_default_is_4096():
    from hindsight_api.config import DEFAULT_CONSOLIDATION_MAX_TOKENS, HindsightConfig

    os.environ.pop("HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS", None)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert DEFAULT_CONSOLIDATION_MAX_TOKENS == 4096
    assert config.consolidation_max_tokens == 4096


def test_recall_budget_env_override():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS"] = "1024"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()

    assert config.consolidation_max_tokens == 1024


def test_recall_budget_rejects_non_integer_value():
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS"] = "plenty"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    with pytest.raises(ValueError, match="invalid literal for int"):
        HindsightConfig.from_env()
