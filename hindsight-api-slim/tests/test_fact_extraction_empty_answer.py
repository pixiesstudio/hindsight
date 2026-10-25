"""
An empty extraction answer must return no facts, also without an event date.

The debug log for an empty answer formatted ``event_date.isoformat()``. The
orchestrator passes ``event_date=None`` when a caller sends an explicit unknown date
(retain/orchestrator.py, "Caller explicitly signalled unknown date"), so an empty
model answer raised AttributeError instead of returning [] — the exception is not
caught by the retry loop, which only handles BadRequestError. Found 2026-10-24 by a
stub self-test in the MIRA language spike.
"""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_LOGGER = "hindsight_api.engine.retain.fact_extraction"


def _make_config():
    """Build a minimal HindsightConfig for fact extraction tests."""
    from hindsight_api.config import HindsightConfig

    cfg = MagicMock(spec=HindsightConfig)
    cfg.retain_llm_max_retries = None
    cfg.llm_max_retries = 3
    cfg.retain_llm_initial_backoff = None
    cfg.llm_initial_backoff = 0.0
    cfg.retain_llm_max_backoff = None
    cfg.llm_max_backoff = 0.0
    cfg.retain_max_completion_tokens = 8192
    cfg.retain_extraction_mode = "concise"
    cfg.retain_extract_causal_links = False
    cfg.retain_mission = None
    cfg.retain_language_hint = False
    return cfg


def _make_llm_config(mock_response):
    """Build a mock LLMProvider that returns the given response."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    token_usage = MagicMock()
    token_usage.__add__ = lambda self, other: self
    llm.call = AsyncMock(return_value=(mock_response, token_usage))
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{"facts": []}, {}], ids=["empty-list", "no-facts-key"])
@pytest.mark.parametrize(
    ("event_date", "logged_date"),
    [
        (None, "date: unknown"),
        (datetime(2023, 1, 1, tzinfo=timezone.utc), "date: 2023-01-01T00:00:00+00:00"),
    ],
    ids=["no-date", "with-date"],
)
async def test_empty_answer_returns_no_facts(response, event_date, logged_date, caplog):
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    llm_config = _make_llm_config(mock_response=response)

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_from_chunk(
            chunk="Nothing worth keeping here.",
            chunk_index=0,
            total_chunks=1,
            event_date=event_date,
            context="notes",
            llm_config=llm_config,
            config=_make_config(),
            agent_name="test-agent",
        )

    assert facts == []
    assert llm_config.call.await_count == 1
    empty_logs = [r.getMessage() for r in caplog.records if "returned empty list" in r.getMessage()]
    assert len(empty_logs) == 1
    assert logged_date in empty_logs[0]
