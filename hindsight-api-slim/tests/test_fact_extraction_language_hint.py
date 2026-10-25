"""
The extraction request names the language of the text, detected in code.

claude-haiku-5-5 follows the language of the (English) instructions rather than of the
text: measured 2026-10-24/25 on the real extraction path (MIRA spikes hindsight-json-*),
German chunks came back German in 0.81-0.86 of answers with thinking off. With one line
naming the language at the end of the user message, 1.000 for German and English
chunks, facts +0.28 per extraction, details unchanged, +3 % cost. Thinking reached the
same language but lost 19 % of facts (default effort) or cost 80 % more (high effort).
User decision 2026-10-25: source language per line. Unclear text gets no line.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.retain.fact_extraction import _build_user_message, _detect_text_language

_DE_HINT = "\n\nSprache des Texts: Deutsch. Schreibe alle Fakten auf Deutsch."
_EN_HINT = "\n\nLanguage of the text: English. Write all facts in English."
_DE_TEXT = (
    "Der Hook wurde gestern geändert, weil die Bank nicht mehr erkannt wurde und alle Einträge in global landeten."
)
_EN_TEXT = "The hook was changed yesterday because the bank was no longer detected and all entries went to global."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_DE_TEXT, "de"),
        (_EN_TEXT, "en"),
        ("Die Funktion hindsight_project_bank liefert global statt mira für das Repo.", "de"),
        ("x = foo(bar, 42)  # TODO", None),
        ("mira global", None),
        ("", None),
    ],
    ids=["german", "english", "german-with-identifiers", "code-only", "two-nouns", "empty"],
)
def test_detect_text_language(text, expected):
    assert _detect_text_language(text) == expected


@pytest.mark.parametrize(("text", "hint"), [(_DE_TEXT, _DE_HINT), (_EN_TEXT, _EN_HINT)], ids=["german", "english"])
def test_user_message_ends_with_language_line(text, hint):
    message = _build_user_message(text, 0, 1, None, "notes", language_hint=True)

    assert message.endswith(f"Text:\n{text}{hint}")


def test_unclear_text_gets_no_language_line():
    message = _build_user_message("x = foo(bar, 42)", 0, 1, None, "notes", language_hint=True)

    assert message.endswith("Text:\nx = foo(bar, 42)")


def test_switch_off_keeps_the_old_message_exactly():
    with_flag_off = _build_user_message(_DE_TEXT, 0, 1, None, "notes", language_hint=False)
    default = _build_user_message(_DE_TEXT, 0, 1, None, "notes")

    assert with_flag_off == default
    assert with_flag_off.endswith(f"Text:\n{_DE_TEXT}")


def _make_config(language_hint: bool):
    from hindsight_api.config import HindsightConfig

    cfg = MagicMock(spec=HindsightConfig)
    cfg.retain_llm_max_retries = None
    cfg.llm_max_retries = 1
    cfg.retain_llm_initial_backoff = None
    cfg.llm_initial_backoff = 0.0
    cfg.retain_llm_max_backoff = None
    cfg.llm_max_backoff = 0.0
    cfg.retain_max_completion_tokens = 8192
    cfg.retain_extraction_mode = "concise"
    cfg.retain_extract_causal_links = False
    cfg.retain_mission = None
    cfg.retain_language_hint = language_hint
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize(("language_hint", "expected_end"), [(True, _DE_HINT), (False, _DE_TEXT)], ids=["on", "off"])
async def test_extraction_path_sends_the_line_per_config(language_hint, expected_end):
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    usage = MagicMock()
    usage.__add__ = lambda self, other: self
    llm.call = AsyncMock(return_value=({"facts": []}, usage))

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        await _extract_facts_from_chunk(
            chunk=_DE_TEXT,
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2026, 10, 25, tzinfo=timezone.utc),
            context="notes",
            llm_config=llm,
            config=_make_config(language_hint),
        )

    user_message = llm.call.await_args.kwargs["messages"][1]["content"]
    assert user_message.endswith(expected_end)


@pytest.mark.parametrize(("env_value", "expected"), [(None, True), ("false", False), ("true", True), ("0", False)])
def test_config_language_hint(env_value, expected, monkeypatch):
    from hindsight_api.config import HindsightConfig, clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    if env_value is None:
        monkeypatch.delenv("HINDSIGHT_API_RETAIN_LANGUAGE_HINT", raising=False)
    else:
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_LANGUAGE_HINT", env_value)
    clear_config_cache()

    assert HindsightConfig.from_env().retain_language_hint is expected
    clear_config_cache()
