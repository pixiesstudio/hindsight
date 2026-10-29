"""
Starting with a --log-level other than the config's must not crash, and must keep every
other setting.

main.py rebuilt the config by listing all fields by hand when --log-level differed from
the config. Two fields added in March 2026 (embeddings_local_default_task,
reflect_wall_timeout) were never added to that list, so the rebuild raised
"HindsightConfig.__init__() missing 2 required positional arguments" and the server did
not start. Measured 2026-10-28 by calling the real constructor with the 176 listed fields.
The existing main() tests replace the config with a MagicMock and never pass --log-level,
so they could not reach the branch; these tests use a real HindsightConfig.
"""

from __future__ import annotations

import dataclasses
import sys
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache

_CLI_FIELDS = {"host", "port", "log_level"}


@pytest.fixture
def real_config(monkeypatch):
    """A real config from the environment, with a known log level and the mock LLM."""
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LOG_LEVEL", "info")
    monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")
    clear_config_cache()
    config = HindsightConfig.from_env()
    yield config
    clear_config_cache()


def _run_main(config: HindsightConfig, argv: list[str]):
    """Run main() up to uvicorn.run with everything that would start or bind replaced."""
    with (
        patch("hindsight_api.main._get_raw_config", return_value=config),
        patch("hindsight_api.main.MemoryEngine", return_value=MagicMock()),
        patch("hindsight_api.main.create_app", return_value=MagicMock()),
        patch("hindsight_api.main.DefaultExtensionContext"),
        patch("hindsight_api.main.print_banner"),
        patch("hindsight_api.banner.print_startup_info"),
        patch("hindsight_api.main.atexit.register"),
        patch("hindsight_api.main.signal.signal"),
        patch.object(HindsightConfig, "configure_logging", autospec=True) as configure_logging,
        patch.object(HindsightConfig, "log_config", autospec=True),
        patch("uvicorn.run") as uvicorn_run,
        patch.object(sys, "argv", argv),
    ):
        from hindsight_api.main import main

        main()
    return configure_logging, uvicorn_run


def test_other_log_level_starts_and_reaches_uvicorn(real_config):
    configure_logging, uvicorn_run = _run_main(real_config, ["hindsight-api", "--log-level", "debug", "--port", "8899"])

    assert uvicorn_run.call_count == 1
    assert uvicorn_run.call_args.kwargs["log_level"] == "debug"
    assert uvicorn_run.call_args.kwargs["port"] == 8899
    configure_logging.assert_called_once()


def test_rebuilt_config_changes_only_the_cli_fields(real_config):
    configure_logging, _ = _run_main(real_config, ["hindsight-api", "--log-level", "debug", "--port", "8899"])

    rebuilt = configure_logging.call_args.args[0]
    assert rebuilt is not real_config
    assert (rebuilt.log_level, rebuilt.port) == ("debug", 8899)
    unchanged = {f.name for f in dataclasses.fields(HindsightConfig)} - _CLI_FIELDS
    differing = sorted(name for name in unchanged if getattr(rebuilt, name) != getattr(real_config, name))
    assert differing == []


def test_same_log_level_keeps_the_original_config(real_config):
    configure_logging, uvicorn_run = _run_main(real_config, ["hindsight-api", "--log-level", "info"])

    assert configure_logging.call_args.args[0] is real_config
    assert uvicorn_run.call_args.kwargs["log_level"] == "info"
