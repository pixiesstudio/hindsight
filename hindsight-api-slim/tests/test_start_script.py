"""
Tests for start.sh --daemon: a slow start is not a failure, and no second instance.

Measured 2026-11-01: a cold restart took 150 s until the port was bound (embedded
PostgreSQL 31.6 s, embedding model 93.8 s), while start.sh gave up after 90 s and
printed "FAILED" with exit 1 for a start that was running fine. The @reboot
autostart ignores that exit code, polls on its own and then calls start.sh a
second time; start.sh only checked /health, so a still-starting first instance
was invisible and a second one would compete for port 8888 and PostgreSQL.

start.sh runs here against a fake `uv` in a temporary HOME and a free port, so the
live instance on 8888 is never touched.
"""

import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

START_SH = Path(__file__).resolve().parents[1] / "start.sh"

FAKE_UV = """#!/usr/bin/env bash
echo "call" >> "$HOME/uv_calls"
case "$FAKE_UV_MODE" in
  slow)  exec sleep 30 ;;
  crash) exit 1 ;;
  serve) exec python3 -m http.server "$HINDSIGHT_PORT" --bind 127.0.0.1 --directory "$HOME/www" ;;
esac
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sandbox(tmp_path):
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / "www").mkdir()
    (home / "www" / "health").write_text("ok", encoding="utf-8")
    uv = home / ".local" / "bin" / "uv"
    uv.write_text(FAKE_UV, encoding="utf-8")
    uv.chmod(0o755)
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "HINDSIGHT_PORT": str(_free_port()),
        "HINDSIGHT_START_WAIT_SECONDS": "2",
    }
    started: list[int] = []
    yield home, env, started
    for pid in started:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run(env: dict, mode: str, started: list[int]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["bash", str(START_SH), "--daemon"],
        env={**env, "FAKE_UV_MODE": mode},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    started.extend(int(p) for p in re.findall(r"PID (\d+)", proc.stdout))
    return proc


def _calls(home: Path) -> int:
    path = home / "uv_calls"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_slow_start_reports_still_starting_not_failed(sandbox):
    home, env, started = sandbox
    proc = _run(env, "slow", started)
    assert proc.returncode == 3
    assert "FAILED" not in proc.stdout
    last = proc.stdout.strip().splitlines()[-1]
    assert re.fullmatch(r"Hindsight: STILL STARTING after 2s \(PID \d+ alive\) — not failed, check .+hindsight\.log", last)
    assert _alive(started[0]) is True


def test_crash_still_reports_crashed(sandbox):
    home, env, started = sandbox
    proc = _run(env, "crash", started)
    assert proc.returncode == 1
    assert re.fullmatch(r"Hindsight: CRASHED after \ds — check .+hindsight\.log", proc.stdout.strip().splitlines()[-1])


def test_healthy_start_reports_ok(sandbox):
    home, env, started = sandbox
    proc = _run(env, "serve", started)
    assert proc.returncode == 0
    assert re.fullmatch(r"Hindsight: OK \(\ds\)", proc.stdout.strip().splitlines()[-1])


def test_second_call_while_starting_does_not_start_twice(sandbox):
    home, env, started = sandbox
    first = _run(env, "slow", started)
    second = _run(env, "slow", started)
    assert first.returncode == 3
    assert second.returncode == 3
    assert _calls(home) == 1
    assert re.search(rf"already starting \(PID {started[0]}\) — not starting a second instance", second.stdout)


def test_stale_pid_file_is_ignored(sandbox):
    home, env, started = sandbox
    dead = subprocess.Popen(["true"])
    dead.wait()
    pid_file = home / ".local" / "share" / "hindsight" / "logs" / "hindsight.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(f"{dead.pid}\n", encoding="utf-8")
    time.sleep(0.1)
    proc = _run(env, "crash", started)
    assert _calls(home) == 1
    assert proc.returncode == 1
    assert "already starting" not in proc.stdout
