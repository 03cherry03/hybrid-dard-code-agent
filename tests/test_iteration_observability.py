import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from iteration_observability import log_event, safe_command, verify_event_stream


def test_concurrent_events_are_complete_ordered_private_and_verifiable(tmp_path):
    def write(index):
        return log_event(tmp_path, "grading_completed", index=index, verdict={"passed": index % 2 == 0})
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(write, range(100)))
    assert len(set(paths)) == 1
    path = Path(paths[0])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "process_metadata"
    assert {row["details"]["index"] for row in rows[1:]} == set(range(100))
    assert verify_event_stream(path)["records"] == 101
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_tampering_and_truncated_records_are_detected(tmp_path):
    path = Path(log_event(tmp_path, "complete", passed=True))
    payload = path.read_text()
    path.write_text(payload.replace('"passed": true', '"passed": false'))
    with pytest.raises(ValueError, match="hash chain"):
        verify_event_stream(path)
    path.write_text(payload[:-5])
    with pytest.raises(ValueError):
        verify_event_stream(path)
    recovered = verify_event_stream(path, allow_incomplete_tail=True)
    assert recovered["records"] == 1 and recovered["incomplete_tail_bytes"] > 0


def test_credentials_scrubbed_without_removing_token_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_PASSWORD", "never-dump-environment")
    path = Path(log_event(tmp_path, "generation", password="credential-one",
        command=["python", "--api-key=credential-two", "--access-token", "credential-three", "--max-tokens", "512"],
        auth="Bearer credential-four", url="https://u:p@example.com/?token=credential-five&x=1",
        token_ids=[42, 43], forced_tokens=2))
    payload = path.read_text()
    assert all(secret not in payload for secret in
        ("never-dump-environment", "credential-one", "credential-two", "credential-three", "credential-four", "credential-five", "u:p"))
    record = json.loads(payload.splitlines()[-1])["details"]
    assert record["token_ids"] == [42, 43] and record["forced_tokens"] == 2
    assert record["command"][-2:] == ["--max-tokens", "512"]


def test_nonfinite_metrics_do_not_break_failure_logging(tmp_path):
    path = log_event(tmp_path, "nonfinite_loss", loss=float("nan"), grad=float("inf"), path=tmp_path)
    assert verify_event_stream(path)["records"] == 2
    row = json.loads(Path(path).read_text().splitlines()[-1])["details"]
    assert row["loss"] == "nan" and row["grad"] == "inf"


def test_new_process_uses_an_independent_stream(tmp_path, monkeypatch):
    first = log_event(tmp_path, "first")
    original = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: original + 1)
    second = log_event(tmp_path, "second")
    assert first != second
    assert verify_event_stream(first)["records"] == verify_event_stream(second)["records"] == 2


def test_inline_and_mapping_credentials_are_removed(tmp_path):
    path = log_event(tmp_path, "failure", hf_token="FAKE_SENTINEL_A",
        error="HF_TOKEN=FAKE_SENTINEL_B password='FAKE_SENTINEL_C' \"api_key\": \"FAKE_SENTINEL_D\"",
        token=42, token_ids=[1, 2])
    payload = Path(path).read_text()
    assert "FAKE_SENTINEL" not in payload
    assert json.loads(payload.splitlines()[-1])["details"]["token"] == 42


def test_write_failure_preserves_caller_and_creates_recovery_stream(tmp_path, monkeypatch, capsys):
    original = Path(log_event(tmp_path, "before"))
    real_write = os.write
    calls = [0]
    def fail_once(fd, data):
        calls[0] += 1
        if calls[0] == 1:
            raise OSError("simulated storage failure")
        return real_write(fd, data)
    monkeypatch.setattr(os, "write", fail_once)
    assert log_event(tmp_path, "lost_event") is None
    assert "logging_write_failed" in capsys.readouterr().err
    health = json.loads((tmp_path / "logging_health.json").read_text())
    assert health["status"] == "degraded" and health["failed_event"] == "lost_event"
    recovered = Path(log_event(tmp_path, "after"))
    assert original != recovered
    assert verify_event_stream(recovered)["records"] == 2


def test_same_thread_reentry_is_deferred_until_outer_event_finishes(tmp_path, monkeypatch):
    path = log_event(tmp_path, "before")
    real_write = os.write
    nested = [False]
    def reenter(fd, data):
        if not nested[0]:
            nested[0] = True
            assert log_event(tmp_path, "nested_signal_like_call") is None
        return real_write(fd, data)
    monkeypatch.setattr(os, "write", reenter)
    log_event(tmp_path, "outer")
    assert verify_event_stream(path)["records"] == 4
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [row["event"] for row in rows[-2:]] == ["outer", "nested_signal_like_call"]


def test_signal_interruption_is_never_swallowed_by_logging_fallback(tmp_path, monkeypatch):
    previous = log_event(tmp_path, "before_signal")
    original_write = os.write
    def interrupt(fd, data):
        raise InterruptedError("requested shutdown")
    monkeypatch.setattr(os, "write", interrupt)
    with pytest.raises(InterruptedError, match="requested shutdown"):
        log_event(tmp_path, "interrupted")
    monkeypatch.setattr(os, "write", original_write)
    recovered = log_event(tmp_path, "outer_failure_handler")
    assert recovered != previous
    assert verify_event_stream(recovered)["records"] == 2
