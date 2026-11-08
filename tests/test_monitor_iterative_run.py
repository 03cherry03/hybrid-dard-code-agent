"""Read-only telemetry, privacy, indexing, and completion without live services."""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import monitor_iterative_run as monitor


def service(*, active="active", pid=100, group="/fixture"):
    return {"available": True, "Id": monitor.DEFAULT_SERVICE, "LoadState": "loaded",
            "ActiveState": active, "SubState": "running" if active == "active" else "dead",
            "MainPID": pid, "ControlGroup": group, "NRestarts": 0,
            "Result": "success", "ExecMainStatus": 0}


def runner_for(state, calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["timeout"] == 3
        assert command[0] in {"systemctl", "nvidia-smi"}
        if command[0] == "systemctl":
            assert command[1:3] == ["--user", "show"]
            return SimpleNamespace(returncode=0, stdout="\n".join(f"{k}={state[k]}" for k in monitor.SERVICE_FIELDS))
        return SimpleNamespace(returncode=0, stdout="0, GPU-fixture, Test GPU, 91, 1234, 24000, 63, 210.5\n")
    return run


class EmptyProcesses:
    def sample(self, state):
        if not state.get("available"):
            raise ValueError("No ownership receipt")
        return {"available": True, "member_pids": [], "processes": [], "errors": []}


@pytest.fixture
def captured_events(monkeypatch):
    captured = []
    def log(output, event, **fields):
        captured.append((Path(output), event, fields))
        return str(Path(output)/"events/fixture.jsonl")
    monkeypatch.setattr(monitor, "log_event", log)
    return captured


def write_proc(root, pid, *, cpu=100, start=123, rss=4, command=None):
    path = root/str(pid)
    path.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 50
    fields[0], fields[1] = "R", "1"
    fields[11], fields[12], fields[17] = str(cpu), "0", "2"
    fields[19], fields[21] = str(start), str(rss)
    (path/"stat").write_text(f"{pid} (python (worker)) " + " ".join(fields))
    (path/"cmdline").write_bytes(b"\0".join(x.encode() for x in (command or ["python", "worker.py"])) + b"\0")
    (path/"environ").write_text("DO_NOT_READ_PROCESS_ENVIRONMENT")


def test_service_and_gpu_queries_are_read_only_and_bounded():
    calls = []
    runner = runner_for(service(), calls)
    observed = monitor.sample_service(monitor.DEFAULT_SERVICE, runner=runner)
    gpu = monitor.sample_gpus(runner=runner)
    assert observed["MainPID"] == 100
    assert gpu["devices"][0]["temperature_celsius"] == 63
    assert gpu["devices"][0]["power_watts"] == 210.5
    assert len(calls) == 2
    assert all(kwargs["timeout"] == 3 for _,kwargs in calls)
    assert not any(any(word in cmd for word in ("restart", "stop", "kill", "start")) for cmd,_ in calls)


def test_unloaded_service_is_not_mistaken_for_completed():
    state = service(active="inactive", pid=0, group="")
    state["LoadState"] = "not-found"
    with pytest.raises(ValueError, match="not loaded"):
        monitor.sample_service(monitor.DEFAULT_SERVICE, runner=runner_for(state, []))


def test_cgroup_cpu_rss_nested_membership_and_command_redaction(tmp_path):
    proc, cgroups = tmp_path/"proc", tmp_path/"cgroups"
    group = cgroups/"fixture"; (group/"child").mkdir(parents=True)
    (group/"cgroup.procs").write_text("100\n")
    (group/"child/cgroup.procs").write_text("101\n")
    write_proc(proc, 100, command=["python", "--api-key=secret-value", "HF_TOKEN=env-secret"])
    write_proc(proc, 101)
    sampler = monitor.ProcessSampler(proc_root=proc, cgroup_root=cgroups)
    first = sampler.sample(service(), monotonic_now=10)
    assert first["member_pids"] == [100, 101]
    assert first["processes"][0]["cpu_percent"] is None
    assert first["processes"][0]["rss_bytes"] == 4 * sampler.page_size
    assert "secret-value" not in json.dumps(first)
    assert "env-secret" not in json.dumps(first)
    assert "DO_NOT_READ_PROCESS_ENVIRONMENT" not in json.dumps(first)
    write_proc(proc, 100, cpu=100+2*sampler.clock_ticks)
    second = sampler.sample(service(), monotonic_now=12)
    assert second["processes"][0]["cpu_percent"] == pytest.approx(100)
    write_proc(proc, 100, cpu=10000, start=999)
    third = sampler.sample(service(), monotonic_now=14)
    assert third["processes"][0]["cpu_percent"] is None


def test_process_inventory_does_not_assume_missing_active_cgroup_is_empty(tmp_path):
    sampler = monitor.ProcessSampler(proc_root=tmp_path/"proc", cgroup_root=tmp_path/"groups")
    with pytest.raises(FileNotFoundError):
        sampler.sample(service())
    assert sampler.sample(service(active="inactive", pid=0, group=""))["member_pids"] == []
    with pytest.raises(ValueError, match="Unsafe"):
        sampler.sample(service(group="/../../other"))


def test_artifacts_hash_small_json_only_and_exclude_raw_source_symlinks_weights(tmp_path, monkeypatch):
    (tmp_path/"round-01").mkdir()
    small = tmp_path/"round-01/result.json"; small.write_text('{"passed":2}')
    log = tmp_path/"round-01/runs.jsonl"; log.write_text('{"code":"opaque"}\n')
    big = tmp_path/"audit/big.json"; big.parent.mkdir()
    with big.open("wb") as stream:
        stream.truncate(monitor.MAX_JSON_BYTES+1)
    binary = tmp_path/"round-01/adapter_model.safetensors"; binary.write_bytes(b"never-open-weights")
    for folder in ["source", "source_logging_v2", "logging", "data", "raw", "cache"]:
        path = tmp_path/folder/"private.json"; path.parent.mkdir(); path.write_text("{}")
    (tmp_path/"linked.json").symlink_to(small)
    (tmp_path/"linked-dir").symlink_to(tmp_path/"round-01", target_is_directory=True)
    real_open = os.open
    def guarded_open(path, flags, *args):
        assert Path(path) not in {log, big, binary}
        return real_open(path, flags, *args)
    monkeypatch.setattr(monitor.os, "open", guarded_open)
    indexer = monitor.ArtifactIndexer(tmp_path)
    first = indexer.scan()
    assert set(first["entries"]) == {"round-01/result.json", "round-01/runs.jsonl", "audit/big.json"}
    assert first["entries"]["round-01/result.json"]["sha256"] == hashlib.sha256(small.read_bytes()).hexdigest()
    assert first["entries"]["round-01/runs.jsonl"]["sha256"] is None
    assert first["entries"]["audit/big.json"]["sha256"] is None
    monkeypatch.setattr(indexer, "_entry", lambda *a: pytest.fail("Unchanged artifact was rehashed"))
    second = indexer.scan()
    assert second["changes"] == []
    assert second["generation"] == first["generation"]


def test_missing_artifact_requires_two_successful_scans_and_restoration(tmp_path):
    path = tmp_path/"decision.json"; path.write_text("{}")
    indexer = monitor.ArtifactIndexer(tmp_path)
    indexer.scan()
    path.unlink()
    missing = indexer.scan()
    assert missing["entries"]["decision.json"]["presence"] == "missing_pending"
    path.write_text("{}")
    restored = indexer.scan()
    assert restored["entries"]["decision.json"]["presence"] == "present"
    path.unlink()
    indexer.scan()
    removed = indexer.scan()
    assert "decision.json" not in removed["entries"]
    assert removed["changes"][0]["action"] == "removed"
    assert removed["changes"][0]["missing_scans"] == 2


def test_logging_health_receipt_is_indexed_without_sensitive_payload(tmp_path):
    path = tmp_path/"round-01/logging_health.json"; path.parent.mkdir()
    path.write_text(json.dumps({"status": "degraded", "error_type": "OSError",
                               "password": "secret", "environment": {"KEY": "private"}}))
    snapshot = monitor.ArtifactIndexer(tmp_path).scan()
    health = snapshot["entries"]["round-01/logging_health.json"]["logging_health"]
    assert health["status"] == "degraded"
    assert health["password"] == "[REDACTED]"
    assert health["environment"] == "[ENVIRONMENT OMITTED]"


def test_completion_requires_two_stable_inactive_pidless_completed_polls():
    tracker = monitor.CompletionTracker()
    state = service(active="inactive", pid=0, group="")
    processes = {"available": True, "member_pids": []}
    iteration = {"status": "completed"}
    artifacts = {"generation": 1, "scan_complete": True, "errors": []}
    assert not tracker.observe(state, processes, iteration, "hash", artifacts)
    assert tracker.observe(state, processes, iteration, "hash", artifacts)
    for changed in [service(active="active", pid=100),
                    service(active="failed", pid=0, group=""),
                    service(active="activating", pid=0, group="")]:
        assert not tracker.observe(changed, processes, iteration, "hash", artifacts)
        assert tracker.stable_polls == 0
    assert not tracker.observe(state, processes, iteration, "hash", artifacts)
    assert not tracker.observe(state, processes, iteration, "hash", {**artifacts, "generation": 2})
    assert not tracker.observe(state, {"available": True, "member_pids": [123]}, iteration, "hash", artifacts)
    assert not tracker.observe(state, {"available": False}, iteration, "hash", artifacts)


def test_sampling_failures_are_events_and_next_sample_recovers(tmp_path, captured_events):
    (tmp_path/"iteration_state.json").write_text('{"status":"running","stage":"train"}')
    calls = []
    succeeding = runner_for(service(active="inactive", pid=0, group=""), calls)
    failure_once = [True]
    def runner(command, **kwargs):
        if command[0] == "systemctl" and failure_once[0]:
            failure_once[0] = False
            raise subprocess.TimeoutExpired(command, 3)
        return succeeding(command, **kwargs)
    instance = monitor.Monitor(tmp_path, monitor.DEFAULT_SERVICE, 10, runner=runner,
                               process_sampler=EmptyProcesses())
    assert not instance.sample()
    assert any(event == "monitor_sample_failure" and fields["component"] == "service"
               for _,event,fields in captured_events)
    assert not instance.sample()
    status = json.loads((tmp_path/"logging/monitor_status.json").read_text())
    assert status["samples"] == 2
    assert status["last_sample"]["service"]["available"]
    assert status["failure_counts"]["service"] == 1


def test_once_writes_only_monitor_artifacts_without_signaling_experiment(tmp_path, captured_events, monkeypatch):
    state_file = tmp_path/"iteration_state.json"
    state_file.write_text('{"status":"completed","stage":"completed"}')
    original = state_file.read_bytes()
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("Must not signal experiment"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a,**k: pytest.fail("Must not launch workers"))
    args = monitor.arguments(["--output", str(tmp_path), "--once"])
    assert monitor.run(args, runner=runner_for(service(active="inactive", pid=0, group=""), []),
                       process_sampler=EmptyProcesses()) == 0
    assert state_file.read_bytes() == original
    status = json.loads((tmp_path/"logging/monitor_status.json").read_text())
    assert status["status"] == "sampled_once"
    assert status["stable_completion_polls"] == 1
    assert not (tmp_path/"logging/monitor_summary.json").exists()
    assert all(path == tmp_path/"logging/monitor" for path,_,_ in captured_events)


def test_duplicate_monitor_lock_fails_before_sampling_or_status_rewrite(tmp_path, captured_events):
    logging = tmp_path/"logging"; logging.mkdir()
    status = logging/"monitor_status.json"; status.write_text('{"owner":"existing"}')
    args = monitor.arguments(["--output", str(tmp_path), "--once"])
    with (logging/"monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(monitor.MonitorAlreadyRunning):
            monitor.run(args, runner=lambda *a,**k: pytest.fail("Must not sample duplicate monitor"))
    assert status.read_text() == '{"owner":"existing"}'
    assert captured_events == []


def test_monitor_metadata_is_private_even_with_permissive_umask(tmp_path, captured_events):
    (tmp_path/"iteration_state.json").write_text('{"status":"running"}')
    logging = tmp_path/"logging"
    logging.mkdir()
    logging.chmod(0o777)
    args = monitor.arguments(["--output", str(tmp_path), "--once"])
    previous_umask = os.umask(0)
    try:
        monitor.run(args, runner=runner_for(service(active="inactive", pid=0, group=""), []),
                    process_sampler=EmptyProcesses())
    finally:
        os.umask(previous_umask)
    assert logging.stat().st_mode & 0o777 == 0o700
    for name in ("monitor_status.json", "artifact_index.json", "monitor.lock"):
        assert (logging/name).stat().st_mode & 0o777 == 0o600


def test_initial_inventory_events_are_batched_without_losing_entries(tmp_path, captured_events):
    (tmp_path/"iteration_state.json").write_text('{"status":"running"}')
    for index in range(129):
        (tmp_path/f"result-{index}.json").write_text("{}")
    instance = monitor.Monitor(tmp_path, monitor.DEFAULT_SERVICE, 10,
        runner=runner_for(service(active="inactive", pid=0, group=""), []), process_sampler=EmptyProcesses())
    instance.sample()
    batches = [fields for _,event,fields in captured_events if event == "artifact_index_changes"]
    assert [len(batch["changes"]) for batch in batches] == [128, 2]
    assert all(batch["total_changes"] == 130 for batch in batches)
    assert len({entry["path"] for batch in batches for entry in batch["changes"]}) == 130


def test_soft_event_writer_failure_is_counted_and_preserves_last_stream(tmp_path, monkeypatch):
    instance = monitor.Monitor(tmp_path, monitor.DEFAULT_SERVICE, 10)
    instance.event_stream = "last-good-stream.jsonl"
    monkeypatch.setattr(monitor, "log_event", lambda *a, **k: None)
    instance.emit("failed_event")
    assert instance.logging_errors == 1
    assert instance.event_stream == "last-good-stream.jsonl"


def test_failed_service_continues_monitoring_until_own_stop_signal(tmp_path, captured_events):
    (tmp_path/"iteration_state.json").write_text('{"status":"failed","stage":"training"}')
    class StopAfterTwo:
        def __init__(self):
            self.waits = []
        def is_set(self):
            return len(self.waits) >= 2
        def wait(self, seconds):
            self.waits.append(seconds)
    stop = StopAfterTwo()
    args = monitor.arguments(["--output", str(tmp_path)])
    monitor.run(args, stop_event=stop, runner=runner_for(service(active="failed", pid=0, group=""), []),
                process_sampler=EmptyProcesses())
    assert len(stop.waits) == 2
    assert all(0 < seconds <= 10 for seconds in stop.waits)
    assert sum(event == "monitor_sample" for _,event,_ in captured_events) == 2
    assert not any(event == "monitor_final_summary" for _,event,_ in captured_events)
    status = json.loads((tmp_path/"logging/monitor_status.json").read_text())
    assert status["status"] == "stopped"


def test_completed_service_final_summary_is_written_after_second_stable_sample(tmp_path, captured_events):
    (tmp_path/"iteration_state.json").write_text('{"status":"completed","stage":"completed"}')
    instance = monitor.Monitor(tmp_path, monitor.DEFAULT_SERVICE, 10,
        runner=runner_for(service(active="inactive", pid=0, group=""), []), process_sampler=EmptyProcesses())
    assert not instance.sample()
    assert instance.sample()
    summary = json.loads((tmp_path/"logging/monitor_summary.json").read_text())
    assert summary["samples"] == 2
    assert summary["status"] == "completed"
    assert "non-atomic" in summary["sampling_limitation"]
    assert any(event == "monitor_final_summary" for _,event,_ in captured_events)


@pytest.mark.parametrize("extra", [["--interval", "0"], ["--interval", "nan"],
                                  ["--service", "../bad.service"], ["--service", "service;stop"]])
def test_invalid_monitor_arguments_are_rejected(tmp_path, extra):
    with pytest.raises(SystemExit):
        monitor.arguments(["--output", str(tmp_path), *extra])
