#!/usr/bin/env python3
"""Observe an iterative experiment without modifying its service or processes.

Samples are periodic, non-atomic observations, not a complete trace of events
between polls. This companion owns only files below OUTPUT/logging. It never
reads process environments, follows artifact symlinks, or hashes model weights.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SERVICE = "hybrid-dard-iterative-20261107.service"
COMMAND_TIMEOUT = 3
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CMDLINE_BYTES = 16384
MAX_CGROUP_PIDS = 4096
MAX_ARTIFACTS = 100000
EXCLUDED_DIRECTORIES = {"logging", "source", "data", "raw", "raws", "cache", "__pycache__"}
ARTIFACT_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}
SERVICE_FIELDS = ("Id", "LoadState", "ActiveState", "SubState", "MainPID",
                  "ControlGroup", "NRestarts", "Result", "ExecMainStatus")
SAMPLING_LIMITATION = (
    "Periodic non-atomic snapshots; short-lived processes and events between polls may be missed. "
    "CPU percent uses successive samples of the same PID/start-time and can exceed 100% for multiple cores. "
    "GPU metrics describe the entire device, not exclusively this service. "
    "Only changed JSON metadata files up to 4 MiB are hashed; binaries are not opened.")


def log_event(output, event, **fields):
    from iteration_observability import log_event as writer
    return writer(output, event, **fields)


def safe_command(argv):
    from iteration_observability import safe_command as scrubber
    result = scrubber(argv)
    for index, value in enumerate(argv):
        if re.match(r"(?i)^[A-Z_]*(?:TOKEN|PASSWORD|PASSWD|SECRET|API_KEY|CREDENTIAL)=", str(value)):
            result[index] = str(value).split("=", 1)[0] + "=[REDACTED]"
    return result


class MonitorAlreadyRunning(RuntimeError):
    pass


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _error(component, exc) -> dict:
    # Exception strings can contain commands or subprocess output.
    result = {"component": component, "error_type": type(exc).__name__}
    if isinstance(exc, subprocess.TimeoutExpired):
        result["timeout_seconds"] = COMMAND_TIMEOUT
    if isinstance(exc, subprocess.CalledProcessError):
        result["returncode"] = exc.returncode
    if isinstance(exc, OSError):
        result["errno"] = exc.errno
    return result


def _query(command: list[str], runner=subprocess.run) -> str:
    result = runner(command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def sample_service(service: str, *, runner=subprocess.run) -> dict:
    output = _query(["systemctl", "--user", "show", service, "--no-pager",
                     "--property=" + ",".join(SERVICE_FIELDS)], runner)
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if values.get("LoadState") != "loaded" or any(key not in values for key in SERVICE_FIELDS):
        raise ValueError("The requested service is not loaded or its state is incomplete")
    for key in ("MainPID", "NRestarts", "ExecMainStatus"):
        values[key] = int(values[key])
        if values[key] < 0:
            raise ValueError("Negative service process/status field")
    return {"available": True, **{key: values[key] for key in SERVICE_FIELDS}}


def sample_gpus(*, runner=subprocess.run) -> dict:
    fields = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw")
    output = _query(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                     "--format=csv,noheader,nounits"], runner)
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        if len(row) != len(fields):
            raise ValueError("Incomplete GPU telemetry row")
        values = dict(zip(fields, (value.strip() for value in row)))
        record = {"index": int(values["index"]), "uuid": values["uuid"], "name": values["name"]}
        for source, target in (("utilization.gpu", "utilization_percent"), ("memory.used", "memory_used_mib"),
                               ("memory.total", "memory_total_mib"), ("temperature.gpu", "temperature_celsius"),
                               ("power.draw", "power_watts")):
            try:
                number = float(values[source])
                record[target] = number if math.isfinite(number) else None
            except ValueError:
                record[target] = None
        rows.append(record)
    if not rows:
        raise ValueError("No GPU telemetry rows")
    return {"available": True, "devices": rows, "scope": "Whole-device instantaneous telemetry"}


def _small_json(path: Path) -> tuple[dict, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("State is not a regular file")
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("State exceeds the small JSON limit")
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("State must contain an object")
    return document, hashlib.sha256(payload).hexdigest()


def scrub_state(value, *, key="", depth=0):
    lowered = key.lower().replace("-", "_")
    if lowered in {"env", "environ", "environment", "environment_variables"}:
        return "[ENVIRONMENT OMITTED]"
    if re.search(r"(?:^|_)(?:password|passwd|secret|api_key|access_token|hf_token|authorization|credentials?)(?:$|_)", lowered):
        return "[REDACTED]"
    if depth > 12:
        return "[DEPTH LIMIT]"
    if isinstance(value, dict):
        return {str(k): scrub_state(v, key=str(k), depth=depth+1) for k, v in value.items()}
    if isinstance(value, list):
        if lowered in {"command", "cmdline", "argv", "child_command"} and all(isinstance(v, str) for v in value):
            return safe_command(value)
        return [scrub_state(v, depth=depth+1) for v in value]
    if isinstance(value, str):
        return safe_command([value])[0]
    return value


class ProcessSampler:
    def __init__(self, *, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
        self.proc_root, self.cgroup_root = Path(proc_root), Path(cgroup_root)
        self.previous = {}
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self.page_size = os.sysconf("SC_PAGE_SIZE")

    def _members(self, service):
        control = service.get("ControlGroup")
        if not control:
            if service.get("MainPID") == 0 and service.get("ActiveState") in {"inactive", "failed"}:
                return []
            raise ValueError("Active service has no observable control group")
        relative = Path(control)
        if not relative.is_absolute() or ".." in relative.parts or control == "/":
            raise ValueError("Unsafe service control group")
        group = self.cgroup_root / control.lstrip("/")
        if group.is_symlink() or not group.resolve().is_relative_to(self.cgroup_root.resolve()):
            raise ValueError("Control group escapes the cgroup filesystem")
        if not group.exists():
            if service.get("MainPID") == 0 and service.get("ActiveState") in {"inactive", "failed"}:
                return []
            raise FileNotFoundError("Active service cgroup is missing")
        members = set()
        walk_errors = []
        for directory, directories, _ in os.walk(group, followlinks=False, onerror=walk_errors.append):
            directories[:] = [name for name in directories if not (Path(directory)/name).is_symlink()]
            path = Path(directory)/"cgroup.procs"
            if path.is_symlink():
                raise ValueError("Symlinked cgroup membership file")
            for line in path.read_text().splitlines():
                pid = int(line)
                if pid <= 0:
                    raise ValueError("Invalid cgroup PID")
                members.add(pid)
                if len(members) > MAX_CGROUP_PIDS:
                    raise ValueError("Service cgroup exceeds bounded PID inventory")
        if walk_errors:
            raise walk_errors[0]
        return sorted(members)

    def sample(self, service: dict, *, monotonic_now=None) -> dict:
        if not service.get("available"):
            raise ValueError("Cannot establish process ownership without service state")
        now = time.monotonic() if monotonic_now is None else monotonic_now
        members = self._members(service)
        records, errors, next_previous = [], [], {}
        for pid in members:
            directory = self.proc_root/str(pid)
            try:
                contents = (directory/"stat").read_text()
                close = contents.rfind(")")
                if close < 0:
                    raise ValueError("Malformed process stat")
                fields = contents[close+2:].split()
                cpu_ticks = int(fields[11]) + int(fields[12])
                start_ticks = int(fields[19])
                identity = (pid, start_ticks)
                before = self.previous.get(identity)
                cpu_percent = None
                if before and now > before[0] and cpu_ticks >= before[1]:
                    cpu_percent = 100 * (cpu_ticks-before[1]) / self.clock_ticks / (now-before[0])
                with (directory/"cmdline").open("rb") as stream:
                    raw = stream.read(MAX_CMDLINE_BYTES + 1)
                command = [part.decode("utf-8", "replace") for part in raw[:MAX_CMDLINE_BYTES].split(b"\0") if part]
                records.append({"pid": pid, "ppid": int(fields[1]), "state": fields[0],
                    "start_ticks": start_ticks, "threads": int(fields[17]),
                    "cpu_seconds": cpu_ticks/self.clock_ticks, "cpu_percent": cpu_percent,
                    "rss_bytes": max(0, int(fields[21])) * self.page_size,
                    "command": safe_command(command), "cmdline_truncated": len(raw) > MAX_CMDLINE_BYTES})
                next_previous[identity] = (now, cpu_ticks)
            except (FileNotFoundError, ProcessLookupError):
                continue  # cgroup membership and /proc are not atomic
            except Exception as exc:
                errors.append({"pid": pid, **_error("process", exc)})
        self.previous = next_previous
        return {"available": True, "control_group": service.get("ControlGroup"),
                "member_pids": members, "processes": records, "errors": errors,
                "membership_races": len(members)-len(records)-len(errors)}


class ArtifactIndexer:
    """Cache hashes by inode/mtime/size and retire missing files after two scans."""
    def __init__(self, output: Path):
        self.output = Path(output)
        self.entries = {}
        self.signatures = {}
        self.missing = {}
        self.generation = 0

    def _entry(self, path, info):
        entry = {"path": str(path.relative_to(self.output)), "size_bytes": info.st_size,
                 "mtime_ns": info.st_mtime_ns, "sha256": None, "presence": "present"}
        if path.suffix != ".json":
            entry["hash_policy"] = "metadata_only_non_json"
        elif info.st_size > MAX_JSON_BYTES:
            entry["hash_policy"] = "metadata_only_json_over_4_mib"
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("Artifact changed before metadata read")
                payload = stream.read(MAX_JSON_BYTES+1)
                after = os.fstat(stream.fileno())
            if (len(payload) > MAX_JSON_BYTES or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns or len(payload) != after.st_size):
                raise ValueError("Artifact changed during metadata read")
            entry.update(sha256=hashlib.sha256(payload).hexdigest(), hash_policy="sha256_small_json")
            if path.name == "logging_health.json":
                health = json.loads(payload)
                if not isinstance(health, dict):
                    raise ValueError("Logging health receipt is not an object")
                entry["logging_health"] = scrub_state(health)
        return entry

    def scan(self) -> dict:
        seen, changes, errors, walk_errors = set(), [], [], []
        count = 0
        for directory, directories, files in os.walk(self.output, followlinks=False, onerror=walk_errors.append):
            directories[:] = [name for name in directories
                              if name not in EXCLUDED_DIRECTORIES and not name.startswith(".")
                              and not name.startswith(("source_", "source-", "audit_large", "audit-large"))
                              and not (Path(directory)/name).is_symlink()]
            for name in sorted(files):
                path = Path(directory)/name
                if name.startswith(".") or path.suffix not in ARTIFACT_SUFFIXES or path.is_symlink():
                    continue
                relative = str(path.relative_to(self.output))
                try:
                    info = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    count += 1
                    if count > MAX_ARTIFACTS:
                        raise ValueError("Artifact scan exceeds bounded inventory")
                    seen.add(relative)
                    signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                    if self.signatures.get(relative) == signature and relative not in self.missing:
                        continue
                    entry = self._entry(path, info)
                    changes.append({"action": "changed" if relative in self.entries else "added", **entry})
                    self.entries[relative], self.signatures[relative] = entry, signature
                    self.missing.pop(relative, None)
                except Exception as exc:
                    errors.append({"path": relative, **_error("artifact", exc)})
            if count > MAX_ARTIFACTS:
                break
        errors.extend(_error("artifact_walk", exc) for exc in walk_errors)
        if not walk_errors and count <= MAX_ARTIFACTS:
            for relative in sorted(set(self.entries)-seen):
                absent = self.missing.get(relative, 0) + 1
                self.missing[relative] = absent
                if absent < 2:
                    self.entries[relative] = {**self.entries[relative], "presence": "missing_pending", "missing_scans": absent}
                    changes.append({"action": "missing_pending", **self.entries[relative]})
                else:
                    changes.append({"action": "removed", **self.entries.pop(relative), "missing_scans": absent})
                    self.signatures.pop(relative, None); self.missing.pop(relative, None)
        if changes:
            self.generation += 1
        return {"generation": self.generation, "entries": dict(sorted(self.entries.items())),
                "changes": changes, "errors": errors, "scan_complete": not walk_errors and count <= MAX_ARTIFACTS,
                "missing_removal_after_scans": 2, "hash_json_max_bytes": MAX_JSON_BYTES}


class CompletionTracker:
    def __init__(self):
        self.signature = None
        self.stable_polls = 0

    def observe(self, service, processes, iteration_state, state_sha, artifacts):
        eligible = (service.get("available") and service.get("LoadState") == "loaded"
                    and service.get("ActiveState") == "inactive" and service.get("MainPID") == 0
                    and processes.get("available") and not processes.get("member_pids")
                    and iteration_state.get("status") == "completed"
                    and artifacts.get("scan_complete") and not artifacts.get("errors"))
        signature = (state_sha, artifacts.get("generation"), service.get("SubState"), service.get("Result")) if eligible else None
        self.stable_polls = self.stable_polls + 1 if signature is not None and signature == self.signature else int(bool(signature))
        self.signature = signature
        return self.stable_polls >= 2


class Monitor:
    def __init__(self, output: Path, service: str, interval: float, *, runner=subprocess.run,
                 process_sampler=None, indexer=None):
        self.output, self.service, self.interval = Path(output), service, interval
        self.runner = runner
        self.process_sampler = process_sampler or ProcessSampler()
        self.indexer = indexer or ArtifactIndexer(self.output)
        self.completion = CompletionTracker()
        self.samples = 0
        self.failure_counts = {}
        self.event_stream = None
        self.started_unix = time.time()
        self.logging_errors = 0

    def emit(self, event, **fields):
        try:
            event_stream = log_event(self.output/"logging/monitor", event, **fields)
            if event_stream is None:
                self.logging_errors += 1
            else:
                self.event_stream = event_stream
        except Exception as exc:
            self.logging_errors += 1
            print("Monitor event writer failed: " + type(exc).__name__, file=sys.stderr, flush=True)

    def sample(self):
        began = time.time()
        errors = []
        try:
            service = sample_service(self.service, runner=self.runner)
        except Exception as exc:
            errors.append(_error("service", exc)); service = {"available": False}
        try:
            gpus = sample_gpus(runner=self.runner)
        except Exception as exc:
            errors.append(_error("gpu", exc)); gpus = {"available": False}
        try:
            processes = self.process_sampler.sample(service)
            errors.extend(processes.get("errors", []))
        except Exception as exc:
            errors.append(_error("process_inventory", exc)); processes = {"available": False}
        try:
            iteration_state, state_sha = _small_json(self.output/"iteration_state.json")
            iteration_state = scrub_state(iteration_state)
        except Exception as exc:
            errors.append(_error("iteration_state", exc)); iteration_state, state_sha = {}, None
        try:
            artifacts = self.indexer.scan()
            errors.extend(artifacts["errors"])
        except Exception as exc:
            errors.append(_error("artifact_index", exc))
            artifacts = {"generation": None, "entries": {}, "changes": [], "errors": [], "scan_complete": False}
        self.samples += 1
        for error in errors:
            component = error["component"]
            self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
            self.emit("monitor_sample_failure", sample=self.samples, **error)
        # Each durable logger write fsyncs. Batching the initial inventory avoids
        # thousands of separate fsyncs delaying a nominal ten-second sample.
        changes = artifacts["changes"]
        for offset in range(0, len(changes), 128):
            self.emit("artifact_index_changes", sample=self.samples, batch_index=offset//128,
                      total_changes=len(changes), changes=changes[offset:offset+128])
        finished = time.time()
        record = {"schema_version": 1, "sample": self.samples, "service_name": self.service,
                  "sample_started_unix": began, "sample_ended_unix": finished,
                  "sample_duration_seconds": finished-began, "configured_interval_seconds": self.interval,
                  "service": service, "gpus": gpus, "processes": processes,
                  "iteration_state": iteration_state, "iteration_state_sha256": state_sha,
                  "artifact_generation": artifacts["generation"], "artifact_count": len(artifacts["entries"]),
                  "errors": errors, "sampling_limitation": SAMPLING_LIMITATION}
        self.emit("monitor_sample", **record)
        done = self.completion.observe(service, processes, iteration_state, state_sha, artifacts)
        index_path = self.output/"logging/artifact_index.json"
        atomic_json(index_path, {"schema_version": 1, "sample": self.samples, "sampled_unix": finished,
            "scan_complete": artifacts["scan_complete"], "generation": artifacts["generation"],
            "entries": artifacts["entries"], "errors": artifacts["errors"],
            "missing_removal_after_scans": 2, "hash_json_max_bytes": MAX_JSON_BYTES,
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES), "binary_files_indexed": False})
        status = {"schema_version": 1, "pid": os.getpid(), "status": "completed" if done else "running",
                  "started_unix": self.started_unix, "updated_unix": finished, "samples": self.samples,
                  "service_name": self.service, "interval_seconds": self.interval,
                  "stable_completion_polls": self.completion.stable_polls,
                  "last_sample": record, "failure_counts": self.failure_counts,
                  "event_stream": self.event_stream, "event_writer_errors": self.logging_errors,
                  "artifact_index_path": str(index_path),
                  "component_logging_health": {name: item["logging_health"] for name, item in
                                               artifacts["entries"].items() if "logging_health" in item}}
        own_health = self.output/"logging/monitor/logging_health.json"
        if own_health.is_file() and not own_health.is_symlink():
            try:
                health, _ = _small_json(own_health)
                status["monitor_logging_health"] = scrub_state(health)
            except Exception as exc:
                status["monitor_logging_health"] = _error("monitor_logging_health", exc)
        if done:
            summary = {"schema_version": 1, "status": "completed", "samples": self.samples,
                "started_unix": self.started_unix, "completed_unix": finished,
                "failure_counts": self.failure_counts, "event_writer_errors": self.logging_errors,
                "completion_rule": "Main service inactive, no main/cgroup PID, completed iteration state, and stable state/artifact inventory for two consecutive polls",
                "artifact_count": len(artifacts["entries"]), "iteration_state_sha256": state_sha,
                "sampling_limitation": SAMPLING_LIMITATION}
            self.emit("monitor_final_summary", **summary)
            atomic_json(self.output/"logging/monitor_summary.json", summary)
        atomic_json(self.output/"logging/monitor_status.json", status)
        return done


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    if not re.fullmatch(r"[A-Za-z0-9_.:@-]+\.service", args.service):
        parser.error("--service must name one systemd service unit")
    return args


def run(args, *, stop_event=None, runner=subprocess.run, process_sampler=None):
    output = args.output.resolve()
    if not output.is_dir():
        raise ValueError("The existing experiment output directory is required")
    logging = output/"logging"
    if logging.is_symlink():
        raise ValueError("Monitor logging directory must not be a symlink")
    logging.mkdir(mode=0o700, exist_ok=True)
    logging.chmod(0o700)
    stop = stop_event or threading.Event()
    descriptor = os.open(logging/"monitor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MonitorAlreadyRunning("Another monitor owns this output") from exc
        monitor = Monitor(output, args.service, args.interval, runner=runner, process_sampler=process_sampler)
        monitor.emit("monitor_started", service=args.service, interval_seconds=args.interval,
                     once=args.once, sampling_limitation=SAMPLING_LIMITATION)
        while not stop.is_set():
            started = time.monotonic()
            try:
                if monitor.sample():
                    return 0
            except Exception as exc:
                monitor.emit("monitor_sampling_failure", **_error("monitor", exc))
            if args.once:
                path = logging/"monitor_status.json"
                if path.is_file():
                    receipt, _ = _small_json(path)
                    receipt.update(status="sampled_once", monitor_stopped_unix=time.time())
                    atomic_json(path, receipt)
                return 0
            stop.wait(max(0.01, args.interval-(time.monotonic()-started)))
        monitor.emit("monitor_stopped", reason="monitor_stop_signal", samples=monitor.samples)
        atomic_json(logging/"monitor_status.json", {"schema_version": 1, "status": "stopped",
            "pid": os.getpid(), "updated_unix": time.time(), "samples": monitor.samples,
            "reason": "Monitor stopped; main experiment was not signaled or modified"})
        return 0


def main(argv=None):
    args = arguments(argv)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        return run(args, stop_event=stop)
    except MonitorAlreadyRunning:
        print("A monitor already owns this experiment output.", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
