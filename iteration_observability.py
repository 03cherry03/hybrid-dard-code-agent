"""Private, append-only, per-process event streams for iterative experiments.

Each write is serialized across threads and fsynced before returning. Source
artifacts remain authoritative; streams add chronology, commands and failures.
Only explicitly allowed rank/GPU environment values are recorded.
"""
from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import uuid

_streams = {}
_streams_lock = threading.RLock()
_local = threading.local()
_failures = {}
_sensitive_key = re.compile(r"(?i)^(?:password|passwd|secret|api[-_]?key|access[-_]?token|refresh[-_]?token|hf[-_]?token|huggingface[-_]?token|github[-_]?token|auth[-_]?token|token|authorization|credential)$")
_sensitive_flag = re.compile(r"(?i)(?:^|[-_])(?:password|passwd|secret|token|api[-_]?key|authorization|credential)(?:$|=)")


def scrub_text(value):
    """Remove common inline authentication forms without hiding code tokens."""
    value = str(value)
    value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)
    value = re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)([?&](?:token|api_key|access_token|password|secret)=)[^&\s]+", r"\1[REDACTED]", value)
    value = re.sub(r'''(?i)(\b(?:password|passwd|secret|api[-_]?key|hf[-_]?token|access[-_]?token|refresh[-_]?token|github[-_]?token|authorization)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;}\]]+)''', r"\1[REDACTED]", value)
    return re.sub(r"\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{20,}\b", "[REDACTED]", value)


def safe_command(argv):
    result, hide_next = [], False
    for value in argv:
        value = str(value)
        if hide_next:
            result.append("[REDACTED]")
            hide_next = False
        elif value.startswith("-") and _sensitive_flag.search(value):
            if "=" in value:
                result.append(value.split("=", 1)[0] + "=[REDACTED]")
            else:
                result.append(value)
                hide_next = True
        else:
            result.append(scrub_text(value))
    return result


def _json_safe(value):
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, Path)):
        return scrub_text(value)
    if isinstance(value, Mapping):
        return {str(k): "[REDACTED]" if _sensitive_key.fullmatch(str(k)) and
                not (str(k) == "token" and not isinstance(v, str)) else
                safe_command(v) if str(k) in ("argv", "command") and isinstance(v, (tuple, list)) else
                _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Never serialize weights/gradients or synchronize GPU tensors for logging.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor_summary_only": True, "shape": list(value.shape),
                "dtype": str(value.dtype), "device": str(getattr(value, "device", "cpu"))}
    return {"unsupported_type": type(value).__name__, "representation": scrub_text(repr(value))[:4096]}


class _Stream:
    def __init__(self, output):
        directory = Path(output) / "events"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        self.path = directory / f"process-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex[:8]}.jsonl"
        self.lock = threading.RLock()
        self.sequence = 0
        self.previous_hash = None

    def append(self, event, fields):
        now = time.time()
        record = {"schema_version": 1, "event": str(event), "sequence": self.sequence,
                  "pid": os.getpid(), "ppid": os.getppid(), "time_unix": now,
                  "time_utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
                  "time_kst": dt.datetime.fromtimestamp(now, dt.timezone(dt.timedelta(hours=9))).isoformat(),
                  "monotonic_ns": time.monotonic_ns(), "previous_sha256": self.previous_hash,
                  "details": _json_safe(fields)}
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        record["record_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Event journal write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.sequence += 1
        self.previous_hash = record["record_sha256"]
        return str(self.path)


def _log_event_strict(output_path, event, **fields):
    key = (os.getpid(), str(Path(output_path).resolve()))
    with _streams_lock:
        stream = _streams.get(key)
        if stream is None:
            stream = _Stream(output_path)
            with stream.lock:
                stream.append("process_metadata", {"argv": safe_command(sys.argv),
                    "executable": sys.executable, "hostname": socket.gethostname(), "cwd": str(Path.cwd()),
                    "execution_context": {k: os.environ[k] for k in
                        ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES") if k in os.environ},
                    "prior_logging_failures_this_process": _failures.get(key[1], 0),
                    "capture_policy": "Explicit pipeline events; no credentials or environment dump; no weight/gradient tensors."})
            _streams[key] = stream
    with stream.lock:
        return stream.append(event, fields)


def _abandon_stream(output_path):
    with _streams_lock:
        return _streams.pop((os.getpid(), str(Path(output_path).resolve())), None)


def _logging_failure(output_path, event, exc):
    """Logging must not prevent cleanup or hide the original experiment error."""
    damaged = _abandon_stream(output_path)
    key = str(Path(output_path).resolve())
    with _streams_lock:
        count = _failures[key] = _failures.get(key, 0)+1
    record = {"status": "degraded", "event": "logging_write_failed", "failed_event": str(event),
              "error_type": type(exc).__name__, "pid": os.getpid(), "time_unix": time.time(),
              "failures_this_process": count, "damaged_stream": str(damaged.path) if damaged else None,
              "policy": "Preserve the experiment error/cleanup; start a new stream on the next write."}
    payload = json.dumps(record, sort_keys=True) + "\n"
    try:
        sys.stderr.write(payload)
        sys.stderr.flush()
    except InterruptedError:
        raise
    except Exception:
        pass
    try:
        path = Path(output_path) / "logging_health.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".logging-health-{os.getpid()}-{threading.get_ident()}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except InterruptedError:
        raise
    except Exception:
        pass


def log_event(output_path, event, **fields):
    """Persist a private event; report I/O failure without breaking cleanup.

    Do not call this from signal handlers. Accidental same-thread reentry is
    deferred until the current append completes, so sequence/hash stay valid.
    """
    if getattr(_local, "active", False):
        _local.pending.append((output_path, event, fields))
        return None
    _local.active = True
    _local.pending = []
    result = None
    try:
        try:
            result = _log_event_strict(output_path, event, **fields)
        except InterruptedError:
            raise
        except Exception as exc:
            _logging_failure(output_path, event, exc)
        while _local.pending:
            pending_output, pending_event, pending_fields = _local.pending.pop(0)
            try:
                _log_event_strict(pending_output, pending_event, **pending_fields)
            except InterruptedError:
                raise
            except Exception as exc:
                _logging_failure(pending_output, pending_event, exc)
        return result
    except BaseException:
        _abandon_stream(output_path)
        raise
    finally:
        _local.active = False
        _local.pending = []


def verify_event_stream(path, *, allow_incomplete_tail=False):
    """Read-only integrity check, used for final review and log consumers."""
    previous = None
    count = 0
    tail_bytes = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                if allow_incomplete_tail:
                    tail_bytes = len(line)
                    break
                raise ValueError("Incomplete trailing event record")
            count += 1
            record = json.loads(line)
            expected = record.pop("record_sha256")
            observed = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if expected != observed or record["sequence"] != count-1 or record["previous_sha256"] != previous:
                raise ValueError(f"Invalid event hash chain at record {count}")
            previous = expected
    return {"records": count, "last_sha256": previous, "incomplete_tail_bytes": tail_bytes}
