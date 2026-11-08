#!/usr/bin/env python3
"""Persistent systemd-owned guard for download, handoff and full experiment failures.

The service restarts this guard after its own abnormal exit. A restart records
and reports the loss, never silently launches a second campaign. SMTP failures
remain in the durable outbox and are retried until accepted. Host/network loss
can delay delivery; the enabled user service resumes notification work at boot.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import all_experiments as pipeline
from scripts import resume_reviewed_experiments as handoff
import experiment_notifications as mail


def record_event(output, event):
    event = {"timestamp_unix": time.time(), **event}
    with (output / "guard_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def exited(child, identity, receipt_path=None):
    """Persist an exit receipt before reaping, avoiding handoff/auxiliary races."""
    current = handoff.process(child.pid)
    if current and current["start_ticks"] == identity["start_ticks"] and current["state"] == "Z":
        code = os.waitstatus_to_exitcode(current["wait_status"])
        if receipt_path:
            pipeline.write_json(receipt_path, {"pid": child.pid, "start_ticks": identity["start_ticks"],
                "exit_code": code, "ended_unix": time.time()})
        child.wait()
        return code
    return None


def cleanup_external(config):
    """Retire the exact gated preparation run only if it is still ours and stopped."""
    if config.get("mode") == "prepared":
        return  # A prepared campaign owns no external preparation processes.
    if config.get("host_boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return  # PID/start ticks are only unique within one boot.
    previous = Path(config["previous_output"])
    gate = pipeline.read_json(previous / "audit_gate.json", {})
    state = pipeline.read_json(previous / "pipeline_state.json", {})
    expected = config["preparation"]
    worker = handoff.process(expected["worker"]["pid"])
    if not worker or worker["start_ticks"] != expected["worker"]["start_ticks"] or worker["state"] not in ("T", "t"):
        return
    parent = handoff.process(expected["supervisor"]["pid"])
    download = handoff.process(expected["download"]["pid"])
    if (not parent or not download
            or parent["start_ticks"] != expected["supervisor"]["start_ticks"]
            or download["start_ticks"] != expected["download"]["start_ticks"]
            or download["parent"] != worker["pid"] or worker["parent"] != parent["pid"]
            or gate.get("paused_worker_pid") != worker["pid"]
            or gate.get("worker_start_ticks") != worker["start_ticks"]
            or gate.get("download_pid") != download["pid"]
            or state.get("invocation_id") != expected["old_invocation_id"]
            or state.get("stage") != "download_models"):
        return
    record = {"worker": worker, "supervisor": parent, "download": download,
              "auxiliary_downloads": [], "previous_output": str(previous),
              "output": config["output"], "planned_maintenance": True, "plan": {"notifications": True}}
    handoff.cancel_downloads(record)
    handoff.retire_previous(record)


def failure_for_exit(output, state, code, *, auxiliary=False):
    # Reuse the inner supervisor's event identity, avoiding duplicate messages.
    if not auxiliary:
        event = pipeline.read_json(output / "pipeline_failure.json")
        supervisor = pipeline.read_json(output / "supervisor.json", {})
        identity = state.get("pipeline", state.get("handoff", {}))
        if (event and event.get("invocation_id") == supervisor.get("invocation_id")
                and supervisor.get("pid") == identity.get("pid")
                and supervisor.get("start_ticks") == identity.get("start_ticks")
                and event.get("output") == str(output)):
            return event
        transfer = ({} if state.get("mode") == "prepared"
                    else pipeline.read_json(output / "audit_handoff.json", {}))
        saved_event = transfer.get("failure_event")
        if (transfer.get("status") in ("handoff_failed", "deadline_exceeded") and saved_event
                and transfer.get("handoff_pid") == identity.get("pid")
                and transfer.get("handoff_start_ticks") == identity.get("start_ticks")
                and saved_event.get("invocation_id") == transfer.get("invocation_id")
                and saved_event.get("output") == str(output)):
            return saved_event
    return pipeline.failure_event(output, state["invocation_id"],
        "auxiliary_download" if auxiliary else "supervisor_lost", code,
        error_type="DownloaderExit" if auxiliary else "UnrecordedSupervisorExit",
        log_path=(str(output / "auxiliary_download.log") if auxiliary
                  else state.get("main_log_path", str(output / "handoff.log"))))


def verify_prepared(config):
    """Verify local checkpoints and the pinned full-integrity receipt, without downloading."""
    from scripts.verify_prepared_models import verify_receipt
    return verify_receipt(Path(config["data_root"]), Path(config["prepared_model_receipt"]),
                          config["prepared_model_receipt_sha256"])


def retry_notifications(output, event=None, *, retry_seconds=60):
    """Keep failed delivery durable and active without rerunning experiments."""
    while True:
        receipt = pipeline.send_failure(output, event) if event else None
        fallback = pipeline.read_json(output / "notification_fallback.json", {})
        if fallback.get("pending"):
            retry = pipeline.send_failure(output, fallback["event"])
            pipeline.write_json(output / "notification_fallback.json", {"event": fallback["event"], **retry})
        result = mail.flush_pending(output, max_records=20, deadline_seconds=45)
        pending = (bool(receipt and not receipt.get("smtp_accepted"))
                   or result.get("pending_remaining") != 0 or not result.get("remaining_scan_complete", True)
                   or bool(result.get("errors"))
                   or pipeline.read_json(output / "notification_fallback.json", {}).get("pending"))
        pipeline.write_json(output / "notification_monitor.json", {
            "updated_unix": time.time(), "pending": bool(pending), "flush_result": result,
            "event_id": event.get("event_id") if event else None,
            "smtp_accepted": receipt.get("smtp_accepted") if receipt else None})
        if not pending:
            return
        time.sleep(retry_seconds)


def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.pop("HYBRID_DARD_DISABLE_EMAIL", None)
    os.environ["PYTHONUNBUFFERED"] = "1"
    config_hash = pipeline.fingerprint(config)
    prepared = config.get("mode") == "prepared"
    with (output / "guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = pipeline.read_json(output / "guard_state.json")
        if (output / "guard_state.json").exists() and (not isinstance(previous, dict)
                or previous.get("status") not in ("starting", "running", "failed", "completed")):
            raise ValueError("Unreadable or invalid guard state; refusing duplicate experiment launch")
        if previous:
            if previous["config_sha256"] != config_hash:
                raise ValueError("Guard settings changed; a new reviewed output is required")
            if previous["status"] in ("starting", "running"):
                event = pipeline.failure_event(output, previous["invocation_id"], "guard_restarted", 1,
                    error_type="GuardProcessLost", log_path=str(output / "guard_service.log"))
                previous.update(status="failed", failure_event=event, ended_unix=time.time())
                pipeline.write_json(output / "guard_state.json", previous)
                record_event(output, {"stage": "guard_restarted", "failure_event": event})
                try:
                    if not prepared:
                        cleanup_external(config)
                except Exception as exc:
                    record_event(output, {"stage": "cleanup_external", "error_type": type(exc).__name__})
            retry_notifications(output, previous.get("failure_event"))
            return 0
        state = {"schema_version": 1, "status": "starting", "config_sha256": config_hash,
                 "invocation_id": "guard-" + uuid.uuid4().hex, "started_unix": time.time(),
                 "pid": os.getpid(), "start_ticks": pipeline.process_start(os.getpid()),
                 "deadline_unix": config["deadline_unix"], "mode": config.get("mode", "handoff"),
                 "stage": "guard_startup"}
        pipeline.write_json(output / "guard_state.json", state)
        old_handlers = pipeline.install_handlers()
        main_child = aux_child = None
        event = None
        logs = []
        try:
            pipeline.enable_subreaper()
            mail.preflight()
            pipeline.remaining_seconds(config)
            if config.get("host_boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                raise RuntimeError("Host rebooted before the recorded campaign could start")
            if prepared:
                state.update(stage="model_validation", model_validation={"status": "running",
                    "started_unix": time.time(), "receipt": config["prepared_model_receipt"]})
                pipeline.write_json(output / "guard_state.json", state)
                record_event(output, {"stage": "model_validation", "status": "started"})
                validation = verify_prepared(config)
                state["model_validation"].update(status="passed", ended_unix=time.time(),
                                                  verification=validation)
                record_event(output, {"stage": "model_validation", "status": "passed",
                                      "verification": validation})
                command = config["pipeline_command"]
                aux_code = 0
            else:
                original = config["preparation"]
                handoff.require_process(original["worker"]["pid"], original["worker"]["start_ticks"], stopped=True)
                handoff.require_process(original["supervisor"]["pid"], original["supervisor"]["start_ticks"])
                current_download = handoff.process(original["download"]["pid"])
                if (not current_download or current_download["start_ticks"] != original["download"]["start_ticks"]
                        or current_download["parent"] != original["worker"]["pid"]):
                    raise RuntimeError("Recorded preparation downloader identity changed")
                # Both downloads reuse their existing content-addressed caches.
                aux_log = (output / "auxiliary_download.log").open("a", buffering=1)
                logs.append(aux_log)
                aux_child = subprocess.Popen(config["auxiliary_command"], cwd=ROOT,
                    env=pipeline.runtime_env(config["data_root"]), stdout=aux_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                aux_identity = handoff.process(aux_child.pid)
                if not aux_identity:
                    raise RuntimeError("Cannot record auxiliary downloader identity")
                receipt_path = output / "auxiliary_download_exit.json"
                auxiliary = {**aux_identity, "exit_status_path": str(receipt_path)}
                gate_path = Path(config["previous_output"]) / "audit_gate.json"
                gate = pipeline.read_json(gate_path)
                if not gate:
                    raise ValueError("Missing gated preparation identity")
                gate["auxiliary_downloads"] = [auxiliary]
                pipeline.write_json(gate_path, gate)
                state["auxiliary"] = auxiliary
                pipeline.write_json(output / "guard_state.json", state)
                record_event(output, {"stage": "auxiliary_started", "process": auxiliary})
                command = config["handoff_command"]
                aux_code = None
            state.update(stage="pipeline_launch" if prepared else "handoff_launch",
                         main_log_path=str(output / ("pipeline.log" if prepared else "handoff.log")))
            pipeline.write_json(output / "guard_state.json", state)
            pipeline.remaining_seconds(config)
            log = Path(state["main_log_path"]).open("a", buffering=1)
            logs.append(log)
            main_child = subprocess.Popen(command, cwd=ROOT,
                env=pipeline.runtime_env(config["data_root"]), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            main_identity = handoff.process(main_child.pid)
            if not main_identity:
                raise RuntimeError("Cannot record campaign child identity")
            state.update(status="running", stage="pipeline" if prepared else "handoff")
            state["pipeline" if prepared else "handoff"] = main_identity
            pipeline.write_json(output / "guard_state.json", state)
            record_event(output, {"stage": "direct_pipeline_launch" if prepared else "handoff_started",
                                  "process": main_identity, "command": command})
            last_heartbeat = 0
            while True:
                if aux_code is None:
                    aux_code = exited(aux_child, aux_identity, receipt_path)
                    if aux_code is not None:
                        record_event(output, {"stage": "auxiliary_exited", "exit_code": aux_code})
                        state["auxiliary_exit_code"] = aux_code
                        pipeline.write_json(output / "guard_state.json", state)
                        if aux_code:
                            event = failure_for_exit(output, state, aux_code, auxiliary=True)
                            break
                code = exited(main_child, main_identity)
                if code is not None:
                    pipeline_state = pipeline.read_json(output / "pipeline_state.json", {})
                    supervisor = pipeline.read_json(output / "supervisor.json", {})
                    record_event(output, {"stage": "coordinator_exited", "exit_code": code})
                    completed_here = (supervisor.get("pid") == main_identity["pid"]
                        and supervisor.get("start_ticks") == main_identity["start_ticks"]
                        and pipeline_state.get("invocation_id") is not None
                        and pipeline_state.get("invocation_id") == supervisor.get("invocation_id")
                        and pipeline_state.get("status") == "completed")
                    if code == 0 and completed_here and aux_code == 0:
                        state.update(status="completed", stage="completed", ended_unix=time.time())
                    else:
                        event = failure_for_exit(output, state, code or 1)
                    break
                if time.time() >= config["deadline_unix"] - 90:
                    event = pipeline.failure_event(output, state["invocation_id"], "campaign_deadline", 124,
                        error_type="DeadlineExceeded", log_path=state["main_log_path"])
                    break
                if time.monotonic() - last_heartbeat >= 15:
                    state["updated_unix"] = time.time()
                    state["remaining_seconds"] = max(0, config["deadline_unix"]-time.time())
                    pipeline.write_json(output / "guard_state.json", state)
                    last_heartbeat = time.monotonic()
                time.sleep(.5)
        except BaseException as exc:
            try:
                record_event(output, {"stage": "guard_exception", "error_type": type(exc).__name__,
                                      "traceback": traceback.format_exc()})
            except OSError:
                pass
            code = 128 + exc.signum if isinstance(exc, pipeline.Interrupted) else 1
            if state.get("stage") == "model_validation":
                state["model_validation"].update(status="failed", ended_unix=time.time(),
                                                  error_type=type(exc).__name__, error=str(exc))
            event = pipeline.failure_event(output, state["invocation_id"],
                state["stage"] if prepared else "campaign_guard", code,
                error_type=type(exc).__name__, log_path=str(output / "guard_service.log"))
        finally:
            for signum in old_handlers:
                signal.signal(signum, signal.SIG_IGN)
            if event:
                state.update(status="failed", failure_event=event, ended_unix=time.time())
                pipeline.write_json(output / "guard_state.json", state)
            try:
                pipeline.cleanup(main_child)
                if event and not prepared:
                    cleanup_external(config)
            except Exception as exc:
                record_event(output, {"stage": "cleanup_failed", "error_type": type(exc).__name__,
                                      "traceback": traceback.format_exc()})
            for log in logs:
                log.close()
            pipeline.write_json(output / "guard_state.json", state)
            pipeline.restore_handlers(old_handlers)
        retry_notifications(output, event)
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    args = p.parse_args(argv)
    stage = "guard_configuration"
    try:
        raw = args.config.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.config_sha256:
            raise ValueError("Guard launch configuration changed")
        config = json.loads(raw)
        mode = config.get("mode", "handoff")
        command_keys = ("pipeline_command",) if mode == "prepared" else ("handoff_command", "auxiliary_command")
        if (Path(config["output"]).resolve() != args.config.parent.resolve()
                or mode not in ("handoff", "prepared")
                or any(not isinstance(config.get(key), list) or not config[key]
                       or any(not isinstance(arg, str) or not arg for arg in config[key])
                       for key in command_keys)
                or (mode == "prepared" and (
                    not isinstance(config.get("prepared_model_receipt"), str)
                    or not config["prepared_model_receipt"]
                    or not isinstance(config.get("prepared_model_receipt_sha256"), str)
                    or len(config["prepared_model_receipt_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in config["prepared_model_receipt_sha256"])))
                or not isinstance(config.get("deadline_unix"), (int, float))
                or not math.isfinite(config["deadline_unix"]) or config["deadline_unix"] <= 0):
            raise ValueError("Invalid guard launch configuration")
        stage = "guard_startup"
        return run(config)
    except Exception as exc:
        output = args.config.parent.resolve()
        os.environ.pop("HYBRID_DARD_DISABLE_EMAIL", None)
        event = pipeline.failure_event(output, "guard-config-" + args.config_sha256[:20],
            stage, 1, error_type=type(exc).__name__, log_path=str(output / "guard_service.log"))
        try:
            pipeline.write_json(output / (stage + "_failure.json"), {
                "event": event, "traceback": traceback.format_exc()})
        except OSError:
            pass
        retry_notifications(output, event)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
