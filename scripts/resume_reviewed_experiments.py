#!/usr/bin/env python3
"""Hand a paused preparation run over to a reviewed, separately frozen campaign.

Waits for the existing downloader; never installs environments or downloads models.
The prior worker must already be stopped with an audit_gate.json identity record.
Run inside tmux/nohup so the handoff can outlive the interactive review session.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
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
from scripts.download_models import MODELS


def process(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        values = raw.rsplit(")", 1)[1].split()
        return {"pid": pid, "state": values[0], "parent": int(values[1]),
                "start_ticks": values[19], "wait_status": int(values[49])}
    except FileNotFoundError:
        return None


def require_process(pid, ticks, *, stopped=False):
    row = process(pid)
    if row is None or row["start_ticks"] != ticks or row["state"] == "Z":
        raise RuntimeError("Recorded process is absent, exited or replaced: " + str(pid))
    if stopped and row["state"] not in ("T", "t"):
        raise RuntimeError("Prior worker is no longer stopped; refusing concurrent experiments")
    return row


def verify_models(root):
    manifest = json.loads((root / "models/manifest.json").read_text())
    if set(manifest) != set(MODELS):
        raise ValueError("Downloaded model manifest does not match the two planned models")
    for repo, revision in MODELS.items():
        entry = manifest[repo]
        snapshot, alias = Path(entry["snapshot"]), Path(entry["alias"])
        if entry["revision"] != revision or snapshot.name != revision or alias.resolve() != snapshot.resolve():
            raise ValueError("Downloaded checkpoint identity mismatch: " + repo)
        for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
            if not (snapshot / filename).is_file():
                raise ValueError("Incomplete checkpoint metadata: " + repo + "/" + filename)
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        for filename in set(index["weight_map"].values()):
            path = snapshot / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing checkpoint shard: " + str(path))
    return manifest


def make_record(previous, output, *, priority_codegen=False, deadline_unix=None, no_notify=False, focused_codegen=False):
    gate = pipeline.read_json(previous / "audit_gate.json")
    state = pipeline.read_json(previous / "pipeline_state.json")
    supervisor = pipeline.read_json(previous / "supervisor.json")
    old_plan = pipeline.read_json(previous / "pipeline_plan.json")
    if not all((gate, state, supervisor, old_plan)):
        raise ValueError("A recorded audit gate and live pipeline identities are required")
    if state["stage"] != "download_models" or state["status"] != "running" or old_plan["scope"] != "functions":
        raise ValueError("Handoff only supports a gated function campaign at download_models")
    if previous == output or (output / "supervisor.json").exists():
        raise ValueError("Use a fresh, separate reviewed output directory")
    worker = require_process(state["worker_pid"], state["worker_start_ticks"], stopped=True)
    if gate["paused_worker_pid"] != worker["pid"] or gate["worker_start_ticks"] != worker["start_ticks"]:
        raise ValueError("Audit gate worker identity mismatch")
    if gate["download_pid"] != state["stage_pid"] or state["invocation_id"] != supervisor["invocation_id"]:
        raise ValueError("Prior stage/invocation identity mismatch")
    parent = process(supervisor["pid"])
    download = process(state["stage_pid"])
    if parent is None or parent["state"] == "Z" or worker["parent"] != parent["pid"]:
        raise ValueError("Prior supervisor identity mismatch")
    if download is None or download["parent"] != worker["pid"]:
        raise ValueError("Recorded downloader is not a child of the gated worker")
    args = argparse.Namespace(data_root=Path(old_plan["data_root"]), output=output,
                              gpus=old_plan["gpus"], prepare=False, scope="functions",
                              priority_codegen=priority_codegen, deadline_unix=deadline_unix, no_notify=no_notify,
                              focused_codegen=focused_codegen)
    plan = pipeline.make_plan(args)
    pipeline.freeze_plan(output, plan)
    return {"schema_version": 1, "previous_output": str(previous), "output": str(output),
            "invocation_id": "audit-" + uuid.uuid4().hex, "created_unix": time.time(),
            "status": "prepared", "old_invocation_id": state["invocation_id"],
            "planned_maintenance": True,
            "worker": worker, "supervisor": parent, "download": download,
            "auxiliary_downloads": gate.get("auxiliary_downloads", []),
            "plan_sha256": pipeline.fingerprint(plan), "plan": plan}


def check_reviewed_plan(record):
    plan = record["plan"]
    args = argparse.Namespace(data_root=Path(plan["data_root"]), output=Path(plan["output"]),
                              gpus=plan["gpus"], prepare=False, scope="functions",
                              priority_codegen=plan.get("priority_codegen", False),
                              focused_codegen=plan.get("focused_codegen", False),
                              deadline_unix=plan.get("deadline_unix"), no_notify=not plan.get("notifications", True))
    saved = pipeline.read_json(Path(plan["output"]) / "pipeline_plan.json")
    if pipeline.fingerprint(saved) != record["plan_sha256"] or pipeline.make_plan(args) != plan:
        raise ValueError("Reviewed source/settings changed; a new review is required before launch")


def downloader_finished(record):
    worker, parent, expected = (record[k] for k in ("worker", "supervisor", "download"))
    record.update(handoff_stage="paused_worker_guard", active_child_pid=worker["pid"],
                  active_child_start_ticks=worker["start_ticks"], active_child_exit_code=None)
    require_process(worker["pid"], worker["start_ticks"], stopped=True)
    record.update(handoff_stage="prior_supervisor_guard", active_child_pid=parent["pid"],
                  active_child_start_ticks=parent["start_ticks"])
    require_process(parent["pid"], parent["start_ticks"])
    record.update(handoff_stage="download_models", active_child_pid=expected["pid"],
                  active_child_start_ticks=expected["start_ticks"])
    actual = process(expected["pid"])
    record["observed_child_process"] = actual
    if actual is None or actual["start_ticks"] != expected["start_ticks"] or actual["parent"] != worker["pid"]:
        raise RuntimeError("Recorded downloader disappeared or changed identity")
    if actual["state"] != "Z":
        return False
    exit_code = os.waitstatus_to_exitcode(actual["wait_status"])
    record["active_child_exit_code"] = exit_code
    if exit_code:
        raise RuntimeError("Existing model downloader failed with exit code " + str(exit_code))
    record["handoff_stage"] = "checkpoint_validation"
    verify_models(Path(record["plan"]["data_root"]))
    return True


def check_auxiliary_downloads(record, *, receipt_grace_seconds=15):
    """Require pinned exit evidence before treating external helpers as done.

    A wrapper that owns a helper should atomically write exit_status_path after
    wait(), containing pid, start_ticks, exit_code and ended_unix. A disappeared
    process without that receipt is unknown, never implicitly successful.
    """
    complete = True
    missing_since = record.setdefault("auxiliary_missing_since", {})
    for expected in record.get("auxiliary_downloads", []):
        record.update(handoff_stage="auxiliary_download", active_child_pid=expected["pid"],
                      active_child_start_ticks=expected["start_ticks"], active_child_exit_code=None)
        current = process(expected["pid"])
        record["observed_child_process"] = current
        same = current is not None and current["start_ticks"] == expected["start_ticks"]
        identity = f"{expected['pid']}:{expected['start_ticks']}"
        path = Path(expected.get("exit_status_path") or
                    Path(record["output"]) / "auxiliary-exits" / f"{expected['pid']}-{expected['start_ticks']}.json")
        receipt = pipeline.read_json(path)
        if receipt is None and path.exists():
            raise RuntimeError("Auxiliary downloader exit receipt is unreadable or invalid JSON")
        if receipt is not None:
            if (not isinstance(receipt, dict) or receipt.get("pid") != expected["pid"]
                    or receipt.get("start_ticks") != expected["start_ticks"]
                    or type(receipt.get("exit_code")) is not int
                    or not isinstance(receipt.get("ended_unix"), (int, float))
                    or not math.isfinite(receipt["ended_unix"]) or receipt["ended_unix"] <= 0):
                raise RuntimeError("Auxiliary downloader exit receipt identity or status is invalid")
            record["active_child_exit_code"] = receipt["exit_code"]
            if receipt["exit_code"]:
                raise RuntimeError(f"Auxiliary downloader {expected['pid']} failed with exit code {receipt['exit_code']}")
            if same and current["state"] != "Z":
                raise RuntimeError("Auxiliary downloader exit receipt precedes process exit")
            missing_since.pop(identity, None)
            continue
        if not same:
            first_missing = missing_since.setdefault(identity, time.monotonic())
            if time.monotonic() - first_missing < receipt_grace_seconds:
                complete = False  # The owning wrapper may be publishing its exit receipt.
                continue
            raise RuntimeError(f"Auxiliary downloader {expected['pid']} disappeared without a verified exit status")
        missing_since.pop(identity, None)
        if expected.get("parent") is not None and current["parent"] != expected["parent"]:
            raise RuntimeError("Auxiliary downloader parent identity changed")
        if current["state"] == "Z":
            code = os.waitstatus_to_exitcode(current["wait_status"])
            record["active_child_exit_code"] = code
            pipeline.write_json(path, {"pid": expected["pid"], "start_ticks": expected["start_ticks"],
                                      "exit_code": code, "ended_unix": time.time(),
                                      "evidence": "observed_proc_zombie_wait_status"})
            if code:
                raise RuntimeError(f"Auxiliary downloader {expected['pid']} failed with exit code {code}")
        elif current["state"] in ("T", "t", "X", "x"):
            raise RuntimeError("Auxiliary downloader is unexpectedly stopped or dead")
        else:
            complete = False
    return complete


def retire_previous(record, *, timeout=60):
    # A pending SIGTERM is delivered on SIGCONT before the paused worker can
    # dispatch another stage. Its existing supervisor records/notifies the stop.
    worker = record["worker"]
    require_process(worker["pid"], worker["start_ticks"], stopped=True)
    previous = Path(record["previous_output"])
    pipeline.write_json(previous / "audit_replacement.json", {
        "reason": "User-authorized maintenance: launch corrected frozen campaign",
        "replacement_output": record["output"], "timestamp_unix": time.time(),
        "planned_maintenance": record.get("planned_maintenance", False),
        "worker": record["worker"], "supervisor": record["supervisor"],
        "replacement_invocation_id": record.get("invocation_id")})
    if record.get("planned_maintenance") is True:
        # A planned replacement is recorded locally. Stop the exact old
        # supervisor before waking its worker, so it cannot send an unsolicited
        # failure email for this intentional maintenance action.
        parent = record["supervisor"]
        require_process(parent["pid"], parent["start_ticks"])
        os.kill(parent["pid"], signal.SIGKILL)
    os.kill(worker["pid"], signal.SIGTERM)
    os.kill(worker["pid"], signal.SIGCONT)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        parent = process(record["supervisor"]["pid"])
        current = process(worker["pid"])
        parent_gone = parent is None or parent["start_ticks"] != record["supervisor"]["start_ticks"] or parent["state"] == "Z"
        worker_gone = current is None or current["start_ticks"] != worker["start_ticks"] or current["state"] == "Z"
        if parent_gone and worker_gone:
            return
        time.sleep(.2)
    raise TimeoutError("Prior supervisor has not exited; refusing overlapping campaigns")


def cancel_downloads(record, *, grace=5):
    """Stop only the frozen downloader identities, including a TERM-ignoring one."""
    items = [record["download"], *record.get("auxiliary_downloads", [])]
    def live(item):
        current = process(item["pid"])
        return current and current["start_ticks"] == item["start_ticks"] and current["state"] != "Z"
    for item in items:
        if live(item):
            try:
                os.kill(item["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
    until = time.monotonic() + grace
    while any(live(item) for item in items) and time.monotonic() < until:
        time.sleep(.1)
    for item in items:
        if live(item):
            try:
                os.kill(item["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(record, *, poll_seconds=15, wait_timeout=86400):
    output = Path(record["output"])
    path = output / "audit_handoff.json"
    enabled = record.get("plan", {}).get("notifications", True)
    os.environ["HYBRID_DARD_DISABLE_EMAIL"] = "0" if enabled else "1"
    record.update(status="preflight", handoff_stage="email_preflight", handoff_pid=os.getpid(),
                  handoff_start_ticks=pipeline.process_start(os.getpid()))
    pipeline.write_json(path, record)
    if enabled:
        pipeline.mail.preflight()  # Authenticate the failure route before waiting for expensive work.
    record.update(status="waiting_for_existing_download", handoff_stage="download_wait")
    pipeline.write_json(path, record)
    print(json.dumps({"status": record["status"], "output": str(output),
                      "download_pid": record["download"]["pid"]}), flush=True)
    deadline = time.monotonic() + wait_timeout
    while True:
        remaining = pipeline.remaining_seconds(record.get("plan", {}))
        if remaining is not None and remaining <= 90:
            raise pipeline.DeadlineExceeded("Download consumed the fixed campaign time budget")
        record["handoff_stage"] = "reviewed_source_validation"
        check_reviewed_plan(record)
        auxiliary_done = check_auxiliary_downloads(record)
        record.update(handoff_stage="download_models", active_child_pid=record["download"]["pid"],
                      active_child_start_ticks=record["download"].get("start_ticks"), active_child_exit_code=None)
        if downloader_finished(record) and auxiliary_done:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing model download did not finish within the handoff deadline")
        pipeline.write_json(output / "handoff_progress.json", {
            "status": "waiting_for_existing_download", "updated_unix": time.time(),
            "deadline_unix": record.get("plan", {}).get("deadline_unix"),
            "remaining_seconds": remaining, "download": process(record["download"]["pid"]),
            "auxiliary_downloads": [process(item["pid"]) for item in record.get("auxiliary_downloads", [])]})
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    record.update(status="retiring_prior_coordinator", handoff_stage="planned_retirement", download_completed_unix=time.time())
    pipeline.write_json(path, record)
    retire_previous(record)
    record["handoff_stage"] = "reviewed_source_validation"
    check_reviewed_plan(record)
    pipeline.remaining_seconds(record.get("plan", {}))
    plan = record["plan"]
    command = [sys.executable, str(ROOT / "scripts/all_experiments.py"), "--scope", "functions",
               "--data-root", plan["data_root"], "--gpus", plan["gpus"], "--output", str(output), "--execute"]
    if plan.get("priority_codegen"):
        command.append("--priority-codegen")
    if plan.get("focused_codegen"):
        command.append("--focused-codegen")
    if plan.get("deadline_unix") is not None:
        command += ["--deadline-unix", str(plan["deadline_unix"])]
    if plan.get("notifications") is False:
        command.append("--no-notify")
    record.update(status="launching_reviewed_pipeline", handoff_stage="pipeline_exec", command=command, launched_unix=time.time())
    pipeline.write_json(path, record)
    print(json.dumps({"status": record["status"], "command": command}), flush=True)
    # Replace the waiter with the existing supervised pipeline. From here its
    # ordinary failure notification/recovery contract applies without a second
    # supervising parent or duplicate failure emails.
    os.execv(sys.executable, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--priority-codegen", action="store_true")
    parser.add_argument("--focused-codegen", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args(argv)
    os.environ["HYBRID_DARD_DISABLE_EMAIL"] = "1" if args.no_notify else "0"
    if args.poll_seconds <= 0 or args.wait_timeout <= 0:
        parser.error("poll-seconds and wait-timeout must be positive")
    args.previous_output, args.output = args.previous_output.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    record = None
    owns_output_lock = False
    stack = ExitStack()
    previous_handlers = pipeline.install_handlers() if args.execute else {}
    try:
        lock = stack.enter_context((args.output / "audit_handoff.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owns_output_lock = True
        # Different destination directories must not retire the same
        # paused source coordinator twice and then start two campaigns.
        previous_lock = stack.enter_context((args.previous_output / "audit_handoff.lock").open("a"))
        fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = make_record(args.previous_output, args.output, priority_codegen=args.priority_codegen,
                             deadline_unix=args.deadline_unix, no_notify=args.no_notify,
                             focused_codegen=args.focused_codegen)
        pipeline.write_json(args.output / "audit_handoff.json", record)
        if not args.execute:
            print(json.dumps({"status": "prepared", "output": str(args.output)}))
            return 0
        run(record, poll_seconds=args.poll_seconds, wait_timeout=args.wait_timeout)
    except (Exception, KeyboardInterrupt) as exc:
        # A second signal must not interrupt durable failure evidence or its alert.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        code = 124 if isinstance(exc, pipeline.DeadlineExceeded) else (128 + exc.signum if isinstance(exc, pipeline.Interrupted) else 1)
        record = record or {"invocation_id": "audit-" + uuid.uuid4().hex}
        timestamp = time.time()
        stage = record.get("handoff_stage", "handoff_preparation")
        record.update(status="deadline_exceeded" if code == 124 else "handoff_failed",
                      error_type=type(exc).__name__, error=str(exc), ended_unix=timestamp)
        error_path = args.output / "handoff_errors.jsonl"
        detail = {"timestamp_unix": timestamp,
                  "timestamp_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                  "invocation_id": record["invocation_id"], "stage": stage,
                  "handoff_exit_code": code,
                  "handoff_pid": os.getpid(), "child_pid": record.get("active_child_pid"),
                  "child_start_ticks": record.get("active_child_start_ticks"),
                  "child_exit_code": record.get("active_child_exit_code"),
                  "observed_child_process": record.get("observed_child_process"),
                  "plan_sha256": record.get("plan_sha256"),
                  "source_sha256": record.get("plan", {}).get("source_sha256"),
                  "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
        try:
            with error_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as storage_error:
            record["error_log_error_type"] = type(storage_error).__name__
        if isinstance(exc, TimeoutError) and record.get("worker"):
            try:
                cancel_downloads(record)
                retire_previous(record)
            except Exception as cleanup_error:
                record["cleanup_error"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
        child_code = record.get("active_child_exit_code")
        reported_code = child_code if type(child_code) is int and child_code != 0 else code
        event = pipeline.failure_event(args.output, record["invocation_id"], "audit_handoff." + stage,
            reported_code, error_type=type(exc).__name__, handoff_exit_code=code,
            signal=-child_code if type(child_code) is int and child_code < 0 else None,
            child_pid=record.get("active_child_pid"), child_exit_code=child_code,
            plan_sha256=record.get("plan_sha256"), log_path=str(error_path))
        record["failure_event"] = event
        if owns_output_lock:
            try:
                pipeline.write_json(args.output / "audit_handoff.json", record)
            except OSError as storage_error:
                record["failure_record_error_type"] = type(storage_error).__name__
        if args.execute:
            pipeline.send_failure(args.output, event)
        print(str(exc), file=sys.stderr)
        return code
    finally:
        # Keep ownership until the failure record is persisted as well.
        stack.close()
        pipeline.restore_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
