#!/usr/bin/env python3
"""One supervised experiment pipeline; preparation is opt-in and execution explicit.

The supervisor survives a failed/OOM-killed pipeline worker and alerts by email.
campaign_guard.py under the user systemd manager covers supervisor/guard loss.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_pipeline import build_stages
import experiment_notifications as mail
from storage import data_root


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def runtime_env(root):
    root = Path(root)
    env = os.environ.copy()
    env.update(HYBRID_DARD_ROOT=str(root), HYBRID_DARD_DATA_ROOT=str(root),
               HF_HOME=str(root / "cache/huggingface"), HF_HUB_CACHE=str(root / "cache/huggingface/hub"),
               HF_DATASETS_CACHE=str(root / "cache/huggingface/datasets"),
               HF_MODULES_CACHE=str(root / "cache/huggingface/modules"),
               XDG_CACHE_HOME=str(root / "cache/xdg"), PIP_CACHE_DIR=str(root / "cache/pip"),
               CONDA_PKGS_DIRS=str(root / "cache/conda"), TORCH_HOME=str(root / "cache/torch"),
               TRITON_CACHE_DIR=str(root / "cache/triton"), CUDA_CACHE_PATH=str(root / "cache/cuda"),
               EVALPLUS_CACHE_DIR=str(root / "cache/evalplus"), BIGCODEBENCH_CACHE_DIR=str(root / "cache/bigcodebench"),
               NLTK_DATA=str(root / "cache/nltk_data"), HYBRID_DARD_NATIVE_BIN=str(root / "envs/function/bin"),
               TMPDIR=str(root / "work/tmp"), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def make_plan(args):
    sources = sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "scripts").glob("*.sh"))
    sources += sorted(ROOT.glob("requirements*.txt"))
    sources += list(ROOT.glob("native_repository_recipes.json"))
    scope = getattr(args, "scope", "functions")
    deadline = getattr(args, "deadline_unix", None)
    if deadline is not None and (not math.isfinite(deadline) or deadline <= 0):
        raise ValueError("deadline-unix must be a positive finite timestamp")
    return {"schema_version": 4, "execution_backend": "local", "scope": scope,
            "data_root": str(args.data_root), "output": str(args.output),
            "gpus": args.gpus, "prepare": args.prepare, "recipient": mail.configuration_status()["recipient"],
            "deadline_unix": deadline, "priority_codegen": getattr(args, "priority_codegen", False),
            "focused_codegen": getattr(args, "focused_codegen", False),
            "notifications": not getattr(args, "no_notify", False),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "stages": build_stages(repo=ROOT, data_root=args.data_root, output=args.output,
                                   gpus=args.gpus, prepare=args.prepare, scope=scope, python=sys.executable,
                                   priority_codegen=getattr(args, "priority_codegen", False), deadline_unix=deadline,
                                   focused_codegen=getattr(args, "focused_codegen", False))}


def freeze_plan(output, plan):
    path = output / "pipeline_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Pipeline settings/source changed. Restore the original settings or use a new --output directory.")
    else:
        write_json(path, plan)


def verify_plan_sources(plan, *, repo=None):
    """Reject code/dependency changes during a long-running frozen campaign."""
    sources = plan.get("source_sha256")
    if sources is None:
        return  # Older synthetic supervision fixtures have no source manifest.
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Frozen source manifest is empty or invalid")
    root = Path(repo or ROOT).resolve()
    changed = []
    for relative, expected in sources.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            changed.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise ValueError("Frozen pipeline sources changed: " + ", ".join(sorted(changed)) +
                         "; use a new reviewed output directory")


def artifact_hashes(stage, root):
    """Only stable artifacts: later grading/analysis must not invalidate generation."""
    command = stage["command"]
    paths = []
    if "--output" in command:
        directory = Path(command[command.index("--output") + 1])
        decision = read_json(directory / "budget_decision.json", {})
        if decision.get("status") == "skipped":
            paths = [directory / "budget_decision.json"]
        elif stage["name"] == "priority_analysis":
            paths = [directory / "priority_summary.json", directory / "priority_summary.md"]
        elif stage["name"] == "focused_analysis":
            paths = [directory / "goal_report.json", directory / "goal_report.md"]
        elif "--execute" in command:
            function = "functions" in command
            paths = [directory / "campaign.json", directory / "execution_status.json"]
            if decision:
                paths.append(directory / "budget_decision.json")
            for index in range(4):
                shard = directory / f"shard-{index}"
                paths += [shard / ("config.json" if function else "run.json"),
                          shard / ("case_results.jsonl" if function else "results.jsonl")]
                if not function:
                    paths.append(shard / "submissions.json")
        elif "--grade" in command:
            paths = [directory / f"shard-{i}" / "grades.jsonl" for i in range(4)]
        elif "--analyze" in command:
            paths = [directory / f"shard-{i}" / "diagnostics.json" for i in range(4)]
            if (directory / "campaign_summary.json").exists():
                paths.append(directory / "campaign_summary.json")
    elif stage["name"] in ("setup_functions", "setup_repositories", "setup_all"):
        environments = {
            "setup_functions": ("inference", "function", "evalplus"),
            "setup_repositories": ("inference", "function", "swebench"),
            "setup_all": ("inference", "function", "evalplus", "swebench"),
        }[stage["name"]]
        paths = [root / "envs/locks" / (name + suffix)
                 for name in environments
                 for suffix in (".freeze.txt", ".conda.txt")]
        if stage["name"] != "setup_functions":
            paths.append(root / "envs/locks/system.json")
    elif stage["name"] == "download_models":
        paths = [root / "models/manifest.json"]
    elif stage["name"] == "prepare_verified_repositories":
        paths = [root / "datasets/swebench_verified/repositories.lock.json"]
    elif stage["name"] == "prepare_multilingual_repositories":
        paths = [root / "datasets/swebench_multilingual/repositories.lock.json"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


class DeadlineExceeded(TimeoutError):
    pass


def remaining_seconds(plan):
    deadline = plan.get("deadline_unix")
    if deadline is None:
        return None
    remaining = deadline - time.time()
    if remaining <= 0:
        raise DeadlineExceeded("The fixed campaign deadline has been reached")
    return remaining


def install_handlers():
    previous = {}
    def handler(signum, frame):
        raise Interrupted(signum)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signum == signal.SIGHUP and signal.getsignal(signum) == signal.SIG_IGN:
            continue  # Preserve nohup's logout protection.
        previous[signum] = signal.signal(signum, handler)
    return previous


def restore_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_worker(plan, output, invocation_id):
    """Run ordered stages, stop on the first infrastructure/process failure."""
    previous = install_handlers()
    state = {"invocation_id": invocation_id, "status": "running", "stage": "worker_start",
             "worker_pid": os.getpid(), "worker_start_ticks": process_start(os.getpid()), "started_unix": time.time()}
    child = None
    try:
        env = runtime_env(plan["data_root"])
        for stage in plan["stages"]:
            remaining_seconds(plan)
            child = None
            name = stage["name"]
            log_path = output / "logs" / (name + ".log")
            state.update(stage=name, command=stage["command"], log_path=str(log_path), stage_pid=None)
            write_json(output / "pipeline_state.json", state)
            verify_plan_sources(plan)
            stamp_path = output / "stage_results" / (name + ".json")
            stamp = read_json(stamp_path, {})
            # Cheap checks always run. Successful costly stages require unchanged artifacts.
            if name not in ("gpu_idle_check", "data_verify", "repository_data_check") and stamp.get("stage_sha256") == fingerprint(stage):
                current = artifact_hashes(stage, Path(plan["data_root"]))
                if stamp.get("artifacts") != current:
                    raise ValueError("Completed stage artifacts changed; refusing to silently skip or replace them")
                print(f"[resume] {name}", flush=True)
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[start] {name}: {log_path}", flush=True)
            timeout = remaining_seconds(plan)
            if timeout is not None:
                timeout -= 60  # Leave bounded process cleanup inside the absolute deadline.
                if timeout <= 0:
                    raise DeadlineExceeded("Only the cleanup reserve remains")
            with log_path.open("a") as log:
                child = subprocess.Popen(stage["command"], cwd=stage["cwd"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["stage_pid"] = child.pid
                write_json(output / "pipeline_state.json", state)
                try:
                    code = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    raise DeadlineExceeded("Stage exceeded the fixed campaign deadline") from exc
            try:
                verify_plan_sources(plan)
            except ValueError:
                if not code:
                    raise
                state["source_changed_during_failed_stage"] = True
            if code:
                state.update(status="failed", exit_code=code, ended_unix=time.time())
                write_json(output / "pipeline_state.json", state)
                return code if code > 0 else 128-code
            artifacts = artifact_hashes(stage, Path(plan["data_root"]))
            write_json(stamp_path, {"stage_sha256": fingerprint(stage), "invocation_id": invocation_id,
                                   "exit_code": 0, "ended_unix": time.time(), "artifacts": artifacts})
            print(f"[done] {name}", flush=True)
        remaining_seconds(plan)
        state.update(status="completed", exit_code=0, ended_unix=time.time())
        write_json(output / "pipeline_state.json", state)
        return 0
    except BaseException as exc:
        try:
            traceback.print_exc()  # Local worker log only; exception text is never mailed.
        except OSError:
            pass
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        code = 124 if isinstance(exc, DeadlineExceeded) else (128 + exc.signum if isinstance(exc, Interrupted) else 1)
        state.update(status="deadline_exceeded" if isinstance(exc, DeadlineExceeded) else
                     ("interrupted" if isinstance(exc, Interrupted) else "failed"), exit_code=code,
                     error_type=type(exc).__name__, ended_unix=time.time())
        try:
            write_json(output / "pipeline_state.json", state)
        except OSError:
            pass
        return code
    finally:
        restore_handlers(previous)


def enable_subreaper():
    """Adopt orphaned descendants, including workers in separate process groups."""
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def descendants():
    """Return PID/start-time pairs belonging to this supervisor, never other jobs."""
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = (int(values[1]), values[19])
        except (OSError, ValueError, IndexError):
            continue
    owned = {os.getpid()}
    while True:
        more = {pid for pid, (parent, _) in rows.items() if parent in owned} - owned
        if not more:
            break
        owned.update(more)
    return {pid: rows[pid][1] for pid in owned if pid != os.getpid()}


def signal_owned(signum):
    for pid, start in descendants().items():
        try:
            current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            if current == start:
                os.kill(pid, signum)
        except (OSError, IndexError):
            pass


def cleanup(worker):
    if worker is not None and worker.poll() is None:
        try:
            os.kill(worker.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    # The subreaper owns descendants even after a stage/coordinator was SIGKILLed.
    signal_owned(signal.SIGINT)
    deadline = time.monotonic() + 3
    while descendants() and time.monotonic() < deadline:
        time.sleep(.1)
    signal_owned(signal.SIGKILL)
    if worker is not None:
        worker.wait(timeout=10)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid <= 0:
                break
        except ChildProcessError:
            break


def failure_event(output, invocation_id, stage, code, **extra):
    return {"event_id": invocation_id + ":" + stage, "invocation_id": invocation_id,
            "stage": stage, "exit_code": code, "host": socket.gethostname(),
            "output": str(output), "timestamp_unix": time.time(), **extra}


def send_failure(output, event):
    if os.environ.get("HYBRID_DARD_DISABLE_EMAIL") == "1":
        receipt = {"smtp_accepted": False, "pending": False, "disabled": True}
        write_json(Path(output) / "local_failure_notice.json", {"event": event, **receipt})
        return receipt
    try:
        receipt = mail.notify(output, event)
        if receipt.get("pending") and not receipt.get("receipt_persisted"):
            write_json(Path(output) / "notification_fallback.json", {"event": event, **receipt})
    except Exception as exc:
        receipt = {"smtp_accepted": False, "pending": True, "error_type": type(exc).__name__}
        try:
            write_json(output / "notification_fallback.json", {"event": event, **receipt})
        except OSError:
            pass
    try:
        print(json.dumps({"failure_stage": event["stage"], "exit_code": event["exit_code"],
                          "smtp_accepted": receipt.get("smtp_accepted", False),
                          "email_pending": receipt.get("pending", True)}, ensure_ascii=False), flush=True)
    except OSError:
        pass
    return receipt


def recover_previous(output):
    state = read_json(output / "pipeline_state.json", {})
    if state.get("status") != "running" or not state.get("invocation_id"):
        return
    pid, ticks = state.get("worker_pid"), state.get("worker_start_ticks")
    still_active = bool(ticks and process_start(pid) == ticks)
    event = failure_event(output, state["invocation_id"], "previous_supervisor_lost", 1,
                          error_type="PriorWorkerStillActive" if still_active else "UnrecordedPreviousExit",
                          log_path=state.get("log_path", str(output / "worker.log")))
    send_failure(output, event)
    if still_active:
        raise RuntimeError("A previous pipeline worker is still active; refusing to start a second pipeline")
    state.update(status="interrupted", recovery_detected_unix=time.time())
    write_json(output / "pipeline_state.json", state)


def supervise(plan, output, invocation_id):
    if plan.get("notifications") is False:
        os.environ["HYBRID_DARD_DISABLE_EMAIL"] = "1"
    elif plan.get("notifications") is True:
        os.environ.pop("HYBRID_DARD_DISABLE_EMAIL", None)
    previous = install_handlers()
    worker = None
    stage = "supervisor_start"
    try:
        enable_subreaper()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                   "--invocation-id", invocation_id, "--plan-sha256", fingerprint(plan)]
        with (output / "worker.log").open("a") as log:
            worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(output / "supervisor.json", {"pid": os.getpid(), "worker_pid": worker.pid,
                                                     "start_ticks": process_start(os.getpid()),
                                                     "invocation_id": invocation_id, "started_unix": time.time()})
            while worker.poll() is None:
                remaining_seconds(plan)
                state = read_json(output / "pipeline_state.json", {})
                if state.get("invocation_id") == invocation_id and state.get("stage") != stage:
                    stage = state["stage"]
                    print(f"[pipeline] {stage}", flush=True)
                time.sleep(.2)
            code = worker.wait()
        state = read_json(output / "pipeline_state.json", {})
        if state.get("invocation_id") != invocation_id:
            state = {}
        stage = state.get("stage", stage)
        if code == 0 and state.get("status") == "completed":
            cleanup(worker)
            print(f"All experiment stages completed: {output}", flush=True)
            return 0
        cleanup(worker)
        code = code or 1
        stage_code = state.get("exit_code", code)
        event = failure_event(output, invocation_id, stage, state.get("exit_code", code),
                              worker_exit_code=code, signal=(-stage_code if stage_code < 0 else (-code if code < 0 else None)),
                              log_path=state.get("log_path", str(output / "worker.log")),
                              error_type=state.get("error_type", "WorkerExit"), command=state.get("command"))
        try:
            write_json(output / "pipeline_failure.json", event)
        except OSError as storage_error:
            event["storage_error_type"] = type(storage_error).__name__
        send_failure(output, event)
        return code if code > 0 else 128-code
    except BaseException as exc:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            cleanup(worker)
        except (OSError, subprocess.SubprocessError):
            pass
        code = 124 if isinstance(exc, DeadlineExceeded) else (128+exc.signum if isinstance(exc, Interrupted) else 1)
        event = failure_event(output, invocation_id, stage, code, error_type=type(exc).__name__,
                              signal=exc.signum if isinstance(exc, Interrupted) else None,
                              log_path=str(output / "worker.log"))
        try:
            write_json(output / "pipeline_failure.json", event)
        except OSError:
            pass
        send_failure(output, event)
        return code
    finally:
        restore_handlers(previous)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=data_root())
    p.add_argument("--output", type=Path)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--scope", choices=("functions", "repositories", "all"), default="functions",
                   help="Evaluation scope; default is independent code generation without library-use tasks")
    p.add_argument("--prepare", action="store_true", help="Prepare only the environments and models required by the selected scope")
    p.add_argument("--deadline-unix", type=float, help="Absolute deadline; never extended by restart")
    p.add_argument("--priority-codegen", action="store_true", help="AR+DARD first; standard diffusion only in remaining time")
    p.add_argument("--focused-codegen", action="store_true", help="One matched multilingual pass of AR,DARD,standard, sized to the fixed deadline")
    p.add_argument("--no-notify", action="store_true", help="Record failures locally without sending email")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--test-email", action="store_true", help="Explicitly send one labelled notification test")
    actions.add_argument("--retry-email", action="store_true", help="Retry pending notifications without running experiments")
    actions.add_argument("--check-email", action="store_true", help="Authenticate SMTP without sending an email")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--check-gpus", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--invocation-id", help=argparse.SUPPRESS)
    p.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.no_notify:
        os.environ["HYBRID_DARD_DISABLE_EMAIL"] = "1"
    elif not args.worker:
        os.environ.pop("HYBRID_DARD_DISABLE_EMAIL", None)
    if args.check_gpus:
        from scripts.four_gpu import resolve_gpus, run_checked
        selected = {gpu["uuid"] for gpu in resolve_gpus(args.gpus)}
        busy = [line for line in run_checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                             "--format=csv,noheader"]).splitlines()
                if line.split(",")[0].strip() in selected]
        if busy:
            print("Selected GPUs are occupied: " + "; ".join(busy), file=sys.stderr)
            return 1
        print("All four selected GPUs have no compute processes.")
        return 0
    args.data_root = args.data_root.expanduser().resolve()
    default_output = {"functions": "code-generation", "repositories": "repository-experiments-local",
                      "all": "all-experiments-local"}[args.scope]
    args.output = (args.output or args.data_root / "results" / default_output).expanduser().resolve()
    if args.worker:
        plan = json.loads((args.output / "pipeline_plan.json").read_text())
        if not args.invocation_id or fingerprint(plan) != args.plan_sha256:
            raise ValueError("Worker plan identity mismatch")
        return run_worker(plan, args.output, args.invocation_id)
    invocation_id = uuid.uuid4().hex
    stage = "startup"
    previous = install_handlers() if args.execute else {}
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.retry_email:
                fallback = read_json(args.output / "notification_fallback.json", {})
                if fallback.get("pending"):
                    receipt = send_failure(args.output, fallback["event"])
                    write_json(args.output / "notification_fallback.json", {"event": fallback["event"], **receipt})
                result = mail.flush_pending(args.output)
                print(json.dumps(result, ensure_ascii=False))
                return int(bool(result.get("pending_remaining", result.get("pending", 0)) or
                                result.get("errors") or not result.get("remaining_scan_complete", True) or
                                result.get("deadline_reached") or
                                read_json(args.output / "notification_fallback.json", {}).get("pending")))
            if args.check_email:
                print(json.dumps(mail.preflight(), ensure_ascii=False))
                return 0
            if args.test_email:
                receipt = mail.notify(args.output, failure_event(args.output, invocation_id, "notification_test", 0,
                                      error_type="ExplicitTest", message="Email configuration test; no experiment failed or ran."))
                print(json.dumps({k: receipt.get(k) for k in ("smtp_accepted", "pending", "receipt_persisted")}, ensure_ascii=False))
                return 0 if receipt.get("smtp_accepted") else 1
            stage = "plan_validation"
            plan = make_plan(args)
            freeze_plan(args.output, plan)
            print(json.dumps({"plan": str(args.output / "pipeline_plan.json"),
                              "scope": plan.get("scope", args.scope),
                              "stages": [s["name"] for s in plan["stages"]],
                              "notification": mail.configuration_status(), "execute": args.execute}, ensure_ascii=False), flush=True)
            if not args.execute:
                return 0
            stage = "email_preflight"
            if not args.no_notify:
                mail.preflight()  # Authenticate only an explicitly enabled notification route.
            stage = "previous_run_recovery"
            recover_previous(args.output)
            return supervise(plan, args.output, invocation_id)
    except (Exception, KeyboardInterrupt) as exc:
        code = 128 + exc.signum if isinstance(exc, Interrupted) else 1
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if args.execute:
            send_failure(args.output, failure_event(args.output, invocation_id, stage, code,
                                                    error_type=type(exc).__name__))
        print(f"Pipeline stopped at {stage}: {type(exc).__name__}", file=sys.stderr)
        return code
    finally:
        restore_handlers(previous)


if __name__ == "__main__":
    raise SystemExit(main())
