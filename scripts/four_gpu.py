#!/usr/bin/env python3
"""Four fixed GPU shards, paired backends per GPU; no implicit downloads.

Without --execute/--grade, only prepare and print the immutable campaign plan.
Reports stay per shard: this coordinator audits coverage, not pooled speed claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, redirect_stdout
import ctypes
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_shards import select_manifest, select_shard, SHARDING_ALGORITHM
from storage import data_root


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_checked(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}: {result.stderr.strip()[:1500]}")
    return result.stdout


def resolve_gpus(spec):
    rows = [line.split(",") for line in run_checked([
        "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]).splitlines()]
    devices = {r[0].strip(): {"uuid": r[1].strip(), "name": r[2].strip()} for r in rows if len(r) == 3}
    devices.update({d["uuid"]: d for d in list(devices.values())})
    requested = [part.strip() for part in spec.split(",")]
    if len(requested) != 4 or any(key not in devices for key in requested):
        raise ValueError("--gpus must select exactly four available indices or full GPU UUIDs")
    selected = [devices[key] for key in requested]
    if len({d["uuid"] for d in selected}) != 4:
        raise ValueError("Each worker requires a distinct GPU")
    return selected


def task_key(task):
    return [task.dataset, task.language, getattr(task, "task_id", getattr(task, "instance_id", None))]


def make_plan(args, gpus):
    root = data_root(args.data_root)
    python = str(root / "envs/inference/bin/python")
    backends = [name.strip() for name in args.backends.split(",")]
    selection = None
    if not backends or len(set(backends)) != len(backends) or any(name not in ("ar", "standard", "dard") for name in backends):
        raise ValueError("--backends must contain distinct choices from ar,standard,dard")
    if args.kind == "functions":
        if args.selection_manifest is not None and (args.limit or args.start):
            raise ValueError("--selection-manifest defines the selection; use --limit 0 --start 0")
        from datasets_code import load_tasks
        from main import DEFAULT_SUITES, cli
        suites = args.suites or DEFAULT_SUITES
        # Reject out-of-protocol selections before reading any dataset files.
        suites = ",".join(cli(["--suites", suites, "--check-data"]).suites)
        with redirect_stdout(io.StringIO()):
            tasks = load_tasks(suites.split(","), root / "datasets/function-cache", args.limit,
                               args.start, data_root=root)
        tasks, selection = select_manifest(tasks, args.selection_manifest)
        base = [python, str(ROOT / "main.py"), "--data-root", str(root), "--suites", suites,
                "--limit", str(args.limit), "--start", str(args.start), "--runner", "local",
                "--backends", ",".join(backends), "--tokens", str(args.tokens),
                "--standard-steps", str(args.standard_steps), "--repairs", str(args.repairs),
                "--tau-c", str(args.tau_c), "--tau-u", str(args.tau_u),
                "--transfer-limit", str(args.transfer_limit),
                "--ar-revision", "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                "--revision", "4fdd4580064ca5d11808069ce78f88d068753c96"]
        if getattr(args, "fail_fast", False):
            base.append("--fail-fast")
        if selection is not None:
            base += ["--selection-manifest", selection["path"], "--selection-manifest-sha256", selection["sha256"]]
        sources = []
    else:
        if args.selection_manifest is not None or args.repairs != 1:
            raise ValueError("--selection-manifest/--repairs apply only to function experiments")
        if backends != ["ar", "standard", "dard"]:
            raise ValueError("--backends applies only to function experiments")
        if args.suites or args.start:
            raise ValueError("--suites/--start apply only to function experiments")
        from repository_agent import load_repo_tasks
        paths = [root / "datasets" / name / "tasks.jsonl" for name in
                 ("swebench_verified", "swebench_multilingual")]
        tasks, sources = load_repo_tasks(paths, limit=args.limit)
        base = [python, str(ROOT / "repository_runner.py"), "run", "--tasks", *map(str, paths),
                "--limit", str(args.limit), "--decoders", "ar,standard,dard",
                "--max-output-tokens", str(args.tokens), "--steps", str(args.standard_steps),
                "--task-timeout", str(args.task_timeout), "--max-steps", str(args.max_steps),
                "--work-root", str(root / "work")]
    if not tasks or len({tuple(task_key(t)) for t in tasks}) != len(tasks):
        raise ValueError("Selection must contain nonempty, unique task identities")
    common = ["--repeats", str(args.repeats), "--seed", str(args.seed), "--warmup", "1",
              "--device", "cuda:0", "--quant", args.quant, "--local-files-only",
              "--temperature", "0", "--top-p", "1", "--block-size", str(args.block_size),
              "--dard-steps", str(args.dard_steps), "--diagnostics", args.diagnostics,
              "--max-step-records", str(args.max_step_records), "--num-shards", "4"]
    workers = []
    for index, gpu in enumerate(gpus):
        shard = select_shard(tasks, 4, index)
        output = str(args.output.resolve() / f"shard-{index}")
        command = [*base, *common, "--shard-index", str(index), "--output", output]
        if args.kind == "functions":
            cli(command[2:])  # Validate all function generation options before freezing the plan.
            seeds = [[args.seed + repeat for repeat in range(args.repeats)] for task in shard]
        else:
            from repository_runner import _case_seed
            seeds = [[_case_seed(task, args.seed, repeat) for repeat in range(args.repeats)] for task in shard]
        workers.append({"index": index, "gpu": gpu, "output": output, "command": command,
                        "task_keys": [task_key(t) for t in shard],
                        "task_seeds": seeds,
                        "task_count": len(shard), "expected_cases": len(shard)*len(backends)*args.repeats,
                        "by_dataset_language": dict(Counter(f"{t.dataset}/{t.language}" for t in shard))})
    source_files = (sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
                    + sorted((ROOT / "scripts").glob("*.sh")))
    source_files += sorted(ROOT.glob("requirements*.txt"))
    source_files += list(ROOT.glob("native_repository_recipes.json"))
    plan = {"schema_version": 2, "execution_backend": "local", "kind": args.kind, "root": str(root), "split": args.split,
            "failure_policy": "fail_fast" if getattr(args, "fail_fast", False) else "wait_all",
            "sharding_algorithm": SHARDING_ALGORITHM, "full_task_sha256": digest([asdict(t) for t in tasks]),
            "task_count": len(tasks), "backends": backends,
            "selection_manifest": selection,
            "expected_cases": len(tasks)*len(backends)*args.repeats,
            "repeats": args.repeats, "seed": args.seed, "sources": sources,
            "workers": workers, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in source_files},
            "measurement_scope": "Four concurrent GPU workers share host CPU, RAM and I/O; throughput campaign, not isolated single-agent latency."}
    return plan, tasks


def save_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    encoded = json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text()) != plan:
            raise ValueError("Campaign/GPU/data/source/settings changed. Restore the original setup or use a new output directory.")


def preflight(plan, tasks):
    python = Path(plan["workers"][0]["command"][0])
    if not python.is_file():
        raise RuntimeError(f"Inference environment missing: {python}; run scripts/setup_env.sh inference")
    manifest = Path(plan["root"]) / "models/manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Model manifest missing; run scripts/download_models.py --download with the inference Python")
    models = json.loads(manifest.read_text())
    required = {"Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                "apple/DiffuCoder-7B-Instruct": "4fdd4580064ca5d11808069ce78f88d068753c96"}
    for name, revision in required.items():
        entry = models.get(name, {})
        snapshot = Path(entry.get("snapshot", "/nonexistent"))
        if entry.get("revision") != revision or not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
            raise RuntimeError(f"Pinned model snapshot missing: {name}")
    selected = {worker["gpu"]["uuid"] for worker in plan["workers"]}
    busy = [line for line in run_checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader"]).splitlines()
            if line.split(",")[0].strip() in selected]
    if busy:
        raise RuntimeError("Selected GPUs already have compute processes; finish those jobs first: " + "; ".join(busy))
    if plan["kind"] == "functions":
        from evaluation import preflight_local
        preflight_local({t.language for t in tasks}, tasks=tasks, data_root=Path(plan["root"]))
    else:
        from local_repository_preparation import verify_prepared_environment
        for task in tasks:
            verify_prepared_environment(task)


def worker_env(plan, worker):
    env = os.environ.copy()
    root = Path(plan["root"])
    env.update(HYBRID_DARD_DATA_ROOT=str(root), HYBRID_DARD_ROOT=str(root),
               HF_HOME=str(root / "cache/huggingface"), HF_HUB_CACHE=str(root / "cache/huggingface/hub"),
               HF_DATASETS_CACHE=str(root / "cache/huggingface/datasets"),
               CONDA_PKGS_DIRS=str(root / "cache/conda"), PIP_CACHE_DIR=str(root / "cache/pip"),
               XDG_CACHE_HOME=str(root / "cache/xdg"), TORCH_HOME=str(root / "cache/torch"),
               EVALPLUS_CACHE_DIR=str(root / "cache/evalplus"), BIGCODEBENCH_CACHE_DIR=str(root / "cache/bigcodebench"),
               NLTK_DATA=str(root / "cache/nltk_data"), HYBRID_DARD_NATIVE_BIN=str(root / "envs/function/bin"),
               TMPDIR=str(root / "work/tmp"), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=worker["gpu"]["uuid"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name, relative in (("HF_MODULES_CACHE", "modules"), ("TRITON_CACHE_DIR", "triton"),
                           ("CUDA_CACHE_PATH", "cuda")):
        path = Path(plan["root"]) / "cache/four-gpu" / worker["gpu"]["uuid"] / relative
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


@contextmanager
def owned_descendant_cleanup():
    """CLI ownership scope for orphaned native tools, including new sessions.

    Importing or calling the lower-level execute() helper does not change the
    caller's child-process policy. Existing children of an embedding caller are
    excluded, and cleanup never uses waitpid(-1) to consume their exit status.
    """
    from scripts.all_experiments import descendants, enable_subreaper, process_start
    from local_execution import _remember_descendants
    protected = descendants()
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot inspect child subreaper state")
    enable_subreaper()
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Campaign interrupted by SIGTERM")
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    previous_sigint = signal.getsignal(signal.SIGINT)
    owned = {}

    def pending():
        _remember_descendants(protected)
        for pid, start in descendants().items():
            if protected.get(pid) != start:
                owned[pid] = start
        active = {}
        for pid, start in owned.items():
            if process_start(pid) != start:
                continue
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                reaped = 0
            if not reaped and process_start(pid) == start:
                active[pid] = start
        return active

    def stop(processes, signum):
        for pid, start in processes.items():
            if process_start(pid) != start:
                continue
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    try:
        yield
    finally:
        # Complete owned-process cleanup even if termination signals repeat.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            active = pending()
            stop(active, signal.SIGINT)
            deadline = time.monotonic() + 3
            while active and time.monotonic() < deadline:
                time.sleep(.05)
                active = pending()
            stop(pending(), signal.SIGKILL)
            deadline = time.monotonic() + 3
            while pending() and time.monotonic() < deadline:
                # Newly adopted grandchildren can appear after their parent dies.
                stop(pending(), signal.SIGKILL)
                time.sleep(.05)
            if pending():
                raise RuntimeError("Owned native processes did not terminate during campaign cleanup")
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)
            if libc.prctl(36, previous.value, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "Cannot restore child subreaper state")


def _terminate_workers(children):
    """Interrupt/reap only the process groups started by this coordinator."""
    def stop(child, signum):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
    cancelled = []
    for index, child in children:
        if child.poll() is None:
            cancelled.append(index)
            stop(child, signal.SIGINT)  # Let runner finally blocks close models/workspaces.
    codes = {}
    for index, child in children:
        try:
            codes[str(index)] = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stop(child, signal.SIGKILL)
            codes[str(index)] = child.wait()
    return codes, cancelled


def _progress_key(row):
    return (row.get("dataset"), row.get("language"), row.get("task_id", row.get("instance_id")),
            row.get("decoder"), row.get("repeat"), row.get("seed"))


class CampaignProgress:
    """Incremental, observational progress; never treat an ETA as completion.

    Only newly appended complete JSONL lines are read. Repeated attempts replace
    the same logical case. Projections require observations for every remaining
    decoder/dataset/language group, so fast AR cases cannot stand in for DARD.
    """

    interval_s = 60.0

    def __init__(self, plan, output, started):
        self.plan, self.output, self.started = plan, Path(output), started
        self.next_check = started + self.interval_s
        self.tails = {}
        self.expected = {}
        for worker in plan["workers"]:
            index = worker["index"]
            self.tails[index] = {"offset": 0, "identity": None, "rows": {}, "invalid_lines": 0}
            self.expected[index] = {
                (*task, decoder, repeat, seeds[repeat])
                for task, seeds in zip(worker["task_keys"], worker["task_seeds"], strict=True)
                for decoder in plan.get("backends", ("ar", "standard", "dard")) for repeat in range(plan["repeats"])}

    def read_new(self, worker):
        tail = self.tails[worker["index"]]
        name = "case_results.jsonl" if self.plan["kind"] == "functions" else "results.jsonl"
        path = Path(worker["output"]) / name
        try:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if tail["identity"] != identity or stat.st_size < tail["offset"]:
                    tail.update(offset=0, identity=identity, rows={}, invalid_lines=0)
                stream.seek(tail["offset"])
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break  # An actively written trailing line is read next time.
                    tail["offset"] = stream.tell()
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        key = _progress_key(row)
                        if key not in self.expected[worker["index"]]:
                            raise ValueError("Unexpected case identity")
                        # Keep only fields needed for monitoring, not generated code.
                        elapsed = (row.get("agent_wall_s", 0) + row.get("offline_grading_wall_s", 0)
                                   if self.plan["kind"] == "functions" else row.get("total_wall_s", 0))
                        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError("Invalid case duration")
                        complete = (row.get("case_completed", True) is True if self.plan["kind"] == "functions"
                                    else row.get("status") in ("submitted", "step_budget", "tool_budget", "deadline"))
                        tail["rows"][key] = {"complete": complete, "elapsed": elapsed}
                    except (ValueError, TypeError, AttributeError):
                        tail["invalid_lines"] += 1
        except FileNotFoundError:
            pass

    def snapshot(self, codes, *, now_unix=None):
        now_unix = time.time() if now_unix is None else now_unix
        samples = defaultdict(list)
        for worker in self.plan["workers"]:
            self.read_new(worker)
            for key, row in self.tails[worker["index"]]["rows"].items():
                if row["complete"] and row["elapsed"] > 0:
                    samples[(key[3], key[0], key[1])].append(row["elapsed"])
        means = {group: sum(values) / len(values) for group, values in samples.items()}
        workers = []
        for worker in self.plan["workers"]:
            index = worker["index"]
            tail = self.tails[index]
            completed = {key for key, row in tail["rows"].items() if row["complete"]}
            remaining = self.expected[index] - completed
            by_group = Counter((key[3], key[0], key[1]) for key in remaining)
            missing = sorted("/".join(group) for group in by_group if group not in means)
            seconds = None if missing else sum(count * means[group] for group, count in by_group.items())
            workers.append({"shard": index, "gpu_uuid": worker["gpu"]["uuid"],
                            "expected_cases": len(self.expected[index]), "completed_cases": len(completed),
                            "remaining_cases": len(remaining),
                            "completed_by_decoder": dict(Counter(key[3] for key in completed)),
                            "exit_code": codes.get(str(index)), "invalid_lines": tail["invalid_lines"],
                            "remaining_case_work_s": seconds, "unsampled_groups": missing})
        complete = all(row["remaining_cases"] == 0 for row in workers)
        projection = (max((row["remaining_case_work_s"] for row in workers), default=0)
                      if all(row["remaining_case_work_s"] is not None for row in workers) else None)
        successful = complete and all(codes.get(str(w["index"])) == 0 for w in self.plan["workers"])
        return {"sampled_unix": now_unix, "invocation_wall_s": time.monotonic() - self.started,
                "expected_cases": self.plan["expected_cases"],
                "completed_cases": sum(row["completed_cases"] for row in workers),
                "coverage_complete": complete, "workers_successful": successful,
                "completed_unix": now_unix if successful else None,
                "remaining_case_work_s": projection,
                "projected_completion_unix": now_unix + projection if projection is not None and not complete else None,
                "sample_counts": {"/".join(group): len(values) for group, values in sorted(samples.items())},
                "workers": workers,
                "projection_note": "Observed mean case work by decoder/dataset/language; excludes future model loads, warmups, reports and other overhead. In-flight cases count as wholly remaining. No deadline guarantee; no estimate until all remaining groups have observations."}

    def tick(self, codes, *, final=False):
        now = time.monotonic()
        if not final and now < self.next_check:
            return
        self.next_check = now + self.interval_s
        try:
            snapshot = self.snapshot(codes)
            if not final:
                try:
                    result = subprocess.run([
                        "nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    if result.returncode:
                        raise RuntimeError("GPU sample failed")
                    selected = {worker["gpu"]["uuid"] for worker in self.plan["workers"]}
                    snapshot["gpu_samples"] = []
                    for line in result.stdout.splitlines():
                        values = [value.strip() for value in line.split(",")]
                        if len(values) == 5 and values[0] in selected:
                            snapshot["gpu_samples"].append(dict(zip(
                                ("uuid", "utilization_percent", "memory_used_mib", "memory_total_mib", "power_w"), values)))
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    snapshot["gpu_sample_error"] = type(exc).__name__
            snapshot["final_observation"] = final
            temporary = self.output / "progress.json.tmp"
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
            temporary.replace(self.output / "progress.json")
            with (self.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(snapshot) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            print(f"[monitor] Progress unavailable: {type(exc).__name__}", flush=True)


def execute(plan, tasks, output, *, fail_fast=False):
    from experiment_journal import append_entry, read_journal
    with ExitStack() as stack:
        locks = Path(plan["root"]) / "work/gpu-locks"
        locks.mkdir(parents=True, exist_ok=True)
        for worker in sorted(plan["workers"], key=lambda w: w["gpu"]["uuid"]):
            stream = stack.enter_context((locks / (worker["gpu"]["uuid"] + ".lock")).open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight(plan, tasks)
        from scripts.all_experiments import verify_plan_sources
        verify_plan_sources(plan, repo=ROOT)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("Campaign interrupted by SIGTERM")
        previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
        stack.callback(signal.signal, signal.SIGTERM, previous_sigterm)
        children = []
        started = time.monotonic()
        progress = CampaignProgress(plan, output, started)
        triggering_failure = None
        cancelled_shards = []
        def save_status(codes, *, error_type=None):
            status = {"exit_codes": codes, "invocation_wall_s": time.monotonic()-started,
                      "failure_policy": "fail_fast" if fail_fast else "wait_all",
                      "triggering_failure": triggering_failure, "cancelled_shards": cancelled_shards,
                      "not_started_shards": [w["index"] for w in plan["workers"]
                                             if w["index"] not in {index for index, _ in children}],
                      "coordinator_error_type": error_type,
                      "note": "Invocation includes model load/warmup; resume covers only remaining work. Cancelled workers are not successful evaluations."}
            (output / "execution_status.json").write_text(json.dumps(status, indent=2) + "\n")
        try:
            for worker in plan["workers"]:
                directory = Path(worker["output"])
                if not read_journal(directory):
                    append_entry(directory, "init", split=plan["split"],
                        purpose="Compare selected backends on fixed disjoint GPU shards",
                        hypothesis="Measure whether DARD improves correctness and elapsed time; no improvement assumed",
                        change="Fixed baseline and shared budgets; four concurrent GPU workers", parent_run=str(output))
                log = stack.enter_context((output / f"shard-{worker['index']}.log").open("a", buffering=1))
                child = subprocess.Popen(worker["command"], cwd=ROOT, env=worker_env(plan, worker),
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((worker["index"], child))
                print(f"Started shard {worker['index']} on {worker['gpu']['uuid']}: PID {child.pid}", flush=True)
            if not fail_fast:
                codes = {str(index): child.wait() for index, child in children}
            else:
                remaining = dict(children)
                codes = {}
                while remaining:
                    for index, child in list(remaining.items()):
                        code = child.poll()
                        if code is None:
                            continue
                        codes[str(index)] = child.wait()
                        del remaining[index]
                        if code != 0:
                            triggering_failure = {"shard": index, "exit_code": code}
                            print(f"Shard {index} exited abnormally ({code}); stopping remaining workers.", flush=True)
                            break
                    if triggering_failure is not None:
                        codes, cancelled_shards = _terminate_workers(children)
                        break
                    if remaining:
                        progress.tick(codes)
                        time.sleep(0.2)
        except BaseException as exc:
            codes, cancelled_shards = _terminate_workers(children)
            try:
                progress.tick(codes, final=True)
                save_status(codes, error_type=type(exc).__name__)
            except OSError:
                pass  # Preserve the original failure even if its status cannot be persisted.
            raise
        progress.tick(codes, final=True)
        save_status(codes)
        return all(code == 0 for code in codes.values())


def audit(plan, output):
    """Verify exact per-shard coverage, never average speedups across shards."""
    from experiment_analysis import write_report
    records, shards = [], []
    for worker in plan["workers"]:
        run = Path(worker["output"])
        name = "case_results.jsonl" if plan["kind"] == "functions" else "results.jsonl"
        rows = [json.loads(line) for line in (run / name).read_text().splitlines() if line.strip()] if (run / name).exists() else []
        def key(row):
            return (row.get("dataset"), row.get("language"), row.get("task_id", row.get("instance_id")),
                    row.get("decoder"), row.get("repeat"), row.get("seed"))
        latest = {key(row): row for row in rows}
        expected = {(*task, decoder, repeat, seeds[repeat]) for task, seeds in zip(worker["task_keys"], worker["task_seeds"], strict=True)
                    for decoder in plan.get("backends", ("ar", "standard", "dard")) for repeat in range(plan["repeats"])}
        missing, extra = expected-latest.keys(), latest.keys()-expected
        records.append({"shard": worker["index"], "expected": len(expected), "recorded": len(latest),
                        "missing": len(missing), "unexpected": len(extra),
                        "missing_examples": sorted(missing)[:10], "unexpected_examples": sorted(extra, key=str)[:10],
                        "incomplete_measurements": sum(row.get("measurement_complete") is False for row in latest.values()),
                        "statuses": dict(Counter(row.get("final_status", row.get("status", "unknown")) for row in latest.values()))})
        diagnostics = write_report(run) if run.exists() else None
        config_path = run / "config.json"
        phases = json.loads(config_path.read_text()).get("phases", []) if config_path.exists() else []
        shards.append({"rows": rows, "diagnostics": diagnostics, "phases": phases})
    report = {"workers": records, "coverage_complete": all(r["missing"] == 0 and r["unexpected"] == 0 for r in records),
              "expected_cases": plan["expected_cases"],
              "note": "Coverage is not correctness or grading completion. campaign_summary.json combines counts and controlled ratios of summed work; no pooled confidence interval is inferred."}
    (output / "campaign_status.json").write_text(json.dumps(report, indent=2) + "\n")
    from campaign_reporting import summarize_campaign
    summarize_campaign(plan, shards, report, output)
    return report


def grade(plan, output):
    if plan["kind"] != "repository":
        raise ValueError("--grade is only for repository campaigns; functions grade within each run")
    if not audit(plan, output)["coverage_complete"]:
        raise ValueError("Finish all generation shards before grading")
    python = str(Path(plan["root"]) / "envs/swebench/bin/python")
    for worker in plan["workers"]:
        for dataset in ("swebench_verified", "swebench_multilingual"):
            command = [sys.executable, str(ROOT / "repository_runner.py"), "grade", "--output", worker["output"],
                       "--dataset", dataset, "--sealed", str(Path(plan["root"]) / "datasets" / dataset / "sealed.jsonl"),
                       "--harness-python", python, "--max-workers", "4"]
            with (output / f"grade-{worker['index']}-{dataset}.log").open("a") as log:
                env = worker_env(plan, worker)
                env["CUDA_VISIBLE_DEVICES"] = ""
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Graded shard {worker['index']} / {dataset}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("functions", "repository"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=data_root())
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--suites", help="Code-generation suites only; default all 10 (library-use evaluation excluded)")
    p.add_argument("--backends", default="ar,standard,dard",
                   help="Function backends to run, e.g. ar,dard for priority coverage or standard for a later campaign")
    p.add_argument("--repairs", type=int, choices=(0, 1, 2), default=1,
                   help="Public-feedback repair attempts; 0 generates once and only runs sealed final grading")
    p.add_argument("--selection-manifest", type=Path,
                   help="JSON schema_version1 exact task_keys selection, before sharding; requires limit0/start0")
    p.add_argument("--limit", type=int, default=0, help="Per suite/source BEFORE four-way sharding; 0 is all")
    p.add_argument("--start", type=int, default=0, help="Per-suite offset, functions only")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("dev", "test"), default="dev", help="Journal declaration, not an automatic data split")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--standard-steps", type=int, default=64)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dard-steps", type=int, default=64)
    p.add_argument("--tau-c", type=float, default=.5)
    p.add_argument("--tau-u", type=float, default=.8)
    p.add_argument("--transfer-limit", type=int, default=0,
                   help="Function DARD commits per step; 0 permits all threshold-eligible positions")
    p.add_argument("--quant", choices=("bf16", "4bit"), default="bf16")
    p.add_argument("--diagnostics", choices=("summary", "steps", "profile"), default="summary")
    p.add_argument("--max-step-records", type=int, default=256)
    p.add_argument("--task-timeout", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--fail-fast", action="store_true",
                   help="On any worker's nonzero exit, stop the other workers and return failure promptly")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--grade", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.limit < 0 or args.start < 0 or args.repeats < 1 or not 0 <= args.seed < 2**31:
        p.error("Invalid selection/repeats/seed")
    if not 16 <= args.tokens <= 1024 or min(args.standard_steps, args.task_timeout, args.max_steps) < 1:
        p.error("tokens must be 16..1024; step/time budgets must be positive")
    from dard_decoder import DARDConfig
    try:
        DARDConfig(block_size=args.block_size, steps_per_block=args.dard_steps,
                   tau_c=args.tau_c, tau_u=args.tau_u, transfer_limit=args.transfer_limit,
                   diagnostics=args.diagnostics, max_step_records=args.max_step_records).validate()
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.grade or args.analyze:
                # CPU postprocessing uses the recorded plan; no GPU probe or
                # repetition of generation options is required.
                plan = json.loads((args.output / "campaign.json").read_text())
                if plan["kind"] != args.kind:
                    raise ValueError("Campaign kind differs from the stored plan")
                if args.grade:
                    with owned_descendant_cleanup():
                        grade(plan, args.output)
                coverage = audit(plan, args.output)
                print(json.dumps(coverage, ensure_ascii=False))
                return 0 if coverage["coverage_complete"] else 1
            plan, tasks = make_plan(args, resolve_gpus(args.gpus))
            save_plan(args.output, plan)
            for worker in plan["workers"]:
                print(f"shard {worker['index']}: {worker['task_count']} tasks / {worker['expected_cases']} cases / {worker['gpu']['uuid']}")
            print(f"Campaign: {args.output / 'campaign.json'}", flush=True)
            if args.execute:
                with owned_descendant_cleanup():
                    success = execute(plan, tasks, args.output, fail_fast=args.fail_fast)
                coverage = audit(plan, args.output)
                return 0 if success and coverage["coverage_complete"] else 1
            print("Plan only. Add --execute with identical options to start; no environment/model/environment download is performed.")
            return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
