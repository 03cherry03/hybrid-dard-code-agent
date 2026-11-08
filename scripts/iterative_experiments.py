#!/usr/bin/env python3
"""Durable four-GPU validation -> LoRA training -> validation controller."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import (AR, CPGRPO, INSTRUCT, adapter_identity, candidate, digest,
    file_sha, freeze_json, initial_candidates, read_json, read_jsonl, round_candidates,
    select_results, summarize, task_id, training_rate, write_json)
from scripts.four_gpu import resolve_gpus, worker_env, owned_descendant_cleanup
from iteration_observability import log_event


class AlreadyRunning(RuntimeError):
    pass


def source_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file()
                  and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)
                  and (p.suffix in (".py", ".json", ".sh") or p.name.startswith("requirements")))


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "iteration_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        verify_plan(plan)
        return plan
    data = output / "data"
    required = ["train_tasks.jsonl", "validation_tasks.jsonl", "holdout_tasks.jsonl",
                "training_examples.jsonl", "split_manifest.json", "data_accounting.json"]
    if not all((data / p).is_file() for p in required):
        raise ValueError("Run scripts/prepare_iterative_data.py first")
    sources = {}
    snapshot = output / "source"
    for path in source_files(ROOT):
        relative = path.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(relative)] = file_sha(destination)
    plan = {"schema_version": 1, "output": str(output), "source": str(snapshot),
            "data_root": str(args.data_root.resolve()), "data": str(data),
            "python": str(args.data_root / "envs/iterative-training/bin/python"),
            "gpus": resolve_gpus(args.gpus), "deadline_unix": args.deadline_unix,
            "max_rounds": args.max_rounds, "patience": args.patience,
            "train_steps": args.train_steps, "training_learning_rates": [5e-5, 2e-5, 1e-4, 1e-5],
            "grading_timeout": 300, "generation_timeout": 120, "accuracy_margin": .03,
            "seed": 20261105, "final_reserve_seconds": 4*3600,
            "data_sha256": {p: file_sha(data / p) for p in required}, "source_sha256": sources,
            "initial_candidates": initial_candidates(), "failure_limit": 3,
            "selection": "Validation only; fastest candidate within AR accuracy margin and faster than AR, otherwise best diffusion accuracy then latency. Deployment retains AR unless joint criteria pass.",
            "training": "Four-rank DDP BF16 frozen DiffuCoder + LoRA, shifted masked diffusion objective, train sources only.",
            "holdout_policy": "Evaluate once after selection freezes; never choose a checkpoint or setting from holdout results.",
            "limitations": ["Finite coordinate search is not proof of a global optimum.",
                            "Previously inspected benchmark sources retain their exposure labels.",
                            "Repeated validation selection does not establish statistical noninferiority."]}
    receipts = [output/"audit/cpgrpo/candidate_model_receipt.json",
                args.data_root/"results/code-generation-dard-core-20261105/audit/model_integrity.json"]
    if not all(p.is_file() for p in receipts):
        raise ValueError("Verified base-model integrity receipts are required")
    plan["model_receipts"] = [{"path": str(p.resolve()), "sha256": file_sha(p)} for p in receipts]
    plan["package_versions"] = {name: importlib.metadata.version(name) for name in
         ("torch", "transformers", "peft", "accelerate", "safetensors", "huggingface-hub")}
    freeze_json(plan_path, plan)
    log_event(output, "plan_prepared", plan_path=plan_path, plan_sha256=file_sha(plan_path),
              source_files=len(sources), gpus=plan["gpus"], package_versions=plan["package_versions"])
    return plan


def verify_plan(plan):
    from scripts.verify_prepared_models import signature
    previous = plan.get("previous_plan")
    if previous and file_sha(previous["path"]) != previous["sha256"]:
        raise ValueError("Archived predecessor plan changed")
    for name, expected in plan.get("package_versions", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Frozen Python package version changed: " + name)
    for name, expected in plan["source_sha256"].items():
        if file_sha(Path(plan["source"]) / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in plan["data_sha256"].items():
        if file_sha(Path(plan["data"]) / name) != expected:
            raise ValueError("Frozen split/corpus changed: " + name)
    for receipt in plan.get("model_receipts", []):
        if file_sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Base model integrity receipt changed")
        for record in read_json(receipt["path"])["files"]:
            if any(record[key] != value for key, value in signature(record["path"]).items()):
                raise ValueError("Pinned base-model file changed: " + record["path"])


def update(plan, stage, **details):
    path = Path(plan["output"]) / "iteration_state.json"
    state = read_json(path, {"schema_version": 1, "failure_count": 0})
    state.update(stage=stage, status="running", updated_unix=time.time())
    state.update(details)
    write_json(path, state)
    log_event(plan["output"], "controller_state", stage=stage, state=state)
    with (Path(plan["output"]) / "iteration_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "stage": stage, **details}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def terminate(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    until = time.monotonic() + 15
    while any(c.poll() is None for c in children) and time.monotonic() < until:
        time.sleep(.2)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)


def wait_children(plan, children, deadline, stage):
    try:
        heartbeat = 0
        while any(child.poll() is None for child in children):
            if time.time() >= deadline:
                raise TimeoutError(stage + " deadline reached")
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(stage + " worker failed; see stage logs")
            if time.time() - heartbeat > 30:
                update(plan, stage, child_pids=[c.pid for c in children], deadline_unix=plan["deadline_unix"])
                heartbeat = time.time()
            time.sleep(2)
        if any(c.returncode for c in children):
            raise RuntimeError(stage + " returned nonzero")
    except BaseException:
        terminate(children)
        raise
    finally:
        log_event(plan["output"], "child_processes_finished", stage=stage,
                  processes=[{"pid": c.pid, "returncode": c.poll()} for c in children])


def evaluate(plan, label, configs, split="validation", deadline=None, tasks_override=None):
    verify_plan(plan)
    output = Path(plan["output"]) / label
    tasks = Path(tasks_override) if tasks_override else Path(plan["data"]) / (split + "_tasks.jsonl")
    configs = list({c["id"]: c for c in configs}.values())
    evaluation = {"schema_version": 1, "split": split, "tasks_path": str(tasks),
        "tasks_sha256": file_sha(tasks), "output": str(output), "data_root": plan["data_root"],
        "candidates": configs, "seed": plan["seed"], "deadline_unix": deadline or plan["deadline_unix"]-120,
        "grading_timeout": plan["grading_timeout"], "generation_timeout": plan["generation_timeout"]}
    # Explicit opt-in preserves older frozen experiments and their hashes.
    allowed_options = {"extraction_version", "counterbalanced_order", "candidate_warmup",
                       "save_grading_verdict", "grading_workers", "grading_queue_size"}
    options = plan.get("evaluation_options", {})
    if set(options) - allowed_options:
        raise ValueError("Unknown evaluation options")
    evaluation.update(options)
    plan_path = output / "evaluation_plan.json"
    freeze_json(plan_path, evaluation)
    rows_expected = read_jsonl(tasks)
    if (output / "results.json").exists():
        saved = read_json(output / "results.json")
        if saved["evaluation_plan_sha256"] != file_sha(plan_path):
            raise ValueError("Evaluation result plan changed")
        log_event(plan["output"], "evaluation_resumed_from_result", stage=label,
                  result_path=output / "results.json", evaluation_plan_sha256=file_sha(plan_path))
        return saved["candidates"]
    update(plan, label, candidate_count=len(configs), expected_cases=len(rows_expected)*len(configs))
    children = []
    with ExitStack() as stack:
        for index, gpu in enumerate(plan["gpus"]):
            log = stack.enter_context((output / f"shard-{index}.log").open("a"))
            env = worker_env({"root": plan["data_root"]}, {"gpu": gpu})
            command = [plan["python"], str(Path(plan["source"])/"scripts/iterative_eval_worker.py"),
                       "--plan", str(plan_path), "--shard", str(index)]
            log_event(plan["output"], "child_launch_requested", stage=label, command=command,
                      shard=index, gpu=gpu, stdout_stderr_path=output / f"shard-{index}.log")
            children.append(subprocess.Popen(command, cwd=plan["source"], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            log_event(plan["output"], "child_started", stage=label, pid=children[-1].pid, shard=index)
        wait_children(plan, children, evaluation["deadline_unix"], label)
    summaries = []
    for config in configs:
        rows = [read_json(p) for p in output.glob(f"shard-*/cases/{config['id']}/*.json")]
        summaries.append(summarize(rows, rows_expected, config))
    write_json(output / "results.json", {"evaluation_plan_sha256": file_sha(plan_path),
               "split": split, "candidates": summaries, "completed_unix": time.time()})
    log_event(plan["output"], "evaluation_completed", stage=label, split=split,
              result_path=output / "results.json", candidates=summaries)
    return summaries


def prepare_teacher_corpus(plan):
    """Only missing TRAIN labels are synthesized; validation/test code is never read."""
    from scripts.iterative_teacher import missing_tasks, augment_training_corpus
    output, data = Path(plan["output"]), Path(plan["data"])
    binding_path = output / "training_corpus.json"
    existing = read_json(binding_path)
    if existing:
        if file_sha(existing["path"]) != existing["sha256"]:
            raise ValueError("Training corpus changed after preparation")
        log_event(output, "training_corpus_reused", corpus=existing)
        return existing
    missing = missing_tasks(data)
    corpus = data / "training_examples.jsonl"
    if missing:
        task_path = data / "teacher_tasks.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in missing)
        if task_path.exists() and task_path.read_text() != payload:
            raise ValueError("Teacher training identities changed")
        if not task_path.exists():
            task_path.write_text(payload)
        cfg = candidate(AR, decoder="ar", temperature=.3, top_p=.95,
                        forward_budget=512, prefix="code_fence")
        evaluate(plan, "train_teacher", [cfg], split="train_teacher", tasks_override=task_path,
                 deadline=plan["deadline_unix"]-plan["final_reserve_seconds"])
        corpus = data / "training_augmented.jsonl"
        receipt = augment_training_corpus(data, output/"train_teacher", corpus)
        write_json(output/"teacher_corpus_summary.json", receipt)
        log_event(output, "teacher_augmentation_completed", receipt_path=output/"teacher_corpus_summary.json",
                  training_examples=receipt["training_examples"], new_labels=receipt["new_verified_synthetic_labels"],
                  unlabeled_count=receipt["unlabeled_count"])
    binding = {"path": str(corpus), "sha256": file_sha(corpus),
               "original_train_sha256": plan["data_sha256"]["training_examples.jsonl"],
               "uses_training_sources_only": True}
    freeze_json(binding_path, binding)
    log_event(output, "training_corpus_frozen", corpus=binding)
    return binding


def train(plan, index, incumbent):
    verify_plan(plan)
    output = Path(plan["output"]) / f"round-{index:02d}" / "training"
    output.mkdir(parents=True, exist_ok=True)
    config = incumbent["candidate"]
    corpus = read_json(Path(plan["output"])/"training_corpus.json",
                       {"path": str(Path(plan["data"])/"training_examples.jsonl"),
                        "sha256": plan["data_sha256"]["training_examples.jsonl"]})
    if file_sha(corpus["path"]) != corpus["sha256"]:
        raise ValueError("Training corpus checksum mismatch")
    binding = {"base_model": config["model"], "revision": config["revision"],
        "initial_adapter": config.get("adapter"), "train_sha256": corpus["sha256"],
        "learning_rate": training_rate(index), "seed": plan["seed"]+index,
        "max_steps": plan["train_steps"], "max_length": 1536, "ranks": 4}
    freeze_json(output / "controller_binding.json", binding)
    state = read_json(output / "training_state.json", {})
    adapter = output / "final_adapter"
    if state.get("status") == "completed" and adapter.exists():
        log_event(plan["output"], "training_resumed_from_completed_adapter", round=index,
                  adapter=adapter_identity(adapter))
        return adapter_identity(adapter)
    command = [plan["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(Path(plan["source"])/"scripts/train_diffusion_lora.py"),
        "--train-jsonl", corpus["path"],
        "--output", str(output), "--base-model", config["model"], "--revision", config["revision"],
        "--max-length", "1536", "--epochs", "2", "--max-steps", str(plan["train_steps"]),
        "--batch-size", "1", "--gradient-accumulation", "4", "--learning-rate", str(training_rate(index)),
        "--rank", "8", "--alpha", "16", "--checkpoint-every", "10", "--seed", str(plan["seed"]+index),
        "--deadline-unix", str(plan["deadline_unix"]-plan["final_reserve_seconds"]), "--local-files-only"]
    if (output / "latest.json").exists():
        command += ["--resume", "latest"]
    elif config.get("adapter"):
        command += ["--initial-adapter", config["adapter"]["path"]]
    env = worker_env({"root": plan["data_root"]}, {"gpu": plan["gpus"][0]})
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in plan["gpus"])
    # 4090 PCIe systems often have no peer-to-peer path; explicit fallbacks keep
    # NCCL collectives reliable while all four ranks perform real gradient work.
    env.update(NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    update(plan, f"round-{index:02d}/training", round=index, learning_rate=training_rate(index))
    with (output / "training.log").open("a") as log:
        log_event(plan["output"], "child_launch_requested", stage=f"round-{index:02d}/training",
                  command=command, gpus=plan["gpus"], stdout_stderr_path=output/"training.log", binding=binding)
        child = subprocess.Popen(command, cwd=plan["source"], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        log_event(plan["output"], "child_started", stage=f"round-{index:02d}/training", pid=child.pid)
        try:
            wait_children(plan, [child], plan["deadline_unix"]-plan["final_reserve_seconds"], f"round-{index:02d}/training")
        except RuntimeError as exc:
            interrupted = read_json(output / "training_state.json", {})
            if interrupted.get("status") == "interrupted" and interrupted.get("stop_reason") == "deadline":
                raise TimeoutError("Trainer saved its deadline checkpoint") from exc
            raise
    state = read_json(output / "training_state.json", {})
    if state.get("status") != "completed":
        raise RuntimeError("Training exited without a completed checkpoint")
    log_event(plan["output"], "training_completed", round=index, state=state,
              adapter=adapter_identity(adapter))
    return adapter_identity(adapter)


def execute(plan):
    verify_plan(plan)
    log_event(plan["output"], "controller_started", plan_sha256=file_sha(Path(plan["output"])/"iteration_plan.json"),
              source=plan["source"], previous_plan=plan.get("previous_plan"), deadline_unix=plan["deadline_unix"])
    output = Path(plan["output"])
    existing = read_json(output / "iteration_state.json", {})
    if existing.get("status") == "completed":
        return
    if existing.get("failure_count", 0) >= plan["failure_limit"]:
        raise RuntimeError("Retry limit reached")
    if time.time() > plan["deadline_unix"]-600:
        raise TimeoutError("Insufficient time before the absolute deadline")
    with ExitStack() as stack:
        lock = stack.enter_context((output / "iteration.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning("Iteration controller already owns this output") from exc
        for gpu in sorted(plan["gpus"], key=lambda g: g["uuid"]):
            path = Path(plan["data_root"])/"work/gpu-locks"/(gpu["uuid"]+".lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True).stdout
        if any(line.split(",")[0].strip() in {g["uuid"] for g in plan["gpus"]} for line in busy.splitlines()):
            raise RuntimeError("Selected GPU already has a compute process")
        started = existing.get("started_unix", time.time())
        update(plan, "prepare_training_corpus", started_unix=started)
        prepare_teacher_corpus(plan)
        update(plan, "baseline_validation", started_unix=started)
        baseline = evaluate(plan, "baseline_validation", plan["initial_candidates"],
                            deadline=plan["deadline_unix"]-plan["final_reserve_seconds"])
        ar = next(r for r in baseline if r["candidate"]["decoder"] == "ar")
        chosen = select_results([r for r in baseline if r["candidate"]["decoder"] != "ar"], ar, plan["accuracy_margin"])
        incumbent = chosen["research_incumbent"]
        freeze_json(output / "baseline_selection.json", {k:v for k,v in chosen.items() if k != "selected_unix"})
        log_event(output, "baseline_selection_frozen", selection=chosen)
        plateau = 0
        completed_rounds = 0
        reason = "max_rounds"
        for index in range(1, plan["max_rounds"]+1):
            decision_path = output / f"round-{index:02d}" / "decision.json"
            saved = read_json(decision_path)
            if saved is not None:
                incumbent, ar, plateau = saved["incumbent"], saved["ar"], saved["plateau"]
                completed_rounds = index
                if plateau >= plan["patience"]:
                    reason = "validation_plateau"
                    break
                continue
            if time.time() >= plan["deadline_unix"] - plan["final_reserve_seconds"] - 7200:
                reason = "deadline_reserve"
                break
            try:
                adapter = train(plan, index, incumbent)
                configs = [ar["candidate"], incumbent["candidate"], *round_candidates(incumbent["candidate"], adapter, index)]
                results = evaluate(plan, f"round-{index:02d}/validation", configs,
                                   deadline=plan["deadline_unix"]-plan["final_reserve_seconds"])
            except TimeoutError as exc:
                write_json(output/f"round-{index:02d}"/"deadline_interruption.json",
                           {"reason": str(exc), "time": time.time(), "incumbent_retained": True})
                reason = "deadline_reserve"
                log_event(output, "round_deadline_interruption", round=index, error=str(exc), incumbent_retained=True)
                break
            ar = next(r for r in results if r["candidate"]["decoder"] == "ar")
            previous = next(r for r in results if r["candidate"]["id"] == incumbent["candidate"]["id"])
            selection = select_results([r for r in results if r["candidate"]["decoder"] != "ar"], ar, plan["accuracy_margin"])
            proposed = selection["research_incumbent"]
            old_goal = previous["pass_rate"]+.03 >= ar["pass_rate"] and previous["mean_generation_seconds"] < ar["mean_generation_seconds"]
            improved = (proposed["candidate"]["id"] != previous["candidate"]["id"] and (
                proposed["passed"] > previous["passed"] or
                (proposed["passed"] == previous["passed"] and proposed["mean_generation_seconds"] < .97*previous["mean_generation_seconds"]) or
                (selection["goal_screen_passed"] and not old_goal)))
            incumbent = proposed if improved else previous
            plateau = 0 if improved else plateau+1
            completed_rounds = index
            write_json(decision_path, {"round": index, "accepted": improved, "incumbent": incumbent,
                "ar": ar, "plateau": plateau, "selection": selection,
                "training_adapter": adapter, "selection_uses": "validation_only", "decided_unix": time.time()})
            log_event(output, "round_selection", round=index, accepted=improved,
                      decision_path=decision_path, selection=selection, plateau=plateau)
            update(plan, "round_complete", round=index, accepted=improved, plateau=plateau,
                   best_pass_rate=incumbent["pass_rate"], best_generation_seconds=incumbent["mean_generation_seconds"])
            if plateau >= plan["patience"]:
                reason = "validation_plateau"
                break
        final_selection = select_results([incumbent], ar, plan["accuracy_margin"])
        frozen = {"rounds": completed_rounds, "stop_reason": reason,
                  "research_candidate": incumbent["candidate"], "deployment_candidate": final_selection["deployment"]["candidate"],
                  "validation": {k:v for k,v in final_selection.items() if k != "selected_unix"},
                  "holdout_not_used_for_selection": True}
        freeze_json(output / "final_selection.json", frozen)
        log_event(output, "final_selection_frozen", selection=frozen)
        holdout = evaluate(plan, "final_holdout", [ar["candidate"], incumbent["candidate"]], split="holdout")
        write_json(output / "iteration_result.json", {"schema_version": 1, **frozen, "holdout": holdout,
                   "holdout_exposure_report": str(Path(plan["data"])/"data_accounting.json"),
                   "completed_unix": time.time(), "source_plan_sha256": file_sha(output/"iteration_plan.json")})
        update(plan, "completed", status="completed", completed_rounds=completed_rounds,
               ended_unix=time.time(), stop_reason=reason, child_pids=[])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--deadline-unix", type=float, default=1794038400)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--post-stop", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.max_rounds <= 32 or not 1 <= args.patience <= args.max_rounds or args.train_steps < 1:
        p.error("Invalid bounded search settings")
    if args.prepare:
        prepare(args)
        return 0
    plan = read_json(args.output / "iteration_plan.json")
    if plan is None:
        raise ValueError("Prepare immutable iteration plan before execution")
    if args.post_stop:
        state = read_json(args.output / "iteration_state.json", {})
        if os.environ.get("SERVICE_RESULT", "success") != "success" and state.get("status") == "running":
            count = state.get("failure_count", 0)+1
            update(plan, "process_lost", status="failed", failure_count=count)
            from experiment_notifications import notify
            notify(args.output, {"event_id": "iterative-process-lost-"+str(count),
                    "stage": state.get("stage"), "error_type": "IterationProcessLost", "output": str(args.output)})
        from experiment_notifications import flush_pending
        log_event(args.output, "service_post_stop", service_result=os.environ.get("SERVICE_RESULT", "success"),
                  exit_code=os.environ.get("EXIT_CODE"), exit_status=os.environ.get("EXIT_STATUS"))
        flush_pending(args.output, max_records=20, deadline_seconds=45)
        return 0
    if not args.execute:
        verify_plan(plan)
        print(json.dumps({"status": "plan_verified", "rounds": plan["max_rounds"]}))
        return 0
    def interrupted(signum, frame):
        raise InterruptedError("Iteration service interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with owned_descendant_cleanup():
            execute(plan)
    except AlreadyRunning:
        return 75
    except BaseException as exc:
        state = read_json(args.output / "iteration_state.json", {})
        count = state.get("failure_count", 0)+1
        update(plan, "failed", status="failed", failure_count=count, error_type=type(exc).__name__)
        write_json(args.output / "iteration_failure.json", {"error_type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(), "failed_unix": time.time()})
        log_event(args.output, "controller_failure", error_type=type(exc).__name__, error=str(exc),
                  traceback=traceback.format_exc(), failure_count=count)
        # Preserve the already requested abnormal-exit notification policy.
        from experiment_notifications import notify
        notify(args.output, {"event_id": "iterative-failure-"+str(count), "stage": state.get("stage"),
                            "error_type": type(exc).__name__, "output": str(args.output)})
        return 75 if count >= plan["failure_limit"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
