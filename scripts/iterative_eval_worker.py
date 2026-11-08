#!/usr/bin/env python3
"""One GPU evaluation shard with bounded asynchronous CPU grading."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import (digest, file_sha, read_json, read_jsonl, write_json,
                              freeze_json, task_id, shard_tasks, validate_candidate, adapter_identity)
from iteration_observability import log_event


class ForwardBudgetExceeded(RuntimeError):
    abort_dard_generation = True


def grading_options(plan):
    """Bound CPU parallelism separately from the queued generation receipts."""
    workers = plan.get("grading_workers", 2)
    capacity = plan.get("grading_queue_size", 4)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("grading_workers must be an integer between 1 and 8")
    if type(capacity) is not int or not workers <= capacity <= 128:
        raise ValueError("grading_queue_size must be an integer between grading_workers and 128")
    return workers, capacity


@contextmanager
def grading_pool(workers):
    """Do not turn a campaign deadline into a wait for queued grader timeouts.

    Running graders retain their original timeout. The CLI's descendant cleanup
    owns their subprocesses on an exceptional exit; pending jobs are cancelled.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def candidate_order(configs, shard, seed, enabled=False):
    """Counterbalance temporal order within model groups, retaining model reuse."""
    if not enabled:
        return list(configs)
    groups = {}
    for config in configs:
        key = (config["model"], config["revision"], digest(config.get("adapter")))
        groups.setdefault(key, []).append(config)
    ordered = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda c: digest([seed, c["id"]]))
        offset = (shard * max(1, len(rows)//4)) % len(rows)
        ordered.extend(rows[offset:]+rows[:offset])
    return ordered


def extract_for_grading(task, raw_output, prefix, plan):
    if plan.get("extraction_version") == "v2":
        from grading_audit import prepare_grade_request
        request = prepare_grade_request(task, raw_output, prefix=prefix,
            timeout=plan["grading_timeout"], extraction_version="v2")
        return request.code, request.receipt()
    from evaluation import strip_code
    return strip_code(prefix+raw_output), {"version": "legacy_strip_code"}


def generation_receipt(generation):
    """Preserve bounded token/decoder diagnostics; code text has its own fields."""
    if generation is None:
        return None
    # Only the visible text is already stored verbatim beside this receipt.
    # Pre-terminal and pre-cleanup strings can differ and must remain auditable.
    return {key: value for key, value in generation.items() if key != "text"}


def record_worker_state(output, record):
    write_json(Path(output) / "worker_state.json", record)
    log_event(output, "evaluation.worker_state", state=record)


def observed_grade(output, task, code, grading_function, timeout, context):
    began = time.perf_counter()
    log_event(output, "evaluation.grading_start", **context, code_sha256=digest(code),
              timeout_seconds=timeout, evaluator=task.final_evaluator)
    try:
        verdict = grading_function(task, code, "local", timeout=timeout)
    except BaseException as exc:
        log_event(output, "evaluation.grading_failure", **context,
                  wall_seconds=time.perf_counter()-began, error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc())
        raise
    log_event(output, "evaluation.grading_end", **context,
              wall_seconds=time.perf_counter()-began, verdict=verdict)
    return verdict


def run(args):
    from datasets_code import Task
    from dard_decoder import DARDConfig
    from evaluation import preflight_local, grade_final, strip_code
    from iterative_data import build_iterative_messages
    from models import create_backend, GenerationDeadlineExceeded
    import torch
    torch.set_num_threads(1)
    plan = read_json(args.plan)
    grading_workers, grading_queue_size = grading_options(plan)
    if file_sha(plan["tasks_path"]) != plan["tasks_sha256"]:
        raise ValueError("Evaluation tasks changed")
    if plan["split"] not in ("validation", "holdout", "smoke", "train_teacher"):
        raise ValueError("Unexpected evaluation split")
    tasks = shard_tasks(read_jsonl(plan["tasks_path"]), args.shard)
    if not tasks:
        raise ValueError("An empty GPU evaluation shard is forbidden")
    task_objects = [Task(**row) for row in tasks]
    output = Path(plan["output"]) / f"shard-{args.shard}"
    output.mkdir(parents=True, exist_ok=True)
    freeze_json(output / "binding.json", {"plan_sha256": file_sha(args.plan), "shard": args.shard})
    log_event(output, "evaluation.plan_verified", plan=args.plan, plan_sha256=file_sha(args.plan),
              split=plan["split"], tasks=len(tasks), candidates=plan["candidates"],
              deadline_unix=plan["deadline_unix"], tasks_sha256=plan["tasks_sha256"])
    log_event(output, "evaluation.environment_probe_start")
    env_receipt = preflight_local({t.language for t in task_objects}, tasks=task_objects,
                                 data_root=Path(plan["data_root"]))
    write_json(output / "environment.json", env_receipt)
    log_event(output, "evaluation.environment_probe_end", receipt=env_receipt)
    pending = {}
    backend = None
    backend_key = None
    completed = 0
    last_generation_completed_unix = None
    def state(stage, **details):
        now = time.monotonic()
        running = ready = 0
        for future in pending:
            if future.done():
                ready += 1
            elif future.running():
                running += 1
        record_worker_state(output, {"status": "running", "stage": stage, "pid": os.getpid(),
                   "shard": args.shard, "completed": completed, "updated_unix": time.time(),
                   "grading_workers": grading_workers, "grading_queue_size": grading_queue_size,
                   "grading_pending": len(pending), "grading_running": running,
                   "grading_ready": ready, "grading_queued": max(0, len(pending)-running-ready),
                   "grading_oldest_pending_seconds": max((now-item[2] for item in pending.values()), default=0.),
                   "last_generation_completed_unix": last_generation_completed_unix, **details})

    def grade(task, code, context):
        # Final grader API is reused for fixed validation scoring, never feedback
        # to generation. Validation is explicitly labelled as selection data.
        return observed_grade(output, task, code, grade_final, plan["grading_timeout"], context)

    def collect(block=False, reason="queue_capacity"):
        nonlocal completed
        if not pending:
            return
        if block and time.time() >= plan["deadline_unix"]:
            state("grading_deadline_reached", waiting_reason=reason)
            raise TimeoutError("Evaluation deadline reached while waiting for grading")
        ready, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
        while block and not ready:
            remaining = plan["deadline_unix"] - time.time()
            if remaining <= 0:
                state("grading_deadline_reached", waiting_reason=reason)
                raise TimeoutError("Evaluation deadline reached while waiting for grading")
            state("waiting_for_grading", waiting_reason=reason)
            ready, _ = wait(pending, timeout=min(5., remaining), return_when=FIRST_COMPLETED)
        for future in ready:
            row, path, _ = pending.pop(future)
            verdict = future.result()
            if verdict.get("passed") not in (True, False) or verdict["status"] in ("infra_error", "not_tested"):
                raise RuntimeError("Grading infrastructure failed: " + str(verdict.get("status")))
            row.update(passed=bool(verdict["passed"]), status=verdict["status"],
                       grading_seconds=verdict.get("grading_wall_s", verdict.get("test_wall_s")),
                       completed_unix=time.time())
            if plan.get("save_grading_verdict"):
                row["grading_verdict"] = verdict
            write_json(path, row)
            log_event(output, "evaluation.case_completed", candidate_id=row["candidate_id"],
                      task_key=row["task_key"], result_path=path, result_sha256=file_sha(path),
                      passed=row["passed"], status=row["status"])
            completed += 1
            state("evaluating", candidate_id=row["candidate_id"], task_id=row["task_id"])

    try:
        with grading_pool(grading_workers) as pool:
            ordered_configs = candidate_order(plan["candidates"], args.shard, plan["seed"],
                                             plan.get("counterbalanced_order", False))
            log_event(output, "evaluation.candidate_order", ids=[c["id"] for c in ordered_configs],
                      counterbalanced=plan.get("counterbalanced_order", False))
            for config in ordered_configs:
                validate_candidate(config)
                log_event(output, "evaluation.candidate_start", candidate=config, config_sha256=digest(config))
                if config.get("adapter") != adapter_identity((config.get("adapter") or {}).get("path")):
                    raise ValueError("Adapter bytes changed after plan freeze")
                remaining = []
                for row, task in zip(tasks, task_objects):
                    path = output / "cases" / config["id"] / (task_id(row) + ".json")
                    saved = read_json(path)
                    if saved is not None:
                        if (saved["task_sha256"] != digest(row) or saved["config_sha256"] != digest(config)
                                or saved.get("infrastructure_error") or type(saved.get("passed")) is not bool):
                            raise ValueError("Saved evaluation binding mismatch")
                        completed += 1
                        log_event(output, "evaluation.case_recovered", candidate_id=config["id"],
                                  task_key=task_id(row), result_path=path, result_sha256=file_sha(path),
                                  passed=saved["passed"], detailed_generation_receipt_available=bool(
                                      saved.get("generation_receipt") and Path(saved["generation_receipt"]).is_file()))
                    else:
                        remaining.append((row, task, path))
                if not remaining:
                    log_event(output, "evaluation.candidate_skipped", candidate_id=config["id"],
                              reason="all_cases_already_recorded")
                    continue
                key = (config["model"], config["revision"], digest(config.get("adapter")))
                if key != backend_key:
                    # Graders hold CPU task/code/context values only. Their
                    # results can be collected after the next model is loaded.
                    collect(False)
                    if backend is not None:
                        log_event(output, "evaluation.model_close_start", model_key=backend_key)
                        backend.close()
                        log_event(output, "evaluation.model_close_end", model_key=backend_key)
                    state("loading_model", candidate_id=config["id"])
                    load_started = time.perf_counter()
                    log_event(output, "evaluation.model_load_start", candidate_id=config["id"],
                              model=config["model"], revision=config["revision"], adapter=config.get("adapter"))
                    backend = create_backend(config["decoder"], model_name=config["model"],
                              revision=config["revision"], quant="bf16", device="cuda:0",
                              local_files_only=True, adapter_path=(config.get("adapter") or {}).get("path"))
                    backend_key = key
                    write_json(output / ("model-" + digest(key)[:16] + ".json"), backend.metadata())
                    log_event(output, "evaluation.model_load_end", candidate_id=config["id"],
                              wall_seconds=time.perf_counter()-load_started, metadata=backend.metadata())
                    log_event(output, "evaluation.warmup_start", candidate_id=config["id"],
                              tokens=32, steps=4, seed=42)
                    warmup = backend.generate("py", "Write a function identity(x) returning x.",
                         "ar" if config["decoder"] == "ar" else "standard", seed=42,
                         num_new_tokens=32, steps=4, dard_config=DARDConfig(temperature=0, top_p=1),
                         temperature=0, top_p=1, max_time=120)
                    log_event(output, "evaluation.warmup_end", candidate_id=config["id"],
                              generation=generation_receipt(warmup))
                if plan.get("candidate_warmup"):
                    from dataclasses import replace
                    warm_config = replace(DARDConfig(**config["dard"]), steps_per_block=4,
                                          diagnostics="summary", max_step_records=0)
                    log_event(output, "evaluation.candidate_warmup_start", candidate_id=config["id"])
                    warm = backend.generate("py", "Write a function identity(x) returning x.",
                        config["decoder"], seed=42, num_new_tokens=config["tokens"], steps=4,
                        dard_config=warm_config, temperature=warm_config.temperature,
                        top_p=warm_config.top_p, max_time=min(120, max(.1, plan["deadline_unix"]-time.time())))
                    log_event(output, "evaluation.candidate_warmup_end", candidate_id=config["id"],
                              generation=generation_receipt(warm), excluded_from_measurements=True)
                for raw, task, path in remaining:
                    if time.time() >= plan["deadline_unix"]:
                        raise TimeoutError("Evaluation deadline reached")
                    while len(pending) >= grading_queue_size:
                        collect(True)
                    collect(False)
                    cfg = DARDConfig(**config["dard"])
                    prefix = "\n```" + {"py": "python", "js": "javascript", "rs": "rust", "cpp": "cpp"}[task.language] + "\n" if config["prefix"] == "code_fence" else ""
                    state("generation", candidate_id=config["id"], task_id=task.task_id)
                    context = {"candidate_id": config["id"], "task_key": task_id(raw),
                               "task_id": task.task_id, "source_id": task.source_id,
                               "language": task.language, "split": plan["split"]}
                    log_event(output, "evaluation.generation_start", **context,
                              config_sha256=digest(config), task_sha256=digest(raw),
                              seed=int(digest([task.source_id, task.language, plan["seed"]])[:8], 16),
                              assistant_prefix=prefix, forward_budget=config["forward_budget"],
                              generation_timeout=plan["generation_timeout"], case_path=path)
                    calls = [0]
                    def budget_hook(module, inputs, kwargs):
                        if calls[0] >= config["forward_budget"]:
                            raise ForwardBudgetExceeded("Total generation forward budget exhausted")
                        calls[0] += 1
                    hook = backend.model.register_forward_pre_hook(budget_hook, with_kwargs=True)
                    began = time.perf_counter()
                    generation = None
                    error = None
                    partial = None
                    failure_duration = None
                    generation_error = None
                    failure_traceback = None
                    effective_timeout = min(plan["generation_timeout"], max(.1, plan["deadline_unix"]-time.time()))
                    timeout_scope = "candidate_limit" if effective_timeout == plan["generation_timeout"] else "campaign_deadline"
                    try:
                        generation = backend.generate_messages(build_iterative_messages(task), config["decoder"],
                            assistant_prefix=prefix, seed=int(digest([task.source_id, task.language, plan["seed"]])[:8], 16),
                            num_new_tokens=config["tokens"], steps=config["standard_steps"], dard_config=cfg,
                            temperature=cfg.temperature, top_p=cfg.top_p,
                            max_time=effective_timeout)
                    except BaseException as exc:
                        generation_error = exc
                        failure_traceback = exc.__traceback__
                    finally:
                        hook.remove()
                    if generation_error is not None:
                        # Match the original failed-generation span, including
                        # hook removal, while keeping disk logging outside it.
                        failure_duration = time.perf_counter()-began
                        exc = generation_error
                        partial = getattr(exc, "generation_diagnostics", None)
                        handled = isinstance(exc, (ForwardBudgetExceeded, GenerationDeadlineExceeded))
                        log_event(output, "evaluation.generation_failure", **context,
                                  error_type=type(exc).__name__, error=str(exc),
                                  traceback="".join(traceback.format_exception(type(exc), exc, failure_traceback)),
                                  wall_seconds=failure_duration, handled=handled,
                                  partial_generation_diagnostics=partial,
                                  partial_generation_diagnostics_available=partial is not None)
                        if not handled:
                            raise exc.with_traceback(exc.__traceback__)
                        if isinstance(exc, GenerationDeadlineExceeded) and timeout_scope == "campaign_deadline":
                            raise TimeoutError("Campaign deadline interrupted a generation; no score recorded") from exc
                        error = type(exc).__name__
                    duration = generation["wall_s"] if generation else failure_duration
                    last_generation_completed_unix = time.time()
                    text = generation["text"] if generation else ""
                    if generation and plan.get("extraction_version") == "v2":
                        text = generation.get("text_with_unfilled_masks", text)
                    code, extraction = extract_for_grading(task, text, prefix, plan) if generation else ("", None)
                    # A crash between generation and grading can trigger a retry.
                    # Preserve both attempts, including their differing timing.
                    receipt_path = output / "generations" / config["id"] / (
                        f"{task_id(raw)}-{os.getpid()}-{time.time_ns()}.json")
                    freeze_json(receipt_path, {"schema_version": 1, **context,
                        "config_sha256": digest(config), "task_sha256": digest(raw),
                        "case_path": str(path), "raw_output": text, "code": code,
                        "extraction": extraction, "effective_timeout": effective_timeout,
                        "timeout_scope": timeout_scope,
                        "generation": generation_receipt(generation), "error_type": error,
                        "partial_generation_diagnostics": partial})
                    log_event(output, "evaluation.generation_end", **context, status="failed" if error else "completed",
                              generation_seconds=duration, forward_calls=calls[0],
                              receipt_path=receipt_path, receipt_sha256=file_sha(receipt_path))
                    record = {"schema_version": 1, "split": plan["split"], "task_key": task_id(raw),
                        "task_sha256": digest(raw), "candidate_id": config["id"], "config_sha256": digest(config),
                        "task_id": task.task_id, "source_id": task.source_id, "language": task.language,
                        "generation_seconds": duration, "forward_calls": calls[0], "infrastructure_error": False,
                        "raw_output": text, "code": code, "code_sha256": digest(code),
                        "forced_tokens": generation.get("forced_tokens", 0) if generation else 0,
                        "visible_output_tokens": generation.get("visible_output_tokens", 0) if generation else 0,
                        "terminal_token_index": generation.get("terminal_token_index") if generation else None,
                        "generation_receipt": str(receipt_path), "generation_receipt_sha256": file_sha(receipt_path),
                        "prefix": prefix, "selection_feedback": plan["split"] == "validation"}
                    record.update(measurement_seed=plan["seed"], extraction=extraction,
                        effective_timeout=effective_timeout, timeout_scope=timeout_scope,
                        incomplete=generation.get("incomplete", False) if generation else True,
                        verification_certified=generation.get("verification_certified") if generation else None,
                        unresolved_mask_tokens=generation.get("unfilled_mask_tokens", 0) if generation else None,
                        model_wall_seconds=generation.get("model_wall_s") if generation else None,
                        gpu_peak_gib=generation.get("gpu_peak_gib") if generation else None,
                        rendered_prompt_sha256=generation.get("rendered_prompt_sha256") if generation else None,
                        messages_sha256=generation.get("messages_sha256") if generation else None,
                        decoder_stop_reason=generation.get("decoder_stop_reason") if generation else None,
                        completion_reverified_tokens=generation.get("completion_reverified_tokens", 0) if generation else 0,
                        completion_rejected_tokens=generation.get("completion_rejected_tokens", 0) if generation else 0)
                    if error:
                        record.update(passed=False, status=error, grading_seconds=0, completed_unix=time.time())
                        write_json(path, record)
                        log_event(output, "evaluation.case_completed", **context, result_path=path,
                                  result_sha256=file_sha(path), passed=False, status=error)
                        completed += 1
                    else:
                        pending[pool.submit(grade, task, code, context)] = (record, path, time.monotonic())
                log_event(output, "evaluation.candidate_generation_complete", candidate_id=config["id"],
                          grading_pending=len(pending))
            while pending:
                collect(True, reason="final_drain")
        record_worker_state(output, {"status": "completed", "completed": completed, "pid": os.getpid(),
                                "updated_unix": time.time(), "shard": args.shard})
    finally:
        if backend is not None:
            log_event(output, "evaluation.model_close_start", model_key=backend_key)
            backend.close()
            log_event(output, "evaluation.model_close_end", model_key=backend_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan, {})
    log_output = Path(plan.get("output", args.plan.parent)) / f"shard-{args.shard}"
    log_event(log_output, "evaluation.process_start", arguments=vars(args), shard=args.shard,
              plan_sha256=file_sha(args.plan))
    from scripts.four_gpu import owned_descendant_cleanup
    received_signal = [None]
    def interrupted(signum, frame):
        received_signal[0] = signum
        raise InterruptedError("Evaluation worker interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with owned_descendant_cleanup():
            run(args)
    except BaseException as exc:
        if received_signal[0] is not None:
            log_event(log_output, "evaluation.signal_received", signal=received_signal[0])
        log_event(log_output, "evaluation.process_failure", error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc(), received_signal=received_signal[0])
        raise
    else:
        log_event(log_output, "evaluation.process_end", status="completed", exit_code=0)


if __name__ == "__main__":
    main()
