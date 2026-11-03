#!/usr/bin/env python3
"""Server benchmark: DiffuCoder published decoding vs independent DARD reimplementation.

This is a FULL-FUNCTION, one-sample code-generation PILOT, NOT official pass@1.
Only Docker tests are allowed; native execution of model-generated code is disabled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from dard_decoder import DARDConfig
from datasets_code import load_tasks
from evaluation import strip_code, preflight_docker, test_task, category

ROOT = Path(__file__).resolve().parent


def cli(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--suites", default="humaneval,mbpp,multiple_rs",
                        help="Comma-separated from humaneval,mbpp,multiple_rs")
    parser.add_argument("--limit", type=int, default=40,
                        help="Number of tasks PER suite; 0 means all (164+500+~160)")
    parser.add_argument("--start", type=int, default=0, help="Offset per suite")
    parser.add_argument("--decoders", default="standard,dard", help="standard, dard, or standard,dard")
    parser.add_argument("--runner", choices=("none", "docker"), default="docker")
    parser.add_argument("--pull-images", action="store_true")
    parser.add_argument("--repairs", type=int, default=1, help="Extra fixes after failed first attempt (requires Docker)")
    parser.add_argument("--tokens", type=int, default=256, help="Fixed dLLM output slots")
    parser.add_argument("--standard-steps", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dard-steps", type=int, default=192, help="DARD steps PER BLOCK")
    parser.add_argument("--tau-c", type=float, default=.10)
    parser.add_argument("--tau-u", type=float, default=.30)
    parser.add_argument("--transfer-limit", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=.3)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--quant", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--model", default="apple/DiffuCoder-7B-Instruct")
    parser.add_argument("--test-timeout", type=int, default=45)
    parser.add_argument("--output", type=Path, default=None, help="Stable output dir; resuming skips completed cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-data", action="store_true", help="Download/check datasets then exit (no model)")
    parser.add_argument("--self-check", action="store_true", help="Check basic imports/params then exit")
    args = parser.parse_args(argv)
    args.suites = [x.strip() for x in args.suites.split(",") if x.strip()]
    args.decoders = [x.strip() for x in args.decoders.split(",") if x.strip()]
    if not args.suites or any(x not in ("humaneval", "mbpp", "multiple_rs") for x in args.suites):
        parser.error("--suites only humaneval,mbpp,multiple_rs")
    if not args.decoders or any(x not in ("standard", "dard") for x in args.decoders):
        parser.error("--decoders only standard,dard")
    if args.limit < 0 or args.start < 0 or args.repairs not in (0, 1, 2):
        parser.error("--limit/--start >= 0; --repairs in {0,1,2}")
    if args.runner == "none" and args.repairs != 0 and not args.check_data and not args.self_check:
        parser.error("--runner none requires --repairs 0 (no test feedback)")
    if args.tokens < 16 or args.tokens > 1024 or args.standard_steps < 1:
        parser.error("--tokens [16,1024]; --standard-steps >= 1")
    DARDConfig(block_size=args.block_size, steps_per_block=args.dard_steps, tau_c=args.tau_c,
               tau_u=args.tau_u, transfer_limit=args.transfer_limit,
               temperature=args.temperature, top_p=args.top_p).validate()
    return args


def _summary(results: list[dict], metadata: dict, output: Path):
    groups = defaultdict(list)
    for row in results:
        groups[(row["decoder"], row["dataset"], row["language"])].append(row)
    group_summaries = {}
    for (dec, ds, lang), rows in groups.items():
        evaluated = [r for r in rows if r["first_pass"] is not None and r["final_pass"] is not None]
        initial = sum(r["first_pass"] is True for r in evaluated)
        final = sum(r["final_pass"] is True for r in evaluated)
        repaired = sum(r["first_pass"] is False and r["final_pass"] is True for r in evaluated)
        fail_initial = len(evaluated) - initial
        group_summaries[f"{dec}/{ds}/{lang}"] = {
            "cases": len(rows), "evaluated": len(evaluated),
            "not_evaluated": len(rows) - len(evaluated),
            "first_pass_count": initial if evaluated else None,
            "final_pass_count": final if evaluated else None,
            "first_pass_rate": round(initial / len(evaluated), 4) if evaluated else None,
            "final_pass_rate": round(final / len(evaluated), 4) if evaluated else None,
            "repair_success_given_initial_failure": round(repaired / fail_initial, 4) if fail_initial else None,
            "time_generation_s": round(sum(r["generation_wall_s"] for r in rows), 3),
            "time_test_s": round(sum(r["test_wall_s"] for r in rows), 3),
            "forward_passes": sum(r["model_forward_passes"] for r in rows),
            "forced_tokens": sum(r["forced_tokens"] for r in rows),
            "promoted_tokens": sum(r.get("promoted_tokens", 0) for r in rows),
            "revoked_tokens": sum(r.get("revoked_tokens", 0) for r in rows),
            "attention_fallback_steps": sum(r.get("attention_fallback_steps", 0) for r in rows),
            "max_gpu_peak_gib": max((r.get("max_gpu_peak_gib", 0) for r in rows), default=0),
            "errors": dict(__import__("collections").Counter(r["final_status"] for r in rows)),
        }
    summary = {"metadata": metadata, "groups": group_summaries}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return summary


def main(argv=None) -> int:
    args = cli(argv)
    if args.self_check:
        import torch
        print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        return 0
    print("[info] DARD is an independent paper-mechanisms implementation, NOT official code.")
    if args.runner == "docker" and not args.check_data:
        preflight_docker(set("rs" if s == "multiple_rs" else "py" for s in args.suites),
                         pull_missing=args.pull_images)
    cache = ROOT / "data" / "cache"
    tasks = load_tasks(args.suites, cache, args.limit, args.start)
    if args.check_data:
        return 0
    if not tasks:
        print("No tasks selected.")
        return 2
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "results" / now
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "case_results.jsonl"
    traces_path = output / "runs.jsonl"
    errors_path = output / "errors.jsonl"
    metadata = {"model": args.model, "quant": args.quant, "runner": args.runner,
                "decoders": args.decoders, "suites": args.suites,
                "limit_per_suite": args.limit, "start_per_suite": args.start,
                "repairs": args.repairs, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p,
                "tokens": args.tokens, "standard_steps": args.standard_steps,
                "dard_settings": {"block_size": args.block_size, "steps_per_block": args.dard_steps,
                                  "tau_c": args.tau_c, "tau_u": args.tau_u,
                                  "transfer_limit": args.transfer_limit},
                "model_load_s": None,
                "benchmark_protocol": "complete_function_pilot_not_official_pass_at_1",
                "dard_status": "independent_paper_mechanisms_not_author_verified",
                "started_utc": datetime.now(timezone.utc).isoformat()}
    if (output / "config.json").exists():
        saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
        keys = ("model", "quant", "runner", "decoders", "suites", "limit_per_suite", "start_per_suite",
                "repairs", "seed", "temperature", "top_p", "tokens", "standard_steps", "dard_settings")
        if any(saved.get(k) != metadata.get(k) for k in keys):
            raise RuntimeError("Resume config mismatch; use a NEW --output directory for changed settings")
        metadata = saved
    else:
        (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    results: list[dict] = []
    if results_path.exists():
        results = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["decoder"], r["dataset"], r["task_id"]) for r in results}
    todo = [(d, t) for t in tasks for d in args.decoders if (d, t.dataset, t.task_id) not in done]
    if not todo:
        print("[resume] All cases already complete.")
        _summary(results, metadata, output)
        return 0
    from models import DiffuCoderBackend
    backend = DiffuCoderBackend(model_name=args.model, quant=args.quant)
    metadata["model_load_s"] = backend.model_load_s
    (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg = DARDConfig(block_size=args.block_size, steps_per_block=args.dard_steps,
                     tau_c=args.tau_c, tau_u=args.tau_u,
                     transfer_limit=args.transfer_limit,
                     temperature=args.temperature, top_p=args.top_p)
    planned = len(todo)
    print(f"[run] {planned} remaining (out of {len(tasks)*len(args.decoders)} total)", flush=True)
    try:
        with results_path.open("a", encoding="utf-8") as case_log, \
             traces_path.open("a", encoding="utf-8") as run_log, \
             errors_path.open("a", encoding="utf-8") as error_log:
            for index, (dec, task) in enumerate(todo, 1):
                print(f"[{index}/{planned}] {dec} {task.dataset} {task.task_id}", flush=True)
                attempts = []
                last_code = ""
                last_feedback = ""
                start = time.perf_counter()
                for attempt in range(args.repairs + 1):
                    try:
                        resp = backend.generate(task.language, task.prompt, dec,
                                                seed=args.seed + task.number + attempt * 100003,
                                                num_new_tokens=args.tokens,
                                                steps=args.standard_steps, dard_config=cfg,
                                                temperature=args.temperature, top_p=args.top_p,
                                                old_code=last_code, error=last_feedback)
                        last_code = strip_code(resp["text"])
                        verdict = test_task(task, last_code, args.runner, args.test_timeout)
                        record = {"decoder": dec, "dataset": task.dataset, "language": task.language,
                                  "task_id": task.task_id, "attempt": attempt+1,
                                  "code": last_code, "generation": {k:v for k,v in resp.items() if k != "text"},
                                  "test": verdict,
                                  "error_category": category(verdict["status"], verdict["feedback"])}
                        run_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        run_log.flush()
                        attempts.append(record)
                        print(f"   attempt={attempt+1} status={verdict['status']} "
                              f"gen={resp['wall_s']:.2f}s test={verdict['test_wall_s']:.2f}s", flush=True)
                        if verdict["passed"] is True or verdict["passed"] is None:
                            break
                        last_feedback = verdict["feedback"] or verdict["status"]
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        error_log.write(json.dumps({"decoder": dec, "task_id": task.task_id,
                                                    "attempt": attempt+1, "exception": repr(exc),
                                                    "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
                        error_log.flush()
                        # GPU out-of-memory or incompatible custom attention mask
                        # is not a model *wrong answer*. Preserve resumable error.
                        print(f"   ERROR: {exc}", flush=True)
                        if "out of memory" in str(exc).lower():
                            raise
                        break
                if not attempts:
                    continue
                first, final = attempts[0], attempts[-1]
                first_test = first["test"]["passed"]
                last_test = final["test"]["passed"]
                row = {"decoder": dec, "dataset": task.dataset, "language": task.language,
                       "task_id": task.task_id, "attempts": len(attempts),
                       "first_pass": first_test, "final_pass": last_test,
                       "first_status": first["test"]["status"], "final_status": final["test"]["status"],
                       "first_error_category": first["error_category"],
                       "last_error_category": final["error_category"],
                       "generation_wall_s": round(sum(a["generation"]["wall_s"] for a in attempts), 3),
                       "test_wall_s": round(sum(a["test"]["test_wall_s"] for a in attempts), 3),
                       "model_forward_passes": sum(a["generation"].get("model_forward_passes",0) for a in attempts),
                       "forced_tokens": sum(a["generation"].get("forced_tokens",0) for a in attempts),
                       "promoted_tokens": sum(a["generation"].get("promoted_tokens",0) for a in attempts),
                       "revoked_tokens": sum(a["generation"].get("revoked_tokens",0) for a in attempts),
                       "attention_fallback_steps": sum(a["generation"].get("attention_fallback_steps",0) for a in attempts),
                       "max_gpu_peak_gib": max((a["generation"].get("gpu_peak_gib",0) for a in attempts), default=0),
                       "total_wall_s": round(time.perf_counter()-start, 3)}
                case_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                case_log.flush()
                results.append(row)
                if index % 5 == 0:
                    _summary(results, metadata, output)
    except KeyboardInterrupt:
        print("Interrupted; saved finished cases.")
    finally:
        _summary(results, metadata, output)
    print("[saved]", output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FATAL:", type(exc).__name__, str(exc), file=sys.stderr)
        raise
