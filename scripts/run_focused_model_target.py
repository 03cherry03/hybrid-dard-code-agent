#!/usr/bin/env python3
"""Focused four-GPU training and target trade-off evaluation.

The controller trains one adapter, compares base and adapter checkpoints on a
small development cohort, then evaluates only the selected operating point.
It deliberately excludes unrelated model canaries and broad hyperparameter
grids. The model gate is never changed automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iterative_tuning import adapter_identity, read_json, read_jsonl
from scripts.iterative_experiments import evaluate

DEFAULT_PYTHON = Path("/data/hybrid-dard/envs/iterative-training/bin/python")
DEFAULT_HF_HOME = Path("/data/hybrid-dard/cache/huggingface")


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    hf = Path(env.get("HF_HOME", str(DEFAULT_HF_HOME))).resolve()
    env.setdefault("HF_HOME", str(hf))
    env.setdefault("HF_HUB_CACHE", str(hf / "hub"))
    env.setdefault("HF_MODULES_CACHE", str(hf / "modules"))
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HUB_CACHE"])
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return env


def cuda_preflight(python: Path, env: dict[str, str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    smi = subprocess.run(["nvidia-smi", "-L"], env=env, text=True,
                         capture_output=True, check=False)
    if smi.returncode != 0:
        raise RuntimeError("nvidia-smi -L failed: " + smi.stderr[-1000:])
    code = ("import json,torch; torch.cuda.init(); print(json.dumps({"
            "'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),"
            "'names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
    check = subprocess.run([str(python), "-c", code], env=env, text=True,
                           capture_output=True, check=False)
    if check.returncode != 0:
        raise RuntimeError("PyTorch CUDA preflight failed: " + check.stderr[-2000:])
    receipt = json.loads([x for x in check.stdout.splitlines() if x.strip()][-1])
    if not receipt["available"] or int(receipt["count"]) < 4:
        raise RuntimeError("Four CUDA devices are required: " + json.dumps(receipt))
    result = {"status": "passed", "nvidia_smi": smi.stdout, "torch": receipt,
              "environment": {k: env[k] for k in (
                  "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE",
                  "HF_MODULES_CACHE", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
                  "TORCH_NCCL_ASYNC_ERROR_HANDLING")}}
    (output / "cuda_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train(args, env: dict[str, str], output: Path) -> Path:
    train_out = output / "training"
    state_path = train_out / "training_state.json"
    adapter = train_out / "final_adapter"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("status") == "completed" and adapter.is_dir():
            config_path = train_out / "training_config.json"
            config = read_json(config_path, {}) if config_path.is_file() else {}
            if config.get("objective_mode") not in (None, "masked_ce"):
                raise RuntimeError("Existing adapter was not trained with masked CE")
            return adapter
        if state.get("status") == "running":
            raise RuntimeError("Training is already running: " + str(train_out))
    train_out.mkdir(parents=True, exist_ok=True)
    command = [str(args.python), "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc_per_node=4", str(ROOT / "scripts" / "train_diffusion_lora.py"),
               "--train-jsonl", str(args.train_jsonl), "--output", str(train_out),
               "--base-model", args.base_model, "--revision", args.revision,
               "--max-length", str(args.max_length), "--epochs", "0",
               "--max-steps", str(args.steps), "--batch-size", "1",
               "--gradient-accumulation", str(args.gradient_accumulation),
               "--learning-rate", str(args.learning_rate), "--rank", str(args.rank),
               "--alpha", str(args.rank * 2),
               "--target-modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
               "--lora-dropout", "0.05", "--checkpoint-every", str(args.checkpoint_every),
               "--balanced-languages",
               "--seed", str(args.seed), "--deadline-unix", str(args.deadline_unix),
               "--deadline-reserve-seconds", str(args.deadline_reserve_seconds),
               "--local-files-only"]
    log_path = output / "training.stdout.log"
    with log_path.open("a") as log:
        log.write(json.dumps({"event": "training_launch", "command": command,
                              "time": time.time()}) + "\n")
        log.flush()
        result = subprocess.run(command, env=env, cwd=str(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed ({result.returncode}); see {log_path}")
    state = read_json(state_path, {})
    if state.get("status") != "completed" or not adapter.is_dir():
        raise RuntimeError("Training did not produce a completed adapter: " + json.dumps(state))
    return adapter


def dev_tasks(base_plan: dict, output: Path) -> Path:
    path = output / "data" / "dev_tasks.jsonl"
    if path.is_file():
        return path
    rows = read_jsonl(Path(base_plan["data"]) / "validation_tasks.jsonl")
    selected = []
    for language in ("py", "js", "cpp", "rs"):
        selected.extend([row for row in rows if row["language"] == language][:12])
    if len(selected) != 48:
        raise ValueError("Development cohort must contain 12 tasks per language")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    return path


def controls(plan: dict) -> tuple[dict, list[dict], list[dict]]:
    configs = plan.get("initial_candidates", [])
    ar = next(copy.deepcopy(c) for c in configs
              if c["decoder"] == "ar" and c.get("tokens") == 512
              and c.get("prefix") == "code_fence")
    native = [copy.deepcopy(c) for c in configs if c["decoder"] == "standard"
              and c.get("tokens") == 512 and c.get("prefix") == "code_fence"
              and c.get("standard_steps") in (16, 32)]
    pool = [c for c in configs if c["decoder"] == "dard"
            and c.get("tokens") == 512 and c.get("prefix") == "code_fence"
            and c.get("dard", {}).get("temperature") == 0
            and c.get("dard", {}).get("top_p") == 1
            and c.get("dard", {}).get("completion_policy") == "resample_then_verify"]
    dard = sorted(pool, key=lambda c: (c["dard"].get("steps_per_block", 0),
                                      c["dard"].get("block_size", 0)))[:3]
    if len(native) < 2 or len(dard) < 2:
        raise ValueError("Base plan lacks focused native/DARD controls")
    return ar, native, dard


def adapter_config(config: dict, adapter: dict) -> dict:
    result = copy.deepcopy(config)
    result["id"] += "-trained"
    result["adapter"] = adapter
    return result


def pick(ar: dict, native: list[dict], dard: list[dict]) -> dict:
    """Select by target feasibility first, then smallest AR gap and latency."""
    fastest_native = min(native, key=lambda row: row["mean_generation_seconds"])
    ar_accuracy = float(ar["pass_rate"])
    candidates = []
    for row in dard:
        native_ok = row["pass_rate"] > fastest_native["pass_rate"]
        faster_than_ar = row["mean_generation_seconds"] < ar["mean_generation_seconds"]
        gap = ar_accuracy - float(row["pass_rate"])
        target_ok = native_ok and faster_than_ar and gap <= 0.05 and gap >= 0
        candidates.append((target_ok, gap, row["mean_generation_seconds"], row))
    feasible = [item for item in candidates if item[0]]
    pool = feasible or candidates
    return min(pool, key=lambda item: (not item[0], max(item[1], 0.0), item[2], -item[3]["pass_rate"]))[3]


def assess(rows: list[dict]) -> dict:
    """Evaluate the requested ordering without changing candidate selection."""
    ar = next(row for row in rows if row["candidate"]["decoder"] == "ar")
    native = min((row for row in rows if row["candidate"]["decoder"] == "standard"),
                 key=lambda row: row["mean_generation_seconds"])
    dard = next(row for row in rows if row["candidate"]["decoder"] == "dard")
    gap = float(ar["pass_rate"]) - float(dard["pass_rate"])
    return {"ar": ar["candidate"]["id"], "native": native["candidate"]["id"],
            "dard": dard["candidate"]["id"], "accuracy_gap_pp": gap * 100,
            "constraints": {
                "ar_gt_dard": ar["pass_rate"] > dard["pass_rate"],
                "dard_gt_native": dard["pass_rate"] > native["pass_rate"],
                "native_lt_dard": native["mean_generation_seconds"] < dard["mean_generation_seconds"],
                "dard_lt_ar": dard["mean_generation_seconds"] < ar["mean_generation_seconds"],
                "gap_within_5pp": 0 <= gap <= .05},
            "target_met": all((ar["pass_rate"] > dard["pass_rate"],
                                dard["pass_rate"] > native["pass_rate"],
                                native["mean_generation_seconds"] < dard["mean_generation_seconds"],
                                dard["mean_generation_seconds"] < ar["mean_generation_seconds"],
                                0 <= gap <= .05))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--base-model", default="apple/DiffuCoder-7B-cpGRPO")
    parser.add_argument("--revision", default="98bc42bfd871eeec92fd0a8439ab34f75b405204")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--deadline-reserve-seconds", type=float, default=900)
    parser.add_argument("--full-seeds", default="20261111,20261112,20261113")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--adapter-path", type=Path,
                        help="Use an existing completed adapter when skipping training")
    parser.add_argument("--full", action="store_true",
                        help="Run full validation, repeated seeds, and one holdout after dev selection")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.base_plan.resolve())
    if args.train_jsonl is None:
        plan_training = Path(plan["data"]) / "training_examples.jsonl"
        iterative_training = Path("/data/hybrid-dard/results/code-generation-iterative-20261107/data/training_examples.jsonl")
        args.train_jsonl = plan_training if plan_training.is_file() else iterative_training
    args.train_jsonl = args.train_jsonl.resolve()
    if not args.train_jsonl.is_file():
        raise FileNotFoundError(args.train_jsonl)
    env = runtime_env()
    cuda_preflight(args.python, env, args.output)
    adapter_path = args.adapter_path.resolve() if args.adapter_path else args.output / "training" / "final_adapter"
    if not args.skip_training:
        adapter_path = train(args, env, args.output)
    elif not adapter_path.is_dir():
        raise FileNotFoundError("Completed adapter not found: " + str(adapter_path))
    adapter = adapter_identity(adapter_path)
    eval_plan = copy.deepcopy(plan)
    eval_plan["output"] = str(args.output)
    eval_plan["python"] = str(args.python)
    eval_plan["deadline_unix"] = args.deadline_unix
    # Preserve the frozen UUID records used by worker_env; replacing them with
    # integer indices would break per-GPU cache isolation during evaluation.
    ar, native, dard = controls(eval_plan)
    trained_native = [adapter_config(c, adapter) for c in native]
    trained_dard = [adapter_config(c, adapter) for c in dard]
    dev_rows = evaluate(eval_plan, "focused_dev", [ar] + native + dard + trained_native + trained_dard,
                        split="validation", tasks_override=dev_tasks(eval_plan, args.output),
                        deadline=args.deadline_unix - 600)
    by_id = {row["candidate"]["id"]: row for row in dev_rows}
    ar_row = by_id[ar["id"]]
    native_rows = [by_id[c["id"]] for c in trained_native]
    dard_rows = [by_id[c["id"]] for c in trained_dard]
    selected = pick(ar_row, native_rows, dard_rows)
    selected_cfg = selected["candidate"]
    fastest_native = min(native_rows, key=lambda row: row["mean_generation_seconds"])["candidate"]
    final_configs = [ar, fastest_native, selected_cfg]
    validation, final, holdout = [], [], []
    if args.full:
        validation = evaluate(eval_plan, "focused_validation", final_configs, split="validation",
                              deadline=args.deadline_unix - 360)
        for seed in (int(x) for x in args.full_seeds.split(",") if x.strip()):
            eval_plan["seed"] = seed
            final.append({"seed": seed, "rows": evaluate(
                eval_plan, f"focused_final_{seed}", final_configs, split="validation",
                deadline=args.deadline_unix - 240)})
        eval_plan["seed"] = args.seed
        holdout = evaluate(eval_plan, "focused_holdout", final_configs, split="holdout",
                           deadline=args.deadline_unix - 120)
    assessments = {"dev": assess([row for row in dev_rows
                                    if row["candidate"]["id"] in {
                                        ar["id"], fastest_native["id"], selected_cfg["id"]}])}
    if validation:
        assessments["validation"] = assess(validation)
    if final:
        assessments["final"] = [{"seed": item["seed"], "assessment": assess(item["rows"])}
                                 for item in final]
    if holdout:
        assessments["holdout"] = assess(holdout)
    result = {"status": "completed", "adapter": adapter, "steps": args.steps,
              "rank": args.rank, "dev": dev_rows, "selected": selected_cfg,
              "validation": validation, "final": final, "holdout": holdout,
              "assessments": assessments,
              "full_evaluation_requested": args.full,
              "holdout_used_for_selection": False, "generated_unix": time.time()}
    (args.output / "focused_target_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "selected": selected_cfg["id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
