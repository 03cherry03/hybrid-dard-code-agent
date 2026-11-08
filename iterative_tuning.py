"""Pure contracts for source-disjoint validation and bounded LoRA search.

Validation scores may select settings/checkpoints; only training examples enter
gradient updates. A final holdout is evaluated once, after selection is frozen.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

SCHEMA = 1
INSTRUCT = ("apple/DiffuCoder-7B-Instruct", "4fdd4580064ca5d11808069ce78f88d068753c96")
CPGRPO = ("apple/DiffuCoder-7B-cpGRPO", "98bc42bfd871eeec92fd0a8439ab34f75b405204")
DREAM_CODER = ("Dream-org/Dream-Coder-v0-Instruct-7B", "5d9e88c723af9045f362748b5284bdf43d9c501e")
AR = ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact changed: {path}")
    else:
        write_json(path, value)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def task_key(task):
    return (task["dataset"], task["language"], task["task_id"])


def task_id(task):
    return digest(task_key(task))[:24]


def shard_tasks(tasks, index, count=4):
    if not 0 <= index < count:
        raise ValueError("Invalid shard")
    groups = defaultdict(list)
    for row in tasks:
        groups[(row["family"], row["language"])].append(row)
    result = []
    offset = 0
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda t: digest(task_key(t)))
        result.extend(row for position, row in enumerate(ordered)
                      if (offset + position) % count == index)
        offset = (offset + len(ordered)) % count
    if not result:
        raise ValueError(f"Shard {index}/{count} is empty for {len(tasks)} selected tasks")
    return result


def adapter_identity(path):
    if not path:
        return None
    root = Path(path).resolve(strict=True)
    required = [root / "adapter_config.json", root / "adapter_model.safetensors"]
    if not all(p.is_file() for p in required):
        raise ValueError("A complete safetensors LoRA adapter is required")
    return {"path": str(root), "sha256": {p.name: file_sha(p) for p in required}}


def validate_candidate(config):
    from dard_decoder import DARDConfig
    if config["decoder"] not in ("ar", "standard", "dard"):
        raise ValueError("Unknown decoder")
    if (config["model"], config["revision"]) not in (AR, INSTRUCT, CPGRPO, DREAM_CODER):
        raise ValueError("Unpinned model")
    if (config["decoder"] == "ar") != (config["model"] == AR[0]):
        raise ValueError("Model/decoder mismatch")
    if not 32 <= config["tokens"] <= 512:
        raise ValueError("Output budget out of range")
    if not 1 <= config["forward_budget"] <= 512:
        raise ValueError("Forward budget out of range")
    if config["prefix"] not in ("none", "code_fence"):
        raise ValueError("Unsupported prefix")
    if not 1 <= config["standard_steps"] <= config["forward_budget"]:
        raise ValueError("Native steps exceed total forward budget")
    DARDConfig(**config["dard"]).validate()
    if config.get("adapter") and config["decoder"] == "ar":
        raise ValueError("AR reference cannot use a diffusion adapter")


def candidate(model, *, decoder="dard", tokens=512, block=128, steps=16,
              temperature=0.3, top_p=0.95, tau_c=0.5, tau_u=0.8,
              prefix="code_fence", native_steps=64, forward_budget=128, adapter=None):
    cfg = {"model": model[0], "revision": model[1], "decoder": decoder,
           "tokens": tokens, "standard_steps": native_steps, "prefix": prefix,
           "forward_budget": forward_budget, "adapter": adapter,
           "dard": {"block_size": block, "steps_per_block": steps,
                    "tau_c": tau_c, "tau_u": tau_u, "temperature": temperature,
                    "top_p": top_p, "transfer_limit": 0, "diagnostics": "summary",
                    "max_step_records": 0, "compact_queries": True,
                    "rectangular_attention": True}}
    validate_candidate(cfg)
    cfg["id"] = decoder + "-" + digest(cfg)[:16]
    return cfg


def initial_candidates():
    result = [candidate(AR, decoder="ar", temperature=0, top_p=1,
                        forward_budget=512, prefix="code_fence")]
    for model in (INSTRUCT, CPGRPO):
        result.extend([
            candidate(model, decoder="standard", tokens=256, native_steps=256,
                      forward_budget=256, temperature=.3 if model == INSTRUCT else .4,
                      prefix="none"),
            candidate(model, decoder="standard", native_steps=64),
            candidate(model, decoder="standard", tokens=256, native_steps=32),
            candidate(model, block=512, steps=64, temperature=0, top_p=1, prefix="none"),
            candidate(model, block=128, steps=16),
            candidate(model, block=64, steps=8, tau_c=.3, tau_u=.7),
            candidate(model, block=128, steps=16, temperature=0, top_p=1),
        ])
    return result


def round_candidates(incumbent, adapter, round_index):
    """Small, prebounded coordinate search; no claim of a global optimum."""
    model = (incumbent["model"], incumbent["revision"])
    old = incumbent["dard"]
    specs = [dict(decoder=incumbent["decoder"], tokens=incumbent["tokens"],
                  block=old["block_size"], steps=old["steps_per_block"],
                  temperature=old["temperature"], top_p=old["top_p"],
                  tau_c=old["tau_c"], tau_u=old["tau_u"], prefix=incumbent["prefix"],
                  native_steps=incumbent["standard_steps"], forward_budget=incumbent["forward_budget"]),
             dict(decoder="standard", native_steps=64),
             dict(decoder="standard", tokens=256, native_steps=32),
             dict(block=128, steps=16, temperature=.3, tau_c=.5, tau_u=.8),
             dict(block=64, steps=8, temperature=.3, tau_c=.3, tau_u=.7),
             dict(block=128, steps=16, temperature=0, top_p=1)]
    # Alternate one threshold coordinate rather than expanding an unbounded grid.
    specs.append(dict(block=128, steps=16, temperature=.3,
                      tau_c=.4 if round_index % 2 else .3, tau_u=.75))
    unique = {}
    for spec in specs:
        value = candidate(model, adapter=adapter, **spec)
        unique[value["id"]] = value
    return list(unique.values())


def summarize(rows, expected_tasks, config):
    expected = {task_id(t) for t in expected_tasks}
    if len(expected) != len(expected_tasks):
        raise ValueError("Duplicate tasks in expected evaluation manifest")
    observed = [r["task_key"] for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Incomplete or duplicate validation cohort")
    for row in rows:
        if row["candidate_id"] != config["id"] or row["infrastructure_error"]:
            raise ValueError("Invalid candidate or infrastructure failure")
        if type(row["passed"]) is not bool or not math.isfinite(row["generation_seconds"]) or row["generation_seconds"] <= 0:
            raise ValueError("Invalid measurement")
    def group(items):
        elapsed = sorted(r["generation_seconds"] for r in items)
        return {"cases": len(items), "passed": sum(r["passed"] for r in items),
                "pass_rate": sum(r["passed"] for r in items) / len(items),
                "mean_generation_seconds": statistics.mean(elapsed),
                "p95_generation_seconds": elapsed[min(len(elapsed)-1, math.ceil(.95*len(elapsed))-1)]}
    value = group(rows)
    value.update(candidate=config, by_language={lang: group([r for r in rows if r["language"] == lang])
                   for lang in sorted({r["language"] for r in rows})})
    return value


def select_results(results, ar, margin=.03):
    if not results or not 0 <= margin < 1:
        raise ValueError("Nonempty candidates and valid accuracy margin required")
    complete = [r for r in results if r["cases"] == ar["cases"]]
    if len(complete) != len(results):
        raise ValueError("Candidates must use identical validation cohorts")
    eligible = [r for r in complete if r["pass_rate"] + margin >= ar["pass_rate"]
                and r["mean_generation_seconds"] < ar["mean_generation_seconds"]]
    research = min(eligible, key=lambda r: (r["mean_generation_seconds"], -r["pass_rate"], r["candidate"]["id"])) if eligible else min(
        complete, key=lambda r: (-r["pass_rate"], r["mean_generation_seconds"], r["candidate"]["id"]))
    return {"research_incumbent": research, "deployment": research if eligible else ar,
            "goal_screen_passed": bool(eligible), "margin": margin,
            "statistical_noninferiority_established": False,
            "selection_data": "validation_only", "selected_unix": time.time()}


def training_rate(round_index):
    return (5e-5, 2e-5, 1e-4, 1e-5)[(round_index-1) % 4]
