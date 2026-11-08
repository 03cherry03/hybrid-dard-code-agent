"""Frozen contract for reproducing the r32/1200 CE-only experiment.

The report selected one DARD operating point on validation and then evaluated
that fixed point on three repeated validation seeds and one holdout.  This
module keeps those choices in code so a rerun cannot silently fall back to a
different candidate grid or a different training objective.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

from diffusion_training import file_sha256, write_language_balanced_jsonl
from iterative_tuning import AR, CPGRPO, adapter_identity, candidate, digest, validate_candidate


EXPERIMENT_ID = "ce-only-diffucoder-dard-r32-1200-v1"
LANGUAGES = ("py", "js", "cpp", "rs")
TRAINING = {
    "base_model": CPGRPO[0],
    "revision": CPGRPO[1],
    "max_length": 512,
    "max_steps": 1200,
    "batch_size": 1,
    "gradient_accumulation": 2,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "max_grad_norm": 1.0,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"),
    "min_mask_prob": 0.15,
    "checkpoint_every": 400,
    "seed": 20261108,
    "world_size": 4,
    "precision": "bfloat16",
    "balanced_languages": True,
}
EVALUATION = {
    "validation_seed": 20261105,
    "repeat_seeds": (20261111, 20261112, 20261113),
    "holdout_seed": 20261108,
    "tokens": 512,
    "prefix": "code_fence",
    "generation_timeout": 120,
    "grading_timeout": 300,
    "grading_workers": 8,
    "grading_queue_size": 64,
    "candidate_warmup": True,
    "counterbalanced_order": True,
    "extraction_version": "v2",
    "cases_per_split": 273,
}

# These hashes are the data identities recorded by the report.  They are
# checked when the corresponding files are supplied, but callers may use a
# separately frozen data manifest for a new, explicitly labelled replication.
REPORT_DATA_SHA256 = {
    "training_examples.jsonl": "184c83f26cac8ab87689dac61a194f13e420e3e4dbf49136f3442f9dd6384b2da",
    "validation_tasks.jsonl": "f97432bb495d3670e4fddfaa3a132a3badb9f714e1e916421e0dac63ef8843ca",
    "holdout_tasks.jsonl": "56f5b21b0a04f5894c58a5cd66b8d82b80968ea3c68f9cc046f74b0113b544b4",
}
REPORT_BALANCED_TRAINING_SHA256 = "725ec184925974e77dae0ac61e65099728deacc281326b958b23ec9235ee8271"
REPORT_TRAINING_SOURCE_SHA256 = {
    "diffusion_training.py": "732a7888017e32d81034a5a01dacacad81494d39cc5d04c88d702d392de15cbd",
    "iteration_observability.py": "f519df284f0935139b1baaedbaa7129aaf8e386b1cb3568ac548e402386b34b6",
    "scripts/train_diffusion_lora.py": "78df396ac2d24dbfee16e1c8572076782edf97a66f96d67451a8e7f8c3f0c37e",
    "tokenizer_contract.py": "846dc1ea8862ed0cc1adf6dc486e3545c4801756aae7d99735dfd7988acfc3c2",
}
REPORT_METRICS = {
    "validation": {
        "ar": (0.7252747252747253, 1.5385061172161172, 3.477754),
        "standard": (0.040293040293040296, 1.3622988131868132, 1.622403),
        "dard": (0.4139194139194139, 1.7093296410256411, 2.674202),
    },
    "holdout": {
        "ar": (0.6593406593406593, 1.6780123406593406, 3.696649),
        "standard": (0.03663003663003663, 1.3809105677655678, 1.754138),
        "dard": (0.3516483516483517, 1.7439661868131868, 2.589458),
    },
}


def _with_id(config):
    config = dict(config)
    config["id"] = config["decoder"] + "-" + digest({k: v for k, v in config.items() if k != "id"})[:16]
    validate_candidate(config)
    return config


def _diffusion_candidate(*, decoder, steps, tau_c=0.5, tau_u=0.8,
                         transfer_limit=0, completion_policy="legacy_force",
                         adapter=None):
    value = candidate(CPGRPO, decoder=decoder, tokens=512, block=128,
                      steps=steps, temperature=0.0, top_p=1.0,
                      tau_c=tau_c, tau_u=tau_u, prefix="code_fence",
                      native_steps=steps if decoder == "standard" else 1,
                      forward_budget=512, adapter=adapter)
    value["dard"].update(completion_policy=completion_policy,
                          completion_verification_steps=2,
                          transfer_limit=transfer_limit,
                          temperature=0.0, top_p=1.0)
    return _with_id({k: v for k, v in value.items() if k != "id"})


def reproduction_candidates(adapter_path=None):
    """Return AR, native-16 and selected DARD configs in report order."""
    adapter = adapter_identity(adapter_path) if adapter_path else None
    ar = candidate(AR, decoder="ar", tokens=512, native_steps=1,
                   temperature=0, top_p=1, prefix="code_fence", forward_budget=512)
    native = _diffusion_candidate(decoder="standard", steps=16,
                                  completion_policy="legacy_force")
    dard = _diffusion_candidate(decoder="dard", steps=64, tau_c=0.4, tau_u=0.75,
                                transfer_limit=8,
                                completion_policy="resample_then_verify")
    # The historical controller appended this suffix after adapter binding.
    if adapter:
        for value in (native, dard):
            value["adapter"] = adapter
            value["id"] += "-trained"
    expected = {"ar": "ar-7f348bc3385bc1e4",
                "standard": "standard-5a4bd2d0c4972b81",
                "dard": "dard-84292bb36a6aa4c8"}
    for value in (ar, native, dard):
        base = value["id"].removesuffix("-trained")
        if base != expected[value["decoder"]]:
            raise AssertionError("Frozen candidate identity changed: " + value["id"])
    return [ar, native, dard]


def balance_training(source, destination):
    """Write and receipt-check the exact language-balanced training corpus."""
    receipt = write_language_balanced_jsonl(source, destination, languages=LANGUAGES)
    return receipt


def check_file(path, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    actual = file_sha256(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def report_contract_manifest(*, output, base_plan, train_jsonl, balanced_jsonl,
                             validation_tasks, holdout_tasks, adapter=None):
    """Build the immutable run metadata written before evaluation."""
    files = {
        "train_jsonl": check_file(train_jsonl, REPORT_DATA_SHA256.get("training_examples.jsonl")),
        "balanced_training_jsonl": check_file(balanced_jsonl, REPORT_BALANCED_TRAINING_SHA256
                                               if Path(train_jsonl).name == "training_examples.jsonl" else None),
        "validation_tasks": check_file(validation_tasks, REPORT_DATA_SHA256["validation_tasks.jsonl"]),
        "holdout_tasks": check_file(holdout_tasks, REPORT_DATA_SHA256["holdout_tasks.jsonl"]),
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "output": str(Path(output).resolve()),
        "base_plan": check_file(base_plan),
        "files": files,
        "training": TRAINING,
        "evaluation": EVALUATION,
        "adapter": adapter,
        "candidates_without_adapter": reproduction_candidates(),
    }


def result_contract(result):
    """Return the small report-facing summary from a full result object."""
    metrics = {}
    for split, rows in (("validation", result.get("validation")),
                        ("holdout", result.get("holdout"))):
        if not rows:
            continue
        selected = rows if isinstance(rows, list) else [rows]
        if split == "validation" and selected and isinstance(selected[0], dict) and "rows" in selected[0]:
            selected = selected[0]["rows"]
        metrics[split] = {row["candidate"]["decoder"]: {
            "cases": row["cases"], "passed": row["passed"],
            "pass_rate": row["pass_rate"],
            "mean_generation_seconds": row["mean_generation_seconds"],
            "p95_generation_seconds": row["p95_generation_seconds"]}
            for row in selected}
    return metrics


def verify_report_artifacts(report_dir):
    """Audit the checked-in report artifacts without loading a model."""
    root = Path(report_dir).resolve(strict=True)
    validation = json.loads((root / "validation" / "results.json").read_text())
    holdout = json.loads((root / "holdout" / "results.json").read_text())
    observed = {}
    for split, payload in (("validation", validation), ("holdout", holdout)):
        rows = payload.get("candidates", [])
        if len(rows) != 3 or any(row.get("cases") != EVALUATION["cases_per_split"] for row in rows):
            raise ValueError(f"{split} does not contain the complete 273-case trio")
        for row in rows:
            decoder = row["candidate"]["decoder"]
            observed.setdefault(split, {})[decoder] = row
            expected = REPORT_METRICS[split][decoder]
            actual = (row["pass_rate"], row["mean_generation_seconds"], row["p95_generation_seconds"])
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)):
                raise ValueError(f"{split}/{decoder} metric changed: {actual} != {expected}")
    configs = reproduction_candidates()
    for split_rows in observed.values():
        for config in configs:
            row = split_rows[config["decoder"]]
            recorded = row["candidate"]
            for key in ("model", "revision", "decoder", "tokens", "standard_steps", "prefix",
                        "forward_budget", "dard"):
                if recorded.get(key) != config.get(key):
                    raise ValueError(f"{split}/{config['decoder']} config changed at {key}")
    summary_path = root / "ce_only_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        expected_summary = result_contract({"validation": [*validation["candidates"]],
                                            "holdout": holdout["candidates"]})
        actual_summary = summary.get("metrics", {})
        for split, modes in expected_summary.items():
            for mode, expected in modes.items():
                actual = actual_summary.get(split, {}).get(mode, {})
                if any(actual.get(key) != value for key, value in expected.items()):
                    raise ValueError("ce_only_summary.json disagrees with raw result files")
    return {"status": "passed", "report_dir": str(root), "cases_per_split": 273,
            "candidate_ids": {split: {mode: row["candidate"]["id"]
                                       for mode, row in rows.items()}
                              for split, rows in observed.items()}}


def verify_bundle(bundle_dir):
    """Verify the portable adapter/data bundle before another team uses it."""
    root = Path(bundle_dir).resolve(strict=True)
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    checked = 0
    for key, recorded in manifest.get("files", {}).items():
        path = root / key
        if not path.is_file():
            raise FileNotFoundError("Bundle file is missing: " + str(path))
        actual = file_sha256(path)
        if actual != recorded["sha256"] or path.stat().st_size != recorded["bytes"]:
            raise ValueError("Bundle file changed: " + key)
        checked += 1
    candidates = reproduction_candidates(root / "adapter")
    expected_ids = manifest["evaluation"]["candidate_ids"]
    actual_ids = {candidate["decoder"]: candidate["id"] for candidate in candidates}
    if actual_ids != expected_ids:
        raise ValueError(f"Bundle candidate IDs changed: {actual_ids} != {expected_ids}")
    if manifest["adapter"]["model_sha256"] != file_sha256(root / "adapter" / "adapter_model.safetensors"):
        raise ValueError("Bundle adapter hash disagrees with experiment manifest")
    for split, modes in REPORT_METRICS.items():
        for mode, (pass_rate, mean_seconds, p95_seconds) in modes.items():
            actual = manifest.get("report_metrics", {}).get(split, {}).get(mode, {})
            if (actual.get("pass_rate"), actual.get("mean_generation_seconds"),
                    actual.get("p95_generation_seconds")) != (pass_rate, mean_seconds, p95_seconds):
                raise ValueError(f"Bundle report metrics disagree at {split}/{mode}")
    return {"status": "passed", "bundle_dir": str(root), "checked_files": checked,
            "adapter_sha256": manifest["adapter"]["model_sha256"],
            "candidate_ids": actual_ids}
