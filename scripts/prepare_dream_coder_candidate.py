#!/usr/bin/env python3
"""Prepare one pinned public Dream-Coder candidate, without loading model weights.

Only a fully checked download receives receipt.json. GPU generation compatibility
is deliberately pending in that receipt and must be tested before evaluation.
Run under a service with RuntimeMaxSec as a second, process-group time bound.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import inspect
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iterative_tuning import DREAM_CODER, write_json
from scripts.verify_prepared_models import _check_shards, _hash_file, signature
from tokenizer_contract import load_verified_tokenizer

MODEL, REVISION = DREAM_CODER
DEADLINE_UNIX = 1794051000.0
PREPARATION_CUTOFF_UNIX = 1794013200.0  # 2026-11-07 10:00 KST; leave GPU search time.
ALLOW_PATTERNS = ("*.safetensors", "*.json", "*.py", "*.txt", "*.model",
                  "*.tiktoken", "*.jinja", "LICENSE*", "README.md")
REQUIRED = ("config.json", "tokenizer_config.json", "model.safetensors.index.json",
            "modeling_dream.py", "configuration_dream.py", "generation_utils.py")


def effective_deadline(now, deadline, max_seconds):
    if not all(math.isfinite(x) for x in (now, deadline, max_seconds)):
        raise ValueError("Download time bounds must be finite")
    if not 0 < max_seconds <= 21600:
        raise ValueError("Preparation limit must be in (0, 21600] seconds")
    end = min(deadline, now + max_seconds, DEADLINE_UNIX, PREPARATION_CUTOFF_UNIX)
    if end <= now:
        raise TimeoutError("Candidate preparation deadline has passed")
    return end


def select_expected_files(info):
    if info.sha != REVISION:
        raise ValueError("Hub returned a different revision")
    result = {}
    for item in info.siblings:
        name = item.rfilename
        if not any(fnmatch.fnmatch(name, pattern) for pattern in ALLOW_PATTERNS):
            continue
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unexpected nested or unsafe candidate path")
        if name in result or not isinstance(item.size, int) or item.size <= 0:
            raise ValueError("Candidate metadata has missing size or duplicate files")
        result[name] = item.size
    if not set(REQUIRED) <= result.keys():
        raise ValueError("Candidate remote file inventory is incomplete")
    return result


def verify_inventory(snapshot, expected):
    snapshot = Path(snapshot)
    if snapshot.name != REVISION:
        raise ValueError("Candidate revision mismatch")
    actual = {p.name for p in snapshot.iterdir() if p.is_file()}
    if actual != set(expected):
        raise ValueError("Candidate downloaded file inventory differs from pinned remote inventory")
    for name, size in expected.items():
        if signature(snapshot / name)["size"] != size:
            raise ValueError("Candidate file size mismatch: " + name)
    return sorted(snapshot / name for name in expected)


def review_native_source(snapshot):
    """Check known Dream coordinates before importing the verified source."""
    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_text())
    expected = {"model_type": "Dream", "architectures": ["DreamModel"],
                "mask_token_id": 151666, "pad_token_id": 151643,
                "eos_token_id": 151643}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate architecture or special-token configuration changed")
    if config.get("auto_map", {}).get("AutoModel") != "modeling_dream.DreamModel":
        raise ValueError("Candidate AutoModel implementation changed")
    modeling = (snapshot / "modeling_dream.py").read_text()
    generation = (snapshot / "generation_utils.py").read_text()
    model_tree, generation_tree = ast.parse(modeling), ast.parse(generation)
    classes = {node.name for node in model_tree.body if isinstance(node, ast.ClassDef)}
    if not {"DreamModel", "DreamSdpaAttention"} <= classes:
        raise ValueError("Candidate lacks the Dream SDPA implementation")
    compact_generation = "".join(generation.split())
    if "torch.cat([logits[:,:1],logits[:,:-1]],dim=1)" not in compact_generation:
        raise ValueError("Candidate shifted prediction convention could not be verified")
    generation_classes = {node.name for node in generation_tree.body if isinstance(node, ast.ClassDef)}
    if not {"DreamGenerationConfig", "DreamGenerationMixin"} <= generation_classes:
        raise ValueError("Candidate native generation API could not be verified")
    return {"status": "verified", "architecture": "DreamModel",
            "attention": "DreamSdpaAttention", "prediction_coordinates": "row_t_minus_1",
            "generation_api": "diffusion_generate", "model_code_patched": False,
            "source_review_scope": "architecture, special IDs, classes and native logit shift"}


def check_generation_config(snapshot):
    """Import verified configuration code; never instantiate a model or CUDA."""
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers import AutoConfig
    cls = get_class_from_dynamic_module("generation_utils.DreamGenerationConfig", str(snapshot),
                                        local_files_only=True)
    model_config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=True)
    config = cls.from_model_config(model_config)
    unused = config.update(max_new_tokens=32, steps=4, temperature=0.1,
                           top_p=0.95, alg="entropy", alg_temp=0.0)
    if unused:
        raise ValueError("Native generation settings rejected: " + ", ".join(sorted(unused)))
    config.validate()
    serialized = config.to_dict()
    if serialized.get("eos_penalty", 0.0) != 0.0:
        raise ValueError("Unreviewed nonzero default EOS penalty")
    return {"status": "verified", "transformers_version": transformers.__version__,
            "class": cls.__name__, "validate_signature": str(inspect.signature(cls.validate)),
            "eos_penalty": serialized.get("eos_penalty", 0.0),
            "model_weights_loaded": False, "gpu_generation": "pending"}


def verify_snapshot(snapshot, expected, *, started, deadline):
    paths = verify_inventory(snapshot, expected)
    # Hash remote code before importing tokenizer/configuration classes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(_hash_file, paths))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = _check_shards(snapshot, index)
    source_review = review_native_source(snapshot)
    tokenizer, tokenizer_receipt = load_verified_tokenizer(snapshot, model_name=MODEL,
                                                           revision=REVISION)
    if tokenizer_receipt["chat_template"]["policy"] != "checkpoint_native":
        raise ValueError("Dream-Coder requires its checkpoint-native chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python function that adds two integers."}],
        tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or "<|im_start|>assistant" not in rendered:
        raise ValueError("Candidate native template lacks an assistant generation boundary")
    del tokenizer
    generation_config = check_generation_config(snapshot)
    from models import terminal_token_policy
    termination = terminal_token_policy("diffusion", tokenizer_receipt["special_token_ids"],
        151643, model_name=MODEL, revision=REVISION)
    if termination["terminal_token_ids"] != [151643]:
        raise ValueError("Candidate native EOS policy mismatch")
    # Bind the receipt to the same files that passed hashes, not files replaced
    # during subsequent tokenizer/source checks.
    for row in files:
        if any(row.get(k) != v for k, v in signature(row["path"]).items()):
            raise ValueError("Candidate file changed after hashing")
    if time.time() >= deadline:
        raise TimeoutError("Candidate verification exceeded the time limit")
    return {"schema_version": 1, "status": "verified", "model": MODEL,
            "revision": REVISION, "snapshot": str(snapshot), "files": files,
            "shards": shards, "started_unix": started, "ended_unix": time.time(),
            "tokenizer_contract": tokenizer_receipt, "native_source_review": source_review,
            "termination": termination,
            "generation_config": generation_config, "gpu_compatibility": "pending",
            "verification": "Pinned remote inventory, content-addressed hashes, shard headers, native tokenizer and CPU generation configuration"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/hybrid-dard/cache/huggingface/hub"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, default=DEADLINE_UNIX)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an existing candidate receipt")
    started = time.time()
    state_path = args.output / "state.json"
    def state(status, **extra):
        record = {"status": status, "model": MODEL, "revision": REVISION,
                  "started_unix": started, "updated_unix": time.time(), **extra}
        write_json(state_path, record)
        print(json.dumps(record), flush=True)
    def interrupted(signum, frame):
        raise TimeoutError("Candidate preparation interrupted or reached its deadline")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    try:
        deadline = effective_deadline(started, args.deadline_unix, args.max_seconds)
        signal.setitimer(signal.ITIMER_REAL, deadline - started)
        from huggingface_hub import HfApi, snapshot_download
        state("reading_pinned_inventory", deadline_unix=deadline)
        info = HfApi(token=False).model_info(MODEL, revision=REVISION, files_metadata=True)
        expected = select_expected_files(info)
        state("downloading", expected_files=len(expected), expected_bytes=sum(expected.values()),
              deadline_unix=deadline)
        snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION,
            cache_dir=str(args.cache_dir), max_workers=4, token=False,
            allow_patterns=list(ALLOW_PATTERNS)))
        state("verifying", snapshot=str(snapshot), deadline_unix=deadline)
        receipt = verify_snapshot(snapshot, expected, started=started, deadline=deadline)
        write_json(receipt_path, receipt)
        state("completed", receipt=str(receipt_path), gpu_compatibility="pending")
        return 0
    except BaseException as exc:
        state("failed", error_type=type(exc).__name__, reason=str(exc)[:1000])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
