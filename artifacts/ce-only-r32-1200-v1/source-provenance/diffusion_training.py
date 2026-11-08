"""Masked diffusion SFT primitives and durable LoRA checkpoint contracts.

Dream predicts target token t from logits row t-1 (the pinned native sampler
shifts logits this way). This module never invokes the model's causal-LM loss.
The objective is a per-example mean over *all* valid completion tokens of
I[token masked] * CE / p, where p is sampled uniformly on [min_mask_prob, 1].
This is a truncated-time conditional denoising objective, not causal SFT.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import torch
import torch.nn.functional as F


OBJECTIVE_VERSION = "dream_shifted_masked_completion_inverse_probability_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode_training_example(row, tokenizer, *, max_length, terminal_token_id):
    """Keep the complete prompt and target; oversized examples are rejected.

    The tokenization boundary is the generation prefix used at inference.
    Chat-template assistant suffix (e.g. im_end/newline) remains in the target;
    one native dlm_pad boundary is supervised. Batch padding is never a target.
    """
    for field in ("example_id", "source_id", "language", "completion"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("Training example requires nonempty " + field)
    if row["language"] not in ("py", "rs", "js", "cpp"):
        raise ValueError("Unsupported training language")
    messages = row.get("messages")
    if (not isinstance(messages, list) or not messages or messages[-1].get("role") != "user"
            or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                   or m["role"] not in ("system", "user", "assistant")
                   or not isinstance(m["content"], str) for m in messages)):
        raise ValueError("Expected conversation messages ending with user")
    if any(token in row["completion"] for token in
           ("<|mask|>", "<|dlm_pad|>", "<|im_start|>", "<|im_end|>")):
        raise ValueError("Completion must not supply model control tokens")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["completion"]}],
                                         tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("Assistant target does not extend the inference chat prefix")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(full[len(prompt):], add_special_tokens=False)
    if not prompt_ids or not target_ids:
        raise ValueError("Empty prompt or target token sequence")
    target_ids.append(int(terminal_token_id))
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {"example_id": row["example_id"], "source_id": row["source_id"],
            "language": row["language"], "input_ids": prompt_ids + target_ids,
            "target_mask": [False] * len(prompt_ids) + [True] * len(target_ids),
            "prompt_tokens": len(prompt_ids), "target_tokens": len(target_ids)}


def load_training_examples(path, tokenizer, *, max_length, terminal_token_id):
    examples, oversized, seen = [], [], set()
    source_ids, languages = set(), {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example = encode_training_example(row, tokenizer, max_length=max_length,
                                              terminal_token_id=terminal_token_id)
            identity = row["example_id"]
            if identity in seen:
                raise ValueError("Duplicate training example_id: " + identity)
            seen.add(identity)
            if example is None:
                oversized.append(identity)
                continue
            examples.append(example)
            source_ids.add(example["source_id"])
            languages[example["language"]] = languages.get(example["language"], 0) + 1
    if not examples:
        raise ValueError("No complete training example fits max_length")
    return examples, {"input_examples": len(seen), "included_examples": len(examples),
        "included_source_count": len(source_ids), "by_language": languages,
        "oversized_examples_dropped": oversized, "truncated_examples": 0,
        "tokenization_policy": "separate_inference_prefix_and_assistant_suffix_v1",
        "target_terminal_tokens_per_example": 1}


def collate_examples(examples, pad_token_id):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(x["input_ids"]) for x in examples)
    ids = torch.full((len(examples), length), int(pad_token_id), dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.zeros_like(ids, dtype=torch.bool)
    for index, example in enumerate(examples):
        size = len(example["input_ids"])
        ids[index, :size] = torch.tensor(example["input_ids"], dtype=torch.long)
        valid[index, :size] = True
        targets[index, :size] = torch.tensor(example["target_mask"], dtype=torch.bool)
    if bool(targets[:, 0].any()) or not bool(targets.any(dim=1).all()):
        raise ValueError("Each target requires a predecessor and at least one supervised token")
    return {"input_ids": ids, "valid_mask": valid, "target_mask": targets,
            "example_ids": [x["example_id"] for x in examples]}


def corrupt_completion(input_ids, target_mask, *, mask_token_id, min_mask_prob,
                       generator=None, probabilities=None):
    if not math.isfinite(min_mask_prob) or not 0 < min_mask_prob <= 1:
        raise ValueError("min_mask_prob must be in (0, 1]")
    if input_ids.shape != target_mask.shape or target_mask.dtype != torch.bool:
        raise ValueError("Expected equally shaped IDs and boolean target mask")
    batch = input_ids.shape[0]
    if probabilities is None:
        probabilities = min_mask_prob + (1-min_mask_prob) * torch.rand(
            (batch,), device=input_ids.device, generator=generator)
    else:
        probabilities = torch.as_tensor(probabilities, device=input_ids.device, dtype=torch.float32)
        if (probabilities.shape != (batch,) or not bool(torch.isfinite(probabilities).all())
                or bool(((probabilities <= 0) | (probabilities > 1)).any())):
            raise ValueError("Expected one probability in (0, 1] per example")
    masked = (torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
              < probabilities[:, None]) & target_mask
    # Do not force one masked token: doing so without changing inclusion weights
    # would bias the inverse-probability objective for short completions.
    return input_ids.masked_fill(masked, int(mask_token_id)), masked, probabilities


def shifted_masked_diffusion_loss(logits, clean_ids, target_mask, masked, probabilities):
    """Return sample-mean 1/p weighted loss with native Dream t-1 alignment."""
    if logits.ndim != 3 or logits.shape[:2] != clean_ids.shape:
        raise ValueError("Full-sequence logits must align with input IDs")
    if clean_ids.shape != target_mask.shape or clean_ids.shape != masked.shape:
        raise ValueError("Loss mask shape mismatch")
    if target_mask.dtype != torch.bool or masked.dtype != torch.bool:
        raise ValueError("Loss masks must be boolean")
    if bool(masked[:, 0].any()) or bool((masked & ~target_mask).any()):
        raise ValueError("Only completion tokens with predecessors can be masked")
    if probabilities.shape != (clean_ids.shape[0],) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Invalid per-example mask probabilities")
    if bool(((probabilities <= 0) | (probabilities > 1)).any()):
        raise ValueError("Mask probabilities must be in (0, 1]")
    counts = target_mask.sum(dim=1)
    if bool((counts == 0).any()) or bool(target_mask[:, 0].any()):
        raise ValueError("Every example needs a nonempty completion after the prompt")
    sample, target_position = torch.where(masked[:, 1:])
    if sample.numel() == 0:
        # Connected zero keeps DDP participation and does not invent a mask.
        return logits.reshape(-1)[0].float() * 0.0
    predicted = logits[sample, target_position].float()
    expected = clean_ids[sample, target_position + 1]
    token_loss = F.cross_entropy(predicted, expected, reduction="none")
    weights = probabilities[sample].float() * counts[sample].float()
    return (token_loss / weights).sum() / clean_ids.shape[0]


def native_attention_mask(valid_mask):
    """Full bidirectional attention over valid positions; no causal triangle."""
    if bool(valid_mask.all()):
        return "full"
    return valid_mask[:, None, :, None] & valid_mask[:, None, None, :]


def capture_rng(device):
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("CUDA checkpoint RNG requires a CUDA device")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def write_checkpoint_manifest(directory, metadata):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checkpoint_manifest.json":
            if path.is_symlink():
                raise ValueError("Checkpoint may not contain symlinks")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files[str(path.relative_to(directory))] = file_sha256(path)
    manifest = {"schema_version": 1, "metadata": metadata, "files": files}
    atomic_json(directory / "checkpoint_manifest.json", manifest)
    fsync_directory(directory)
    return manifest


def verify_checkpoint(directory, *, expected_fingerprint=None):
    directory = Path(directory)
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported checkpoint manifest")
    if expected_fingerprint and manifest.get("metadata", {}).get("training_fingerprint") != expected_fingerprint:
        raise ValueError("Checkpoint training configuration or dataset fingerprint mismatch")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p.name != "checkpoint_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Checkpoint file set changed")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe checkpoint manifest path")
        path = directory / relative
        if path.is_symlink() or file_sha256(path) != expected:
            raise ValueError("Checkpoint file SHA256 mismatch: " + name)
    return manifest


def resolve_resume(output, value):
    if value is None:
        return None
    if value != "latest":
        return Path(value).resolve()
    output = Path(output)
    pointer = json.loads((output / "latest.json").read_text())
    relative = Path(pointer["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe latest checkpoint path")
    path = output / relative
    if file_sha256(path / "checkpoint_manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Latest checkpoint manifest SHA256 mismatch")
    return path.resolve()
