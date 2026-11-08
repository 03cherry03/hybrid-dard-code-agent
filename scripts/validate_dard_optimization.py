#!/usr/bin/env python3
"""Validate compact/rectangular DARD on synthetic inputs, never benchmark tasks.

Must run with the inference Python after the baseline has released all GPUs.
Locks and CUDA/cache environment are established before importing torch.
The BF16 tolerance gate is numerical validation, not evidence of code accuracy
or end-to-end generation speed. The caller owns failure email notifications.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "apple/DiffuCoder-7B-Instruct"
REVISION = "4fdd4580064ca5d11808069ce78f88d068753c96"
GATES = {"mean_kl_max": .02, "top1_agreement_min": .90,
         "finite_required": True, "native_bitwise_equal_required": True}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def distribution_metrics(reference, actual, chunk_rows=8):
    """Compare detached CPU BF16 blocks without materializing full FP32 logits."""
    import torch
    if len(reference) != len(actual):
        raise ValueError("Different consumed logit blocks")
    count = values = agree = 0
    kl_sum = absolute_sum = maximum = 0.0
    for expected_block, actual_block in zip(reference, actual):
        if expected_block.shape != actual_block.shape or expected_block.ndim != 2:
            raise ValueError("Consumed logit block shape mismatch")
        for start in range(0, expected_block.shape[0], chunk_rows):
            expected = expected_block[start:start+chunk_rows].float()
            observed = actual_block[start:start+chunk_rows].float()
            if not bool(torch.isfinite(expected).all() and torch.isfinite(observed).all()):
                return {"finite": False, "rows": count}
            difference = (expected - observed).abs()
            maximum = max(maximum, difference.max().item())
            absolute_sum += difference.double().sum().item()
            expected_logp = expected.log_softmax(-1)
            observed_logp = observed.log_softmax(-1)
            kl_sum += (expected_logp.exp() * (expected_logp - observed_logp)).sum(-1).double().sum().item()
            agree += int((expected.argmax(-1) == observed.argmax(-1)).sum())
            count += expected.shape[0]
            values += expected.numel()
    if not count:
        raise ValueError("No consumed logit rows to compare")
    return {"finite": True, "rows": count, "max_abs_logit_difference": maximum,
            "mean_abs_logit_difference": absolute_sum / values,
            "mean_kl_reference_to_optimized": max(0.0, kl_sum / count),
            "top1_agreement": agree / count}


def accepts(metrics):
    return (metrics.get("finite") is True
            and metrics["mean_kl_reference_to_optimized"] <= GATES["mean_kl_max"]
            and metrics["top1_agreement"] >= GATES["top1_agreement_min"])


def consumed_cpu(decoder, ids, state, confidence):
    import torch
    orig, verifier, calls, fallback = decoder._forward(ids, state, confidence)
    if calls != 1 or fallback:
        raise RuntimeError("Numerical validation requires one genuine model forward")
    if decoder.config.compact_queries and not bool((state == 0).all()):
        original = orig
    else:
        original = orig[torch.where(state == 0)[0]]
    return original.detach().cpu().clone(), verifier.detach().cpu().clone()


def decoders(model, mask_id, block_size):
    from dard_decoder import DARDConfig, DARDDecoder
    shared = dict(block_size=block_size, steps_per_block=64, temperature=0, top_p=1,
                  diagnostics="summary", allow_attention_fallback=False)
    return {
        "legacy": DARDDecoder(model, mask_id, DARDConfig(
            **shared, compact_queries=False, rectangular_attention=False)),
        "optimized": DARDDecoder(model, mask_id, DARDConfig(
            **shared, compact_queries=True, rectangular_attention=True)),
    }


def cpu_validation():
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from dard_model_adapter import install_dream_rectangular_attention
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION,
                                       trust_remote_code=True, local_files_only=True)
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], MODEL,
                                                revision=REVISION, local_files_only=True)
    tiny_config = type(config)(vocab_size=128, hidden_size=64, intermediate_size=128,
                              num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                              max_position_embeddings=256, mask_token_id=127, pad_token_id=0)
    tiny_config._attn_implementation = "sdpa"
    torch.manual_seed(124)
    model = model_class(tiny_config).eval()
    install_dream_rectangular_attention(model)
    pair = decoders(model, 127, 6)
    ids = torch.tensor([[1, 2, 3, 127, 5, 6, 7, 127, 9]])
    state = torch.tensor([0, 1, 1, 2, 0, 2])
    confidence = torch.tensor([0., .6, .6, .9, 0., .95])
    with torch.inference_mode():
        legacy = consumed_cpu(pair["legacy"], ids, state, confidence)
        optimized = consumed_cpu(pair["optimized"], ids, state, confidence)
    metrics = distribution_metrics(legacy, optimized)
    metrics.update(model_class=type(model).__name__, layers=4,
                   query_heads=8, kv_heads=2,
                   status="passed" if metrics["max_abs_logit_difference"] <= 2e-6
                   and metrics["top1_agreement"] == 1.0 else "failed")
    if metrics["status"] != "passed":
        raise RuntimeError("Float32 actual-Dream compact/rectangular parity failed: " + json.dumps(metrics))
    return metrics


def cuda_health(gpus):
    import torch
    if torch.cuda.device_count() != len(gpus):
        raise RuntimeError("CUDA visibility does not match the four locked GPUs")
    result = []
    for index, gpu in enumerate(gpus):
        device = torch.device("cuda", index)
        with torch.cuda.device(device), torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                values = (torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64) / 128).to(dtype)
                identity = torch.eye(64, device=device, dtype=dtype)
                product = values @ identity
                torch.cuda.synchronize(device)
                if not torch.equal(product, values):
                    raise RuntimeError(f"CUDA {dtype} identity product failed on {gpu['uuid']}")
                del values, identity, product
            result.append({"index": index, **gpu, "status": "passed", "dtypes": ["float32", "bfloat16"]})
            torch.cuda.empty_cache()
    return result


def gpu_validation(report, output):
    import torch
    from models import DiffuCoderBackend
    from dard_model_adapter import install_dream_rectangular_attention, uninstall_dream_rectangular_attention
    backend = None
    try:
        backend = DiffuCoderBackend(revision=REVISION, local_files_only=True, device="cuda:0", quant="bf16")
        model = backend.model
        prompt = ("You are a coding assistant. Produce a small pure function. "
                  "This is a synthetic numerical validation prompt, not a benchmark problem.\n"
                  "The function receives a sequence of integers and returns a sequence of integers.\n")
        prefix = backend.tokenizer(prompt, return_tensors="pt")["input_ids"].to(backend.device)
        block = 512
        mask_id = backend.mask_id
        canvas = torch.full((1, block), mask_id, dtype=torch.long, device=backend.device)
        all_masked_ids = torch.cat((prefix, canvas), dim=1)
        with torch.inference_mode():
            uninstall_dream_rectangular_attention(model)
            native_before = model(all_masked_ids, use_cache=False, num_logits_to_keep=block+1).logits.cpu()
            receipt = install_dream_rectangular_attention(model)
            native_after = model(all_masked_ids, use_cache=False, num_logits_to_keep=block+1).logits.cpu()
        native_equal = torch.equal(native_before, native_after)
        report["native_bf16_bitwise_equal"] = native_equal
        report["adapter"] = receipt
        report["model_metadata"] = backend.metadata()
        if not native_equal:
            raise RuntimeError("Native BF16 path changed after installing the adapter")
        del native_before, native_after
        pair = decoders(model, mask_id, block)
        seed_code = backend.tokenizer("def synthetic_function(values):\n    return list(values)\n",
                                      add_special_tokens=False)["input_ids"]
        fill = torch.tensor((seed_code * (block // len(seed_code) + 1))[:block],
                            device=backend.device, dtype=torch.long)
        patterns = {
            "half_masked": ([0, 0, 1, 2] * 128),
            "one_eighth_masked": ([0, 1, 2, 2, 2, 2, 2, 2] * 64),
            "verification_only": ([1, 2, 2, 2] * 128),
        }
        report["fixed_state_cases"] = []
        for name, entries in patterns.items():
            state = torch.tensor(entries, device=backend.device)
            confidence = torch.where(state == 0, 0., torch.where(state == 1, .65, .92))
            body = torch.where(state == 0, mask_id, fill)
            ids = torch.cat((prefix, body[None]), dim=1)
            with torch.inference_mode():
                reference = consumed_cpu(pair["legacy"], ids, state, confidence)
                optimized = consumed_cpu(pair["optimized"], ids, state, confidence)
            metrics = distribution_metrics(reference, optimized)
            del reference, optimized
            timings = {key: [] for key in pair}
            # The two preceding forwards warm both paths. Alternate order to
            # avoid assigning every first-in-pair launch to the same path.
            with torch.inference_mode():
                for repeat in range(3):
                    order = ("legacy", "optimized") if repeat % 2 == 0 else ("optimized", "legacy")
                    for key in order:
                        torch.cuda.synchronize(backend.device)
                        started = time.perf_counter()
                        outputs = pair[key]._forward(ids, state, confidence)
                        torch.cuda.synchronize(backend.device)
                        timings[key].append(time.perf_counter() - started)
                        del outputs
            legacy_time = statistics.median(timings["legacy"])
            optimized_time = statistics.median(timings["optimized"])
            record = {"name": name, "prefix_tokens": prefix.shape[1], "block_tokens": block,
                      "masked_tokens": entries.count(0), "metrics": metrics,
                      "status": "passed" if accepts(metrics) else "failed",
                      "synchronized_forward_seconds": timings,
                      "legacy_median_seconds": legacy_time,
                      "optimized_median_seconds": optimized_time,
                      "fixed_state_forward_speedup": legacy_time / optimized_time,
                      "layouts": {key: {field: decoder.last_diagnostics.get(field) for field in
                          ("last_packed_input_length", "last_packed_attention_mask_elements",
                           "packed_query_rows", "packed_key_rows", "omitted_original_query_rows")}
                          for key, decoder in pair.items()}}
            report["fixed_state_cases"].append(record)
            write_json(output / "validation_report.json", report)
            print(json.dumps({"validation_case": name, "status": record["status"],
                              "metrics": metrics, "forward_speedup": record["fixed_state_forward_speedup"]}), flush=True)
        if any(case["status"] != "passed" for case in report["fixed_state_cases"]):
            raise RuntimeError("BF16 consumed-logit approximation gate failed")
    finally:
        if backend is not None:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.data_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "running", "started_unix": time.time(),
              "model": MODEL, "revision": REVISION, "acceptance_gates": GATES,
              "scope": "Synthetic fixed-state numerical and compute-cost validation; no benchmark examples.",
              "limitations": ["BF16 rectangular/compact GEMMs need not be bitwise equal.",
                              "Passing this gate does not establish code-generation accuracy.",
                              "Fixed-state forward timings are not end-to-end generation speed claims."],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ("dard_attention.py", "dard_decoder.py", "dard_model_adapter.py",
                               "models.py", "scripts/validate_dard_optimization.py")}}
    write_json(output / "validation_report.json", report)
    write_json(output / "validation_plan.json", report)
    try:
        from scripts.four_gpu import resolve_gpus, worker_env, run_checked
        gpus = resolve_gpus(args.gpus)
        report["gpus"] = gpus
        with ExitStack() as stack:
            locks = root / "work/gpu-locks"
            locks.mkdir(parents=True, exist_ok=True)
            for gpu in sorted(gpus, key=lambda item: item["uuid"]):
                lock = stack.enter_context((locks / (gpu["uuid"] + ".lock")).open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selected = {gpu["uuid"] for gpu in gpus}
            busy = [line for line in run_checked(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader"]).splitlines() if line.split(",")[0].strip() in selected]
            if busy:
                raise RuntimeError("Selected GPUs still have compute processes: " + "; ".join(busy))
            if "torch" in sys.modules:
                raise RuntimeError("Validation must start before torch is imported")
            os.environ.update(worker_env({"root": str(root)}, {"gpu": gpus[0]}))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in gpus)
            def interrupted(signum, frame):
                raise KeyboardInterrupt("Validation interrupted by SIGTERM")
            previous = signal.signal(signal.SIGTERM, interrupted)
            stack.callback(signal.signal, signal.SIGTERM, previous)
            import torch
            torch.set_num_threads(1)
            report["cpu_actual_dream"] = cpu_validation()
            report["cuda_health"] = cuda_health(gpus)
            write_json(output / "validation_report.json", report)
            gpu_validation(report, output)
        report.update(status="passed", ended_unix=time.time())
        write_json(output / "validation_report.json", report)
        return 0
    except BaseException as exc:
        report.update(status="failed", ended_unix=time.time(), error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        write_json(output / "validation_report.json", report)
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
