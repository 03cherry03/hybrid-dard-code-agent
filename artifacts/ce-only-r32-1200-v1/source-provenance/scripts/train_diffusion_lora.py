#!/usr/bin/env python3
"""torchrun worker for actual shifted-logit DiffuCoder masked-diffusion LoRA.

No benchmark examples are selected here. The controller supplies a separately
prepared training JSONL and owns GPU allocation and an external hard deadline.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import functools
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from diffusion_training import (OBJECTIVE_VERSION, atomic_json, canonical_sha256,
    capture_rng, collate_examples, corrupt_completion, file_sha256, fsync_directory,
    load_training_examples, native_attention_mask, resolve_resume, restore_rng,
    shifted_masked_diffusion_loss, verify_checkpoint, write_checkpoint_manifest)
from iteration_observability import log_event


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", default="apple/DiffuCoder-7B-cpGRPO")
    p.add_argument("--revision", default="98bc42bfd871eeec92fd0a8439ab34f75b405204")
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--epochs", type=int, default=0, help="0: cycle shuffled epochs until max-steps")
    p.add_argument("--max-steps", type=int, default=100, help="Optimizer steps, not microbatches")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=8, dest="lora_rank")
    p.add_argument("--alpha", type=int, default=16, dest="lora_alpha")
    p.add_argument("--min-mask-prob", type=float, default=0.15)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--resume", nargs="?", const="latest")
    p.add_argument("--initial-adapter", type=Path)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--deadline-reserve-seconds", type=float, default=90)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)
    if any(getattr(args, n) <= 0 for n in ("max_length", "max_steps", "batch_size",
           "gradient_accumulation", "lora_rank", "lora_alpha", "checkpoint_every")):
        p.error("Lengths, steps, batch sizes, accumulation and LoRA sizes must be positive")
    if args.epochs < 0 or args.warmup_steps < 0:
        p.error("Epochs and warmup must be nonnegative")
    if (not all(math.isfinite(getattr(args, n)) for n in ("learning_rate", "weight_decay",
            "max_grad_norm", "min_mask_prob", "deadline_unix", "deadline_reserve_seconds"))
            or args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0
            or not 0 < args.min_mask_prob <= 1 or args.deadline_reserve_seconds < 0):
        p.error("Invalid finite training hyperparameters")
    if args.resume and args.initial_adapter:
        p.error("--resume and --initial-adapter are mutually exclusive")
    if not args.revision or len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        p.error("A pinned 40-character lowercase checkpoint revision is required")
    return args


class DistributedContext:
    def __init__(self, output):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        log_event(output, "training.device_init_start", rank=self.rank,
                  local_rank=self.local_rank, world_size=self.world_size,
                  allowed_cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not torch.cuda.is_available() or self.local_rank >= torch.cuda.device_count():
            raise RuntimeError("Training requires an available local CUDA device")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        log_event(output, "training.device_init_end", device=str(self.device),
                  device_name=torch.cuda.get_device_name(self.device))
        log_event(output, "training.distributed_init_start", backend="nccl",
                  world_size=self.world_size, timeout_seconds=180,
                  enabled=self.world_size > 1)
        if self.world_size > 1:
            dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        log_event(output, "training.distributed_init_end", rank=self.rank,
                  world_size=self.world_size, enabled=self.world_size > 1)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def any(self, flag):
        value = torch.tensor(int(bool(flag)), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def reduce_mean(self, value):
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32).detach().clone()
        if self.world_size > 1:
            dist.all_reduce(value)
            value /= self.world_size
        return float(value.item())


def save_checkpoint(args, context, model, tokenizer, optimizer, scheduler, counters, config):
    directory = args.output / "checkpoints" / ("step-%08d" % counters["global_step"])
    fingerprint = canonical_sha256(config)
    began = time.monotonic()
    log_event(args.output, "training.checkpoint_start", checkpoint=directory,
              rank=context.rank, counters=counters, training_fingerprint=fingerprint)
    if directory.exists():
        verify_checkpoint(directory, expected_fingerprint=fingerprint)
        log_event(args.output, "training.checkpoint_end", checkpoint=directory,
                  rank=context.rank, reused=True, wall_seconds=time.monotonic()-began,
                  manifest_sha256=file_sha256(directory / "checkpoint_manifest.json"))
        return directory
    token = [uuid.uuid4().hex if context.rank == 0 else None]
    if context.world_size > 1:
        dist.broadcast_object_list(token, src=0)
    temporary = directory.with_name("." + directory.name + "." + token[0] + ".tmp")
    if context.rank == 0:
        temporary.mkdir(parents=True, exist_ok=False)
    context.barrier()
    torch.save(capture_rng(context.device), temporary / ("rng-rank-%04d.pt" % context.rank))
    if context.rank == 0:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "counters": counters}, temporary / "training_state.pt")
        atomic_json(temporary / "training_config.json", config)
    context.barrier()
    if context.rank == 0:
        write_checkpoint_manifest(temporary, {"training_fingerprint": fingerprint,
            "world_size": context.world_size, "counters": counters,
            "base_model": args.base_model, "base_revision": args.revision})
        os.replace(temporary, directory)
        fsync_directory(directory.parent)
        atomic_json(args.output / "latest.json", {"checkpoint": str(directory.relative_to(args.output)),
            "manifest_sha256": file_sha256(directory / "checkpoint_manifest.json"),
            "global_step": counters["global_step"]})
    context.barrier()
    log_event(args.output, "training.checkpoint_end", checkpoint=directory,
              rank=context.rank, reused=False, counters=counters,
              wall_seconds=time.monotonic()-began,
              manifest_sha256=file_sha256(directory / "checkpoint_manifest.json"))
    return directory


def record_training_state(output, record):
    atomic_json(Path(output) / "training_state.json", record)
    log_event(output, "training.state", state=record)


def run(args, context):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    from tokenizer_contract import configure_chat_template, validate_diffucoder_tokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    stop_requested = [False]
    received_signal = [None]
    def request_stop(signum, frame):
        stop_requested[0] = True
        received_signal[0] = signum
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    log_event(args.output, "training.run_start", arguments=vars(args), rank=context.rank,
              deadline_unix=args.deadline_unix,
              checkpoint_reserve_seconds=args.deadline_reserve_seconds)
    if time.time() >= args.deadline_unix - args.deadline_reserve_seconds:
        raise TimeoutError("No time remains before the checkpoint reserve")
    random.seed(args.seed + context.rank)
    torch.manual_seed(args.seed + context.rank)
    torch.cuda.manual_seed(args.seed + context.rank)
    log_event(args.output, "training.tokenizer_load_start", model=args.base_model,
              revision=args.revision, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision,
        local_files_only=args.local_files_only, trust_remote_code=True)
    template = configure_chat_template(tokenizer, model_name=args.base_model, revision=args.revision)
    tokenizer_receipt = validate_diffucoder_tokenizer(tokenizer)
    tokenizer_receipt["chat_template"] = template
    log_event(args.output, "training.tokenizer_load_end", receipt=tokenizer_receipt)
    token_ids = tokenizer_receipt["special_token_ids"]
    log_event(args.output, "training.data_load_start", train_jsonl=args.train_jsonl,
              train_jsonl_sha256=file_sha256(args.train_jsonl), max_length=args.max_length)
    examples, dataset_receipt = load_training_examples(args.train_jsonl, tokenizer,
        max_length=args.max_length, terminal_token_id=token_ids["<|dlm_pad|>"])
    sampler = DistributedSampler(examples, num_replicas=context.world_size, rank=context.rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    batches_per_epoch = len(sampler) // args.batch_size
    if batches_per_epoch < 1:
        raise ValueError("Training set is smaller than world_size * batch_size after length filtering")
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    planned_steps = min(args.max_steps, args.epochs * updates_per_epoch) if args.epochs else args.max_steps
    log_event(args.output, "training.data_load_end", dataset_receipt=dataset_receipt,
              retained_examples=len(examples), batches_per_epoch_per_rank=batches_per_epoch,
              planned_steps=planned_steps, rank=context.rank)
    resume = resolve_resume(args.output, args.resume)
    log_event(args.output, "training.resume_resolved", requested=args.resume,
              checkpoint=resume, initial_adapter=args.initial_adapter)
    initial_files = None
    if resume:
        verify_checkpoint(resume)
        # The initial adapter is ancestry, not an input to a resumed step. Keep
        # its frozen hashes without requiring --initial-adapter with --resume.
        initial_files = json.loads((resume / "training_config.json").read_text())["initial_adapter_files"]
    elif args.initial_adapter:
        if (args.initial_adapter / "checkpoint_manifest.json").exists():
            verify_checkpoint(args.initial_adapter)
        initial_files = {n: file_sha256(args.initial_adapter / n) for n in
                         ("adapter_config.json", "adapter_model.safetensors")}
    config = {"schema_version": 1, "objective": OBJECTIVE_VERSION,
        "loss_semantics": "sample_mean(sum_masked_completion_CE_div_p / total_completion_tokens); logits[t-1] predicts token[t]",
        "time_distribution": "uniform_min_mask_prob_to_one", "min_mask_prob": args.min_mask_prob,
        "base_model": args.base_model, "base_revision": args.revision,
        "train_jsonl_sha256": file_sha256(args.train_jsonl), "dataset": dataset_receipt,
        "tokenizer": tokenizer_receipt, "max_length": args.max_length,
        "epochs_limit": args.epochs, "max_steps": args.max_steps, "planned_steps": planned_steps,
        "batch_size_per_rank": args.batch_size, "world_size": context.world_size,
        "gradient_accumulation": args.gradient_accumulation, "batches_per_epoch_per_rank": batches_per_epoch,
        "examples_per_epoch_global": batches_per_epoch * args.batch_size * context.world_size,
        "drop_last_examples_per_epoch": len(examples) - batches_per_epoch * args.batch_size * context.world_size,
        "epoch_policy": "DistributedSampler(seed + epoch), drop_last; final partial accumulation window normalized by its actual microbatch count",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": min(args.warmup_steps, planned_steps), "max_grad_norm": args.max_grad_norm,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "lora_dropout": 0.0,
        "peft_task_type": None, "base_dtype": "bfloat16", "seed": args.seed,
        "initial_adapter_files": initial_files,
        "source_sha256": {name: file_sha256(ROOT / name) for name in
            ("diffusion_training.py", "scripts/train_diffusion_lora.py", "tokenizer_contract.py",
             "iteration_observability.py")}}
    fingerprint = canonical_sha256(config)
    log_event(args.output, "training.config_verified", config=config,
              training_fingerprint=fingerprint, rank=context.rank)
    existing = args.output / "training_config.json"
    if existing.exists() and canonical_sha256(json.loads(existing.read_text())) != fingerprint:
        raise ValueError("Output already contains a different training configuration")
    if context.rank == 0:
        atomic_json(existing, config)
        record_training_state(args.output, {"status": "loading", "updated_unix": time.time(),
                    "training_fingerprint": fingerprint, "planned_steps": planned_steps})
    if resume:
        verify_checkpoint(resume, expected_fingerprint=fingerprint)
    elif (args.output / "latest.json").exists():
        raise ValueError("Output has a checkpoint; pass --resume latest or use a new output")
    model_load_started = time.monotonic()
    log_event(args.output, "training.model_load_start", model=args.base_model,
              revision=args.revision, device=str(context.device), dtype="bfloat16",
              attention_implementation="sdpa", rank=context.rank)
    base = AutoModel.from_pretrained(args.base_model, revision=args.revision,
        local_files_only=args.local_files_only, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": str(context.device)}, attn_implementation="sdpa")
    if getattr(base.config, "model_type", "").lower() != "dream":
        raise ValueError("This shifted masked-diffusion objective requires a Dream backbone")
    if int(base.config.mask_token_id) != token_ids["<|mask|>"]:
        raise ValueError("Model/tokenizer mask-token mismatch")
    resolved = getattr(base.config, "_commit_hash", None)
    if resolved and resolved != args.revision:
        raise ValueError("Loaded base model revision does not match the pinned revision")
    base.config.use_cache = False
    base.requires_grad_(False)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    log_event(args.output, "training.model_load_end", rank=context.rank,
              wall_seconds=time.monotonic()-model_load_started,
              model_type=base.config.model_type, resolved_revision=resolved,
              gradient_checkpointing=True, use_cache=False)
    adapter_path = resume or args.initial_adapter
    log_event(args.output, "training.adapter_init_start", adapter=adapter_path,
              lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, targets=config["lora_targets"])
    if adapter_path:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True,
                                         autocast_adapter_dtype=True)
        loaded = model.peft_config["default"]
        if loaded.r != args.lora_rank or loaded.lora_alpha != args.lora_alpha or set(loaded.target_modules) != set(config["lora_targets"]):
            raise ValueError("Loaded adapter architecture does not match training configuration")
        if loaded.revision and loaded.revision != args.revision:
            raise ValueError("Loaded adapter has a different base revision")
    else:
        model = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=0.0, target_modules=config["lora_targets"], bias="none", task_type=None,
            base_model_name_or_path=args.base_model, revision=args.revision))
    model.peft_config["default"].base_model_name_or_path = args.base_model
    model.peft_config["default"].revision = args.revision
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Only LoRA parameters may require gradients")
    log_event(args.output, "training.adapter_init_end", rank=context.rank,
              trainable_parameters=sum(p.numel() for _, p in trainable),
              trainable_parameter_names=[name for name, _ in trainable],
              base_model=args.base_model, base_revision=args.revision)
    model.train()
    log_event(args.output, "training.ddp_wrap_start", enabled=context.world_size > 1)
    wrapped = DistributedDataParallel(model, device_ids=[context.local_rank],
        output_device=context.local_rank, broadcast_buffers=False, find_unused_parameters=False) if context.world_size > 1 else model
    log_event(args.output, "training.ddp_wrap_end", enabled=context.world_size > 1)
    parameters = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], planned_steps)
    counters = {"global_step": 0, "epoch": 0, "next_batch": 0, "microbatches_seen_per_rank": 0,
                "examples_seen_global": 0}
    if resume:
        log_event(args.output, "training.resume_start", checkpoint=resume, rank=context.rank,
                  training_fingerprint=fingerprint)
        saved = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        counters = saved["counters"]
        restore_rng(torch.load(resume / ("rng-rank-%04d.pt" % context.rank),
                               map_location="cpu", weights_only=True), context.device)
        log_event(args.output, "training.resume_end", checkpoint=resume, rank=context.rank,
                  counters=counters, optimizer_restored=True, scheduler_restored=True,
                  rank_rng_restored=True)
    context.barrier()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    checkpoint = resume
    stop_reason = "max_steps"
    log_event(args.output, "training.optimization_start", rank=context.rank,
              counters=counters, planned_steps=planned_steps)
    while counters["global_step"] < planned_steps:
        if context.any(stop_requested[0] or time.time() >= args.deadline_unix - args.deadline_reserve_seconds):
            stop_reason = "signal" if stop_requested[0] else "deadline"
            break
        epoch = counters["epoch"]
        sampler.set_epoch(epoch)
        log_event(args.output, "training.epoch_start", rank=context.rank,
                  epoch=epoch, next_batch=counters["next_batch"],
                  sampler_seed=args.seed+epoch, batches_per_epoch=batches_per_epoch)
        loader = DataLoader(examples, batch_size=args.batch_size, sampler=sampler, drop_last=True,
            num_workers=0, collate_fn=functools.partial(collate_examples, pad_token_id=token_ids["<|dlm_pad|>"]),
            generator=torch.Generator().manual_seed(args.seed + epoch))
        start_batch = counters["next_batch"]
        iterator = iter(loader)
        for _ in range(start_batch):
            next(iterator)
        batch_index = start_batch
        while batch_index < batches_per_epoch and counters["global_step"] < planned_steps:
            if context.any(stop_requested[0] or time.time() >= args.deadline_unix - args.deadline_reserve_seconds):
                stop_reason = "signal" if stop_requested[0] else "deadline"
                break
            window = min(args.gradient_accumulation, batches_per_epoch - batch_index)
            loss_total, masked_total, target_total = 0.0, 0, 0
            log_event(args.output, "training.optimizer_step_start", rank=context.rank,
                      global_step=counters["global_step"]+1, epoch=epoch,
                      next_batch=batch_index, accumulation_microbatches=window)
            step_started = time.monotonic()
            for micro in range(window):
                batch = next(iterator)
                log_event(args.output, "training.microbatch_start", rank=context.rank,
                          global_step=counters["global_step"]+1, epoch=epoch,
                          batch_index=batch_index, microbatch_index=micro,
                          example_ids=batch["example_ids"])
                clean = batch["input_ids"].to(context.device)
                target = batch["target_mask"].to(context.device)
                valid = batch["valid_mask"].to(context.device)
                noisy, masked, probs = corrupt_completion(clean, target,
                    mask_token_id=token_ids["<|mask|>"], min_mask_prob=args.min_mask_prob)
                sync = nullcontext() if micro == window - 1 or context.world_size == 1 else wrapped.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = wrapped(input_ids=noisy, attention_mask=native_attention_mask(valid),
                            use_cache=False, return_dict=True, num_logits_to_keep=0).logits
                        loss = shifted_masked_diffusion_loss(logits, clean, target, masked, probs)
                    if context.any(not bool(torch.isfinite(loss.detach()))):
                        raise FloatingPointError("Nonfinite masked-diffusion loss on at least one rank")
                    (loss / window).backward()
                local_loss = float(loss.detach())
                local_masked = int(masked.sum())
                local_target = int(target.sum())
                loss_total += local_loss
                masked_total += local_masked
                target_total += local_target
                batch_index += 1
                counters["microbatches_seen_per_rank"] += 1
                counters["examples_seen_global"] += args.batch_size * context.world_size
                log_event(args.output, "training.microbatch_end", rank=context.rank,
                          global_step=counters["global_step"]+1, epoch=epoch,
                          batch_index=batch_index-1, microbatch_index=micro,
                          example_ids=batch["example_ids"],
                          input_shape=list(clean.shape), loss=local_loss,
                          masked_tokens=local_masked, target_tokens=local_target,
                          backward_completed=True, gradients_synchronized=micro == window-1 or context.world_size == 1)
                del logits, loss, noisy, clean, target, valid, masked, probs
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if context.any(not bool(torch.isfinite(grad_norm))):
                raise FloatingPointError("Nonfinite LoRA gradient norm on at least one rank")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            counters["global_step"] += 1
            counters["next_batch"] = batch_index
            if batch_index == batches_per_epoch:
                counters["epoch"] += 1
                counters["next_batch"] = 0
            metric = {"global_step": counters["global_step"], "epoch": epoch,
                "next_batch": counters["next_batch"], "loss": context.reduce_mean(loss_total / window),
                "grad_norm": context.reduce_mean(grad_norm), "learning_rate": scheduler.get_last_lr()[0],
                "masked_tokens_mean_per_rank": context.reduce_mean(masked_total),
                "target_tokens_mean_per_rank": context.reduce_mean(target_total),
                "step_wall_s": time.monotonic() - step_started,
                "max_memory_allocated_gib": context.reduce_mean(torch.cuda.max_memory_allocated(context.device) / 1024**3),
                "updated_unix": time.time()}
            log_event(args.output, "training.optimizer_step_end", rank=context.rank,
                      metric=metric, counters=counters,
                      local_loss=loss_total/window, local_masked_tokens=masked_total,
                      local_target_tokens=target_total)
            if context.rank == 0:
                with (args.output / "training_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                record_training_state(args.output, {"status": "running", **counters,
                    "latest_metric": metric, "training_fingerprint": fingerprint,
                    "planned_steps": planned_steps, "updated_unix": time.time()})
                print(json.dumps(metric, sort_keys=True), flush=True)
            if counters["global_step"] % args.checkpoint_every == 0:
                checkpoint = save_checkpoint(args, context, model, tokenizer, optimizer, scheduler, counters, config)
        if stop_reason in ("signal", "deadline"):
            break
    if received_signal[0] is not None:
        log_event(args.output, "training.signal_received", signal=received_signal[0],
                  rank=context.rank, deferred_to_optimizer_boundary=True)
    log_event(args.output, "training.optimization_end", rank=context.rank, counters=counters,
              stop_reason=stop_reason, planned_steps=planned_steps,
              deadline_unix=args.deadline_unix, signal_requested=stop_requested[0])
    checkpoint = save_checkpoint(args, context, model, tokenizer, optimizer, scheduler, counters, config)
    if context.rank == 0:
        final_adapter = None
        if counters["global_step"] >= planned_steps:
            final_adapter = args.output / "final_adapter"
            temporary_link = args.output / (".final_adapter." + uuid.uuid4().hex)
            temporary_link.symlink_to(checkpoint.relative_to(args.output), target_is_directory=True)
            os.replace(temporary_link, final_adapter)
            fsync_directory(args.output)
        record_training_state(args.output, {"status": "completed" if counters["global_step"] >= planned_steps else "interrupted",
            "stop_reason": stop_reason, **counters, "planned_steps": planned_steps,
            "checkpoint": str(checkpoint), "final_adapter": str(final_adapter) if final_adapter else None,
            "training_fingerprint": fingerprint,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "invocation_wall_s": time.time() - started, "updated_unix": time.time()})
    log_event(args.output, "training.run_end", rank=context.rank,
              status="completed" if counters["global_step"] >= planned_steps else "interrupted",
              stop_reason=stop_reason, counters=counters, checkpoint=checkpoint,
              invocation_wall_s=time.time()-started)
    return 0 if counters["global_step"] >= planned_steps else 75


def main(argv=None):
    args = arguments(argv)
    context = None
    log_event(args.output, "training.process_start", arguments=vars(args),
              rank=int(os.environ.get("RANK", 0)),
              local_rank=int(os.environ.get("LOCAL_RANK", 0)),
              world_size=int(os.environ.get("WORLD_SIZE", 1)),
              allowed_cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        context = DistributedContext(args.output)
        code = run(args, context)
        log_event(args.output, "training.process_end", rank=context.rank,
                  exit_code=code, status="completed" if code == 0 else "interrupted")
        return code
    except BaseException as exc:
        log_event(args.output, "training.process_failure",
                  rank=context.rank if context is not None else int(os.environ.get("RANK", 0)),
                  error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if context is not None:
            atomic_json(args.output / ("failure-rank-%04d.json" % context.rank), {
                "status": "failed", "rank": context.rank, "error_type": type(exc).__name__,
                "error": str(exc)[:4000], "updated_unix": time.time()})
        if context is None or context.rank == 0:
            record_training_state(args.output, {"status": "failed", "error_type": type(exc).__name__,
                "error": str(exc)[:4000], "updated_unix": time.time()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            log_event(args.output, "training.distributed_cleanup_start")
            dist.destroy_process_group()
            log_event(args.output, "training.distributed_cleanup_end")


if __name__ == "__main__":
    raise SystemExit(main())
