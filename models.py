"""Matched single-GPU backends; importing this module never loads/downloads a model."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import uuid
from typing import Any

from dard_decoder import DARDConfig, DARDDecoder
from tokenizer_contract import (configure_chat_template, validate_diffusion_tokenizer,
                                is_pinned_dream_coder)

DEFAULT_AR_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DIFFUSION_MODEL = "apple/DiffuCoder-7B-Instruct"
LANGUAGE_NAMES = {"py": "Python", "rs": "Rust", "js": "JavaScript", "cpp": "C++"}
PROMPT_VERSION = "complete_function_shared_v2"
MESSAGES_PROMPT_VERSION = "shared_messages_v1"


@contextmanager
def _diagnostic_stage(diagnostics, name):
    """Host wall spans; CUDA work is only complete where explicitly synchronized."""
    diagnostics["phase"] = name
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        timings = diagnostics["stage_wall_s"]
        timings[name] = round(timings.get(name, 0.0) + (time.perf_counter_ns()-started)/1e9, 9)


class GenerationDeadlineExceeded(TimeoutError):
    """Cooperative deadline reached; a worker process must enforce a hard limit.

    A GPU kernel already running cannot be interrupted by a Python hook. This
    exception therefore means no completed response is accepted past the limit.
    """


def _sha256(value: Any) -> str:
    encoded = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_messages(language: str, instruction: str, old_code: str = "", error: str = "") -> list[dict[str, str]]:
    """Identical task/repair policy across model families, with no model-specific hints."""
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    user = (
        f"Implement the requested {LANGUAGE_NAMES[language]} function exactly as specified. "
        "Return the complete source for the requested function(s), including necessary imports "
        "and helper definitions. Preserve the required names and signatures. "
        "Do not include tests, a main entry point, explanations, or Markdown fences.\n\n"
        + instruction)
    if error:
        # Shared limits, with the task and requested interface never truncated.
        user += ("\n\nPREVIOUS SOURCE FAILED. Return the ENTIRE corrected source.\n"
                 "Previous source:\n" + old_code[:8500]
                 + "\n\nCompiler/test feedback:\n" + error[:4500])
    return [{"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": user}]


def ar_generation_options(*, num_new_tokens: int, temperature: float, top_p: float,
                          eos_token_ids: list[int], pad_token_id: int | None) -> dict[str, Any]:
    """Fresh config prevents inheriting checkpoint-specific sampling defaults."""
    options: dict[str, Any] = {
        "max_new_tokens": num_new_tokens, "do_sample": temperature > 0,
        "use_cache": True, "num_beams": 1, "num_return_sequences": 1,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "eos_token_id": eos_token_ids or None, "pad_token_id": pad_token_id,
        "return_dict_in_generate": True, "output_scores": False,
    }
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p, top_k=0)
    return options


def _as_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def terminal_token_policy(family: str, vocabulary: dict[str, int], *configured_eos: Any,
                          model_name: str = "", revision: str | None = None) -> dict[str, Any]:
    """Use the native DiffuCoder padding boundary; preserve AR EOS behavior.

    Apple's published DiffuCoder example splits at ``<|dlm_pad|>``. Its inherited
    Dream generation config also names ``<|endoftext|>`` as EOS, but the native
    diffusion sampler does not stop on it. Treating every chat marker as an EOS
    in postprocessing silently introduces a different termination policy.
    """
    observed_ids = {token for value in configured_eos for token in _as_ids(value)}
    for marker in ("<|im_end|>", "<|endoftext|>", "<|dlm_pad|>"):
        if marker in vocabulary:
            observed_ids.add(int(vocabulary[marker]))
    if family == "diffusion" and is_pinned_dream_coder(model_name, revision):
        # Pinned Dream-Coder's official example splits at tokenizer.eos_token;
        # <|im_end|> is a chat marker, not its native padding/EOS boundary.
        if vocabulary.get("<|endoftext|>") != 151643:
            raise ValueError("Pinned Dream-Coder EOS token mapping changed")
        terminals = [151643]
        policy = "dream_coder_native_eos_v1"
    elif family == "diffusion" and "<|dlm_pad|>" in vocabulary:
        terminals = [int(vocabulary["<|dlm_pad|>"])]
        policy = "diffucoder_native_dlm_pad_v1"
    else:
        terminals = sorted(observed_ids)
        policy = "configured_eos_and_chat_end_v1"
    return {"terminal_token_ids": terminals, "termination_policy": policy,
            "observed_end_marker_ids": sorted(observed_ids)}


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _nvidia_uuid(value: Any) -> str | None:
    """PyTorch CUuuid strings omit the GPU- prefix used by NVIDIA tooling."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith(("GPU-", "MIG-")):
        return text
    try:
        return "GPU-" + str(uuid.UUID(text))
    except (ValueError, AttributeError):
        return None


def _adapter_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(name: str) -> str:
    path = Path(name)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name[len("models--"):].replace("--", "/")
    return name.rstrip("/")


def load_local_lora_for_inference(model, adapter_path: str | Path, *, model_name: str,
                                  revision: str | None = None):
    """Merge a local LoRA into the native BF16 Dream instance before wrapping attention.

    Keeping the native model object preserves diffusion_generate and its top-level
    forward hooks. The source checkpoint and adapter files are never modified.
    """
    path = Path(adapter_path).expanduser().resolve()
    config_path, weights_path = path / "adapter_config.json", path / "adapter_model.safetensors"
    if not path.is_dir() or not config_path.is_file() or not weights_path.is_file():
        raise ValueError("Local LoRA requires adapter_config.json and adapter_model.safetensors")
    config = json.loads(config_path.read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("Only local LoRA adapters are supported")
    base_name = config.get("base_model_name_or_path")
    if not isinstance(base_name, str) or _checkpoint_identity(base_name) != _checkpoint_identity(model_name):
        raise ValueError("LoRA base model does not match the requested checkpoint")
    if config.get("revision") and revision and config["revision"] != revision:
        raise ValueError("LoRA base revision does not match the loaded checkpoint")
    base_path = Path(base_name)
    if base_path.parent.name == "snapshots" and revision and base_path.name != revision:
        raise ValueError("LoRA snapshot revision does not match the loaded checkpoint")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Loading a trained LoRA requires the peft package") from exc
    files = {p.name: _adapter_file_digest(p) for p in (config_path, weights_path)}
    wrapped = PeftModel.from_pretrained(model, str(path), is_trainable=False,
                                       local_files_only=True, autocast_adapter_dtype=False)
    merged = wrapped.merge_and_unload(safe_merge=True).eval()
    # Binding must still describe the exact bytes consumed if a producer raced.
    if any(_adapter_file_digest(path / name) != digest for name, digest in files.items()):
        raise ValueError("LoRA checkpoint changed while being loaded")
    return merged, {"path": str(path), "files_sha256": files, "peft_type": "LORA",
                    "base_model": base_name, "base_revision": revision,
                    "rank": config.get("r"), "target_modules": config.get("target_modules"),
                    "merged_for_inference": True, "safe_merge": True}


class _Backend:
    family = ""

    def __init__(self, model_name: str, quant: str = "bf16", device: str = "cuda:0",
                 revision: str | None = None, local_files_only: bool = False, *,
                 adapter_path: str | Path | None = None,
                 chat_template_override: str | None = None,
                 chat_template_sha256: str | None = None):
        if quant not in ("4bit", "bf16"):
            raise ValueError("quant must be 4bit or bf16")
        if adapter_path is not None and (self.family != "diffusion" or quant != "bf16"):
            raise ValueError("Local LoRA inference currently requires a BF16 diffusion backend")
        try:
            import torch
            from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig, GenerationConfig)
        except ImportError as exc:
            raise RuntimeError("Install CUDA PyTorch and requirements.txt first") from exc
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or requested_device.index is None:
            raise ValueError("An explicit single CUDA device is required, e.g. cuda:0")
        if not torch.cuda.is_available() or requested_device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
        self.torch = torch
        self.device = requested_device
        self.model_name = model_name
        self.quant = quant
        self.revision = revision
        self._generation_config_class = GenerationConfig
        self._dard_decoder = None
        self._dard_config = None
        self._closed = False
        self.model = None
        self.tokenizer = None
        adapter_receipt = None
        lora_receipt = None
        started = time.perf_counter()
        remote_code = self.family == "diffusion"
        kwargs: dict[str, Any] = {
            "trust_remote_code": remote_code, "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True, "device_map": {"": str(self.device)},
            "attn_implementation": "sdpa", "local_files_only": local_files_only,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        factory = AutoModelForCausalLM if self.family == "ar" else AutoModel
        try:
            with torch.cuda.device(self.device):
                self.model = factory.from_pretrained(model_name, **kwargs).eval()
                if adapter_path is not None:
                    self.model, lora_receipt = load_local_lora_for_inference(
                        self.model, adapter_path, model_name=model_name,
                        revision=getattr(self.model.config, "_commit_hash", None) or revision)
                if self.family == "diffusion":
                    from dard_model_adapter import install_dream_rectangular_attention
                    adapter_receipt = install_dream_rectangular_attention(self.model)
                actual_devices = {str(p.device) for p in self.model.parameters()}
                if actual_devices != {str(self.device)}:
                    raise RuntimeError(f"Model must reside entirely on {self.device}; found {actual_devices}")
                self.device_name = torch.cuda.get_device_name(self.device)
                torch.cuda.synchronize(self.device)
            resolved_revision = getattr(self.model.config, "_commit_hash", None)
            tokenizer_kwargs = {"trust_remote_code": remote_code, "local_files_only": local_files_only}
            if resolved_revision or revision:
                tokenizer_kwargs["revision"] = resolved_revision or revision
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            template_receipt = configure_chat_template(
                self.tokenizer, model_name=model_name, revision=resolved_revision or revision,
                template_override=chat_template_override,
                expected_template_sha256=chat_template_sha256)
            self.chat_template = self.tokenizer.get_chat_template()
            if not self.chat_template:
                raise ValueError("Tokenizer has no native chat template")
            self.mask_id = getattr(self.model.config, "mask_token_id", None)
            if self.family == "diffusion" and self.mask_id is None:
                raise ValueError("Diffusion model has no mask_token_id")
            vocabulary = self.tokenizer.get_vocab()
            tokenizer_receipt = (validate_diffusion_tokenizer(
                self.tokenizer, model_name=model_name, revision=resolved_revision or revision)
                                 if self.family == "diffusion" else None)
            termination = terminal_token_policy(
                self.family, vocabulary, getattr(self.model.config, "eos_token_id", None),
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
                model_name=model_name, revision=resolved_revision or revision)
            self.terminal_ids = termination["terminal_token_ids"]
            self.termination_policy = termination["termination_policy"]
            self.observed_end_marker_ids = termination["observed_end_marker_ids"]
            self.pad_token_id = self.tokenizer.pad_token_id
            if self.pad_token_id is None and self.terminal_ids:
                self.pad_token_id = self.terminal_ids[0]
            self.model_load_s = round(time.perf_counter() - started, 6)
            props = torch.cuda.get_device_properties(self.device)
            versions = {}
            for name in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            count_method = "transformers_num_parameters" if hasattr(self.model, "num_parameters") else "parameter_numel"
            parameter_count = (self.model.num_parameters() if hasattr(self.model, "num_parameters")
                               else sum(p.numel() for p in self.model.parameters()))
            self._metadata = {
                "family": self.family, "model": model_name, "model_name": model_name, "requested_revision": revision,
                "resolved_revision": resolved_revision,
                "tokenizer_revision": getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash") or resolved_revision,
                "tokenizer": getattr(self.tokenizer, "name_or_path", model_name),
                "tokenizer_class": type(self.tokenizer).__name__,
                "tokenizer_vocab_sha256": _sha256(vocabulary),
                "chat_template_sha256": _sha256(self.chat_template),
                "chat_template_contract": template_receipt,
                "tokenizer_contract": tokenizer_receipt,
                "model_config_sha256": _sha256(self.model.config.to_dict()),
                "model_class": type(self.model).__name__, "parameter_count": int(parameter_count),
                "parameter_count_method": count_method,
                "parameter_count_note": "Transformers logical count when available; fallback numel may count packed 4bit storage.",
                "quant": quant, "compute_dtype": "bfloat16", "attention_implementation": "sdpa",
                "device": str(self.device), "gpu_name": self.device_name,
                "gpu_uuid": _nvidia_uuid(getattr(props, "uuid", None)),
                "gpu_uuid_raw": _optional_string(getattr(props, "uuid", None)),
                "gpu_pci_bus_id": _optional_string(getattr(props, "pci_bus_id", None)),
                "gpu_capability": list(torch.cuda.get_device_capability(self.device)),
                "gpu_total_gib": round(props.total_memory / 1024**3, 3),
                "cuda_build": torch.version.cuda, "versions": versions,
                "model_load_s": self.model_load_s, "prompt_version": PROMPT_VERSION,
                "batch_size": 1, **termination,
                "output_special_token_policy": ("tokenizer_skip_special_tokens_before_padding"
                                                  if self.family == "diffusion" else "preserve_pre_eos_tokens"),
                "ar_kv_cache": self.family == "ar", "local_files_only": local_files_only,
                "dard_logit_projection": ("needed_tail_rows_if_explicitly_supported_v1"
                                           if self.family == "diffusion" else None),
                "dard_rectangular_attention_available": bool(
                    getattr(self.model, "_dard_rectangular_attention", False)),
                "dard_attention_adapter": adapter_receipt,
                "trained_adapter": lora_receipt,
                "standard_diffusion_policy": ("pinned_entropy_fixed_steps_full_slots"
                                               if self.family == "diffusion" else None),
            }
        except Exception:
            self.close()
            raise
        print(f"[model] {model_name} ({quant}) on {self.device_name} / {self.device}, "
              f"loaded in {self.model_load_s:.2f}s", flush=True)

    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def close(self) -> None:
        """Release decoder and model references before switching backends."""
        if getattr(self, "_closed", False):
            return
        self._dard_decoder = None
        self._dard_config = None
        self.model = None
        self.tokenizer = None
        self._closed = True
        gc.collect()
        with self.torch.cuda.device(self.device):
            self.torch.cuda.empty_cache()

    def _generate_tokens(self, tokens, attention_mask, decoder: str, *, num_new_tokens: int,
                         steps: int, dard_config: DARDConfig, temperature: float, top_p: float):
        if decoder == "ar" and self.family == "ar":
            options = ar_generation_options(
                num_new_tokens=num_new_tokens, temperature=temperature, top_p=top_p,
                eos_token_ids=self.terminal_ids, pad_token_id=self.pad_token_id)
            result = self.model.generate(input_ids=tokens, attention_mask=attention_mask,
                                         generation_config=self._generation_config_class(**options))
            return result.sequences[0, tokens.shape[1]:], {"forced_tokens": 0, "blocks": []}
        if self.family != "diffusion":
            raise ValueError(f"Decoder {decoder} is incompatible with {self.family} backend")
        if decoder == "standard":
            result = self.model.diffusion_generate(
                tokens, attention_mask=attention_mask, max_new_tokens=num_new_tokens, steps=steps,
                temperature=temperature, top_p=top_p if temperature > 0 else 1.0, top_k=None,
                alg="entropy", alg_temp=0.0, return_dict_in_generate=True, output_history=False)
            return result.sequences[0, tokens.shape[1]:], {"steps": steps, "forced_tokens": 0, "blocks": []}
        if decoder == "dard":
            if dard_config.temperature != temperature or dard_config.top_p != top_p:
                raise ValueError("DARD and shared sampling configuration must match")
            if self._dard_decoder is None or self._dard_config != dard_config:
                self._dard_decoder = DARDDecoder(self.model, int(self.mask_id), dard_config,
                                                 eos_token_ids=tuple(self.terminal_ids))
                self._dard_config = dard_config
            generated, stats = self._dard_decoder.generate(tokens, num_new_tokens=num_new_tokens)
            return generated[0], stats
        raise ValueError(f"Unknown decoder: {decoder}")

    def generate(self, language: str, instruction: str, decoder: str, *, seed: int,
                 num_new_tokens: int, steps: int, dard_config: DARDConfig,
                 temperature: float, top_p: float, old_code: str = "", error: str = "",
                 max_time: float | None = None, deadline: float | None = None,
                 max_input_tokens: int | None = None, assistant_prefix: str = "") -> dict[str, Any]:
        return self.generate_messages(
            build_messages(language, instruction, old_code, error), decoder,
            seed=seed, num_new_tokens=num_new_tokens, steps=steps, dard_config=dard_config,
            temperature=temperature, top_p=top_p, max_time=max_time, deadline=deadline,
            max_input_tokens=max_input_tokens, prompt_version=PROMPT_VERSION,
            assistant_prefix=assistant_prefix)

    def generate_messages(self, messages: list[dict[str, str]], decoder: str, *, seed: int,
                          num_new_tokens: int, steps: int, dard_config: DARDConfig,
                          temperature: float, top_p: float, max_time: float | None = None,
                          deadline: float | None = None, max_input_tokens: int | None = None,
                          prompt_version: str = MESSAGES_PROMPT_VERSION,
                          assistant_prefix: str = "") -> dict[str, Any]:
        """Record completed and interrupted generations without accepting partial code."""
        mode = getattr(dard_config, "diagnostics", "summary")
        started_ns = time.perf_counter_ns()
        diagnostics = {"schema_version": 1, "mode": mode, "complete": False,
            "phase": "validation", "instrumentation_affects_timing": mode != "summary",
            "stage_wall_s": {}, "forward_shapes": {"calls": 0, "sum_input_tokens": 0,
                "max_input_tokens": 0, "attention_mask_max_elements": 0,
                "unknown_shape_calls": 0}, "forward_events": [], "forward_records_dropped": 0,
            "timing_semantics": "host_wall; device_transfer_and_sync and successful model span synchronize CUDA; per-forward synchronization only in profile",
            "instrumentation_policy": "common lightweight metadata in summary; additional per-step records or synchronization are diagnostic-only",
            "forward_profile_wall_s": None if mode != "profile" else 0.0}
        self.last_generation_diagnostics = None
        if decoder == "dard" and self._dard_decoder is not None:
            self._dard_decoder.last_diagnostics = None
        try:
            result = self._generate_messages_impl(messages, decoder, seed=seed,
                num_new_tokens=num_new_tokens, steps=steps, dard_config=dard_config,
                temperature=temperature, top_p=top_p, max_time=max_time, deadline=deadline,
                max_input_tokens=max_input_tokens, prompt_version=prompt_version,
                diagnostics=diagnostics, assistant_prefix=assistant_prefix)
            decoder_diagnostics = result.pop("diagnostics", None)
            if decoder_diagnostics is not None:
                diagnostics["decoder"] = decoder_diagnostics
            diagnostics.update(complete=True, phase="complete", measurement_complete=True)
            result.update(diagnostics=diagnostics, diagnostics_mode=mode,
                          instrumentation_affects_timing=mode != "summary", measurement_complete=True)
            self.last_generation_diagnostics = copy.deepcopy({k:v for k,v in result.items() if k != "text"})
            return result
        except BaseException as exc:
            diagnostics.update(error_type=type(exc).__name__, error=str(exc)[:3000], measurement_complete=False)
            if decoder == "dard" and self._dard_decoder is not None:
                partial = getattr(self._dard_decoder, "last_diagnostics", None)
                if partial is not None:
                    diagnostics["decoder"] = copy.deepcopy(partial)
            partial = {"wall_s": round((time.perf_counter_ns()-started_ns)/1e9, 6),
                       "model_forward_passes": diagnostics["forward_shapes"]["calls"],
                       "diagnostics_mode": mode, "instrumentation_affects_timing": mode != "summary",
                       "diagnostics": diagnostics, "measurement_complete": False}
            if "prompt_tokens" in diagnostics:
                partial["prompt_tokens"] = diagnostics["prompt_tokens"]
            for name in ("messages_sha256", "rendered_prompt_sha256"):
                if name in diagnostics:
                    partial[name] = diagnostics[name]
            if "model" in diagnostics["stage_wall_s"]:
                partial["model_wall_s"] = diagnostics["stage_wall_s"]["model"]
                diagnostics["partial_model_timing_semantics"] = "host_wall; GPU work may be unfinished on failure"
            if diagnostics.get("gpu_peak_reset"):
                try:
                    partial["gpu_peak_gib"] = self.torch.cuda.max_memory_allocated(self.device) / 1024**3
                except Exception as memory_error:
                    diagnostics["gpu_peak_unavailable"] = type(memory_error).__name__
            partial["generation_budget_tokens"] = num_new_tokens
            self.last_generation_diagnostics = copy.deepcopy(partial)
            # Keep the original exception type so deadline/OOM handling still works.
            exc.generation_diagnostics = copy.deepcopy(partial)
            raise

    def _generate_messages_impl(self, messages: list[dict[str, str]], decoder: str, *, seed: int,
                          num_new_tokens: int, steps: int, dard_config: DARDConfig,
                          temperature: float, top_p: float, max_time: float | None = None,
                          deadline: float | None = None, max_input_tokens: int | None = None,
                          prompt_version: str = MESSAGES_PROMPT_VERSION, diagnostics: dict,
                          assistant_prefix: str = "") -> dict[str, Any]:
        """Generate from the exact shared conversation using the native wrapper.

        No message or token truncation occurs. ``deadline`` is an absolute
        ``time.monotonic()`` timestamp; ``max_time`` bounds this request in
        seconds. The earlier limit wins. Both include prompt preparation and
        output decoding. A parent process should enforce a hard wall limit.
        """
        if self._closed:
            raise RuntimeError("Backend is closed")
        if not isinstance(assistant_prefix, str):
            raise ValueError("assistant_prefix must be text")
        dard_config.validate()
        if (num_new_tokens < 1 or steps < 1 or temperature < 0 or
                not math.isfinite(temperature) or not 0 < top_p <= 1):
            raise ValueError("Invalid generation or sampling budget")
        if (not isinstance(messages, list) or not messages or
                any(not isinstance(item, dict) or set(item) != {"role", "content"} or
                    item["role"] not in {"system", "user", "assistant"} or
                    not isinstance(item["content"], str) for item in messages)):
            raise ValueError("messages must be nonempty role/content dictionaries with text content")
        if max_input_tokens is not None and (not isinstance(max_input_tokens, int) or max_input_tokens < 1):
            raise ValueError("max_input_tokens must be a positive integer")
        if max_time is not None and (not math.isfinite(max_time) or max_time <= 0):
            raise ValueError("max_time must be finite and positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if max_time is not None:
            relative_deadline = time.monotonic() + max_time
            deadline = relative_deadline if deadline is None else min(deadline, relative_deadline)

        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise GenerationDeadlineExceeded("Generation request exceeded its monotonic deadline")

        with _diagnostic_stage(diagnostics, "seed_and_cuda_setup"):
            check_deadline()
            messages = copy.deepcopy(messages)
            torch = self.torch
            torch.random.default_generator.manual_seed(seed)
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(seed)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            diagnostics["gpu_peak_reset"] = True
        started = time.perf_counter()
        with _diagnostic_stage(diagnostics, "prompt_render"):
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered += assistant_prefix
        with _diagnostic_stage(diagnostics, "tokenize"):
            encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with _diagnostic_stage(diagnostics, "device_transfer_and_sync"):
            tokens = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            torch.cuda.synchronize(self.device)
        diagnostics["prompt_tokens"] = int(tokens.shape[1])
        diagnostics["messages_sha256"] = _sha256(messages)
        diagnostics["rendered_prompt_sha256"] = _sha256(rendered)
        diagnostics["assistant_prefix_sha256"] = _sha256(assistant_prefix)
        diagnostics["phase"] = "context_budget_check"
        check_deadline()
        if max_input_tokens is not None and tokens.shape[1] > max_input_tokens:
            raise ValueError(f"Prompt exceeds shared input budget ({max_input_tokens}); no truncation applied")
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and tokens.shape[1] + num_new_tokens > context_limit:
            raise ValueError(f"Prompt plus output budget exceeds model context ({context_limit}); no truncation applied")
        calls = [0]
        pending_forward = []
        mode = diagnostics["mode"]
        max_records = getattr(dard_config, "max_step_records", 256)

        def count_forward(_module, _inputs, kwargs):
            check_deadline()
            if mode == "profile":
                # A failed pre-call synchronization has not invoked the model.
                torch.cuda.synchronize(self.device)
                check_deadline()
            calls[0] += 1
            shapes = diagnostics["forward_shapes"]
            shapes["calls"] += 1
            tensor = kwargs.get("input_ids")
            if tensor is None:
                tensor = kwargs.get("inputs_embeds")
            if tensor is None and _inputs:
                tensor = _inputs[0]
            shape = list(tensor.shape) if hasattr(tensor, "shape") else None
            length = int(shape[1]) if shape and len(shape) >= 2 else None
            if length is None:
                shapes["unknown_shape_calls"] += 1
            else:
                shapes["sum_input_tokens"] += int(shape[0]) * length
                shapes["max_input_tokens"] = max(shapes["max_input_tokens"], length)
            attention = kwargs.get("attention_mask")
            elements = int(attention.numel()) if hasattr(attention, "numel") else None
            if elements is not None:
                shapes["attention_mask_max_elements"] = max(shapes["attention_mask_max_elements"], elements)
            event = None
            if mode != "summary":
                if len(diagnostics["forward_events"]) < max_records:
                    event = {"call": calls[0], "input_shape": shape, "input_tokens": length,
                        "attention_mask_elements": elements, "use_cache": kwargs.get("use_cache"),
                        "has_past_key_values": kwargs.get("past_key_values") is not None}
                    diagnostics["forward_events"].append(event)
                else:
                    diagnostics["forward_records_dropped"] += 1
            if mode == "profile":
                pending_forward.append((time.perf_counter_ns(), event))

        def finish_forward(_module, _inputs, kwargs, output):
            if not pending_forward:
                return
            begin, event = pending_forward.pop()
            try:
                torch.cuda.synchronize(self.device)
            except Exception as sync_error:
                diagnostics["forward_profile_sync_error"] = type(sync_error).__name__
                if output is not None:
                    raise
                # always_call hooks also run during unwinding. Preserve the
                # original forward error if the CUDA context is already broken.
                return
            span = (time.perf_counter_ns()-begin)/1e9
            diagnostics["forward_profile_wall_s"] += span
            if event is not None:
                event.update(synchronized_wall_s=round(span, 9), returned_output=output is not None)

        hook = self.model.register_forward_pre_hook(count_forward, with_kwargs=True)
        post_hook = None
        try:
            if mode == "profile":
                post_hook = self.model.register_forward_hook(finish_forward, with_kwargs=True, always_call=True)
            model_started = time.perf_counter()
            with _diagnostic_stage(diagnostics, "model"):
                with torch.inference_mode():
                    generated, stats = self._generate_tokens(
                        tokens, attention_mask, decoder, num_new_tokens=num_new_tokens, steps=steps,
                        dard_config=dard_config, temperature=temperature, top_p=top_p)
                torch.cuda.synchronize(self.device)
                check_deadline()
                model_elapsed = time.perf_counter() - model_started
        finally:
            hook.remove()
            if post_hook is not None:
                post_hook.remove()
        with _diagnostic_stage(diagnostics, "output_decode"):
            token_ids = generated.detach().cpu().tolist()
            stop_index = next((i for i, token in enumerate(token_ids) if token in self.terminal_ids), len(token_ids))
            raw_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            text_before_cleanup = self.tokenizer.decode(token_ids[:stop_index], skip_special_tokens=False)
            # Chat/EOS control tokens can precede DiffuCoder's native padding
            # boundary. Remove tokenizer-declared control tokens from rendered
            # source without discarding subsequent content. The full sequence,
            # pre-cleanup text and remaining MASK count are retained below.
            text = (self.tokenizer.decode(token_ids[:stop_index], skip_special_tokens=True)
                    if self.family == "diffusion" else text_before_cleanup).strip()
            # A shared evaluation view must not silently erase unresolved MASKs.
            # Preserve historical `text` for existing callers; new blind graders
            # opt into this view for every decoder, without inspecting its name.
            visible_ids = token_ids[:stop_index]
            text_with_unfilled_masks = text
            if self.mask_id is not None and self.mask_id in visible_ids:
                parts, begin = [], 0
                for position, token_id in enumerate(visible_ids):
                    if token_id == self.mask_id:
                        parts.append(self.tokenizer.decode(visible_ids[begin:position], skip_special_tokens=True))
                        parts.append(self.tokenizer.decode([self.mask_id], skip_special_tokens=False))
                        begin = position+1
                parts.append(self.tokenizer.decode(visible_ids[begin:], skip_special_tokens=True))
                text_with_unfilled_masks = "".join(parts).strip()
            terminal_id = token_ids[stop_index] if stop_index < len(token_ids) else None
            terminal_text = (self.tokenizer.decode([terminal_id], skip_special_tokens=False)
                             if terminal_id is not None else None)
            end_markers = []
            for token_id in getattr(self, "observed_end_marker_ids", self.terminal_ids):
                positions = [i for i, token in enumerate(token_ids) if token == token_id]
                if positions:
                    end_markers.append({"token_id": token_id,
                        "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                        "first_index": positions[0], "count": len(positions),
                        "terminates": token_id in self.terminal_ids})
            visible_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            check_deadline()
        elapsed = time.perf_counter() - started
        stats = dict(stats)
        decoder_forward_count = stats.pop("model_forward_passes", None)
        return {
            **stats, "text": text, "wall_s": round(elapsed, 6),
            "text_with_unfilled_masks": text_with_unfilled_masks,
            "mask_rendering_policy": "preserve_literal_MASK_remove_other_declared_control_tokens",
            "decoder_stop_reason": stats.get("stop_reason"), "output_sha256": _sha256(text),
            "model_wall_s": round(model_elapsed, 6), "prompt_tokens": int(tokens.shape[1]),
            "output_tokens": visible_tokens, "visible_output_tokens": visible_tokens,
            "generated_tokens": len(token_ids), "output_slots": len(token_ids),
            "generated_token_ids": token_ids,
            "raw_text_before_terminal": raw_text,
            "raw_text_before_terminal_sha256": _sha256(raw_text),
            "text_before_special_token_cleanup": text_before_cleanup,
            "output_special_token_policy": ("tokenizer_skip_special_tokens_before_padding"
                                              if self.family == "diffusion" else "preserve_pre_eos_tokens"),
            "termination_policy": getattr(self, "termination_policy", "configured_eos_and_chat_end_v1"),
            "terminal_token_id": terminal_id,
            "terminal_token_index": stop_index if terminal_id is not None else None,
            "terminal_token_text": terminal_text,
            "tokens_after_terminal": len(token_ids)-stop_index-1 if terminal_id is not None else 0,
            "end_marker_observations": end_markers,
            "generation_budget_tokens": num_new_tokens, "code_bytes": len(text.encode("utf-8")),
            "output_token_measure": "retokenized_visible_output_before_source_extraction",
            "stop_reason": "eos" if stop_index < len(token_ids) else ("length" if len(token_ids) >= num_new_tokens else "decoder_stop"),
            "budget_exhausted": stop_index == len(token_ids) and len(token_ids) >= num_new_tokens,
            "unfilled_mask_tokens": sum(token == self.mask_id for token in token_ids[:stop_index]) if self.mask_id is not None else 0,
            "model_forward_passes": calls[0], "decoder_reported_forward_passes": decoder_forward_count,
            "forward_count_measure": "top_level_forward_invocations_including_failed_probes",
            "gpu_peak_gib": round(torch.cuda.max_memory_allocated(self.device) / 1024**3, 6),
            "messages_sha256": _sha256(messages), "rendered_prompt_sha256": _sha256(rendered),
            "chat_template_sha256": _sha256(self.chat_template), "prompt_version": prompt_version,
            "assistant_prefix": assistant_prefix, "assistant_prefix_sha256": _sha256(assistant_prefix),
            "assistant_prefix_is_input": True,
            "max_input_tokens": max_input_tokens, "max_time_s": max_time,
            "deadline_enforcement": "before_forward_and_after_completion" if deadline is not None else None,
        }


class ARBackend(_Backend):
    family = "ar"

    def __init__(self, model_name: str = DEFAULT_AR_MODEL, quant: str = "bf16",
                 device: str = "cuda:0", revision: str | None = None, local_files_only: bool = False, *,
                 chat_template_override: str | None = None, chat_template_sha256: str | None = None):
        super().__init__(model_name, quant, device, revision, local_files_only,
                         chat_template_override=chat_template_override,
                         chat_template_sha256=chat_template_sha256)


class DiffuCoderBackend(_Backend):
    family = "diffusion"

    def __init__(self, model_name: str = DEFAULT_DIFFUSION_MODEL, quant: str = "bf16",
                 device: str = "cuda:0", revision: str | None = None, local_files_only: bool = False, *,
                 adapter_path: str | Path | None = None, chat_template_override: str | None = None,
                 chat_template_sha256: str | None = None):
        super().__init__(model_name, quant, device, revision, local_files_only,
                         adapter_path=adapter_path, chat_template_override=chat_template_override,
                         chat_template_sha256=chat_template_sha256)


def create_backend(decoder: str, *, model_name: str, quant: str = "bf16", device: str = "cuda:0",
                   revision: str | None = None, local_files_only: bool = False,
                   adapter_path: str | Path | None = None, chat_template_override: str | None = None,
                   chat_template_sha256: str | None = None):
    if decoder not in ("ar", "standard", "dard"):
        raise ValueError(f"Unknown decoder: {decoder}")
    backend_class = ARBackend if decoder == "ar" else DiffuCoderBackend
    if decoder == "ar" and adapter_path is not None:
        raise ValueError("Local LoRA inference currently requires a BF16 diffusion backend")
    extra = {"adapter_path": adapter_path} if decoder != "ar" else {}
    return backend_class(model_name=model_name, quant=quant, device=device,
                         revision=revision, local_files_only=local_files_only,
                         chat_template_override=chat_template_override,
                         chat_template_sha256=chat_template_sha256, **extra)
