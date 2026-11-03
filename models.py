"""Shared DiffuCoder model backend for published standard decoder and DARD research reimplementation."""
from __future__ import annotations

import time
from typing import Any

from dard_decoder import DARDDecoder, DARDConfig


class DiffuCoderBackend:
    def __init__(self, model_name: str = "apple/DiffuCoder-7B-Instruct", quant: str = "4bit",
                 dtype: str = "bf16", device_map: str = "auto"):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install CUDA PyTorch and requirements.txt first") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. This project intentionally refuses silent CPU fallback.")
        self.torch = torch
        self.model_name = model_name
        self.quant = quant
        self.device_name = torch.cuda.get_device_name(0)
        print(f"[model] CUDA GPU: {self.device_name}; CUDA build: {torch.version.cuda}", flush=True)
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, device_map=device_map)
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        elif quant not in ("bf16", "none"):
            raise ValueError("--quant must be 4bit or bf16")
        self.model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        placement = getattr(self.model, "hf_device_map", None)
        if placement and any(str(value) in ("cpu", "disk") for value in placement.values()):
            raise RuntimeError("Model is offloaded to CPU/disk. Increase available GPU memory or use --quant 4bit.")
        self.device = next(self.model.parameters()).device
        self.mask_id = getattr(self.model.config, "mask_token_id", None)
        if self.mask_id is None:
            raise RuntimeError("Model config has no mask_token_id, incompatible with Dream masked diffusion")
        self.model_load_s = round(time.perf_counter() - t0, 3)
        print(f"[model] {model_name} loaded ({quant}) in {self.model_load_s:.1f}s", flush=True)

    @staticmethod
    def prompt(language: str, instruction: str, old_code: str = "", error: str = "") -> str:
        lang = "Python" if language == "py" else "Rust"
        user = (f"Implement the requested {lang} function exactly as specified. "
                "Return only the complete source for the function(s) requested; "
                "preserve the expected name and signature; no test code, no main(), "
                "no explanation, no markdown fences.\n\n"
                + instruction)
        if error:
            user += ("\n\nPREVIOUS SOURCE FAILED; return the ENTIRE fixed function source.\n"
                     "Prior source:\n" + old_code[:8500] + "\n\nCompiler/test feedback:\n" + error[:4500])
        return ("<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
                "<|im_start|>user\n" + user.strip() + "<|im_end|>\n<|im_start|>assistant\n")

    def generate(self, language: str, instruction: str, decoder: str, *, seed: int,
                 num_new_tokens: int, steps: int, dard_config: DARDConfig,
                 temperature: float, top_p: float,
                 old_code: str = "", error: str = "") -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompt = self.prompt(language, instruction, old_code, error)
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            if decoder == "standard":
                result = self.model.diffusion_generate(
                    tokens, max_new_tokens=num_new_tokens, steps=steps,
                    temperature=temperature, top_p=top_p,
                    alg="entropy", alg_temp=0.0,
                    return_dict_in_generate=True, output_history=False)
                generated = result.sequences[0, tokens.shape[1]:]
                stats: dict[str, Any] = {
                    "model_forward_passes": steps, "steps": steps,
                    "forced_tokens": 0, "blocks": []}
            elif decoder == "dard":
                generated, stats = DARDDecoder(self.model, int(self.mask_id), dard_config).generate(
                    tokens, num_new_tokens=num_new_tokens)
                generated = generated[0]
            else:
                raise ValueError(f"Unknown decoder: {decoder}")
        torch.cuda.synchronize()
        elapsed = round(time.perf_counter() - start, 3)
        text = self.tokenizer.decode(generated.tolist(), skip_special_tokens=False)
        for marker in ("<|dlm_pad|>", "<|im_end|>", "<|endoftext|>"):
            text = text.split(marker, 1)[0]
        return {
            "text": text.strip(), "wall_s": elapsed,
            "prompt_tokens": int(tokens.shape[1]),
            "output_tokens": int(generated.numel()),
            "output_token_measure": "fixed_generation_slots_not_visible_text_length",
            "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
            **stats,
        }
