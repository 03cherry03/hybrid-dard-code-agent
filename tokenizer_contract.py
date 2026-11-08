"""Offline diffusion tokenizer checks and versioned chat-template provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CPGRPO_MODEL = "apple/DiffuCoder-7B-cpGRPO"
CPGRPO_REVISION = "98bc42bfd871eeec92fd0a8439ab34f75b405204"
DREAM_CODER_MODEL = "Dream-org/Dream-Coder-v0-Instruct-7B"
DREAM_CODER_REVISION = "5d9e88c723af9045f362748b5284bdf43d9c501e"
CPGRPO_TEMPLATE_SOURCE = (
    "https://huggingface.co/apple/DiffuCoder-7B-cpGRPO/blob/"
    + CPGRPO_REVISION + "/README.md")
# The pinned model card supplies this Qwen role-delimiter format. It does not
# ship a tokenizer chat_template. Keep the fallback explicit and checkpoint-bound.
CPGRPO_CHAT_TEMPLATE = (
    "{%- if messages and messages[0]['role'] != 'system' %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pinned_cpgrpo(model_name: str, revision: str | None) -> bool:
    if model_name == CPGRPO_MODEL:
        return revision == CPGRPO_REVISION
    path = Path(model_name)
    return (path.name == CPGRPO_REVISION and path.parent.name == "snapshots"
            and path.parent.parent.name == "models--apple--DiffuCoder-7B-cpGRPO"
            and revision in (None, CPGRPO_REVISION))


def is_pinned_dream_coder(model_name: str, revision: str | None) -> bool:
    if model_name == DREAM_CODER_MODEL:
        return revision == DREAM_CODER_REVISION
    path = Path(model_name)
    return (path.name == DREAM_CODER_REVISION and path.parent.name == "snapshots"
            and path.parent.parent.name == "models--Dream-org--Dream-Coder-v0-Instruct-7B"
            and revision in (None, DREAM_CODER_REVISION))


def configure_chat_template(tokenizer, *, model_name: str, revision: str | None = None,
                            template_override: str | None = None,
                            expected_template_sha256: str | None = None) -> dict:
    """Retain native templates; only the pinned cpGRPO may use our fallback.

    Explicit overrides require their expected hash, so caller configuration can
    bind training and inference to the same rendering contract.
    """
    if template_override is not None:
        if not expected_template_sha256:
            raise ValueError("An explicit chat-template override requires its SHA256")
        template = template_override
        source = "explicit_sha256_bound_override"
        source_url = None
    else:
        try:
            template = tokenizer.get_chat_template()
        except ValueError:
            template = None
        if template:
            source, source_url = "checkpoint_native", None
        elif _pinned_cpgrpo(model_name, revision):
            template = CPGRPO_CHAT_TEMPLATE
            source, source_url = "cpgrpo_model_card_qwen_v1", CPGRPO_TEMPLATE_SOURCE
        else:
            raise ValueError("Tokenizer has no native chat template or pinned fallback")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Chat template must be a nonempty string")
    digest = text_sha256(template)
    if expected_template_sha256 and digest != expected_template_sha256:
        raise ValueError("Chat template SHA256 mismatch")
    # Do not mutate checkpoint files; only this loaded tokenizer instance changes.
    tokenizer.chat_template = template
    return {"policy": source, "source_url": source_url,
            "template_sha256": digest, "model": model_name, "revision": revision}


def validate_tokenizer_assets(snapshot: str | Path) -> dict:
    """Accept either a serialized tokenizer or the native slow BPE assets."""
    snapshot = Path(snapshot)
    if not (snapshot / "tokenizer_config.json").is_file():
        raise ValueError("Candidate required file missing: tokenizer_config.json")
    for layout, names in (("serialized", ("tokenizer.json",)),
                          ("bpe_vocab_merges", ("vocab.json", "merges.txt")),
                          ("sentencepiece", ("tokenizer.model",))):
        if all((snapshot / name).is_file() for name in names):
            return {"layout": layout, "files": list(names)}
    raise ValueError("Candidate has no supported complete tokenizer asset set")


def validate_diffucoder_tokenizer(tokenizer) -> dict:
    """Verify actual load semantics, special-token IDs and source round trips."""
    vocab = tokenizer.get_vocab()
    required = ("<|mask|>", "<|dlm_pad|>", "<|im_start|>", "<|im_end|>")
    ids = {}
    for token in required:
        token_id = vocab.get(token)
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("DiffuCoder tokenizer missing special token: " + token)
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError("DiffuCoder special token does not encode atomically: " + token)
        ids[token] = token_id
    if len(set(ids.values())) != len(ids):
        raise ValueError("DiffuCoder special-token IDs collide")
    if tokenizer.pad_token_id != ids["<|dlm_pad|>"]:
        raise ValueError("DiffuCoder padding token disagrees with native padding boundary")
    samples = ("def add(a, b):\n    return a + b\n",
               "fn add(a: i64, b: i64) -> i64 { a + b }\n",
               "function add(a, b) { return a + b; }\n",
               "long add(long a, long b) { return a + b; }\n")
    for source in samples:
        token_ids = tokenizer.encode(source, add_special_tokens=False)
        if tokenizer.decode(token_ids, skip_special_tokens=False) != source:
            raise ValueError("DiffuCoder tokenizer source roundtrip failed")
    serialized = json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"status": "verified", "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": len(vocab), "vocab_sha256": text_sha256(serialized),
            "special_token_ids": ids, "roundtrip_languages": ["py", "rs", "js", "cpp"]}


def validate_diffusion_tokenizer(tokenizer, *, model_name: str, revision: str | None) -> dict:
    """Keep DiffuCoder semantics; admit only the pinned Dream-Coder EOS contract.

    Dream-Coder's native PAD/EOS is <|endoftext|>. It has no DiffuCoder
    <|dlm_pad|> boundary. Never add tokens or relabel its existing vocabulary.
    """
    if not is_pinned_dream_coder(model_name, revision):
        path = Path(model_name)
        if (model_name == DREAM_CODER_MODEL or
                path.parent.parent.name == "models--Dream-org--Dream-Coder-v0-Instruct-7B"):
            raise ValueError("Dream-Coder tokenizer requires its exact pinned revision")
        return validate_diffucoder_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    ids = {"<|endoftext|>": 151643, "<|im_start|>": 151644,
           "<|im_end|>": 151645, "<|mask|>": 151666}
    for token, expected_id in ids.items():
        if vocab.get(token) != expected_id or tokenizer.encode(token, add_special_tokens=False) != [expected_id]:
            raise ValueError("Pinned Dream-Coder special-token mapping changed: " + token)
    if "<|dlm_pad|>" in vocab:
        raise ValueError("Pinned Dream-Coder unexpectedly contains a DiffuCoder padding token")
    if tokenizer.pad_token_id != ids["<|endoftext|>"] or tokenizer.eos_token_id != ids["<|endoftext|>"]:
        raise ValueError("Dream-Coder native padding/EOS boundary mismatch")
    samples = ("def add(a, b):\n    return a + b\n",
               "fn add(a: i64, b: i64) -> i64 { a + b }\n",
               "function add(a, b) { return a + b; }\n",
               "long add(long a, long b) { return a + b; }\n")
    for source in samples:
        token_ids = tokenizer.encode(source, add_special_tokens=False)
        if tokenizer.decode(token_ids, skip_special_tokens=False) != source:
            raise ValueError("Dream-Coder tokenizer source roundtrip failed")
    serialized = json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"status": "verified", "tokenizer_class": type(tokenizer).__name__,
            "contract": "pinned_dream_coder_native_eos_v1", "vocab_size": len(vocab),
            "vocab_sha256": text_sha256(serialized), "special_token_ids": ids,
            "native_pad_token_id": tokenizer.pad_token_id, "native_eos_token_id": tokenizer.eos_token_id,
            "roundtrip_languages": ["py", "rs", "js", "cpp"]}


def load_verified_tokenizer(snapshot: str | Path, *, model_name: str, revision: str):
    """Load a fully local tokenizer and return a JSON-serializable receipt."""
    from transformers import AutoTokenizer
    assets = validate_tokenizer_assets(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                             trust_remote_code=True)
    template = configure_chat_template(tokenizer, model_name=model_name, revision=revision)
    receipt = validate_diffusion_tokenizer(tokenizer, model_name=model_name, revision=revision)
    receipt.update(assets=assets, chat_template=template)
    return tokenizer, receipt
