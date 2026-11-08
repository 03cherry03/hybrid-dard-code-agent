import json

import pytest
from jinja2 import Environment

from tokenizer_contract import (CPGRPO_CHAT_TEMPLATE, CPGRPO_MODEL, CPGRPO_REVISION,
    configure_chat_template, text_sha256, validate_diffucoder_tokenizer,
    validate_tokenizer_assets, validate_diffusion_tokenizer, is_pinned_dream_coder,
    DREAM_CODER_MODEL, DREAM_CODER_REVISION)


class TokenizerFixture:
    pad_token_id = 2

    def __init__(self, template=None):
        self.chat_template = template
        self.vocab = {"<|mask|>": 1, "<|dlm_pad|>": 2, "<|im_start|>": 3, "<|im_end|>": 4}

    def get_chat_template(self):
        if self.chat_template is None:
            raise ValueError("missing template")
        return self.chat_template

    def get_vocab(self):
        return self.vocab

    def encode(self, source, add_special_tokens=False):
        return [self.vocab[source]] if source in self.vocab else [1000 + ord(c) for c in source]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i - 1000) for i in ids)


def test_native_template_is_preserved_and_bound():
    tokenizer = TokenizerFixture("native template")
    receipt = configure_chat_template(tokenizer, model_name="example/native")
    assert tokenizer.chat_template == "native template"
    assert receipt["policy"] == "checkpoint_native"
    assert receipt["template_sha256"] == text_sha256("native template")


def test_cpgrpo_fallback_renders_the_published_role_delimiters():
    tokenizer = TokenizerFixture()
    receipt = configure_chat_template(tokenizer, model_name=CPGRPO_MODEL, revision=CPGRPO_REVISION)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write add."}]
    rendered = Environment().from_string(tokenizer.chat_template).render(
        messages=messages, add_generation_prompt=True)
    assert rendered == ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        "<|im_start|>user\nWrite add.<|im_end|>\n<|im_start|>assistant\n")
    assert CPGRPO_REVISION in receipt["source_url"]
    assert receipt["template_sha256"] == text_sha256(CPGRPO_CHAT_TEMPLATE)


@pytest.mark.parametrize("name,revision", [(CPGRPO_MODEL, None), (CPGRPO_MODEL, "different"),
                                          ("example/other", CPGRPO_REVISION)])
def test_missing_template_never_silently_uses_an_unrelated_fallback(name, revision):
    with pytest.raises(ValueError, match="no native chat template"):
        configure_chat_template(TokenizerFixture(), model_name=name, revision=revision)


def test_explicit_template_override_requires_and_checks_its_hash():
    tokenizer = TokenizerFixture("native")
    with pytest.raises(ValueError, match="requires its SHA256"):
        configure_chat_template(tokenizer, model_name="example", template_override="new")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        configure_chat_template(tokenizer, model_name="example", template_override="new",
                                expected_template_sha256="0" * 64)
    assert tokenizer.chat_template == "native"
    configure_chat_template(tokenizer, model_name="example", template_override="new",
                            expected_template_sha256=text_sha256("new"))
    assert tokenizer.chat_template == "new"


def test_slow_bpe_tokenizer_assets_do_not_require_tokenizer_json(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "DreamTokenizer"}))
    (tmp_path / "vocab.json").write_text("{}")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    assert validate_tokenizer_assets(tmp_path)["layout"] == "bpe_vocab_merges"
    (tmp_path / "merges.txt").unlink()
    with pytest.raises(ValueError, match="complete tokenizer asset"):
        validate_tokenizer_assets(tmp_path)


def test_special_tokens_and_multilingual_source_roundtrip():
    receipt = validate_diffucoder_tokenizer(TokenizerFixture())
    assert receipt["status"] == "verified"
    assert receipt["roundtrip_languages"] == ["py", "rs", "js", "cpp"]
    tokenizer = TokenizerFixture()
    tokenizer.pad_token_id = 1
    with pytest.raises(ValueError, match="padding token"):
        validate_diffucoder_tokenizer(tokenizer)


def test_failed_roundtrip_is_rejected():
    tokenizer = TokenizerFixture()
    tokenizer.decode = lambda *args, **kwargs: "changed source"
    with pytest.raises(ValueError, match="roundtrip failed"):
        validate_diffucoder_tokenizer(tokenizer)


def dream_tokenizer():
    tokenizer = TokenizerFixture("native Dream template")
    tokenizer.vocab = {"<|endoftext|>": 151643, "<|im_start|>": 151644,
                       "<|im_end|>": 151645, "<|mask|>": 151666}
    tokenizer.pad_token_id = tokenizer.eos_token_id = 151643
    return tokenizer


def test_pinned_dream_native_contract_preserves_tokens_and_template():
    tokenizer = dream_tokenizer()
    before = dict(tokenizer.vocab), tokenizer.chat_template
    receipt = validate_diffusion_tokenizer(tokenizer, model_name=DREAM_CODER_MODEL,
                                          revision=DREAM_CODER_REVISION)
    assert receipt["native_eos_token_id"] == receipt["native_pad_token_id"] == 151643
    assert receipt["special_token_ids"]["<|im_end|>"] != receipt["native_eos_token_id"]
    assert (tokenizer.vocab, tokenizer.chat_template) == before
    assert "<|dlm_pad|>" not in tokenizer.vocab


@pytest.mark.parametrize("revision", [None, "wrong", "0" * 40])
def test_dream_contract_cannot_be_used_with_an_unpinned_revision(revision):
    with pytest.raises(ValueError, match="exact pinned revision"):
        validate_diffusion_tokenizer(dream_tokenizer(), model_name=DREAM_CODER_MODEL, revision=revision)


def test_pinned_dream_snapshot_identity_checks_repository_namespace():
    path = "/cache/models--Dream-org--Dream-Coder-v0-Instruct-7B/snapshots/" + DREAM_CODER_REVISION
    assert is_pinned_dream_coder(path, None)
    assert not is_pinned_dream_coder(path.replace("Dream-org", "another-owner"), None)
    assert not is_pinned_dream_coder(path, "wrong")


@pytest.mark.parametrize("attribute", ["pad_token_id", "eos_token_id"])
def test_dream_chat_end_cannot_replace_native_pad_or_eos(attribute):
    tokenizer = dream_tokenizer()
    setattr(tokenizer, attribute, 151645)
    with pytest.raises(ValueError, match="padding/EOS boundary"):
        validate_diffusion_tokenizer(tokenizer, model_name=DREAM_CODER_MODEL, revision=DREAM_CODER_REVISION)


def test_dream_changed_vocabulary_and_roundtrip_are_rejected():
    tokenizer = dream_tokenizer()
    tokenizer.vocab["<|mask|>"] = 999
    with pytest.raises(ValueError, match="mapping changed"):
        validate_diffusion_tokenizer(tokenizer, model_name=DREAM_CODER_MODEL, revision=DREAM_CODER_REVISION)
    tokenizer = dream_tokenizer()
    tokenizer.decode = lambda *a, **k: "wrong"
    with pytest.raises(ValueError, match="roundtrip failed"):
        validate_diffusion_tokenizer(tokenizer, model_name=DREAM_CODER_MODEL, revision=DREAM_CODER_REVISION)


def test_diffucoder_contract_is_unchanged_by_dispatcher():
    tokenizer = TokenizerFixture()
    assert validate_diffusion_tokenizer(tokenizer, model_name=CPGRPO_MODEL, revision=CPGRPO_REVISION) == validate_diffucoder_tokenizer(tokenizer)
    with pytest.raises(ValueError, match="missing special token"):
        validate_diffucoder_tokenizer(dream_tokenizer())


def test_dream_terminal_policy_uses_only_native_eos_and_preserves_other_backends():
    from models import terminal_token_policy
    vocab = dream_tokenizer().vocab
    dream = terminal_token_policy("diffusion", vocab, 151643, model_name=DREAM_CODER_MODEL,
                                  revision=DREAM_CODER_REVISION)
    assert dream["terminal_token_ids"] == [151643]
    assert dream["termination_policy"] == "dream_coder_native_eos_v1"
    assert dream["observed_end_marker_ids"] == [151643, 151645]
    assert terminal_token_policy("ar", vocab, 151643)["terminal_token_ids"] == [151643, 151645]
    assert terminal_token_policy("diffusion", {**vocab, "<|dlm_pad|>": 151667}, 151643)["terminal_token_ids"] == [151667]
