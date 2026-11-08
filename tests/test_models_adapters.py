import json
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import models
from tests.test_models import FakeTokenizer, fake_cuda, fixture_backend, generate


def adapter_fixture(path, **overrides):
    config = {"peft_type": "LORA", "base_model_name_or_path": "apple/DiffuCoder-7B-cpGRPO",
              "revision": "pinned", "r": 8, "target_modules": ["q_proj", "v_proj"]}
    config.update(overrides)
    (path / "adapter_config.json").write_text(json.dumps(config))
    (path / "adapter_model.safetensors").write_bytes(b"fixture weights")


def test_local_lora_preserves_native_model_after_safe_merge_and_records_hashes(tmp_path):
    adapter_fixture(tmp_path)
    base = SimpleNamespace(eval=lambda: base, diffusion_generate=Mock())
    wrapper = SimpleNamespace(merge_and_unload=Mock(return_value=base))
    peft = ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=Mock(return_value=wrapper))
    with patch.dict("sys.modules", {"peft": peft}):
        merged, receipt = models.load_local_lora_for_inference(
            base, tmp_path, model_name="apple/DiffuCoder-7B-cpGRPO", revision="pinned")
    assert merged is base
    assert merged.diffusion_generate is base.diffusion_generate
    wrapper.merge_and_unload.assert_called_once_with(safe_merge=True)
    assert peft.PeftModel.from_pretrained.call_args.kwargs["local_files_only"] is True
    assert set(receipt["files_sha256"]) == {"adapter_config.json", "adapter_model.safetensors"}
    assert receipt["merged_for_inference"] is True


@pytest.mark.parametrize("changes,expected", [
    ({"base_model_name_or_path": "other/model"}, "base model"),
    ({"revision": "wrong"}, "base revision"),
    ({"peft_type": "PREFIX_TUNING"}, "Only local LoRA")])
def test_incompatible_adapter_is_rejected_before_loading(tmp_path, changes, expected):
    adapter_fixture(tmp_path, **changes)
    with pytest.raises(ValueError, match=expected):
        models.load_local_lora_for_inference(None, tmp_path,
            model_name="apple/DiffuCoder-7B-cpGRPO", revision="pinned")


def test_adapter_support_rejects_ar_and_quantized_merge():
    with pytest.raises(ValueError, match="BF16 diffusion"):
        models.create_backend("ar", model_name="example", adapter_path="local")
    with pytest.raises(ValueError, match="BF16 diffusion"):
        models.DiffuCoderBackend(quant="4bit", adapter_path="local")


def test_assistant_prefix_is_prefilled_without_becoming_generated_output():
    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            self.rendered = text
            return super().__call__(text, **kwargs)
    backend = fixture_backend("ar")
    backend.tokenizer = RecordingTokenizer()
    with fake_cuda():
        plain = generate(backend, "ar")
        prefixed = generate(backend, "ar", assistant_prefix="```python\n")
    assert backend.tokenizer.rendered.endswith("```python\n")
    assert prefixed["text"] == plain["text"]
    assert prefixed["generated_token_ids"] == plain["generated_token_ids"]
    assert prefixed["rendered_prompt_sha256"] != plain["rendered_prompt_sha256"]
    assert prefixed["messages_sha256"] == plain["messages_sha256"]
    assert prefixed["assistant_prefix_is_input"] is True
    assert prefixed["assistant_prefix"] == "```python\n"
