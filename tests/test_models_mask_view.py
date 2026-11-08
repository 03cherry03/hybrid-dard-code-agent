"""CPU-only output views keep visible MASKs without changing legacy rendering."""
from unittest.mock import patch

import pytest
import torch

import models
from tests.test_models import FakeTokenizer, fake_cuda, fixture_backend, generate


def render(ids, *, decoder="standard", stats=None, backend=None):
    backend = backend or fixture_backend("ar" if decoder == "ar" else "diffusion")
    with fake_cuda(), patch.object(backend, "_generate_tokens", return_value=(
            torch.tensor(ids), stats or {})):
        return generate(backend, decoder)


@pytest.mark.parametrize("decoder", ["standard", "dard"])
def test_visible_masks_remain_in_common_view_but_legacy_text_is_unchanged(decoder):
    result = render([7, 98, 98, 8, 99], decoder=decoder)
    assert result["text"] == "def f():\n    return 1"
    assert result["text_with_unfilled_masks"] == "def f():\n<|mask|><|mask|>    return 1"
    assert result["unfilled_mask_tokens"] == 2
    assert result["output_sha256"] == models._sha256(result["text"])
    assert result["generated_token_ids"] == [7, 98, 98, 8, 99]


@pytest.mark.parametrize("decoder", ["standard", "dard"])
def test_mask_after_first_terminal_is_diagnostic_only(decoder):
    result = render([7, 8, 99, 98, 98], decoder=decoder)
    assert result["text_with_unfilled_masks"] == result["text"]
    assert result["unfilled_mask_tokens"] == 0
    assert result["terminal_token_index"] == 2
    assert result["raw_text_before_terminal"].endswith("<|mask|><|mask|>")


@pytest.mark.parametrize("decoder", ["ar", "standard", "dard"])
def test_no_mask_preserves_exact_existing_output_for_every_decoder(decoder):
    result = render([7, 8, 99], decoder=decoder)
    assert result["text_with_unfilled_masks"] == result["text"] == "def f():\n    return 1"
    assert result["unfilled_mask_tokens"] == 0


def test_native_padding_boundary_keeps_mask_but_removes_other_control_tokens():
    backend = fixture_backend("diffusion")

    class NativePadTokenizer(FakeTokenizer):
        def decode(self, ids, *, skip_special_tokens):
            if skip_special_tokens:
                ids = [token for token in ids if token != 97]
            return "".join("<|dlm_pad|>" if token == 97 else super(
                NativePadTokenizer, self).decode([token], skip_special_tokens=skip_special_tokens)
                for token in ids)

    backend.tokenizer = NativePadTokenizer()
    backend.terminal_ids = [97]
    result = render([7, 99, 98, 0, 8, 97, 98], backend=backend)
    assert result["text_with_unfilled_masks"] == "def f():\n<|mask|>    return 1"
    assert result["text"] == "def f():\n    return 1"
    assert result["unfilled_mask_tokens"] == 1
    assert result["terminal_token_index"] == 5
    assert "<|im_end|>" in result["text_before_special_token_cleanup"]
    assert "<pad>" in result["text_before_special_token_cleanup"]


def test_mask_at_each_edge_without_terminal_stays_visible_and_uncertified():
    result = render([98, 7, 8, 98], decoder="dard", stats={
        "incomplete": True, "verification_certified": False,
        "unresolved_mask_tokens": 2, "termination_cause": "completion_rejected"})
    assert result["text_with_unfilled_masks"] == "<|mask|>def f():\n    return 1\n<|mask|>"
    assert result["text"] == "def f():\n    return 1"
    assert result["unfilled_mask_tokens"] == result["unresolved_mask_tokens"] == 2
    assert result["incomplete"] is True
    assert result["verification_certified"] is False
    assert result["termination_cause"] == "completion_rejected"
    assert result["terminal_token_index"] is None
