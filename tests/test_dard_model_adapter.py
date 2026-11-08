"""Rectangular Dream attention parity and isolation contracts (CPU, no downloads)."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from dard_attention import M, C, U, build_two_stream_visibility, packed_inputs
from dard_model_adapter import (DreamRectangularAttention,
                                install_dream_rectangular_attention,
                                uninstall_dream_rectangular_attention)


class RotaryFixture(nn.Module):
    def forward(self, value, position_ids):
        frequencies = position_ids[..., None].float() * torch.tensor([1.0, .1, .01, .001])
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(value.dtype), angles.sin().to(value.dtype)


class DreamSdpaAttention(nn.Module):
    """Independent full-square reference with biases, GQA and genuine RoPE."""
    def __init__(self, heads=4, kv_heads=2):
        super().__init__()
        self.num_heads, self.num_key_value_heads = heads, kv_heads
        self.num_key_value_groups = heads // kv_heads
        self.head_dim, self.hidden_size = 8, 8 * heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, 8 * kv_heads, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, 8 * kv_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryFixture()
        self.native_calls = 0

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None):
        self.native_calls += 1
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch, length, self.num_heads, 8).transpose(1, 2)
        k = self.k_proj(hidden_states).reshape(batch, length, self.num_key_value_heads, 8).transpose(1, 2)
        v = self.v_proj(hidden_states).reshape(batch, length, self.num_key_value_heads, 8).transpose(1, 2)
        cos, sin = position_embeddings or self.rotary_emb(v, position_ids)
        q = q * cos[:, None] + torch.cat((-q[..., 4:], q[..., :4]), dim=-1) * sin[:, None]
        k = k * cos[:, None] + torch.cat((-k[..., 4:], k[..., :4]), dim=-1) * sin[:, None]
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        result = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        result = result.transpose(1, 2).contiguous().reshape(batch, length, self.hidden_size)
        return self.o_proj(result), None, past_key_value


class LayerFixture(nn.Module):
    def __init__(self, kv_heads=2):
        super().__init__()
        self.self_attn = DreamSdpaAttention(kv_heads=kv_heads)
        self.input_norm = nn.LayerNorm(32)
        self.output_norm = nn.LayerNorm(32)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))

    def forward(self, hidden, mask, positions):
        hidden = hidden + self.self_attn(self.input_norm(hidden), mask, positions)[0]
        return hidden + self.mlp(self.output_norm(hidden))


class ModelFixture(nn.Module):
    def __init__(self, kv_heads=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([LayerFixture(kv_heads) for _ in range(4)])
        self.embed = nn.Embedding(32, 32)
        self.output = nn.Linear(32, 64)

    def forward(self, ids, mask=None, positions=None):
        hidden = self.embed(ids)
        for layer in self.model.layers:
            hidden = layer(hidden, mask, positions)
        return self.output(hidden)


class DreamModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(124)

    def _inputs(self):
        state = torch.tensor([M, C, C, U, M, U])
        confidence = torch.tensor([0., .6, .6, .9, 0., .95])
        prefix = 3
        ids = torch.tensor([[1, 2, 3, 31, 5, 6, 7, 31, 9]])
        joined, positions = packed_inputs(ids, state, confidence, 31)
        mask = build_two_stream_visibility(prefix, state, confidence)
        return joined, positions, mask, prefix + 2 * state.numel()

    def test_four_layers_match_square_reference_with_grouped_and_ungrouped_heads(self):
        ids, positions, square, keys = self._inputs()
        for kv_heads in (1, 2, 4):
            with self.subTest(kv_heads=kv_heads):
                reference = ModelFixture(kv_heads).eval()
                adapted = copy.deepcopy(reference)
                install_dream_rectangular_attention(adapted)
                with torch.inference_mode():
                    expected = reference(ids, square, positions)
                    actual = adapted(ids, square[..., :keys], positions)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_rectangular_projection_sizes_and_native_delegation_are_isolated(self):
        ids, positions, square, keys = self._inputs()
        original = DreamSdpaAttention().eval()
        wrapper = DreamRectangularAttention(original)
        hidden = torch.randn(1, ids.shape[1], 32)
        observed = {}
        handles = [getattr(original, name).register_forward_pre_hook(
            lambda module, args, name=name: observed.setdefault(name, []).append(args[0].shape[1]))
            for name in ("q_proj", "k_proj", "v_proj")]
        try:
            expected = wrapper(hidden, square, positions)[0]
            wrapper(hidden, square[..., :keys], positions)
            again = wrapper(hidden, square, positions)[0]
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed["q_proj"], [ids.shape[1]] * 3)
        self.assertEqual(observed["k_proj"], [ids.shape[1], keys, ids.shape[1]])
        self.assertEqual(observed["v_proj"], [ids.shape[1], keys, ids.shape[1]])
        self.assertEqual(original.native_calls, 2)
        self.assertTrue(torch.equal(again, expected))

    def test_precomputed_rope_matches_internal_rope(self):
        ids, positions, square, keys = self._inputs()
        original = DreamSdpaAttention().eval()
        wrapper = DreamRectangularAttention(original)
        hidden = torch.randn(1, ids.shape[1], 32)
        embeddings = original.rotary_emb(hidden, positions)
        implicit = wrapper(hidden, square[..., :keys], positions)[0]
        explicit = wrapper(hidden, square[..., :keys], position_embeddings=embeddings)[0]
        self.assertTrue(torch.equal(implicit, explicit))

    def test_install_idempotent_uninstall_restores_weights_and_state_dict_paths(self):
        model = ModelFixture().eval()
        original_layers = [layer.self_attn for layer in model.model.layers]
        state_keys = list(model.state_dict())
        parameters = list(model.parameters())
        receipt = install_dream_rectangular_attention(model)
        self.assertEqual(receipt["layers"], 4)
        install_dream_rectangular_attention(model)
        self.assertEqual(parameters, list(model.parameters()))
        self.assertTrue(model._dard_rectangular_attention)
        for original, layer in zip(original_layers, model.model.layers):
            self.assertIs(layer.self_attn.original, original)
            self.assertIs(layer.self_attn.rotary_emb, original.rotary_emb)
            self.assertFalse(layer.self_attn.training)
        uninstall_dream_rectangular_attention(model)
        self.assertFalse(model._dard_rectangular_attention)
        self.assertEqual(state_keys, list(model.state_dict()))
        for original, layer in zip(original_layers, model.model.layers):
            self.assertIs(layer.self_attn, original)

    def test_install_failure_is_atomic(self):
        model = ModelFixture().eval()
        model.model.layers[-1].self_attn = nn.Identity()
        first = model.model.layers[0].self_attn
        with self.assertRaises(TypeError):
            install_dream_rectangular_attention(model)
        self.assertIs(model.model.layers[0].self_attn, first)
        self.assertFalse(hasattr(model, "_dard_rectangular_attention"))

    def test_unsafe_rectangular_calls_fail_fast(self):
        ids, positions, square, keys = self._inputs()
        wrapper = DreamRectangularAttention(DreamSdpaAttention().eval())
        hidden = torch.randn(1, ids.shape[1], 32)
        for options in ({"use_cache": True}, {"past_key_value": object()},
                        {"output_attentions": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                wrapper(hidden, square[..., :keys], positions, **options)
        with self.assertRaises(ValueError):
            wrapper(hidden, square[..., :keys].float(), positions)
        with self.assertRaises(ValueError):
            wrapper(hidden, square[..., :-1, :keys], positions)
        with self.assertRaises(ValueError):
            wrapper(hidden, square[..., :keys])
        wrapper.train()
        with self.assertRaises(ValueError):
            wrapper(hidden, square[..., :keys], positions)


class ValidationMetricTests(unittest.TestCase):
    def test_gate_handles_empty_blocks_and_softmax_offset_invariance(self):
        from scripts.validate_dard_optimization import accepts, distribution_metrics
        expected = (torch.empty(0, 3), torch.tensor([[1., 2., 3.], [-1., 1., 0.]]))
        observed = (torch.empty(0, 3), expected[1] + 100.)
        metrics = distribution_metrics(expected, observed, chunk_rows=1)
        self.assertTrue(accepts(metrics))
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["max_abs_logit_difference"], 100.)
        self.assertEqual(metrics["mean_kl_reference_to_optimized"], 0.)

    def test_nonfinite_or_changed_distribution_does_not_pass_gate(self):
        from scripts.validate_dard_optimization import accepts, distribution_metrics
        expected = (torch.tensor([[0., 3.]]),)
        for observed in ((torch.tensor([[3., 0.]]),), (torch.tensor([[0., float("nan")]]),)):
            with self.subTest(observed=observed):
                self.assertFalse(accepts(distribution_metrics(expected, observed)))
        with self.assertRaises(ValueError):
            distribution_metrics((torch.empty(0, 3),), (torch.empty(0, 3),))


if __name__ == "__main__":
    unittest.main()
