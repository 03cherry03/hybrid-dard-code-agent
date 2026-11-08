"""Offline CPU tests for dependency isolation, state transitions and termination."""
import contextlib
import io
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import dard_decoder

from dard_decoder import (DARDConfig, DARDDecoder, M, C, U,
                          build_two_stream_visibility, adaptive_weights,
                          filtered_sample, verify_probabilities)
from datasets_code import Task
from evaluation import _syntax_check, strip_code, category
from main import cli, _summary


class ConstantDream:
    def __init__(self, token=7, mask_id=23, prefer_mask=False):
        self.calls = 0
        self.token, self.mask_id, self.prefer_mask = token, mask_id, prefer_mask

    def __call__(self, x, attention_mask=None, position_ids=None, **kwargs):
        self.calls += 1
        if isinstance(attention_mask, torch.Tensor):
            assert attention_mask.shape[-2:] == (x.shape[1], x.shape[1])
            assert position_ids.shape == x.shape
        logits = torch.zeros((1, x.shape[1], 24))
        logits[..., self.token] = 12 if not self.prefer_mask else 90
        if self.prefer_mask:
            logits[..., self.mask_id] = 100
        return SimpleNamespace(logits=logits)


class MixingDream:
    """Four real SDPA/residual layers: forbidden tokens can leak indirectly."""
    def __init__(self, reject_custom_mask=False):
        self.reject_custom_mask = reject_custom_mask
        self.calls = 0

    def __call__(self, x, attention_mask=None, position_ids=None, **kwargs):
        self.calls += 1
        if isinstance(attention_mask, torch.Tensor) and self.reject_custom_mask:
            raise TypeError("4D attention mask is not supported")
        # Deterministic token embeddings; no downloads, CUDA or model weights.
        dimensions = torch.arange(1, 17, dtype=torch.float32)
        hidden = torch.sin((x.float()[..., None] + 1) * dimensions * .17)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1]).unsqueeze(0)
        hidden += .05 * torch.cos(position_ids.float()[..., None] * dimensions * .11)
        mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
        for _ in range(4):
            qkv = hidden.unsqueeze(1)
            attended = torch.nn.functional.scaled_dot_product_attention(
                qkv, qkv, qkv, attn_mask=mask, dropout_p=0, is_causal=False)
            hidden = torch.tanh(hidden + attended.squeeze(1))
        return SimpleNamespace(logits=hidden)


class TestDARD(unittest.TestCase):
    def test_completion_policy_validation_and_paper_equal_threshold_grid(self):
        for tau_c in (.4, .5, .6):
            for tau_u in (.6, .7, .8):
                DARDConfig(tau_c=tau_c, tau_u=tau_u).validate()
        for options in ({"completion_policy": "unknown"},
                        {"completion_verification_steps": -1},
                        {"completion_verification_steps": True},
                        {"completion_verification_steps": 33},
                        {"completion_policy": "force_then_verify", "completion_verification_steps": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DARDConfig(**options).validate()
        DARDConfig(completion_policy="resample_then_verify",
                   completion_verification_steps=1).validate()
        with self.assertRaises(ValueError):
            DARDConfig(completion_policy="resample_then_verify",
                       completion_verification_steps=0).validate()

    def test_equal_threshold_has_no_candidate_interval_and_keeps_strict_boundary(self):
        class TwoChoiceDream(ConstantDream):
            def __init__(self, probability):
                super().__init__()
                self.probability = probability
            def __call__(self, x, **kwargs):
                result = super().__call__(x, **kwargs)
                result.logits.fill_(-torch.inf)
                result.logits[..., 7] = math.log(self.probability)
                result.logits[..., 8] = math.log(1-self.probability)
                return result
        for probability, certified in ((.5, False), (.75, True)):
            decoder = DARDDecoder(TwoChoiceDream(probability), 23, DARDConfig(
                block_size=2, steps_per_block=3, tau_c=.5, tau_u=.5,
                temperature=0, diagnostics="steps", completion_policy="verified_only"))
            output, stats = decoder.generate(torch.tensor([[1, 2]]), 2)
            self.assertEqual(stats["verification_certified"], certified)
            self.assertTrue(all(e["state_after"]["C"] == 0 for e in stats["diagnostics"]["events"]
                                if e["type"] == "step"))
            self.assertEqual(output.tolist(), [[7, 7]] if certified else [[23, 23]])

    def test_default_completion_preserves_legacy_tokens_rng_and_forward_counts(self):
        outcomes = []
        for options in ({}, {"completion_policy": "legacy_force"}):
            decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
                block_size=3, steps_per_block=1, temperature=.7, **options))
            torch.manual_seed(47)
            output, stats = decoder.generate(torch.tensor([[1, 2]]), 6)
            outcomes.append((output, stats, torch.get_rng_state()))
        self.assertTrue(torch.equal(outcomes[0][0], outcomes[1][0]))
        self.assertEqual(outcomes[0][1], outcomes[1][1])
        self.assertTrue(torch.equal(outcomes[0][2], outcomes[1][2]))
        self.assertEqual(outcomes[0][0].tolist(), [[7]*6])
        self.assertEqual(outcomes[0][1]["model_forward_passes"], 4)
        self.assertFalse(outcomes[0][1]["incomplete"])
        self.assertFalse(outcomes[0][1]["verification_certified"])

    def test_verified_only_keeps_masks_stops_next_block_and_reports_incomplete(self):
        model = ConstantDream(prefer_mask=True)
        decoder = DARDDecoder(model, 23, DARDConfig(block_size=3, steps_per_block=1,
            temperature=0, completion_policy="verified_only"))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 9)
        self.assertEqual(output.tolist(), [[23]*3])
        self.assertEqual(model.calls, 1)
        self.assertTrue(stats["incomplete"])
        self.assertFalse(stats["verification_certified"])
        self.assertEqual(stats["unresolved_mask_tokens"], 3)
        self.assertEqual(stats["completion_unverified_tokens"], 3)
        self.assertEqual(stats["forced_tokens"], 0)
        self.assertEqual(stats["termination_cause"], "unverified_at_decode_budget")
        self.assertTrue(stats["diagnostics"]["incomplete"])
        self.assertEqual(len(stats["blocks"]), 1)

    def test_new_eos_without_verification_is_retained_and_not_certified(self):
        decoder = DARDDecoder(ConstantDream(), 23, DARDConfig(block_size=3,
            steps_per_block=1, temperature=0, completion_policy="verified_only"), eos_token_ids=(7,))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 6)
        self.assertEqual(output.tolist(), [[7, 7, 7]])
        self.assertTrue(stats["incomplete"])
        self.assertFalse(stats["verification_certified"])
        self.assertEqual(stats["unresolved_mask_tokens"], 0)
        self.assertEqual(stats["completion_unverified_tokens"], 1)
        self.assertEqual(stats["stop_reason"], "incomplete")

    def test_already_verified_eos_needs_no_completion_forward_under_any_policy(self):
        for policy in ("legacy_force", "verified_only", "force_then_verify"):
            model = ConstantDream()
            decoder = DARDDecoder(model, 23, DARDConfig(block_size=3,
                steps_per_block=4, temperature=0, completion_policy=policy), eos_token_ids=(7,))
            output, stats = decoder.generate(torch.tensor([[1, 2]]), 9)
            self.assertEqual(output.tolist(), [[7]])
            self.assertEqual(model.calls, 2)
            self.assertFalse(stats["incomplete"])
            self.assertTrue(stats["verification_certified"])
            self.assertEqual(stats["completion_verification_steps"], 0)
            self.assertEqual(stats["forced_tokens"], 0)

    def test_force_then_verify_rejects_and_preserves_unresolved_masks(self):
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=3, steps_per_block=1, temperature=0,
            completion_policy="force_then_verify", completion_verification_steps=2))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 9)
        self.assertEqual(output.tolist(), [[23]*3])
        self.assertTrue(stats["incomplete"])
        self.assertFalse(stats["verification_certified"])
        self.assertEqual(stats["forced_tokens"], 3)
        self.assertEqual(stats["completion_reverified_tokens"], 3)
        self.assertEqual(stats["completion_rejected_tokens"], 3)
        self.assertEqual(stats["completion_remasked_tokens"], 3)
        self.assertEqual(stats["completion_verification_steps"], 1)
        self.assertEqual(stats["unresolved_mask_tokens"], 3)
        self.assertEqual(stats["termination_cause"], "completion_rejected")
        self.assertEqual(stats["model_forward_passes"], 3)

    def test_resample_then_verify_never_marks_rejected_tokens_as_forced_complete(self):
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=3, steps_per_block=1, temperature=0,
            completion_policy="resample_then_verify", completion_verification_steps=2))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 9)
        self.assertEqual(output.tolist(), [[23] * 3])
        self.assertTrue(stats["incomplete"])
        self.assertFalse(stats["verification_certified"])
        self.assertEqual(stats["forced_tokens"], 0)
        self.assertGreaterEqual(stats["completion_reverified_tokens"], 3)
        self.assertGreaterEqual(stats["completion_remasked_tokens"], 3)
        self.assertEqual(stats["termination_cause"], "completion_rejected")

    def test_force_then_verify_certifies_eos_before_discarding_masked_tail(self):
        class AcceptVerifier(ConstantDream):
            def __init__(self):
                super().__init__(prefer_mask=True)
            def __call__(self, x, attention_mask=None, **kwargs):
                result = super().__call__(x, attention_mask=attention_mask, **kwargs)
                if isinstance(attention_mask, torch.Tensor):
                    result.logits.zero_()
                    result.logits[..., 7] = 12
                return result
        model = AcceptVerifier()
        decoder = DARDDecoder(model, 23, DARDConfig(block_size=3, steps_per_block=1,
            transfer_limit=1, temperature=0, diagnostics="steps",
            completion_policy="force_then_verify", completion_verification_steps=2), eos_token_ids=(7,))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 6)
        self.assertEqual(output.tolist(), [[7]])
        self.assertFalse(stats["incomplete"])
        self.assertTrue(stats["verification_certified"])
        self.assertEqual(stats["completion_verification_steps"], 1)
        self.assertEqual(stats["completion_reverified_tokens"], 1)
        self.assertEqual(stats["completion_rejected_tokens"], 0)
        self.assertEqual(stats["model_forward_passes"], 3)
        self.assertEqual(stats["stop_reason"], "eos")
        events = [e for e in stats["diagnostics"]["events"] if e["type"] == "completion_verification"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["token_records"][0]["token_id"], 7)
        self.assertTrue(events[0]["token_records"][0]["verified_u"])

    def test_completion_verification_respects_budget_when_candidates_never_promote(self):
        class UncertainVerifier(ConstantDream):
            def __init__(self):
                super().__init__(prefer_mask=True)
            def __call__(self, x, attention_mask=None, **kwargs):
                result = super().__call__(x, attention_mask=attention_mask, **kwargs)
                if isinstance(attention_mask, torch.Tensor):
                    result.logits.fill_(-torch.inf)
                    result.logits[..., 7] = math.log(.7)
                    result.logits[..., 8] = math.log(.3)
                return result
        decoder = DARDDecoder(UncertainVerifier(), 23, DARDConfig(block_size=3,
            steps_per_block=1, temperature=0, completion_policy="force_then_verify",
            completion_verification_steps=2))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 6)
        self.assertEqual(output.tolist(), [[7]*3])
        self.assertTrue(stats["incomplete"])
        self.assertFalse(stats["verification_certified"])
        self.assertEqual(stats["completion_verification_steps"], 2)
        self.assertEqual(stats["completion_reverified_tokens"], 6)
        self.assertEqual(stats["completion_rejected_tokens"], 6)
        self.assertEqual(stats["completion_remasked_tokens"], 0)
        self.assertEqual(stats["unresolved_mask_tokens"], 0)
        self.assertEqual(stats["model_forward_passes"], 4)

    def test_completion_forwards_preserve_external_budget_and_deadline_exceptions(self):
        class BudgetStop(RuntimeError):
            abort_dard_generation = True
        for error in (BudgetStop("forward budget"), TimeoutError("generation deadline")):
            class InterruptedVerifier(ConstantDream):
                def __call__(self, *args, **kwargs):
                    if self.calls >= 2:
                        raise error
                    return super().__call__(*args, **kwargs)
            model = InterruptedVerifier(prefer_mask=True)
            decoder = DARDDecoder(model, 23, DARDConfig(block_size=3, steps_per_block=1,
                temperature=0, completion_policy="force_then_verify", allow_attention_fallback=True))
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)) as caught:
                decoder.generate(torch.tensor([[1, 2]]), 3)
            self.assertIs(caught.exception, error)
            self.assertEqual(model.calls, 2)
            self.assertEqual(decoder.last_diagnostics["model_forward_attempts"], 3)
            self.assertEqual(decoder.last_diagnostics["blocks"][0]["forced_tokens"], 3)
            self.assertFalse(decoder.last_diagnostics["complete"])

    def test_masked_mixing_preserves_logits_sampling_and_rng(self):
        generator = torch.Generator().manual_seed(91)
        for dtype in (torch.float32, torch.bfloat16):
            original = torch.randn(9, 64, generator=generator).to(dtype)
            shadow = torch.randn(9, 64, generator=generator).to(dtype)
            original[:, 2] = shadow[:, 2] = -torch.inf
            weights = torch.rand(9, generator=generator).clamp(.05, .99)
            masked = torch.tensor([0, 3, 5, 8])
            expected = (weights[:, None] * original.float()
                        + (1-weights[:, None]) * shadow.float())[masked]
            actual = dard_decoder._mix_masked_logits(original, shadow, weights, masked)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(actual.shape, (4, 64))
            for temperature in (0, .7):
                results = []
                for logits in (expected, actual):
                    torch.manual_seed(312)
                    with patch.object(dard_decoder, '_validate_logits',
                                      wraps=dard_decoder._validate_logits) as validate:
                        ids, confidence = filtered_sample(logits, temperature, .9,
                                                          forbidden_token_ids=(63,))
                    self.assertEqual(validate.call_count, 1)
                    results.append((ids, confidence, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))

    def test_masked_mixing_preserves_decoder_states_and_forward_budget(self):
        cfg = DARDConfig(block_size=5, steps_per_block=7, transfer_limit=2,
                         tau_c=.03, tau_u=.05, temperature=.7, top_p=.9,
                         diagnostics='steps', compact_queries=False)
        mixed_sizes = []

        def full_mixing(original, shadow, weights, masked):
            mixed_sizes.append((len(weights), len(masked)))
            return (weights[:, None] * original.float()
                    + (1-weights[:, None]) * shadow.float())[masked]

        expected_decoder = DARDDecoder(MixingDream(), 15, cfg)
        actual_decoder = DARDDecoder(MixingDream(), 15, cfg)
        torch.manual_seed(29)
        with patch.object(dard_decoder, '_mix_masked_logits', side_effect=full_mixing):
            expected, expected_stats = expected_decoder.generate(torch.tensor([[1, 2, 3]]), 8)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(29)
        actual, actual_stats = actual_decoder.generate(torch.tensor([[1, 2, 3]]), 8)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(expected_stats, actual_stats)
        self.assertTrue(any(sampled < size for size, sampled in mixed_sizes))

    def test_tail_projection_preserves_shifted_predictions_and_forced_sampling(self):
        class TailDream(MixingDream):
            def __call__(self, x, num_logits_to_keep=0, **kwargs):
                result = super().__call__(x, **kwargs)
                self.projected_rows.append(num_logits_to_keep or x.shape[1])
                result.logits = result.logits[:, -num_logits_to_keep:]
                return result

            def __init__(self):
                super().__init__()
                self.projected_rows = []

        full, tail = MixingDream(), TailDream()
        cfg = DARDConfig(block_size=3, steps_per_block=2, transfer_limit=2,
                         tau_c=.8, tau_u=.9, temperature=.7, top_p=1)
        decoders = [DARDDecoder(m, 15, cfg) for m in (full, tail)]
        ids = torch.tensor([[1, 2, 3, 4, 15, 15]])
        state, confidence = torch.tensor([C, M, M]), torch.tensor([.2, 0., 0.])
        full_prediction = decoders[0]._forward(ids, state, confidence)
        tail_prediction = decoders[1]._forward(ids, state, confidence)
        self.assertTrue(torch.equal(full_prediction[0], tail_prediction[0]))
        self.assertTrue(torch.equal(full_prediction[1], tail_prediction[1]))
        outcomes = []
        for decoder in decoders:
            torch.manual_seed(71)
            output, stats = decoder.generate(torch.tensor([[1, 2, 3, 4, 5]]), 5)
            outcomes.append((output, stats, torch.get_rng_state()))
        self.assertTrue(torch.equal(outcomes[0][0], outcomes[1][0]))
        self.assertTrue(torch.equal(outcomes[0][2], outcomes[1][2]))
        for key in ('model_forward_passes', 'forced_tokens', 'forced_forward_passes',
                    'candidate_proposals', 'revoked_tokens'):
            self.assertEqual(outcomes[0][1][key], outcomes[1][1][key])
        self.assertEqual(outcomes[1][1]['forced_tokens'], 5)
        self.assertEqual(outcomes[1][1]['logit_projection_policy'], 'needed_tail_rows_v1')
        self.assertEqual(outcomes[0][1]['logit_projection_policy'], 'full_sequence_compatibility')
        self.assertLess(outcomes[1][1]['diagnostics']['projected_logit_rows'],
                        outcomes[0][1]['diagnostics']['projected_logit_rows'])
        # Compact banks need B+|M| rows; forced fill needs B+1 for Dream's shift.
        self.assertEqual(tail.projected_rows, [5, 4, 4, 4, 4, 3, 3, 3])

    def test_projection_requires_explicit_support_and_validates_returned_width(self):
        class IgnoreTail(ConstantDream):
            def __call__(self, x, num_logits_to_keep=0, **kwargs):
                return super().__call__(x, **kwargs)
        supported = DARDDecoder(IgnoreTail(), 23, DARDConfig())
        with self.assertRaisesRegex(RuntimeError, 'plain-logit shape'):
            supported._forward(torch.tensor([[1, 2, 23]]), torch.tensor([M]), torch.tensor([0.]))
        compatible = DARDDecoder(ConstantDream(), 23, DARDConfig())
        self.assertEqual(compatible._projection_options(2), {})
        self.assertEqual(compatible.logit_projection_policy, 'full_sequence_compatibility')

    def test_tail_projection_keeps_isolated_context_fallback_alignment(self):
        class TailDream(MixingDream):
            def __call__(self, x, num_logits_to_keep=0, **kwargs):
                result = super().__call__(x, **kwargs)
                result.logits = result.logits[:, -num_logits_to_keep:]
                return result
        cfg = DARDConfig(allow_attention_fallback=True)
        plain = DARDDecoder(MixingDream(reject_custom_mask=True), 15, cfg)
        tail = DARDDecoder(TailDream(reject_custom_mask=True), 15, cfg)
        x = torch.tensor([[1, 2, 3, 4, 15]])
        state, conf = torch.tensor([C, U, M]), torch.tensor([.5, .9, 0.])
        with contextlib.redirect_stdout(io.StringIO()):
            expected = plain._forward(x, state, conf)
            actual = tail._forward(x, state, conf)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(expected[2:], actual[2:])
        self.assertEqual(actual[2:], (5, True))

    def test_all_masked_decoder_forward_preserves_native_predictions(self):
        # The former prefix+self context changed the pretrained distribution
        # even before any candidate existed. Test the end-to-end adapter here.
        decoder = DARDDecoder(MixingDream(), 15, DARDConfig())
        ids = torch.tensor([[1, 2, 15, 15, 15, 15]])
        state, confidence = torch.zeros(4, dtype=torch.long), torch.zeros(4)
        native = decoder._plain_block_logits(ids, 4)
        original, shadow, calls, fallback = decoder._forward(ids, state, confidence)
        self.assertTrue(torch.allclose(original, native, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(shadow, native, atol=1e-6, rtol=1e-6))
        self.assertEqual(calls, 1)
        self.assertFalse(fallback)

    def test_low_predecessor_is_removed_from_verifier_residual(self):
        decoder = DARDDecoder(MixingDream(), 15, DARDConfig())
        state, conf = torch.tensor([C, C, M]), torch.tensor([.2, .7, 0.])
        ids = torch.tensor([[1, 2, 3, 4, 15]])
        packed, _ = decoder._packed_inputs(ids, state, conf)
        # Direct residual exclusion; paper context does not promise complete
        # transitive self-information isolation through all reliable keys.
        self.assertEqual(packed[0, -2].item(), 15)

    def test_dream_shift_and_single_shared_prefix(self):
        class PositionDream:
            def __call__(self, ids, attention_mask, position_ids, **kwargs):
                self.ids, self.positions = ids.clone(), position_ids.clone()
                logits = torch.zeros((1, ids.shape[1], 24))
                logits[..., 0] = position_ids
                logits[..., 1] = ids
                return SimpleNamespace(logits=logits)
        model = PositionDream()
        decoder = DARDDecoder(model, 23, DARDConfig(compact_queries=False))
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # P=5, B=3
        state, conf = torch.tensor([C, C, U]), torch.tensor([.2, .7, .9])
        orig, shad, calls, fallback = decoder._forward(ids, state, conf)
        self.assertEqual(model.ids.shape[1], 5 + 4*3)
        self.assertEqual(orig[:, 0].tolist(), [4, 5, 6])
        self.assertEqual(shad[:, 0].tolist(), [4, 5, 6])
        self.assertEqual(orig[:, 1].tolist(), [5, 6, 7])
        self.assertEqual(shad[:, 1].tolist(), [5, 23, 23])
        self.assertEqual(calls, 1)
        self.assertFalse(fallback)

    def test_compact_predictions_keep_predecessor_alignment_and_sparse_mixing(self):
        ids = torch.tensor([[1, 2, 3, 15, 5, 15]])
        state, conf = torch.tensor([C, M, U, M]), torch.tensor([.5, 0., .9, 0.])
        legacy = DARDDecoder(MixingDream(), 15, DARDConfig(compact_queries=False))
        compact = DARDDecoder(MixingDream(), 15, DARDConfig())
        old_orig, old_shadow, *_ = legacy._forward(ids, state, conf)
        new_orig, new_shadow, *_ = compact._forward(ids, state, conf)
        masked = torch.where(state == M)[0]
        self.assertEqual(new_orig.shape, (2, 16))
        torch.testing.assert_close(new_orig, old_orig[masked], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(new_shadow, old_shadow, rtol=1e-6, atol=1e-6)
        weights = torch.tensor([.1, .2, .3, .4])
        expected = dard_decoder._mix_masked_logits(old_orig, old_shadow, weights, masked)
        actual = dard_decoder._mix_masked_logits(new_orig, new_shadow, weights, masked)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(compact.last_diagnostics['omitted_original_query_rows'], 2)
        self.assertLess(compact.last_diagnostics['max_actual_input_length'],
                        legacy.last_diagnostics['max_actual_input_length'])

    def test_compact_decoding_preserves_greedy_transitions_and_reduces_rows(self):
        records = []
        for compact in (False, True):
            model = ConstantDream()
            decoder = DARDDecoder(model, 23, DARDConfig(
                block_size=8, steps_per_block=8, transfer_limit=2, temperature=0,
                compact_queries=compact))
            output, stats = decoder.generate(torch.tensor([[1, 2]]), 8)
            records.append((output, stats, torch.get_rng_state()))
        self.assertTrue(torch.equal(records[0][0], records[1][0]))
        self.assertTrue(torch.equal(records[0][2], records[1][2]))
        for name in ('model_forward_passes', 'candidate_proposals', 'promoted_tokens',
                     'revoked_tokens', 'forced_tokens', 'steps'):
            self.assertEqual(records[0][1][name], records[1][1][name])
        self.assertGreater(records[1][1]['diagnostics']['omitted_original_query_rows'], 0)
        self.assertLess(records[1][1]['diagnostics']['projected_logit_rows'],
                        records[0][1]['diagnostics']['projected_logit_rows'])

    def test_confidence_is_unwarped_for_both_sampling_modes(self):
        logits = torch.tensor([[0., 1., 2.]])
        expected = torch.softmax(logits, -1)[0, 2].item()
        for temperature, top_p in [(0, 1), (.01, .01), (.3, .5)]:
            ids, confidence = filtered_sample(logits, temperature, top_p)
            self.assertEqual(ids.item(), 2)
            self.assertAlmostEqual(confidence.item(), expected, places=6)
            self.assertTrue(torch.equal(confidence, verify_probabilities(logits, ids)))
        ids, confidence = filtered_sample(logits, 0, 1, forbidden_token_ids=(2,))
        self.assertEqual(ids.item(), 1)
        self.assertAlmostEqual(confidence.item(), torch.softmax(logits, -1)[0, 1].item(), places=6)

    def test_sampling_rejects_invalid_or_mask_only_distribution(self):
        for logits in [torch.tensor([[float('nan'), 0.]]),
                       torch.tensor([[float('inf'), 0.]]),
                       torch.tensor([[-float('inf'), -float('inf')]])]:
            with self.assertRaises(ValueError):
                filtered_sample(logits, 0, 1)
        with self.assertRaisesRegex(ValueError, "non-forbidden"):
            filtered_sample(torch.tensor([[-float('inf'), 0.]]), 0, 1, forbidden_token_ids=(1,))

    def test_u_reverification_revocation_and_confidence_refresh(self):
        class ScriptedDecoder(DARDDecoder):
            predictions = [[(7, .8), (7, .5)], [(7, .4), (7, .6)],
                           [(7, .1), (7, .9)], [(8, .9), (7, .1)],
                           [(8, .95), (9, .95)], [(8, .95), (9, .95)]]
            def __init__(self):
                super().__init__(None, 15, DARDConfig(block_size=2, steps_per_block=8,
                                  transfer_limit=2, tau_c=.2, tau_u=.7, temperature=0, top_p=1,
                                  diagnostics='steps'))
                self.snapshots = []
            def _forward(self, ids, state, confidence):
                step = len(self.snapshots)
                self.snapshots.append((state.clone(), confidence.clone()))
                logits = torch.empty((2, 16))
                for i, (token, probability) in enumerate(self.predictions[step]):
                    logits[i].fill_(math.log((1-probability)/15))
                    logits[i, token] = math.log(probability)
                return logits, logits.clone(), 1, False
        decoder = ScriptedDecoder()
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 2)
        self.assertEqual(output.tolist(), [[8, 9]])
        self.assertEqual([s.tolist() for s, _ in decoder.snapshots],
                         [[M, M], [U, C], [C, C], [M, U], [U, M], [U, U]])
        self.assertTrue(torch.allclose(decoder.snapshots[2][1], torch.tensor([.4, .6])))
        self.assertEqual(stats['revoked_tokens'], 2)
        self.assertEqual(stats['demoted_tokens'], 1)
        self.assertEqual(stats['u_reverified_tokens'], 5)
        self.assertEqual(stats['forced_tokens'], 0)
        events = stats['diagnostics']['events']
        for event, (state, _) in zip(events, decoder.snapshots):
            self.assertEqual(event['state_before'], {
                name: int((state == value).sum()) for name, value in [('M', M), ('C', C), ('U', U)]})
            self.assertEqual(sum(event['state_after'].values()), 2)
        self.assertEqual(events[1]['transitions']['U_to_C'], 1)
        self.assertEqual(events[2]['transitions']['C_to_M'], 1)
        self.assertEqual(events[3]['transitions']['U_to_M'], 1)
        self.assertEqual(sum(event['revisited_tokens'] for event in events), 2)
        self.assertEqual(stats['diagnostics']['blocks'][0]['revisited_tokens'], 2)
        self.assertEqual(stats['diagnostics']['blocks'][0]['max_position_visits'], 2)
        self.assertEqual(events[-1]['verified_after'], 2)
        self.assertEqual(events[-1]['stop_reason'], 'verified')

    def test_full_decoder_with_fake_dream(self):
        fake = ConstantDream()
        decoder = DARDDecoder(fake, 23, DARDConfig(block_size=4, steps_per_block=12,
                              transfer_limit=2, temperature=0, top_p=1))
        output, stats = decoder.generate(torch.tensor([[1, 2, 3]]), 7)
        self.assertEqual(output.tolist(), [[7]*7])
        self.assertEqual(stats['forced_tokens'], 0)
        self.assertEqual(stats['promoted_tokens'], 7)
        self.assertEqual(len(stats['blocks']), 2)
        self.assertEqual(stats['model_forward_passes'], fake.calls)
        self.assertGreater(stats['u_reverified_tokens'], 0)

    def test_mask_prediction_cannot_cause_infinite_completion(self):
        fake = ConstantDream(prefer_mask=True)
        decoder = DARDDecoder(fake, 23, DARDConfig(block_size=5, steps_per_block=3,
                              transfer_limit=2, temperature=0, top_p=1))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 5)
        self.assertEqual(output.tolist(), [[7]*5])
        self.assertEqual(stats['forced_tokens'], 5)
        self.assertEqual(stats['forced_forward_passes'], 3)
        self.assertEqual(fake.calls, 1 + math.ceil(5/2))
        self.assertEqual(stats['candidate_proposals'], 0)

    def test_new_u_tokens_are_verified_before_exit(self):
        fake = ConstantDream()
        decoder = DARDDecoder(fake, 23, DARDConfig(block_size=3, transfer_limit=3,
                              steps_per_block=4, temperature=0, top_p=1))
        output, stats = decoder.generate(torch.tensor([[1]]), 3)
        self.assertEqual(output.tolist(), [[7]*3])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(stats['u_reverified_tokens'], 3)

    def test_terminal_stops_after_verified_prefix(self):
        fake = ConstantDream()
        decoder = DARDDecoder(fake, 23, DARDConfig(block_size=4, transfer_limit=4,
                              steps_per_block=4, temperature=0, top_p=1), eos_token_ids=(7,))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 12)
        self.assertEqual(output.tolist(), [[7]])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(len(stats['blocks']), 1)
        self.assertEqual(stats['stop_reason'], 'eos')
        self.assertEqual(stats['forced_tokens'], 0)

    def test_fallback_is_explicit_and_does_not_repeat_mask_work(self):
        model = MixingDream(reject_custom_mask=True)
        ids = torch.tensor([[1, 2, 3, 4, 15]])
        state, conf = torch.tensor([C, U, M]), torch.tensor([.5, .9, 0.])
        strict = DARDDecoder(model, 15, DARDConfig())
        with self.assertRaisesRegex(RuntimeError, 'fallback is disabled'):
            strict._forward(ids, state, conf)
        self.assertEqual(model.calls, 1)
        diagnostic = DARDDecoder(model, 15, DARDConfig(allow_attention_fallback=True))
        with contextlib.redirect_stdout(io.StringIO()):
            _, before, calls, fallback = diagnostic._forward(ids, state, conf)
        self.assertEqual(calls, 1 + 2 + 2)
        self.assertTrue(fallback)
        changed = ids.clone()
        changed[0, 2] = 10
        with patch('dard_decoder.build_two_stream_visibility', side_effect=AssertionError('unused')):
            _, after, calls, fallback = diagnostic._forward(changed, state, conf)
        self.assertEqual(calls, 4)
        self.assertTrue(torch.equal(before[0], after[0]))  # own C physically masked
        self.assertTrue(torch.equal(before[1], after[1]))  # U never sees C

    def test_oom_never_enables_fallback(self):
        def model(*args, **kwargs):
            raise RuntimeError('CUDA out of memory')
        decoder = DARDDecoder(model, 15, DARDConfig(allow_attention_fallback=True))
        with self.assertRaisesRegex(RuntimeError, 'out of memory'):
            decoder._forward(torch.tensor([[1, 15]]), torch.tensor([M]), torch.tensor([0.]))
        self.assertFalse(decoder._attention_fallback)

    def test_paper_prior_ratio_mixing_and_distance_half_life(self):
        plain = adaptive_weights(12, [], prior=.1, decay=8, device='cpu')
        bad = adaptive_weights(12, [(5, -1)], prior=.1, decay=8, device='cpu')
        good = adaptive_weights(12, [(5, +1)], prior=.1, decay=8, device='cpu')
        self.assertTrue(torch.equal(plain, torch.ones(12)))
        self.assertTrue(torch.equal(good, torch.ones(12)))
        self.assertAlmostEqual(float(bad[5]), .1/1.1, places=6)
        self.assertAlmostEqual(float(bad[6]), .1/(2**(-1/8)+.1), places=6)
        both = adaptive_weights(12, [(3, +1), (5, -1)], prior=.1, decay=8, device='cpu')
        self.assertAlmostEqual(float(both[4]), (2**(-1/8)+.1)/(2*2**(-1/8)+.1), places=6)

    def test_exact_mixing_endpoints_do_not_create_nan_from_infinity(self):
        original = torch.tensor([[1., 2., -torch.inf], [-torch.inf, -torch.inf, 3.]])
        shadow = torch.tensor([[-torch.inf, 4., -torch.inf], [2., 3., -torch.inf]])
        mixed = dard_decoder._mix_masked_logits(original, shadow, torch.tensor([1., 0.]), torch.arange(2))
        self.assertTrue(torch.equal(mixed[0], original[0]))
        self.assertTrue(torch.equal(mixed[1], shadow[1]))
        self.assertFalse(bool(torch.isnan(mixed).any()))

    def test_uncapped_threshold_commits_more_than_eight_positions(self):
        decoder = DARDDecoder(ConstantDream(), 23, DARDConfig(
            block_size=24, steps_per_block=3, transfer_limit=0, temperature=0, top_p=1,
            diagnostics='steps'))
        result, stats = decoder.generate(torch.tensor([[1, 2]]), 24)
        self.assertEqual(result.tolist(), [[7]*24])
        self.assertEqual(stats['diagnostics']['events'][0]['proposed'], 24)
        self.assertEqual(stats['model_forward_passes'], 2)
        self.assertEqual(stats['forced_tokens'], 0)

    def test_greedy_fixed_point_skips_identical_forward_repetitions(self):
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=16, steps_per_block=64, transfer_limit=0, temperature=0,
            diagnostics='steps'))
        result, stats = decoder.generate(torch.tensor([[1, 2]]), 16)
        self.assertEqual(result.tolist(), [[7]*16])
        self.assertEqual(stats['steps'], 1)
        self.assertEqual(stats['forced_forward_passes'], 2)
        self.assertEqual(stats['model_forward_passes'], 3)
        self.assertEqual(stats['diagnostics']['blocks'][0]['forced_completion']['reason'], 'fixed_point')

    def test_exact_two_state_cycle_enters_bounded_completion(self):
        class CyclingDream(ConstantDream):
            def __call__(self, x, attention_mask=None, **kwargs):
                self.token = 8 if isinstance(attention_mask, torch.Tensor) else 7
                return super().__call__(x, attention_mask=attention_mask, **kwargs)
        decoder = DARDDecoder(CyclingDream(), 23, DARDConfig(
            block_size=1, steps_per_block=64, temperature=0, diagnostics='steps'))
        result, stats = decoder.generate(torch.tensor([[1, 2]]), 1)
        self.assertEqual(result.tolist(), [[7]])
        self.assertEqual(stats['steps'], 2)
        self.assertEqual(stats['model_forward_passes'], 3)
        self.assertEqual(stats['diagnostics']['blocks'][0]['cycle_period'], 2)
        self.assertEqual(stats['diagnostics']['blocks'][0]['forced_completion']['reason'], 'deterministic_cycle')

    def test_bad_config_and_prompt(self):
        for config in [DARDConfig(tau_c=.6, tau_u=.2),
                       DARDConfig(allow_attention_fallback='yes'),
                       DARDConfig(compact_queries='yes'),
                       DARDConfig(rectangular_attention=1),
                       DARDConfig(temperature=float('inf')),
                       DARDConfig(diagnostics='unknown'),
                       DARDConfig(max_step_records=-1),
                       DARDConfig(max_step_records=True),
                       DARDConfig(max_step_records=1.5)]:
            with self.assertRaises(ValueError):
                config.validate()
        decoder = DARDDecoder(ConstantDream(), 23, DARDConfig())
        for prompt in [torch.empty((1, 0), dtype=torch.long), torch.tensor([[1, 23]]),
                       torch.tensor([[1], [2]])]:
            with self.assertRaises(ValueError):
                decoder.generate(prompt, 2)

    def test_diagnostics_modes_preserve_samples_rng_and_forward_count(self):
        records = []
        for mode in ('summary', 'steps', 'profile'):
            torch.manual_seed(91)
            model = ConstantDream()
            decoder = DARDDecoder(model, 23, DARDConfig(
                block_size=4, steps_per_block=5, transfer_limit=2,
                temperature=.8, top_p=1, diagnostics=mode))
            output, stats = decoder.generate(torch.tensor([[1, 2]]), 7)
            records.append((output, torch.get_rng_state(), model.calls))
            diag = stats['diagnostics']
            self.assertIs(diag, decoder.last_diagnostics)
            self.assertTrue(diag['complete'])
            self.assertFalse(diag['partial'])
            self.assertEqual(diag['model_forward_attempts'], model.calls)
            self.assertEqual(diag['model_forward_succeeded'], model.calls)
            self.assertEqual(diag['model_forward_failed'], 0)
            self.assertEqual(diag['instrumentation_affects_timing'], mode != 'summary')
            json.dumps(diag, allow_nan=False)
            if mode == 'profile':
                self.assertEqual(diag['timing_semantics'], 'cpu_wall')
                self.assertGreater(diag['component_wall_s']['model_forward'], 0)
                self.assertGreater(diag['component_wall_s']['layout_construction'], 0)
                self.assertGreater(diag['component_wall_s']['verification'], 0)
                self.assertGreater(diag['component_wall_s']['mixing_and_resampling'], 0)
            else:
                self.assertEqual(diag['component_wall_s'], {})
        for output, rng, calls in records[1:]:
            self.assertTrue(torch.equal(output, records[0][0]))
            self.assertTrue(torch.equal(rng, records[0][1]))
            self.assertEqual(calls, records[0][2])
        self.assertEqual(records[0][2], 6)  # no instrumentation model probes

    def test_summary_never_collects_extra_tensor_summaries_or_cuda_sync(self):
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=3, steps_per_block=2, transfer_limit=2, temperature=0))
        with patch.object(decoder, '_tensor_summary', side_effect=AssertionError('extra reduction')), \
                patch('torch.cuda.synchronize', side_effect=AssertionError('extra sync')):
            output, stats = decoder.generate(torch.tensor([[1]]), 3)
        self.assertEqual(output.tolist(), [[7, 7, 7]])
        self.assertEqual(stats['diagnostics']['events'], [])
        self.assertEqual(stats['diagnostics']['records_dropped'], 0)
        self.assertEqual(stats['diagnostics']['blocks'][0]['state_counts'], {'M': 3, 'C': 0, 'U': 0})
        self.assertEqual(stats['diagnostics']['blocks'][0]['zero_eligible_steps'], 1)
        forced = stats['diagnostics']['blocks'][0]['forced_completion']
        self.assertEqual(len(forced['token_records']), 3)
        self.assertEqual(sorted(r['block_position'] for r in forced['token_records']), [0, 1, 2])
        self.assertTrue(all(r['token_id'] == 7 and 0 < r['probability'] < .001
                            for r in forced['token_records']))
        self.assertEqual(forced['token_records_dropped'], 0)

    def test_forced_records_are_bounded_and_preserved_when_later_forward_fails(self):
        class InterruptedDream(ConstantDream):
            def __call__(self, *args, **kwargs):
                if self.calls == 2:
                    raise RuntimeError('forced fill interrupted')
                return super().__call__(*args, **kwargs)
        decoder = DARDDecoder(InterruptedDream(prefer_mask=True), 23, DARDConfig(
            block_size=5, steps_per_block=2, transfer_limit=2, temperature=0))
        with self.assertRaisesRegex(RuntimeError, 'forced fill interrupted'):
            decoder.generate(torch.tensor([[1, 2]]), 5)
        forced = decoder.last_diagnostics['blocks'][0]['forced_completion']
        self.assertEqual(forced['filled_tokens'], 2)
        self.assertEqual(len(forced['token_records']), 2)
        self.assertFalse(forced['complete'])
        json.dumps(decoder.last_diagnostics, allow_nan=False)
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=256, steps_per_block=1, transfer_limit=256, temperature=0))
        _, stats = decoder.generate(torch.tensor([[1, 2]]), 600)
        blocks = stats['diagnostics']['blocks']
        self.assertEqual(sum(len(b['forced_completion']['token_records']) for b in blocks), 512)
        self.assertEqual(sum(b['forced_completion']['token_records_dropped'] for b in blocks), 88)
        last_logged = blocks[1]['forced_completion']['token_records']
        self.assertTrue(all(256 <= r['output_position'] < 512 for r in last_logged))

    def test_profile_sync_failure_does_not_count_uninvoked_model(self):
        model = ConstantDream()
        decoder = DARDDecoder(model, 23, DARDConfig(diagnostics='profile'))
        fake_ids = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        with patch('torch.cuda.synchronize', side_effect=RuntimeError('prior CUDA failure')):
            with self.assertRaisesRegex(RuntimeError, 'prior CUDA'):
                decoder._call_model(fake_ids)
        self.assertEqual(model.calls, 0)
        self.assertEqual(decoder.last_diagnostics['model_forward_attempts'], 0)
        self.assertEqual(decoder.last_diagnostics['model_forward_failed'], 0)

    def test_profile_post_sync_failure_surfaces_without_overwriting_forward_error(self):
        fake_ids = SimpleNamespace(device=SimpleNamespace(type='cuda'))
        for forward_fails in (False, True):
            def model(ids):
                if forward_fails:
                    raise ValueError('original forward failure')
                return SimpleNamespace(logits=None)
            decoder = DARDDecoder(model, 23, DARDConfig(diagnostics='profile'))
            with patch('torch.cuda.synchronize', side_effect=[None, RuntimeError('async CUDA failure')]):
                expected = ValueError if forward_fails else RuntimeError
                with self.assertRaises(expected):
                    decoder._call_model(fake_ids)
            self.assertEqual(decoder.last_diagnostics['model_forward_attempts'], 1)
            self.assertEqual(decoder.last_diagnostics['model_forward_failed'], 1)
            self.assertEqual(decoder.last_diagnostics['timing_sync_error'], 'RuntimeError')

    def test_step_trace_cap_is_global_and_forced_summary_survives_cap(self):
        decoder = DARDDecoder(ConstantDream(prefer_mask=True), 23, DARDConfig(
            block_size=3, steps_per_block=2, transfer_limit=2, temperature=0,
            diagnostics='steps', max_step_records=1))
        output, stats = decoder.generate(torch.tensor([[1, 2]]), 5)
        diag = stats['diagnostics']
        self.assertEqual(output.tolist(), [[7]*5])
        self.assertEqual(len(diag['events']), 1)
        self.assertTrue(diag['trace_truncated'])
        self.assertEqual(diag['records_dropped'], 6)  # 2 fixed-point steps + 2 begin + 3 fill
        self.assertTrue(diag['complete'])
        self.assertEqual([b['forced_completion']['filled_tokens'] for b in diag['blocks']], [3, 2])
        self.assertTrue(all(b['forced_completion']['complete'] for b in diag['blocks']))
        self.assertEqual(diag['model_forward_attempts'], 5)
        first = diag['events'][0]
        self.assertEqual(first['state_before'], {'M': 3, 'C': 0, 'U': 0})
        self.assertEqual(first['state_after'], first['state_before'])
        self.assertEqual(first['eligible'], 0)
        self.assertTrue(first['stalled'])
        self.assertEqual(first['prefix_length'], 2)
        self.assertEqual(first['packed_input_length'], 14)
        self.assertEqual(first['packed_attention_mask_elements'], 196)
        json.dumps(diag, allow_nan=False)

    def test_partial_diagnostics_preserve_failed_forward_and_reset_on_next_request(self):
        class InterruptedDream(ConstantDream):
            def __call__(self, *args, **kwargs):
                if self.calls == 1:
                    self.calls += 1
                    raise TimeoutError('experiment deadline')
                return super().__call__(*args, **kwargs)
        model = InterruptedDream()
        decoder = DARDDecoder(model, 23, DARDConfig(
            block_size=2, transfer_limit=2, diagnostics='steps', temperature=0))
        with self.assertRaisesRegex(TimeoutError, 'deadline'):
            decoder.generate(torch.tensor([[1]]), 2)
        diag = decoder.last_diagnostics
        self.assertFalse(diag['complete'])
        self.assertTrue(diag['partial'])
        self.assertEqual(diag['model_forward_attempts'], 2)
        self.assertEqual(diag['model_forward_succeeded'], 1)
        self.assertEqual(diag['model_forward_failed'], 1)
        self.assertEqual(diag['error']['phase'], 'model_forward')
        self.assertEqual(diag['blocks'][0]['candidate_proposals'], 2)
        self.assertEqual(diag['events'][-1]['type'], 'error')
        self.assertEqual(diag['events'][-1]['error_type'], 'TimeoutError')
        json.dumps(diag, allow_nan=False)
        with self.assertRaises(ValueError):
            decoder.generate(torch.tensor([[23]]), 2)
        self.assertIsNot(diag, decoder.last_diagnostics)
        self.assertEqual(decoder.last_diagnostics['model_forward_attempts'], 0)
        self.assertEqual(decoder.last_diagnostics['blocks'], [])
        self.assertEqual(decoder.last_diagnostics['error']['phase'], 'input_validation')

    def test_fallback_and_strict_errors_count_failed_probe(self):
        class RejectPacked(ConstantDream):
            def __call__(self, x, attention_mask=None, **kwargs):
                if isinstance(attention_mask, torch.Tensor):
                    self.calls += 1
                    raise TypeError('unsupported packed attention')
                return super().__call__(x, attention_mask=attention_mask, **kwargs)
        model = RejectPacked()
        decoder = DARDDecoder(model, 23, DARDConfig(
            block_size=2, transfer_limit=2, temperature=0,
            allow_attention_fallback=True, diagnostics='steps'))
        with contextlib.redirect_stdout(io.StringIO()):
            _, stats = decoder.generate(torch.tensor([[1]]), 2)
        diag = stats['diagnostics']
        self.assertEqual(diag['model_forward_attempts'], 6)  # native all-M + failed packed + 4 reference
        self.assertEqual(diag['model_forward_attempts'], model.calls)
        self.assertEqual(diag['model_forward_failed'], 1)
        self.assertTrue(diag['attention_fallback_used'])
        self.assertTrue(any(e['type'] == 'attention_fallback' for e in diag['events']))
        steps = [e for e in diag['events'] if e['type'] == 'step']
        self.assertEqual(steps[0]['failed_forward_attempts'], 0)
        self.assertEqual(steps[0]['forward_calls'], 1)
        self.assertEqual(steps[1]['failed_forward_attempts'], 1)
        self.assertEqual(steps[1]['forward_calls'], 5)
        strict = DARDDecoder(RejectPacked(), 23, DARDConfig(diagnostics='steps'))
        with self.assertRaisesRegex(RuntimeError, 'fallback is disabled'):
            strict.generate(torch.tensor([[1]]), 2)
        self.assertEqual(strict.last_diagnostics['model_forward_attempts'], 2)
        self.assertEqual(strict.last_diagnostics['model_forward_failed'], 1)
        self.assertTrue(strict.last_diagnostics['partial'])

    def test_fences_and_static(self):
        self.assertEqual(strip_code("```python\nprint('hi')\n```"), "print('hi')\n")
        task = Task('humaneval', 'py', 'HumanEval/0', 0, '', '')
        self.assertEqual(_syntax_check(task, 'def f(:\n')['status'], 'syntax_error')
        self.assertIsNone(_syntax_check(task, 'def f(): return 1\n'))
        rust = Task('multiple_rs', 'rs', 'HumanEval_0', 0, '', '')
        self.assertEqual(_syntax_check(rust, 'fn main() {}')['status'], 'output_format_error')
        self.assertEqual(category('test_failed', 'AssertionError'), 'assertion_failure')

    def test_no_runner_with_repairs_fails(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli(['--runner', 'none'])
        args = cli(['--runner', 'none', '--repairs', '0', '--limit', '2'])
        self.assertEqual(args.limit, 2)

    def test_summary_no_false_failure_for_not_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'decoder': 'dard', 'dataset': 'humaneval', 'language': 'py', 'task_id': 'HumanEval/0',
                     'first_pass': None, 'final_pass': None, 'generation_wall_s': 1.0,
                     'test_wall_s': 0.0, 'model_forward_passes': 5, 'forced_tokens': 0,
                     'final_status': 'not_tested'}]
            info = _summary(rows, {'runner': 'none'}, Path(tmp))
            group = info['groups']['dard/humaneval/py']
            self.assertIsNone(group['final_pass_rate'])
            self.assertEqual(group['not_evaluated'], 1)


if __name__ == '__main__':
    unittest.main()
