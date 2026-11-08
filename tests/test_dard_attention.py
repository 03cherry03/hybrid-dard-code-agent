"""Numerical and direct-context contracts for shifted Dream DARD attention."""
from types import SimpleNamespace
from itertools import product
import unittest

import torch

from dard_attention import M, C, U, build_two_stream_visibility, packed_inputs, compact_packed_inputs


class BidirectionalFixture:
    """Four genuine SDPA/residual layers, including duplicated RoPE coordinates."""
    def __call__(self, ids, attention_mask=None, position_ids=None):
        dimensions = torch.arange(1, 17, dtype=torch.float32)
        hidden = torch.sin((ids.float()[..., None]+1)*dimensions*.17)
        if position_ids is None:
            position_ids = torch.arange(ids.shape[1])[None]
        hidden += .05*torch.cos(position_ids.float()[..., None]*dimensions*.11)
        for _ in range(4):
            qkv = hidden[:, None]
            keys = qkv if attention_mask is None else qkv[:, :, :attention_mask.shape[-1]]
            attended = torch.nn.functional.scaled_dot_product_attention(
                qkv, keys, keys, attn_mask=attention_mask, dropout_p=0, is_causal=False)
            hidden = torch.tanh(hidden+attended[:, 0])
        return SimpleNamespace(logits=hidden)


class DreamDardAttentionTests(unittest.TestCase):
    def test_every_query_selects_one_key_per_position_and_no_predictor_keys(self):
        state = torch.tensor([M, C, C, U, U])
        conf = torch.tensor([0., .4, .5, .9, .8])
        p, n = 3, len(state)
        mask = build_two_stream_visibility(p, state, conf)[0, 0]
        self.assertEqual(mask.shape, (p+4*n, p+4*n))
        self.assertTrue(bool(mask[:, :p].all()))
        self.assertTrue(bool((mask[:, p:p+n] ^ mask[:, p+n:p+2*n]).all()))
        self.assertFalse(bool(mask[:, p+2*n:].any()))

    def test_original_and_shadow_context_follow_paper_direct_choices(self):
        state = torch.tensor([M, C, C, U, U])
        conf = torch.tensor([0., .4, .5, .9, .8])
        p, n = 2, len(state)
        mask = build_two_stream_visibility(p, state, conf)[0, 0]
        # A prefix row retains masked replacements, not just its own prefix.
        self.assertEqual(mask[0, p:p+n].tolist(), [False, False, False, True, True])
        # Original M sees all original keys; C sees more-confident C and U.
        self.assertEqual(mask[p, p:p+n].tolist(), [True]*n)
        self.assertEqual(mask[p+1, p:p+n].tolist(), [False, True, True, True, True])
        self.assertEqual(mask[p+2, p:p+n].tolist(), [False, False, True, True, True])
        # Original U context keeps other reliable U tokens, unlike old adapter.
        self.assertEqual(mask[p+3, p:p+n].tolist(), [False, False, False, True, True])
        # Shadow U and verification query exclude their own original token.
        self.assertEqual(mask[p+n+3, p:p+n].tolist(), [False, False, False, False, True])
        self.assertTrue(torch.equal(mask[p+n:p+2*n], mask[p+3*n:]))

    def test_equal_confidence_candidates_use_paper_greater_equal_rule(self):
        state, conf = torch.tensor([C, C]), torch.tensor([.4, .4])
        mask = build_two_stream_visibility(1, state, conf)[0, 0]
        self.assertTrue(bool(mask[1:3, 1:3].all()))
        self.assertEqual(mask[3:5, 1:3].tolist(), [[False, True], [True, False]])

    def test_predecessor_embeddings_mask_only_forbidden_verifier_predecessors(self):
        p, state = 2, torch.tensor([C, U, C, M])
        conf = torch.tensor([.4, .9, .5, 0.])
        ids = torch.tensor([[2, 3, 4, 5, 6, 15]])
        joined, positions = packed_inputs(ids, state, conf, 15)
        n = len(state)
        self.assertEqual(joined[0, p+n:p+2*n].tolist(), [15]*n)
        self.assertEqual(joined[0, p+2*n:p+3*n].tolist(), [3, 4, 5, 6])
        self.assertEqual(joined[0, p+3*n:].tolist(), [3, 15, 5, 15])
        self.assertEqual(positions[0, p+n:p+2*n].tolist(), [2, 3, 4, 5])
        self.assertEqual(positions[0, p+2*n:].tolist(), [1, 2, 3, 4, 1, 2, 3, 4])

    def test_all_masked_predictors_match_native_bidirectional_logits(self):
        model = BidirectionalFixture()
        for p, n in [(1, 1), (2, 4), (7, 5)]:
            with self.subTest(prefix=p, block=n):
                ids = torch.tensor([list(range(1, p+1))+[15]*n])
                state = torch.full((n,), M)
                conf = torch.zeros(n)
                native = model(ids).logits[:, p-1:p+n-1]
                joined, positions = packed_inputs(ids, state, conf, 15)
                actual = model(joined, build_two_stream_visibility(p, state, conf), positions).logits
                torch.testing.assert_close(actual[:, p+2*n:p+3*n], native, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(actual[:, p+3*n:], native, rtol=1e-6, atol=1e-6)

    def test_invalid_inputs_fail_before_building_visibility(self):
        with self.assertRaises(ValueError):
            build_two_stream_visibility(0, torch.tensor([M]), torch.zeros(1))
        with self.assertRaises(ValueError):
            build_two_stream_visibility(1, torch.tensor([3]), torch.zeros(1))
        with self.assertRaises(ValueError):
            build_two_stream_visibility(1, torch.tensor([M]), torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):
            packed_inputs(torch.ones((2, 3), dtype=torch.long), torch.tensor([M]), torch.zeros(1), 15)

    def test_compact_rows_and_rectangular_keys_preserve_all_consumed_predictions(self):
        model = BidirectionalFixture()
        for states in product((M, C, U), repeat=3):
            for rectangular in (False, True):
                with self.subTest(states=states, rectangular=rectangular):
                    p, n = 2, len(states)
                    state = torch.tensor(states)
                    confidence = torch.tensor([0., .5, .9])
                    ids = torch.tensor([[2, 4, 6, 8, 10]])
                    ids[0, p:][state == M] = 15
                    full_ids, full_positions = packed_inputs(ids, state, confidence, 15)
                    full_mask = build_two_stream_visibility(p, state, confidence)
                    full = model(full_ids, full_mask, full_positions).logits
                    joined, positions, mask, masked = compact_packed_inputs(
                        ids, state, confidence, 15, rectangular=rectangular)
                    keep = torch.cat((torch.arange(p+2*n), p+2*n+masked,
                                      torch.arange(p+3*n, p+4*n)))
                    self.assertTrue(torch.equal(joined, full_ids[:, keep]))
                    self.assertTrue(torch.equal(positions, full_positions[:, keep]))
                    keys = torch.arange(p+2*n) if rectangular else keep
                    self.assertTrue(torch.equal(mask, full_mask[:, :, keep][:, :, :, keys]))
                    self.assertEqual(joined.shape[1], p+3*n+masked.numel())
                    compact = model(joined, mask, positions).logits
                    # Verifies context rows too: their parity proves deleted
                    # predictor rows cannot re-enter K/V in later layers.
                    torch.testing.assert_close(compact, full[:, keep], rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
