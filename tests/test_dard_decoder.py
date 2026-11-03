"""CPU-only tests: never downloads models, never executes generated programs."""
import tempfile
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dard_decoder import (DARDConfig, DARDDecoder, M, C, U,
                          build_two_stream_visibility, adaptive_weights,
                          filtered_sample, verify_probabilities)
from datasets_code import Task, load_mbpp, load_humaneval, load_rust
from evaluation import _syntax_check, strip_code, category
from main import cli, _summary


class TestDARD(unittest.TestCase):
    def test_vis_ordering_and_no_self_leak(self):
        # Prefix=2; generation: C_high, C_low, M
        state = torch.tensor([C, C, M])
        conf = torch.tensor([.7, .2, 0.])
        visible = build_two_stream_visibility(2, state, conf)[0, 0]
        l = 5
        self.assertEqual(visible.shape, (10, 10))
        self.assertTrue(bool(visible[3, 2]))        # original C_low attends C_high
        self.assertFalse(bool(visible[2, 3]))       # original C_high cannot attend C_low
        # Verification target index 1: shadow query at l+p-1+1 = 7
        self.assertTrue(bool(visible[7, 2]))        # can attend higher-confidence C_high
        self.assertFalse(bool(visible[7, 3]))       # cannot attend own C_low
        # Verification target index 0: shadow row6, no lower C context
        self.assertFalse(bool(visible[6, 3]))
        # Masked token's shadow query must not attend any tentative original C.
        self.assertFalse(bool(visible[8, 2]))
        self.assertFalse(bool(visible[8, 3]))
        self.assertTrue(bool(visible.any(-1).all()))
        # Original queries never attend shadow keys.
        self.assertFalse(bool(visible[:l, l:].any()))

    def test_mixing_local_revocations(self):
        no_events = adaptive_weights(12, [], base=.9, impact=.65, decay=3, device="cpu")
        self.assertAlmostEqual(float(no_events[2]), .9)
        bad = adaptive_weights(12, [(5, -1)], base=.9, impact=.65, decay=3, device="cpu")
        good = adaptive_weights(12, [(5, +1)], base=.7, impact=.15, decay=3, device="cpu")
        self.assertLess(float(bad[5]), float(no_events[5]))
        self.assertGreater(float(good[5]), .7)
        self.assertTrue(bool((bad >= .05).all()))

    def test_sample_and_verify(self):
        l = torch.tensor([[0., 0., 9.]])
        ids, conf = filtered_sample(l, temperature=0, top_p=.95)
        self.assertEqual(ids.tolist(), [2])
        self.assertGreater(float(conf[0]), .99)
        prob = verify_probabilities(l, ids)
        self.assertGreater(float(prob[0]), .99)

    def test_full_decoder_with_fake_dream(self):
        class FakeDream:
            def __init__(self):
                self.calls = 0
            def __call__(self, x, attention_mask=None, position_ids=None, use_cache=False):
                self.calls += 1
                if isinstance(attention_mask, torch.Tensor):
                    self_test = attention_mask
                    assert self_test.shape[-2:] == (x.shape[1], x.shape[1])
                    assert position_ids.shape == x.shape
                logits = torch.zeros((1, x.shape[1], 24))
                logits[:, :, 7] = 12
                return SimpleNamespace(logits=logits)
        fake = FakeDream()
        cfg = DARDConfig(block_size=4, steps_per_block=12, tau_c=.1, tau_u=.3,
                         transfer_limit=2, temperature=0, top_p=1)
        result, stats = DARDDecoder(fake, 23, cfg).generate(torch.tensor([[1, 2, 3]]), 7)
        self.assertEqual(result.shape, (1, 7))
        self.assertEqual(result.tolist(), [[7]*7])
        self.assertEqual(stats["forced_tokens"], 0)
        self.assertEqual(stats["promoted_tokens"], 7)
        self.assertEqual(len(stats["blocks"]), 2)
        self.assertGreater(stats["model_forward_passes"], 0)


    def test_fallback_on_unsupported_4d_attention(self):
        class FakeNoCustomMask:
            def __call__(self, x, attention_mask=None, position_ids=None, use_cache=False):
                if isinstance(attention_mask, torch.Tensor):
                    raise TypeError("4D attention mask is not supported")
                out = torch.zeros((1, x.shape[1], 16))
                out[:, :, 7] = 10
                return SimpleNamespace(logits=out)
        d = DARDDecoder(FakeNoCustomMask(), 15,
                        DARDConfig(block_size=3, steps_per_block=10,
                                   transfer_limit=2, temperature=0, top_p=1))
        tokens, stats = d.generate(torch.tensor([[1, 2]]), 3)
        self.assertEqual(tokens.tolist(), [[7, 7, 7]])
        self.assertGreater(stats["attention_fallback_steps"], 0)
        self.assertGreater(stats["model_forward_passes"], stats["steps"])

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            DARDConfig(tau_c=.6, tau_u=.2).validate()

    def test_fences_and_static(self):
        self.assertEqual(strip_code("```python\nprint('hi')\n```"), "print('hi')\n")
        task = Task("humaneval", "py", "HumanEval/0", 0, "", "")
        self.assertEqual(_syntax_check(task, "def f(:\n")["status"], "syntax_error")
        self.assertIsNone(_syntax_check(task, "def f(): return 1\n"))
        rust = Task("multiple_rs", "rs", "HumanEval_0", 0, "", "")
        self.assertEqual(_syntax_check(rust, "fn main() {}") ["status"], "output_format_error")
        self.assertEqual(category("test_failed", "AssertionError"), "assertion_failure")

    def test_no_runner_with_repairs_fails(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli(["--runner", "none"])
        args = cli(["--runner", "none", "--repairs", "0", "--limit", "2"])
        self.assertEqual(args.limit, 2)

    def test_summary_no_false_failure_for_not_tested(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"decoder": "dard", "dataset": "humaneval", "language": "py", "task_id": "HumanEval/0",
                     "first_pass": None, "final_pass": None, "generation_wall_s": 1.0,
                     "test_wall_s": 0.0, "model_forward_passes": 5, "forced_tokens": 0,
                     "final_status": "not_tested"}]
            info = _summary(rows, {"runner":"none"}, Path(tmp))
            group = info["groups"]["dard/humaneval/py"]
            self.assertIsNone(group["final_pass_rate"])
            self.assertEqual(group["not_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
