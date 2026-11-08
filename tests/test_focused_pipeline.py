"""Focused target pipeline tests that do not require CUDA or model weights."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_diffusion_lora import LanguageBalancedDistributedSampler


class FocusedPipelineTests(unittest.TestCase):
    def test_language_sampler_balances_each_epoch(self):
        rows = ([{"language": "py"}] * 3 + [{"language": "js"}]
                + [{"language": "cpp"}] * 2 + [{"language": "rs"}])
        sampler = LanguageBalancedDistributedSampler(rows, num_replicas=1, rank=0, seed=11)
        sampled = [rows[index]["language"] for index in sampler]
        self.assertEqual(Counter(sampled), {"py": 3, "js": 3, "cpp": 3, "rs": 3})
        sampler.set_epoch(1)
        self.assertEqual(len(list(sampler)), 12)


if __name__ == "__main__":
    unittest.main()
