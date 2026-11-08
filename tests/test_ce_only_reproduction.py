import tempfile
import unittest
from pathlib import Path

from ce_only_reproduction import (EVALUATION, TRAINING, balance_training,
    reproduction_candidates)
from scripts.reproduce_ce_only import prepare_training_source, training_command


class CEOnlyContractTests(unittest.TestCase):
    def test_report_candidates_and_seeds_are_frozen(self):
        candidates = reproduction_candidates()
        self.assertEqual([c["id"] for c in candidates], [
            "ar-7f348bc3385bc1e4",
            "standard-5a4bd2d0c4972b81",
            "dard-84292bb36a6aa4c8",
        ])
        self.assertEqual(candidates[-1]["dard"], {
            "block_size": 128, "steps_per_block": 64,
            "tau_c": 0.4, "tau_u": 0.75, "temperature": 0.0,
            "top_p": 1.0, "transfer_limit": 8,
            "diagnostics": "summary", "max_step_records": 0,
            "compact_queries": True, "rectangular_attention": True,
            "completion_policy": "resample_then_verify",
            "completion_verification_steps": 2,
        })
        self.assertEqual(EVALUATION["repeat_seeds"], (20261111, 20261112, 20261113))
        self.assertEqual(TRAINING["max_steps"], 1200)
        self.assertEqual(TRAINING["lora_rank"], 32)

    def test_training_command_explicitly_freezes_ce_arguments(self):
        args = type("Args", (), {"python": Path("/opt/python"),
                                  "deadline_unix": 1234.0,
                                  "deadline_reserve_seconds": 900.0})()
        command = training_command(args, Path("train.jsonl"), Path("run/training"))
        self.assertIn("--min-mask-prob", command)
        self.assertEqual(command[command.index("--min-mask-prob") + 1], "0.15")
        self.assertEqual(command[command.index("--max-steps") + 1], "1200")
        self.assertIn("--balanced-languages", command)
        self.assertIn("gate_proj,up_proj,down_proj", ",".join(command))

    def test_training_source_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            snapshot = prepare_training_source(output)
            self.assertTrue((snapshot / "scripts" / "train_diffusion_lora.py").is_file())
            (snapshot / "diffusion_training.py").write_text("changed")
            with self.assertRaises(ValueError):
                prepare_training_source(output)

    def test_balanced_training_matches_report_hash_for_source_fixture(self):
        source = Path("/data/hybrid-dard/results/code-generation-iterative-20261107/data/training_examples.jsonl")
        if not source.is_file():
            self.skipTest("historical training source is not present")
        with tempfile.TemporaryDirectory() as temporary:
            receipt = balance_training(source, Path(temporary) / "balanced_training.jsonl")
            self.assertEqual(receipt["source_rows"], 1593)
            self.assertEqual(receipt["balanced_rows"], 3648)
            self.assertEqual(receipt["balanced_sha256"],
                             "725ec184925974e77dae0ac61e65099728deacc281326b958b23ec9235ee8271")


if __name__ == "__main__":
    unittest.main()
