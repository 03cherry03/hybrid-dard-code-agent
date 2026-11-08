import unittest

from iterative_tuning import CPGRPO
from scripts.repair_target_experiments import (
    NATIVE_STEPS, campaign_metadata, repair_grid, select_dev_rows,
)
from scripts.tradeoff_experiments import shared_context


class RepairTargetExperimentTests(unittest.TestCase):
    def test_grid_has_complete_native_curve_for_every_dard_context(self):
        values = repair_grid()
        self.assertEqual(len(values), len({v["id"] for v in values}))
        dards = [v for v in values if v["decoder"] == "dard"]
        self.assertEqual(len(dards), 52)
        for dard in dards:
            native = [v for v in values if v["decoder"] == "standard"
                      and shared_context(v) == shared_context(dard)]
            self.assertEqual({v["standard_steps"] for v in native}, set(NATIVE_STEPS))

    def test_grid_contains_repaired_and_diagnostic_completion_policies(self):
        values = repair_grid()
        policies = {v["dard"]["completion_policy"] for v in values if v["decoder"] == "dard"}
        self.assertEqual(policies, {"resample_then_verify", "verified_only", "legacy_force"})
        self.assertEqual({v["tokens"] for v in values if v["decoder"] == "dard"}, {256, 512})
        self.assertEqual({v["prefix"] for v in values if v["decoder"] == "dard"},
                         {"none", "code_fence"})

    def test_campaign_metadata_freezes_protocol_and_fresh_seeds(self):
        metadata = campaign_metadata(CPGRPO)
        self.assertEqual(metadata["candidate_count"], 70)
        self.assertEqual(metadata["grading_workers"], 8)
        self.assertEqual(metadata["grading_queue_size"], 64)
        self.assertEqual(metadata["final_seed_ids"], [20261111, 20261112, 20261113])
        self.assertTrue(metadata["no_holdout_selection"])

    def test_dev_selection_excludes_legacy_quality_claims_and_incomplete_rows(self):
        def row(candidate_id, policy, valid=True, rate=.5, seconds=1.):
            return {"candidate": {"id":candidate_id, "decoder":"dard",
                                   "dard":{"completion_policy":policy}},
                    "candidate_id":candidate_id, "valid_for_selection":valid,
                    "missing_or_invalid_cases":0, "infrastructure_error_count":0,
                    "pass_rate":rate, "mean_generation_seconds":seconds,
                    "completion_qualified_pass_rate":rate}
        rows = [row("legacy", "legacy_force", rate=.9),
                row("incomplete", "resample_then_verify", valid=False),
                row("repair", "resample_then_verify", rate=.6)]
        selected = select_dev_rows(rows)
        self.assertEqual([r["candidate_id"] for r in selected], ["repair"])


if __name__ == "__main__":
    unittest.main()
