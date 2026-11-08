"""Reporting must expose missing pairs, diagnostic fallbacks, and repeat dependence."""
import tempfile
import unittest
from pathlib import Path
from reporting import summarize


def case(decoder, task='0', repeat=0, **overrides):
    result = dict(decoder=decoder, dataset='humaneval', family='humaneval', language='py',
                  task_id=task, source_id=task, repeat=repeat, seed=42 + repeat,
                  first_pass=True, final_pass=True, case_completed=True,
                  generation_wall_s=2 if decoder == 'ar' else 1,
                  first_generation_wall_s=2 if decoder == 'ar' else 1,
                  test_wall_s=1, total_wall_s=3 if decoder == 'ar' else 2,
                  final_status='passed', model_forward_passes=1, forced_tokens=0,
                  messages_sha256='same', warmup_done=True, resource_check_available=True,
                  resource_contended=False, task_sha256='same-task', model_identity_verified=True)
    result.update(overrides)
    return result


class TestReporting(unittest.TestCase):
    def report(self, rows, expected):
        with tempfile.TemporaryDirectory() as path:
            return summarize(rows, {'expected_groups': {'ar/humaneval/py': expected, 'dard/humaneval/py': expected},
                                    'bootstrap_draws': 40}, Path(path))

    def test_incomplete_pairs_never_certify_controlled_comparison(self):
        report = self.report([case('ar'), case('dard')], expected=2)
        value = report['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertEqual(value['paired_evaluated_cases'], 1)
        self.assertFalse(value['eligible_for_controlled_comparison'])
        self.assertIn('incomplete_paired_evaluation', value['comparison_issues'])

    def test_fallback_prompt_mismatch_and_resource_contention_are_explicit(self):
        report = self.report([case('ar'), case('dard', attention_fallback_steps=1,
                                                messages_sha256='different', resource_contended=True)], 1)
        issues = report['paired_comparisons']['dard_vs_ar/humaneval/py']['comparison_issues']
        self.assertIn('dard_diagnostic_attention_fallback', issues)
        self.assertIn('different_shared_prompt_content', issues)
        self.assertIn('other_compute_process_on_selected_gpu', issues)

    def test_detailed_instrumentation_and_missing_measurements_are_explicit(self):
        rows = [case('ar', diagnostics_mode='steps'), case('dard', measurement_complete=False)]
        report = self.report(rows, 1)
        issues = report['paired_comparisons']['dard_vs_ar/humaneval/py']['comparison_issues']
        self.assertIn('diagnostic_instrumentation_affects_timing', issues)
        self.assertIn('incomplete_generation_measurements', issues)
        self.assertEqual(report['groups']['dard/humaneval/py']['incomplete_measurement_cases'], 1)

    def test_repeated_trials_are_one_problem_cluster(self):
        rows = [case(d, repeat=r) for d in ('ar', 'dard') for r in range(3)]
        value = self.report(rows, 3)['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertEqual(value['unique_problem_clusters'], 1)
        self.assertEqual(value['first_pass_delta_cluster_bootstrap_95ci'], [None, None])

    def test_missing_fingerprints_do_not_certify_fairness(self):
        rows = [case('ar', messages_sha256=None),
                case('dard', messages_sha256=None, task_sha256=None, model_identity_verified=False)]
        value = self.report(rows, 1)['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertFalse(value['eligible_for_controlled_comparison'])
        self.assertIn('missing_prompt_fingerprint', value['comparison_issues'])
        self.assertIn('missing_or_different_task_fingerprint', value['comparison_issues'])
        self.assertIn('unresolved_model_or_tokenizer_identity', value['comparison_issues'])

    def test_failed_solutions_are_included_in_all_case_speed_and_quality(self):
        rows = [case('ar'), case('dard', first_pass=False, final_pass=False, final_status='test_failed')]
        value = self.report(rows, 1)['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertEqual(value['first_pass_rate_delta_candidate_minus_ar'], -1)
        self.assertFalse(value['observed_first_pass_within_tolerance'])
        self.assertEqual(value['initial_generation_speedup_ar_over_candidate'], 2)
        self.assertEqual(value['jointly_successful_cases'], 0)
        self.assertIsNone(value['jointly_successful_total_time_speedup'])

    def test_empty_expected_groups_remain_visible(self):
        report = self.report([], 4)
        self.assertEqual(report['groups']['dard/humaneval/py']['expected_cases'], 4)
        self.assertEqual(report['groups']['dard/humaneval/py']['evaluated'], 0)
        self.assertFalse(report['paired_comparisons']['dard_vs_ar/humaneval/py']['eligible_for_controlled_comparison'])

    def test_both_planned_counts_and_manifest_are_required(self):
        rows = [case('ar'), case('dard')]
        with tempfile.TemporaryDirectory() as path:
            report = summarize(rows, {'expected_groups': {'ar/humaneval/py': 2, 'dard/humaneval/py': 1}}, Path(path))
        value = report['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertIn('incomplete_paired_evaluation', value['comparison_issues'])
        self.assertIn('missing_experiment_manifest', value['comparison_issues'])


if __name__ == '__main__':
    unittest.main()
