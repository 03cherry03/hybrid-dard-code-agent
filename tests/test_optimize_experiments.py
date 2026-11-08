"""Outcome-independent splits, honest selection, pinned artifacts and failure boundaries."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import optimize_experiments as optimize
from scripts.focused_selection import build_selection
from scripts.all_experiments import write_json
from scripts.verify_prepared_models import _hash_file
from tests.test_focused_selection import tasks_fixture


def rows(passed, seconds=1., backend='dard', n=64):
    return [{'dataset': 'fixture', 'language': optimize.LANGUAGES[i % 4], 'task_id': str(i),
             'source_id': 'fixture/'+str(i//4), 'repeat': 0, 'seed': 42, 'decoder': backend,
             'final_pass': i < passed, 'generation_wall_s': seconds, 'case_completed': True,
             'measurement_complete': True, 'failure_kind': None, 'visible_output_tokens': 16 if i < n//2 else 100,
             'final_status': 'passed' if i < passed else 'test_failed'} for i in range(n)]


def candidate(passed, seconds, *, model='fixture', n=64):
    return {'model': model, 'revision': 'pinned', 'dard': rows(passed, seconds, n=n),
            'standard': rows(passed, 3., 'standard', n=n), 'quality': rows(passed, 6., 'standard', n=n)}


class SelectionTests(unittest.TestCase):
    def test_partition_is_disjoint_at_source_and_language_level(self):
        catalog = build_selection(tasks_fixture())
        dev, final = optimize.partition(catalog, 2)
        self.assertEqual(dev['source_count'], 4)
        self.assertEqual(final['source_count'], 10)
        for family in optimize.FAMILIES:
            self.assertTrue(set(dev['source_ids_by_family'][family]).isdisjoint(final['source_ids_by_family'][family]))
            old = {x['source_id'] for x in catalog['calibration_sources']}
            self.assertFalse(old.intersection(dev['source_ids_by_family'][family]))
            self.assertFalse(old.intersection(final['source_ids_by_family'][family]))
        self.assertTrue(set(map(tuple, dev['task_keys'])).isdisjoint(map(tuple, final['task_keys'])))
        self.assertEqual(set(key[1] for key in dev['task_keys']), set(optimize.LANGUAGES))

    def test_partition_choice_ignores_task_order_and_outcome_fields(self):
        catalog = build_selection(tasks_fixture())
        dev, _ = optimize.partition(catalog, 2)
        changed = copy.deepcopy(catalog)
        changed['evaluation_sources'].reverse()
        for source in changed['evaluation_sources']:
            source['passed'] = False
            source['generation_wall_s'] = 100000
        self.assertEqual(optimize.partition(changed, 2)[0], dev)

    def test_rejects_missing_language_and_empty_final_family(self):
        catalog = build_selection(tasks_fixture())
        with self.assertRaises(ValueError):
            optimize.partition(catalog, 7)
        for source in catalog['evaluation_sources']:
            source['task_keys'].pop()
        with self.assertRaisesRegex(ValueError, 'four languages'):
            optimize.partition(catalog, 2)

    def test_fastest_candidate_must_first_meet_accuracy_margin(self):
        decision = optimize.choose(rows(60, 2., 'ar'), {
            'too_inaccurate': candidate(57, .1), 'fast': candidate(59, 1.), 'accurate_slow': candidate(61, 3.)})
        self.assertEqual(decision['selected'], 'fast')
        self.assertTrue(decision['goal_screen_passed'])
        self.assertFalse(decision['screening_is_statistical_noninferiority'])

    def test_goal_unmet_fallback_prioritizes_accuracy_over_speed(self):
        decision = optimize.choose(rows(60, 1., 'ar'), {
            'fast_weak': candidate(30, .1), 'slower_better': candidate(50, 2.)})
        self.assertEqual(decision['selected'], 'slower_better')
        self.assertFalse(decision['goal_screen_passed'])
        self.assertIn('goal_unmet', decision['selection_reason'])

    def test_legacy_winner_retains_both_legacy_execution_flags(self):
        legacy = candidate(60, 1.5)
        legacy.update(compact_queries=False, rectangular_attention=False)
        decision = optimize.choose(rows(60, 1., 'ar'), {'optimized': candidate(50, .3), 'legacy_instruct': legacy})
        self.assertEqual(decision['selected'], 'legacy_instruct')
        self.assertFalse(decision['compact_queries'])
        self.assertFalse(decision['rectangular_attention'])

    def test_equal_latency_does_not_claim_faster(self):
        decision = optimize.choose(rows(60, 1., 'ar'), {'same': candidate(60, 1.)})
        self.assertFalse(decision['goal_screen_passed'])

    def test_paired_identity_mismatch_or_duplicate_is_rejected(self):
        reference = rows(60, 1., 'ar')
        value = candidate(60, .5)
        value['dard'][0]['task_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identical unique'):
            optimize.choose(reference, {'bad': value})
        value = candidate(60, .5)
        value['standard'][0] = value['standard'][1]
        with self.assertRaises(ValueError):
            optimize.choose(reference, {'bad': value})

    def test_failure_times_are_included_and_regressions_counted(self):
        value = candidate(40, 1.)
        value['dard'][-1]['generation_wall_s'] = 65.
        decision = optimize.choose(rows(60, 1., 'ar'), {'candidate': value})
        report = decision['candidates'][0]
        self.assertEqual(report['dard']['mean_generation_seconds'], 2.)
        self.assertEqual(report['paired_vs_ar']['regression'], 20)
        self.assertEqual(sum(x['cases'] for x in report['dard']['by_language'].values()), 64)

    def test_final_length_buckets_use_same_ar_task_cohort_for_all_methods(self):
        data = rows(60, 1., 'ar') + rows(59, .5, 'dard') + rows(40, 2., 'standard')
        for row in data:
            if row['decoder'] != 'ar':
                row['visible_output_tokens'] = 400
        report = optimize.final_result(data, {})
        self.assertTrue(report['observed_full_cohort_criteria_met'])
        self.assertFalse(report['statistical_noninferiority_established'])
        self.assertEqual(set(report['length_buckets']), {'0_to_64', '65_to_128'})
        self.assertTrue(all(report['length_buckets']['0_to_64'][b]['cases'] == 32 for b in ('ar', 'dard', 'standard')))
        self.assertFalse(report['accuracy_margin_supported_by_source_bootstrap'])

    def test_source_cluster_support_requires_controlled_complete_cohort(self):
        data = rows(60, 1., 'ar') + rows(59, .5) + rows(40, 2., 'standard')
        comparisons = {'dard_vs_ar/all': {'complete_known_controlled_cohort': True,
            'pass_at_1_delta_source_cluster_bootstrap_95ci': [-.02, .01]}}
        self.assertTrue(optimize.final_result(data, {'comparisons': comparisons})['accuracy_margin_supported_by_source_bootstrap'])
        comparisons['dard_vs_ar/all']['complete_known_controlled_cohort'] = False
        self.assertFalse(optimize.final_result(data, {'comparisons': comparisons})['accuracy_margin_supported_by_source_bootstrap'])


class ReceiptAndLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_candidate_receipt_accepts_snapshot_symlinks_and_detects_mutation(self):
        import hashlib
        blob = self.root/'models--candidate'/'blobs'
        snapshot = self.root/'models--candidate'/'snapshots'/optimize.CPGRPO_REVISION
        blob.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        payload = b'nonempty checkpoint fixture'
        content = blob/hashlib.sha256(payload).hexdigest()
        content.write_bytes(payload)
        file = snapshot/'weights.safetensors'
        file.symlink_to(content)
        receipt = self.root/'receipt.json'
        write_json(receipt, {'status': 'verified', 'model': optimize.CPGRPO,
            'revision': optimize.CPGRPO_REVISION, 'snapshot': str(snapshot), 'files': [_hash_file(file)]})
        args = Namespace(candidate_model_receipt=receipt, cpgrpo_revision=optimize.CPGRPO_REVISION, output=self.root)
        result = optimize.verify_candidate_receipt(args)
        self.assertEqual(result['files_verified'], 1)
        write_json(self.root/'candidate_model_decision.json', result)
        content.write_bytes(b'changed checkpoint fixture')
        with self.assertRaisesRegex(ValueError, 'changed after'):
            optimize.verify_candidate_receipt(args)

    def test_incomplete_case_cannot_enter_selection(self):
        output = self.root/'shard-0'
        output.mkdir()
        plan = {'backends': ['dard'], 'workers': [{'output': str(output), 'task_keys': [['fixture', 'py', '0']]}]}
        row = rows(1, n=1)[0]
        row['measurement_complete'] = False
        (output/'case_results.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            optimize.read_rows(self.root, plan)

    def test_wrong_answer_is_valid_measurement(self):
        output = self.root/'shard-0'
        output.mkdir()
        plan = {'backends': ['dard'], 'workers': [{'output': str(output), 'task_keys': [['fixture', 'py', '0']]}]}
        row = rows(0, n=1)[0]
        (output/'case_results.jsonl').write_text(json.dumps(row)+'\n')
        self.assertFalse(optimize.read_rows(self.root, plan)[0]['final_pass'])

    def test_post_stop_records_and_emails_killed_running_process_once(self):
        args = Namespace(output=self.root)
        write_json(self.root/'optimization_state.json', {'status': 'running', 'invocation_id': 'fixture', 'failure_count': 0})
        with patch.object(optimize.pipeline, 'send_failure') as send, patch('experiment_notifications.flush_pending') as flush:
            self.assertEqual(optimize.post_stop(args), 0)
            self.assertEqual(optimize.post_stop(args), 0)
        send.assert_called_once()
        self.assertEqual(json.loads((self.root/'optimization_state.json').read_text())['failure_count'], 1)
        self.assertEqual(flush.call_count, 2)

    def test_post_stop_reports_startup_kill_before_first_state_write(self):
        args = Namespace(output=self.root)
        with patch.dict('os.environ', {'SERVICE_RESULT': 'oom-kill'}), \
                patch.object(optimize.pipeline, 'send_failure') as send, \
                patch('experiment_notifications.flush_pending'):
            self.assertEqual(optimize.post_stop(args), 0)
            self.assertEqual(optimize.post_stop(args), 0)
        send.assert_called_once()
        state = json.loads((self.root/'optimization_state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['failure_count'], 1)

    def test_restart_cap_runs_only_notification_flush(self):
        args = Namespace(output=self.root, deadline_unix=100.)
        state = {'invocation_id': 'fixture', 'failure_count': 3}
        with patch.object(optimize.time, 'time', return_value=10.), patch('experiment_notifications.flush_pending',
                return_value={'pending_remaining': 0, 'remaining_scan_complete': True}) as flush:
            self.assertEqual(optimize.notification_only(args, state), 0)
        flush.assert_called_once()
        self.assertEqual(state['status'], 'blocked')

    def test_startup_failure_counter_caps_before_repeating_prepare(self):
        argv = ['--output', str(self.root), '--baseline-output', str(self.root/'baseline'),
                '--deadline-unix', '1794038400', '--execute']
        with patch.object(optimize, 'prepare', side_effect=ValueError('changed frozen source')) as prepare, \
                patch.object(optimize.pipeline, 'send_failure') as send, \
                patch.object(optimize, 'notification_only', return_value=0) as only:
            for _ in range(3):
                with self.assertRaisesRegex(ValueError, 'changed frozen source'):
                    optimize.main(argv)
            self.assertEqual(optimize.main(argv), 0)
        self.assertEqual(prepare.call_count, 3)
        self.assertEqual(send.call_count, 3)
        only.assert_called_once()
        self.assertEqual(json.loads((self.root/'optimization_state.json').read_text())['failure_count'], 3)

    def test_optional_timeout_excludes_diagnostic_and_preserves_completed_work(self):
        args = Namespace(output=self.root)
        state = {'invocation_id': 'fixture'}
        with patch.object(optimize.time, 'time', return_value=100.), \
                patch.object(optimize, 'run_campaign', side_effect=TimeoutError('budget')) as run, \
                patch.object(optimize.pipeline, 'send_failure') as send:
            self.assertIsNone(optimize.optional_campaign(args, state, 'dev/quality', deadline=1000., estimated_seconds=10.))
            self.assertIsNone(optimize.optional_campaign(args, state, 'dev/quality', deadline=1000., estimated_seconds=10.))
        run.assert_called_once()
        send.assert_called_once()
        self.assertEqual(json.loads((self.root/'dev/quality/optional_stage.json').read_text())['status'], 'timed_out')

    def test_expired_budget_fails_before_entering_operation(self):
        with patch.object(optimize.time, 'time', return_value=100.):
            with self.assertRaises(TimeoutError):
                with optimize.bounded(99.):
                    self.fail('expired work started')


if __name__ == '__main__':
    unittest.main()
