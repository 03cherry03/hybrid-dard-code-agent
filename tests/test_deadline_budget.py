"""Offline tests for measured campaign planning and non-extending deadlines."""
import copy
import unittest

from scripts.deadline_budget import estimate_seconds, resolve_deadline, select_repeats


def fixture():
    command = ['python', 'main.py', '--tokens', '512', '--quant', 'bf16',
               '--temperature', '0', '--top-p', '1', '--repairs', '1',
               '--diagnostics', 'summary', '--warmup', '1', '--block-size', '128',
               '--dard-steps', '192', '--standard-steps', '512']
    workers, shards, targets = [], [], []
    for index in range(4):
        keys = [['fast', 'py', f'fast-{index}'], ['slow', 'rs', f'slow-{index}']]
        workers.append({'index': index, 'command': command.copy(), 'task_keys': keys,
                        'task_seeds': [[42], [42]]})
        rows, phases = [], []
        for phase_id, backend in enumerate(('ar', 'dard', 'standard')):
            phase_rows = []
            for dataset, language, task_id in keys:
                factor = {'ar': 1, 'dard': 2, 'standard': 4}[backend]
                agent = factor * (1 if dataset == 'fast' else 10)
                phase_rows.append({'dataset': dataset, 'language': language, 'task_id': task_id,
                                   'decoder': backend, 'repeat': 0, 'seed': 42, 'phase_id': phase_id,
                                   'case_completed': True, 'measurement_complete': True,
                                   'final_pass': dataset == 'fast', 'failure_kind': None,
                                   'agent_wall_s': agent, 'offline_grading_wall_s': 2,
                                   'total_wall_s': agent + 1})
            load, warmup = 2+index, 3
            phases.append({'phase_id': phase_id, 'decoder': backend, 'completed': True,
                           'model_load_s': load, 'warmup_s': warmup,
                           'phase_wall_s': load+warmup+sum(r['agent_wall_s']+3 for r in phase_rows)})
            rows.extend(phase_rows)
        shards.append({'rows': rows, 'phases': phases})
        counts = (1, 5) if index == 0 else (5, 1)
        target_keys = [[dataset, language, f'target-{index}-{dataset}-{task}']
                       for (dataset, language), count in zip((('fast', 'py'), ('slow', 'rs')), counts)
                       for task in range(count)]
        targets.append({'index': index, 'command': command.copy(), 'task_keys': target_keys,
                        'by_dataset_language': {'fast/py': counts[0], 'slow/rs': counts[1]}})
    return {'workers': workers, 'repeats': 1}, shards, targets


class DeadlineBudgetTests(unittest.TestCase):
    def test_critical_shard_uses_suite_counts_and_all_grading_load_warmup_costs(self):
        plan, shards, workers = fixture()
        result = estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)
        self.assertEqual(result['max_observed_case_seconds'],
                         {'ar/fast/py': 4., 'ar/slow/rs': 13.,
                          'dard/fast/py': 5., 'dard/slow/rs': 23.})
        self.assertEqual(result['max_load_warmup_seconds_per_phase'], {'ar': 8., 'dard': 8.})
        self.assertEqual(result['workers'][0]['base_by_backend_seconds'], {'ar': 77., 'dard': 128.})
        self.assertEqual(result['base_wall_seconds'], 205.)
        self.assertEqual(result['estimated_wall_seconds'], 307.5)
        self.assertFalse(result['estimate_is_guarantee'])
        self.assertEqual(result['calibration_cases'], 16)

    def test_slowest_observed_suite_case_is_used_instead_of_mean(self):
        plan, shards, workers = fixture()
        row = next(r for r in shards[1]['rows'] if r['decoder'] == 'dard' and r['dataset'] == 'slow')
        row['agent_wall_s'] += 100
        shards[1]['phases'][1]['phase_wall_s'] += 100
        result = estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)
        self.assertEqual(result['max_observed_case_seconds']['dard/slow/rs'], 123.)
        self.assertEqual(result['base_wall_seconds'], 705.)

    def test_selects_largest_complete_repeat_count_and_reports_infeasible(self):
        plan, shards, workers = fixture()
        for available, selected in [(1000, 3), (615, 2), (400, 1), (307.5, 1), (307.49, 0), (0, 0)]:
            with self.subTest(available=available):
                result = select_repeats(plan, shards, workers, ['ar', 'dard'], available)
                self.assertEqual(result['selected_repeats'], selected)
                self.assertEqual(result['fits_estimate'], bool(selected))
                self.assertFalse(result['estimate_is_guarantee'])
        self.assertEqual(select_repeats(plan, shards, workers, ['ar', 'dard'], 1000,
                                        max_repeats=1)['selected_repeats'], 1)

    def test_unmeasured_suite_is_not_replaced_by_another_suites_runtime(self):
        plan, shards, workers = fixture()
        workers[0]['task_keys'].append(['new', 'cpp', 'unseen'])
        workers[0]['by_dataset_language']['new/cpp'] = 1
        with self.assertRaisesRegex(ValueError, 'Unmeasured'):
            estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)

    def test_ungraded_incomplete_missing_and_duplicate_calibration_fail_closed(self):
        for mutation in ('ungraded', 'incomplete', 'failure', 'missing', 'duplicate', 'no_phase', 'phase_failed'):
            plan, shards, workers = fixture()
            if mutation == 'ungraded': shards[0]['rows'][0]['final_pass'] = None
            if mutation == 'incomplete': shards[0]['rows'][0]['measurement_complete'] = False
            if mutation == 'failure': shards[0]['rows'][0]['failure_kind'] = 'generation_error'
            if mutation == 'missing': shards[0]['rows'].pop(0)
            if mutation == 'duplicate': shards[0]['rows'].append(copy.deepcopy(shards[0]['rows'][0]))
            if mutation == 'no_phase': shards[0]['phases'].pop(0)
            if mutation == 'phase_failed': shards[0]['phases'][0]['completed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)

    def test_short_smoke_budget_cannot_estimate_full_generation(self):
        plan, shards, workers = fixture()
        for worker in plan['workers']:
            worker['command'][worker['command'].index('--tokens')+1] = '128'
        with self.assertRaisesRegex(ValueError, 'budget/settings'):
            estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)

    def test_backend_specific_options_do_not_require_optional_standard(self):
        plan, shards, workers = fixture()
        for worker in workers:
            worker['command'][worker['command'].index('--standard-steps')+1] = '256'
        self.assertGreater(estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)['estimated_wall_seconds'], 0)
        with self.assertRaisesRegex(ValueError, 'budget/settings'):
            estimate_seconds(plan, shards, workers, ['standard'], 1)

    def test_dard_threshold_or_parallel_commit_changes_require_new_calibration(self):
        for flag, old, new in [('--tau-c', '.5', '.4'), ('--tau-u', '.8', '.7'),
                               ('--transfer-limit', '0', '8')]:
            plan, shards, workers = fixture()
            for worker in [*plan['workers'], *workers]:
                worker['command'] += [flag, old]
            workers[0]['command'][-1] = new
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'budget/settings'):
                estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)
            # AR timing does not depend on the DARD setting.
            self.assertGreater(estimate_seconds(plan, shards, workers, ['ar'], 1)['estimated_wall_seconds'], 0)

    def test_inputs_are_not_mutated_and_invalid_timing_is_rejected(self):
        plan, shards, workers = fixture()
        previous = copy.deepcopy((plan, shards, workers))
        estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)
        self.assertEqual((plan, shards, workers), previous)
        for bad in (-1, float('nan'), float('inf'), None, True):
            with self.subTest(value=bad):
                shards[0]['rows'][0]['offline_grading_wall_s'] = bad
                with self.assertRaises(ValueError):
                    estimate_seconds(plan, shards, workers, ['ar', 'dard'], 1)

    def test_resume_preserves_original_absolute_deadline(self):
        original = resolve_deadline(now_unix=1000, hours=40)
        resumed = resolve_deadline(now_unix=1000+39*3600, hours=40, existing=original)
        self.assertEqual(resumed['deadline_unix'], original['deadline_unix'])
        self.assertEqual(resumed['started_unix'], 1000)
        self.assertEqual(resumed['remaining_seconds'], 3600)
        self.assertFalse(resumed['expired'])
        enlarged = resolve_deadline(now_unix=2000, hours=80, existing=original)
        self.assertEqual(enlarged['deadline_unix'], original['deadline_unix'])
        expired = resolve_deadline(now_unix=original['deadline_unix'], existing=original)
        self.assertTrue(expired['expired'])
        self.assertEqual(expired['remaining_seconds'], 0)

    def test_reduced_budget_uses_original_start_and_deadline_validation(self):
        original = resolve_deadline(now_unix=1000, hours=40)
        reduced = resolve_deadline(now_unix=2000, hours=1, existing=original)
        self.assertEqual(reduced['deadline_unix'], 4600)
        with self.assertRaises(ValueError):
            resolve_deadline(now_unix=2000, existing={'started_unix': 1000, 'deadline_unix': 900})
        plan, shards, workers = fixture()
        for repeats in (0, 4, 1.5, True):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                estimate_seconds(plan, shards, workers, ['ar'], repeats)


if __name__ == '__main__':
    unittest.main()
