"""Focused campaign integration with real planning/estimation and mocked GPU work."""
from argparse import Namespace
from contextlib import ExitStack
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets_code import Task
from scripts import focused_experiments as focused
from scripts import four_gpu
from scripts.all_experiments import read_json, write_json
from scripts.deadline_budget import estimate_seconds
from scripts.focused_selection import build_selection, selection_manifest
from tests.test_focused_selection import tasks_fixture


class FocusedExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = Namespace(output=self.root/'functions', data_root=self.root/'data',
                              gpus='0,1,2,3', deadline_unix=100000.)
        self.tasks = [Task(t.dataset, t.language, t.task_id, int(t.source_id.split('/')[-1]),
                           t.prompt, t.test_source, family=t.family, source_id=t.source_id)
                      for t in tasks_fixture()]
        self.gpus = [{'uuid': f'GPU-{i}', 'name': 'fixture'} for i in range(4)]
        self.calls = []
        self.plans = []

    def _simulate_campaign(self, argv):
        self.calls.append(list(argv))
        args = four_gpu.parser().parse_args(argv)
        plan, selected = four_gpu.make_plan(args, self.gpus)
        four_gpu.save_plan(args.output, plan)
        self.plans.append(plan)
        for worker in plan['workers']:
            path = Path(worker['output'])
            rows, phases = [], []
            for phase_id, backend in enumerate(focused.BACKENDS):
                phase_rows = []
                for dataset, language, task_id in worker['task_keys']:
                    cost = {'ar': 1., 'dard': 3., 'standard': 8.}[backend]
                    cost *= {'py': 1, 'rs': 2, 'js': 1.2, 'cpp': 1.5}[language]
                    phase_rows.append({'dataset': dataset, 'language': language, 'task_id': task_id,
                        'decoder': backend, 'repeat': 0, 'seed': 42, 'phase_id': phase_id,
                        'case_completed': True, 'measurement_complete': True, 'final_pass': False,
                        'failure_kind': None, 'agent_wall_s': cost, 'offline_grading_wall_s': 1.,
                        'total_wall_s': cost+.2})
                phases.append({'phase_id': phase_id, 'decoder': backend, 'completed': True,
                               'model_load_s': 3., 'warmup_s': 5.,
                               'phase_wall_s': 8.+sum(r['agent_wall_s']+1.2 for r in phase_rows)})
                rows.extend(phase_rows)
            write_json(path/'config.json', {'phases': phases})
            (path/'case_results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        write_json(args.output/'execution_status.json', {'exit_codes': {str(i): 0 for i in range(4)}})
        write_json(args.output/'campaign_summary.json', {
            'coverage_complete': True, 'backends': focused.BACKENDS,
            'groups': {f'{b}/all': {'complete_evaluated_cohort': True} for b in focused.BACKENDS}})
        return 0

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(focused, 'load_tasks', return_value=self.tasks))
        stack.enter_context(patch('datasets_code.load_tasks', return_value=self.tasks))
        stack.enter_context(patch.object(four_gpu, 'resolve_gpus', return_value=self.gpus))
        stack.enter_context(patch.object(four_gpu, 'main', side_effect=self._simulate_campaign))
        stack.enter_context(patch.object(focused.time, 'time', return_value=1000.))
        return stack

    def test_real_planning_and_estimation_keep_all_three_singlepass_methods_and_languages(self):
        with self.patches(), patch.object(four_gpu, 'audit') as audit:
            self.assertEqual(focused.run(self.args), 0)
        audit.assert_not_called()  # Normal campaign main already owns the analysis.
        self.assertEqual(len(self.calls), 2)
        calibration, evaluation = self.plans
        self.assertEqual(calibration['task_count'], 16)
        self.assertEqual(evaluation['task_count'], 56)
        self.assertEqual(evaluation['expected_cases'], 168)
        for plan in self.plans:
            self.assertEqual(plan['backends'], ['ar', 'dard', 'standard'])
            self.assertEqual(plan['repeats'], 1)
            self.assertEqual({key[1] for w in plan['workers'] for key in w['task_keys']}, {'py','rs','js','cpp'})
            for worker in plan['workers']:
                command = worker['command']
                for flag, value in [('--repairs', '0'), ('--tokens', '512'), ('--standard-steps', '64'),
                                    ('--dard-steps', '64'), ('--block-size', '512'),
                                    ('--tau-c', '0.5'), ('--tau-u', '0.8'), ('--transfer-limit', '0')]:
                    self.assertEqual(command[command.index(flag)+1], value)
        decision = read_json(self.args.output/'budget_decision.json')
        self.assertEqual(decision['status'], 'completed')
        self.assertEqual(decision['selected_source_count'], 14)
        self.assertEqual(decision['expected_cases'], 168)
        self.assertEqual(decision['estimate']['calibration_cases'], 48)
        self.assertTrue(decision['all_backends_required'])
        self.assertTrue(decision['generation_budgets_unchanged'])
        self.assertEqual(decision['generation_settings'], focused.DEFAULT_GENERATION_SETTINGS)
        self.assertEqual(read_json(self.root/'generation_settings.json'), decision['generation_settings'])
        path = self.root/'evaluation_selection.json'
        self.assertEqual(decision['selection_manifest_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        selected = read_json(path)
        calibration_selected = read_json(self.root/'calibration_selection.json')
        for family in ('humaneval', 'mbpp'):
            self.assertTrue(set(selected['source_ids_by_family'][family]).isdisjoint(
                            calibration_selected['source_ids_by_family'][family]))
        # Resolve estimated candidate shards exactly as the launched coordinator.
        catalog = build_selection(self.tasks)
        candidate = focused.candidate_workers(evaluation, self.tasks, selection_manifest(catalog))
        self.assertEqual([w['task_keys'] for w in candidate], [w['task_keys'] for w in evaluation['workers']])

    def test_choose_cohort_uses_actual_conservative_estimate_api(self):
        with self.patches():
            focused.run(self.args)
            calibration, shards = focused.calibration_records(self.root/'calibration')
            catalog = build_selection(self.tasks)
            base = self.plans[-1]
            estimates = {}
            for count in range(2, 15):
                workers = focused.candidate_workers(base, self.tasks, selection_manifest(catalog, source_limit=count))
                estimates[count] = estimate_seconds(calibration, shards, workers, focused.BACKENDS, 1)['estimated_wall_seconds']
            available = estimates[8]
            selected, choice = focused.choose_cohort(catalog, self.tasks, base, calibration, shards, available)
        expected = max(c for c, seconds in estimates.items() if seconds <= available)
        self.assertEqual(choice['selected_source_count'], expected)
        self.assertEqual(selected['source_count'], expected)
        self.assertEqual(choice['estimate']['repeats'], 1)
        self.assertEqual(choice['estimate']['backends'], focused.BACKENDS)
        self.assertTrue(choice['fits_estimate'])

    def test_largest_fitting_prefix_does_not_assume_runtime_monotonicity(self):
        with self.patches():
            focused.run(self.args)
        catalog = build_selection(self.tasks)
        checked = []
        def estimate(_plan, _shards, workers, backends, repeats):
            count = sum(w['task_count'] for w in workers)//4
            checked.append(count)
            self.assertEqual(backends, focused.BACKENDS)
            self.assertEqual(repeats, 1)
            return {'estimated_wall_seconds': 20 if count == 13 else 100}
        with patch.object(focused, 'estimate_seconds', side_effect=estimate):
            manifest, choice = focused.choose_cohort(catalog, self.tasks, self.plans[-1], {}, [], 30)
        self.assertEqual(checked, [14, 13])
        self.assertEqual(manifest['source_count'], 13)
        self.assertTrue(choice['fits_estimate'])

    def test_even_infeasible_smallest_cohort_retains_every_family_language_and_method(self):
        with self.patches():
            focused.run(self.args)
        catalog = build_selection(self.tasks)
        with patch.object(focused, 'estimate_seconds', return_value={'estimated_wall_seconds': 100}):
            manifest, choice = focused.choose_cohort(catalog, self.tasks, self.plans[-1], {}, [], 0)
        self.assertEqual(manifest['source_count'], 2)
        self.assertEqual(manifest['task_count'], 8)
        self.assertEqual(choice['expected_cases'], 24)
        self.assertFalse(choice['fits_estimate'])
        self.assertTrue(choice['deadline_at_risk'])
        self.assertTrue(all(manifest['source_ids_by_family'].values()))
        self.assertEqual({k[1] for k in manifest['task_keys']}, {'py','rs','js','cpp'})

    def test_resume_retains_selection_deadline_and_skips_calibration(self):
        with self.patches():
            focused.run(self.args)
            original = (self.root/'evaluation_selection.json').read_bytes()
            with patch.object(focused, 'choose_cohort', side_effect=AssertionError('must retain choice')):
                self.assertEqual(focused.run(self.args), 0)
            self.assertEqual(len(self.calls), 3)
            changed = copy.copy(self.args)
            changed.deadline_unix += 1
            with self.assertRaisesRegex(ValueError, 'retain its original deadline'):
                focused.run(changed)
        self.assertEqual((self.root/'evaluation_selection.json').read_bytes(), original)
        self.assertEqual(self.plans[-1]['workers'], self.plans[-2]['workers'])

    def test_modified_manifest_blocks_resume_before_model_work(self):
        with self.patches():
            focused.run(self.args)
            path = self.root/'evaluation_selection.json'
            changed = read_json(path)
            changed['task_keys'].pop()
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'Frozen experiment selection/settings changed'):
                focused.run(self.args)
        self.assertEqual(len(self.calls), 2)

    def test_custom_parallel_settings_are_shared_and_frozen_before_any_model_work(self):
        self.args.standard_steps = 128
        self.args.dard_steps = 96
        self.args.block_size = 64
        self.args.tau_c = .4
        self.args.tau_u = .7
        self.args.transfer_limit = 0
        with self.patches():
            focused.run(self.args)
            for plan in self.plans:
                for worker in plan['workers']:
                    command = worker['command']
                    for flag, value in [('--standard-steps', '128'), ('--dard-steps', '96'),
                                        ('--block-size', '64'), ('--tau-c', '0.4'),
                                        ('--tau-u', '0.7'), ('--transfer-limit', '0')]:
                        self.assertEqual(command[command.index(flag)+1], value)
            saved = (self.root/'generation_settings.json').read_bytes()
            for name, value in [('standard_steps', 64), ('tau_c', .5), ('transfer_limit', 4)]:
                changed = copy.copy(self.args)
                setattr(changed, name, value)
                with self.subTest(setting=name), self.assertRaisesRegex(ValueError, 'Frozen experiment'):
                    focused.run(changed)
                self.assertEqual(len(self.calls), 2)
                self.assertEqual((self.root/'generation_settings.json').read_bytes(), saved)

    def test_execution_exception_is_not_hidden_by_partial_audit_failure(self):
        with self.patches():
            focused.run(self.args)
            with patch.object(four_gpu, 'main', side_effect=RuntimeError('generation failed')), \
                    patch.object(four_gpu, 'audit', side_effect=ValueError('report failed')) as audit, \
                    self.assertRaisesRegex(RuntimeError, 'generation failed'):
                focused.run(self.args)
        audit.assert_called_once()
        self.assertEqual(read_json(self.args.output/'budget_decision.json')['status'], 'failed')
        self.assertEqual(read_json(self.args.output/'partial_report_error.json')['error_type'], 'ValueError')

    def test_expired_deadline_launches_neither_calibration_nor_evaluation(self):
        self.args.deadline_unix = 2000.
        with self.patches(), self.assertRaises(TimeoutError):
            focused.run(self.args)
        self.assertEqual(self.calls, [])

    def test_analysis_requires_known_grades_for_all_three_backends(self):
        with self.patches():
            focused.run(self.args)
        arguments = ['functions', '--output', str(self.args.output), '--analyze']
        with patch('campaign_reporting.write_goal_report'):
            self.assertEqual(focused.main(arguments), 0)
            path = self.args.output/'campaign_summary.json'
            report = read_json(path)
            report['groups']['standard/all']['complete_evaluated_cohort'] = False
            write_json(path, report)
            self.assertEqual(focused.main(arguments), 1)
            report['groups']['standard/all']['complete_evaluated_cohort'] = True
            report['backends'] = ['ar', 'dard']
            write_json(path, report)
            self.assertEqual(focused.main(arguments), 1)

    def test_calibration_nonzero_has_durable_failed_stage_without_starting_evaluation(self):
        with self.patches(), patch.object(four_gpu, 'main', return_value=7) as launch:
            self.assertEqual(focused.run(self.args), 7)
        launch.assert_called_once()
        state = read_json(self.args.output/'focused_state.json')
        failure = read_json(self.args.output/'focused_failure.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['stage'], 'calibration')
        self.assertEqual(state['exit_code'], 7)
        self.assertEqual(failure['stage'], 'calibration')
        self.assertEqual(failure['error_type'], 'CampaignExitFailure')
        self.assertFalse((self.args.output/'budget_decision.json').exists())
        self.assertEqual(len((self.args.output/'focused_failures.jsonl').read_text().splitlines()), 1)

    def test_setup_and_calibration_exceptions_record_exact_phase_and_local_traceback(self):
        for phase in ('load_prepared_tasks', 'calibration', 'validate_calibration', 'estimate_evaluation_cohort'):
            with self.subTest(phase=phase):
                args = copy.copy(self.args)
                args.output = self.root/phase/'functions'
                with self.patches() as stack:
                    if phase == 'load_prepared_tasks':
                        stack.enter_context(patch.object(focused, 'load_tasks', side_effect=ValueError('bad prepared data')))
                    elif phase == 'calibration':
                        stack.enter_context(patch.object(four_gpu, 'main', side_effect=ValueError('calibration launch failed')))
                    elif phase == 'validate_calibration':
                        stack.enter_context(patch.object(focused, 'calibration_records', side_effect=ValueError('bad calibration rows')))
                    else:
                        stack.enter_context(patch.object(focused, 'choose_cohort', side_effect=ValueError('incompatible calibration')))
                    with self.assertRaises(ValueError):
                        focused.run(args)
                failure = read_json(args.output/'focused_failure.json')
                self.assertEqual(failure['stage'], phase)
                self.assertEqual(failure['error_type'], 'ValueError')
                self.assertIn('Traceback (most recent call last)', failure['traceback'])
                self.assertEqual(failure['traceback_scope'], 'local_artifact_only_not_notification_body')
                self.assertEqual(read_json(args.output/'focused_state.json')['status'], 'failed')

    def test_new_frozen_output_can_use_authorized_extended_deadline_without_altering_original(self):
        with self.patches():
            focused.run(self.args)
            original = (self.args.output/'budget_decision.json').read_bytes()
            extended = copy.copy(self.args)
            extended.output = self.root/'explicit-extension'/'functions'
            extended.deadline_unix += 36000
            self.assertEqual(focused.run(extended), 0)
        self.assertEqual((self.args.output/'budget_decision.json').read_bytes(), original)
        self.assertEqual(read_json(extended.output/'budget_decision.json')['deadline_unix'], extended.deadline_unix)
        for command in self.calls:
            self.assertEqual(command[command.index('--tokens')+1], '512')
            self.assertEqual(command[command.index('--block-size')+1], '512')
            self.assertEqual(command[command.index('--dard-steps')+1], '64')

    def test_extended_dataset_cannot_exceed_fixed_1440_evaluation_tasks(self):
        expanded = []
        for family in ('humaneval', 'mbpp'):
            examples = [t for t in self.tasks if t.family == family and t.source_id.endswith('/0')
                        and t.dataset in focused.SUITES]
            for index in range(205):
                for task in examples:
                    expanded.append(replace(task, task_id=f'{task.dataset}-{index}',
                                    source_id=('HumanEval/' if family == 'humaneval' else 'MBPP/')+str(index)))
        catalog = build_selection(expanded)
        self.assertGreater(len(catalog['evaluation_sources']), 360)
        base = {'workers': [{'index': i} for i in range(4)]}
        with patch.object(focused, 'estimate_seconds', return_value={'estimated_wall_seconds': 1}) as estimate:
            manifest, choice = focused.choose_cohort(catalog, expanded, base, {}, [], 100)
        estimate.assert_called_once()
        self.assertEqual(manifest['task_count'], 1440)
        self.assertEqual(choice['expected_cases'], 4320)

    def test_failure_history_is_append_only_across_resume_attempts(self):
        with self.patches(), patch.object(four_gpu, 'main', return_value=7):
            focused.run(self.args)
            focused.run(self.args)
        failures = [json.loads(line) for line in (self.args.output/'focused_failures.jsonl').read_text().splitlines()]
        self.assertEqual(len(failures), 2)
        self.assertNotEqual(failures[0]['invocation_id'], failures[1]['invocation_id'])
        self.assertEqual(failures[-1], read_json(self.args.output/'focused_failure.json'))


if __name__ == '__main__':
    unittest.main()
