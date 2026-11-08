"""Offline priority orchestration checks with durable artifacts and no GPU/model work."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import priority_experiments as priority
from scripts.all_experiments import write_json
from tests.test_deadline_budget import fixture


class PriorityExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan, self.shards, self.targets = fixture()
        self.calibration = self.root / 'function-calibration'
        self._save_calibration(self.calibration)
        self.calls = []
        self.gpus = [{'uuid': f'GPU-{i}', 'name': 'fixture'} for i in range(4)]

    def args(self, mode='priority', deadline=100000):
        return Namespace(mode=mode, output=self.root / f'functions-{mode}',
                         data_root=self.root / 'data', gpus='0,1,2,3',
                         calibration=self.calibration, deadline_unix=deadline)

    def _save_calibration(self, directory, backends=('ar', 'dard', 'standard')):
        plan = copy.deepcopy(self.plan)
        for index, worker in enumerate(plan['workers']):
            output = directory / f'shard-{index}'
            output.mkdir(parents=True, exist_ok=True)
            worker['output'] = str(output)
            rows = [r for r in self.shards[index]['rows'] if r['decoder'] in backends]
            (output / 'case_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write_json(output / 'config.json', {
                'phases': [p for p in self.shards[index]['phases'] if p['decoder'] in backends]})
        write_json(directory / 'campaign.json', plan)
        write_json(directory / 'execution_status.json', {'exit_codes': {str(i): 0 for i in range(4)}})

    def _report(self, directory, backends, *, complete=True):
        groups = {}
        for backend in backends:
            groups[f'{backend}/all'] = {
                'complete_evaluated_cohort': complete, 'expected_cases': 24,
                'recorded_latest_cases': 24 if complete else 1,
                'evaluated_cases': 24 if complete else 1, 'passed_cases': 10 if complete else 1,
                'failed_cases': 14 if complete else 0,
                'summed_latest_case_seconds': {'generation_wall_s': 1, 'total_wall_s': 2},
                'phase_accounting': {'summed_phase_wall_s': 3},
            }
        write_json(directory / 'campaign_summary.json', {
            'coverage_complete': complete, 'groups': groups, 'backends': list(backends),
            'expected_cases': 24*len(backends), 'recorded_latest_cases': 24*len(backends),
        })

    def fake_campaign(self, argv):
        self.calls.append(list(argv))
        directory = Path(argv[argv.index('--output')+1])
        backends = argv[argv.index('--backends')+1].split(',')
        limit = int(argv[argv.index('--limit')+1])
        if limit:
            self._save_calibration(directory, backends)
        else:
            write_json(directory / 'campaign.json', {'workers': self.targets, 'backends': backends})
            self._report(directory, backends)
        return 0

    def patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(priority.time, 'time', return_value=1000.))
        stack.enter_context(patch.object(priority.four_gpu, 'resolve_gpus', return_value=self.gpus))
        stack.enter_context(patch.object(priority.four_gpu, 'make_plan', return_value=({'workers': self.targets}, [])))
        stack.enter_context(patch.object(priority.four_gpu, 'main', side_effect=self.fake_campaign))
        return stack

    def test_full_priority_finishes_before_optional_standard_and_writes_result_index(self):
        with self.patches(), patch.object(priority.four_gpu, 'audit') as audit:
            self.assertEqual(priority.run(self.args()), 0)
            self.assertEqual(priority.run(self.args('standard')), 0)
        choices = [(c[c.index('--backends')+1], c[c.index('--limit')+1]) for c in self.calls]
        self.assertEqual(choices, [('ar,dard', '0'), ('standard', '2'), ('standard', '0')])
        self.assertTrue(all(c[c.index('--tokens')+1] == '512' for c in self.calls))
        self.assertTrue(all(c[c.index('--dard-steps')+1] == '192' for c in self.calls))
        audit.assert_not_called()  # Every normal four_gpu.main return already audited.
        index = json.loads((self.root / 'priority_summary.json').read_text())
        self.assertTrue(index['required_campaigns_complete'])
        self.assertTrue(index['all_available_campaigns_complete'])
        self.assertEqual(index['cross_campaign_comparisons'], [])
        self.assertTrue((self.root / 'priority_summary.md').is_file())
        budget = json.loads((self.root / 'budget_decision.json').read_text())
        self.assertEqual(budget['priority']['status'], 'completed')
        self.assertEqual(budget['standard']['status'], 'completed')
        self.assertEqual(budget['deadline_unix'], 100000)

    def test_standard_skips_calibration_when_insufficient_time_remains(self):
        self._report(self.root / 'functions-priority', ['ar', 'dard'])
        with self.patches():
            self.assertEqual(priority.run(self.args('standard', deadline=5000)), 0)
        self.assertEqual(self.calls, [])
        saved = json.loads((self.root / 'functions-standard/budget_decision.json').read_text())
        self.assertEqual(saved['status'], 'skipped')
        self.assertEqual(saved['selected_repeats'], 0)
        self.assertTrue(json.loads((self.root / 'priority_summary.json').read_text())['required_campaigns_complete'])

    def test_standard_refuses_incomplete_priority_even_when_resuming(self):
        for saved in (False, True):
            args = self.args('standard')
            self._report(self.root / 'functions-priority', ['ar', 'dard'], complete=False)
            if saved:
                write_json(args.output / 'budget_decision.json', {
                    'status': 'scheduled', 'backends': ['standard'], 'deadline_unix': args.deadline_unix,
                    'selected_repeats': 1})
            with self.subTest(resume=saved), self.patches(), self.assertRaisesRegex(ValueError, 'before complete'):
                priority.run(args)
        self.assertEqual(self.calls, [])

    def test_priority_infeasible_estimate_attempts_one_complete_repeat_and_marks_risk(self):
        # 100 seconds remain after reserving 1800; the fixture estimates >300.
        with self.patches():
            self.assertEqual(priority.run(self.args(deadline=2900)), 0)
        saved = json.loads((self.root / 'functions-priority/budget_decision.json').read_text())
        self.assertEqual(saved['selected_repeats'], 1)
        self.assertTrue(saved['deadline_at_risk'])
        self.assertFalse(saved['fits_estimate'])
        self.assertTrue(saved['all_tasks_retained'])
        self.assertTrue(saved['generation_budgets_unchanged'])
        self.assertEqual(self.calls[0][self.calls[0].index('--limit')+1], '0')
        self.assertEqual(self.calls[0][self.calls[0].index('--repeats')+1], '1')

    def test_standard_skips_full_campaign_when_measured_estimate_does_not_fit(self):
        self._report(self.root / 'functions-priority', ['ar', 'dard'])
        original = priority.select_repeats
        def no_optional_budget(*args, **kwargs):
            return original(*args[:4], 0, **kwargs)
        with self.patches(), patch.object(priority, 'select_repeats', side_effect=no_optional_budget):
            self.assertEqual(priority.run(self.args('standard')), 0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][self.calls[0].index('--limit')+1], '2')
        saved = json.loads((self.root / 'functions-standard/budget_decision.json').read_text())
        self.assertEqual(saved['status'], 'skipped')
        self.assertEqual(saved['selected_repeats'], 0)

    def test_resume_preserves_repeat_choice_and_rejects_deadline_reset(self):
        args = self.args()
        write_json(args.output / 'budget_decision.json', {'status': 'scheduled', 'backends': ['ar', 'dard'],
                   'selected_repeats': 2, 'deadline_unix': args.deadline_unix})
        with self.patches(), patch.object(priority, 'calibration_records') as calibration:
            self.assertEqual(priority.run(args), 0)
            calibration.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'deadline/backend decision changed'):
                priority.run(self.args(deadline=100001))
        self.assertEqual(self.calls[0][self.calls[0].index('--repeats')+1], '2')
        self.assertEqual(len(self.calls), 1)

    def test_campaign_exception_preserved_when_partial_audit_also_fails(self):
        args = self.args()
        write_json(args.output / 'budget_decision.json', {'status': 'scheduled', 'backends': ['ar', 'dard'],
                   'selected_repeats': 1, 'deadline_unix': args.deadline_unix})
        write_json(args.output / 'campaign.json', {'workers': self.targets})
        with self.patches(), patch.object(priority.four_gpu, 'main', side_effect=RuntimeError('model failed')), \
                patch.object(priority.four_gpu, 'audit', side_effect=ValueError('partial report failed')) as audit, \
                self.assertRaisesRegex(RuntimeError, 'model failed'):
            priority.run(args)
        audit.assert_called_once()
        saved = json.loads((args.output / 'budget_decision.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('model failed', saved['error'])
        self.assertIn('partial report failed', saved['partial_analysis_error'])

    def test_expired_absolute_deadline_does_not_launch_saved_campaign(self):
        args = self.args(deadline=1000)
        write_json(args.output / 'budget_decision.json', {'status': 'scheduled', 'backends': ['ar', 'dard'],
                   'selected_repeats': 1, 'deadline_unix': args.deadline_unix})
        with self.patches(), self.assertRaises(TimeoutError):
            priority.run(args)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
