"""Offline tests of the shared agent loop, repeat scheduling, and retryable logs."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from datasets_code import Task
import main as bench
from models import build_messages
from reporting import latest_cases


class FakeBackend:
    def __init__(self, runtime, decoder, model_name):
        self.runtime, self.decoder, self.model_name = runtime, decoder, model_name
        runtime['active'] += 1
        runtime['max_active'] = max(runtime['active'], runtime['max_active'])
        runtime['loads'].append(decoder)

    def metadata(self):
        revision = self.runtime.get('revisions', {}).get(self.decoder, 'fixture-commit')
        return {'model_name': self.model_name, 'resolved_revision': revision,
                'tokenizer_revision': 'fixture-commit', 'model_config_sha256': 'fixture-config',
                'chat_template_sha256': 'fixture-chat',
                'tokenizer_vocab_sha256': 'fixture-vocab', 'gpu_uuid': 'GPU-fixture'}

    def close(self):
        self.runtime['active'] -= 1

    def generate(self, language, instruction, decoder, *, seed, num_new_tokens, steps,
                 dard_config, temperature, top_p, old_code, error):
        self.runtime['calls'].append((decoder, instruction, seed, num_new_tokens, temperature, top_p, error))
        if error and self.runtime.get('fail_repair', False):
            self.runtime['fail_repair'] = False
            raise RuntimeError('mock inference infrastructure failure')
        messages = build_messages(language, instruction, old_code, error)
        return {'text': 'def f(x):\n    return x * 2\n' if error else 'def f(x):\n    return x + 1\n',
                'wall_s': {'ar': .04, 'standard': .03, 'dard': .02}[decoder],
                'model_wall_s': .01, 'model_forward_passes': 2, 'forced_tokens': 0,
                'generated_tokens': 16, 'visible_output_tokens': 12, 'output_slots': 16,
                'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}


def fixture_task(number=0):
    return Task('humaneval', 'py', f'HumanEval/{number}', number,
                f'Implement f(x) returning twice x. Task {number}.', 'assert f(2) == 4',
                'f', family='humaneval', source_id=f'HumanEval/{number}')


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'out'
        self.runtime = {'loads': [], 'calls': [], 'active': 0, 'max_active': 0}
        self.tasks = [fixture_task(0), fixture_task(1)]
        self.infra_once = False

    def simulate_test(self, task, code, runner, timeout):
        if runner == 'none':
            return {'passed': None, 'status': 'not_tested', 'feedback': '--runner none', 'test_wall_s': 0}
        if self.infra_once:
            self.infra_once = False
            return {'passed': None, 'status': 'infra_error', 'feedback': 'Local interpreter unavailable', 'test_wall_s': .01}
        passed = 'x * 2' in code
        return {'passed': passed, 'status': 'passed' if passed else 'test_failed',
                'feedback': '' if passed else 'AssertionError', 'test_wall_s': .01}

    def run_bench(self, extra=()):
        args = ['--suites', 'humaneval', '--limit', '2', '--backends', 'ar,standard,dard',
                '--runner', 'local', '--repairs', '1', '--tokens', '16', '--repeats', '3',
                '--bootstrap-draws', '30', '--output', str(self.output), *extra]
        with patch.object(bench, 'preflight_local', return_value={'py': {'python': 'fixture'}}), \
             patch.object(bench, 'load_tasks', return_value=self.tasks), \
             patch.object(bench, '_environment', return_value={'host': 'fixture'}), \
             patch.object(bench, '_gpu_processes', return_value={'available': True, 'other_pids': []}), \
             patch.object(bench, 'test_task', side_effect=self.simulate_test), \
             patch.object(bench, 'grade_final', side_effect=self.simulate_test), \
             patch('models.create_backend', side_effect=lambda decoder, **kw: FakeBackend(self.runtime, decoder, kw['model_name'])), \
             contextlib.redirect_stdout(io.StringIO()):
            return bench.main(args)

    def test_common_repair_pipeline_counterbalanced_order_and_resume(self):
        self.assertEqual(self.run_bench(), 0)
        expected_order = ['ar', 'standard', 'dard', 'standard', 'dard', 'ar', 'dard', 'ar', 'standard']
        self.assertEqual(self.runtime['loads'], expected_order)
        self.assertEqual(self.runtime['max_active'], 1)
        self.assertEqual(self.runtime['active'], 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r['attempts'] == 2 and r['first_pass'] is False and r['final_pass'] is True for r in rows))
        self.assertTrue(all(r['warmup_done'] and r['case_completed'] for r in rows))
        prompts = defaultdict(set)
        for row in rows:
            prompts[(row['task_id'], row['repeat'])].add(row['messages_sha256'])
        self.assertTrue(all(len(values) == 1 for values in prompts.values()))
        traces = [json.loads(line) for line in (self.output / 'runs.jsonl').read_text().splitlines()]
        self.assertEqual(len(traces), 36)  # nine warmups are not scored/logged as attempts
        self.assertEqual(traces[0]['raw_output_sha256'], hashlib.sha256(traces[0]['raw_output'].encode()).hexdigest())
        self.assertTrue(traces[1]['code_changed_from_previous'])
        self.assertIn('-    return x + 1', traces[1]['repair_diff'])
        self.assertTrue(all(r['repair_code_changes'] == 1 for r in rows))
        report = json.loads((self.output / 'summary.json').read_text())
        comparison = report['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertEqual(comparison['paired_evaluated_cases'], 6)
        self.assertEqual(comparison['unique_problem_clusters'], 2)
        self.assertEqual(comparison['initial_generation_speedup_ar_over_candidate'], 2.0)
        self.assertTrue(comparison['eligible_for_controlled_comparison'])
        dard_effect = report['paired_comparisons']['dard_vs_standard/humaneval/py']
        self.assertEqual(dard_effect['initial_generation_speedup_standard_over_candidate'], 1.5)
        self.assertTrue(dard_effect['eligible_for_controlled_comparison'])
        self.assertEqual(self.run_bench(), 0)
        self.assertEqual(self.runtime['loads'], expected_order)
        self.assertEqual(len(latest_cases(self.output / 'case_results.jsonl')), 18)

    def test_live_progress_does_not_rebootstrap_every_five_cases(self):
        self.tasks = [fixture_task(index) for index in range(12)]
        with patch.object(bench, '_summary', wraps=bench._summary) as summary:
            self.assertEqual(self.run_bench(['--backends', 'ar', '--repeats', '1', '--warmup', '0']), 0)
        # One complete report at the phase boundary and one at the run end.
        self.assertEqual(summary.call_count, 2)
        progress = json.loads((self.output / 'progress.json').read_text())
        self.assertEqual(progress['phase_cases_processed'], 10)
        self.assertEqual(progress['completed_cases'], 10)
        self.assertEqual(progress['expected_cases'], 12)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['groups']['ar/humaneval/py']['completed'], 12)

    def test_repair_inference_error_is_incomplete_and_retryable(self):
        self.tasks = [fixture_task()]
        self.runtime['fail_repair'] = True
        options = ['--backends', 'ar', '--repeats', '1', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 1)
        row = latest_cases(self.output / 'case_results.jsonl')[0]
        self.assertFalse(row['case_completed'])
        self.assertIs(row['first_pass'], False)
        self.assertIsNone(row['final_pass'])
        self.assertEqual(self.runtime['active'], 0)
        self.assertEqual(self.run_bench(options), 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['case_completed'])
        self.assertTrue(rows[0]['final_pass'])
        self.assertEqual(len((self.output / 'case_results.jsonl').read_text().splitlines()), 2)

    def test_infrastructure_verdict_is_retried_on_resume(self):
        self.tasks = [fixture_task()]
        self.infra_once = True
        options = ['--backends', 'ar', '--repeats', '1', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 1)
        self.assertFalse(latest_cases(self.output / 'case_results.jsonl')[0]['case_completed'])
        self.assertEqual(self.run_bench(options), 0)
        self.assertTrue(latest_cases(self.output / 'case_results.jsonl')[0]['final_pass'])

    def test_fail_fast_persists_infrastructure_error_before_next_case_or_backend(self):
        self.infra_once = True
        options = ['--fail-fast', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 1)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['case_completed'])
        self.assertEqual(rows[0]['failure_kind'], 'evaluation_infrastructure')
        self.assertEqual(self.runtime['loads'], ['ar'])
        self.assertEqual(self.runtime['active'], 0)
        self.assertEqual(self.run_bench(options), 0)
        self.assertEqual(len(latest_cases(self.output / 'case_results.jsonl')), 18)

    def test_fail_fast_keeps_wrong_answers_as_normal_completed_results(self):
        self.assertEqual(self.run_bench(['--fail-fast', '--repairs', '0', '--warmup', '0']), 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r['case_completed'] and r['final_pass'] is False for r in rows))

    def test_single_generation_has_no_feedback_execution_and_grades_exactly_once(self):
        backend = FakeBackend(self.runtime, 'ar', 'fixture')
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '0'])
        with patch.object(bench, 'test_task', side_effect=AssertionError('Public feedback must not run')) as feedback, \
             patch.object(bench, 'grade_final', side_effect=self.simulate_test) as grade, \
             contextlib.redirect_stdout(io.StringIO()):
            row = bench._run_case(backend, self.tasks[0], 'ar', 0, 42, args, bench._dard_config(args),
                                  io.StringIO(), io.StringIO(), {'phase_id': 0})
        backend.close()
        feedback.assert_not_called()
        grade.assert_called_once()
        self.assertEqual(len(self.runtime['calls']), 1)
        self.assertTrue(row['case_completed'])
        self.assertIsNone(row['failure_kind'])
        self.assertIsNone(row['feedback_first_pass'])
        self.assertIsNone(row['feedback_final_pass'])
        self.assertEqual(row['feedback_protocol'], 'disabled_single_generation')
        self.assertEqual(row['feedback_first_error_category'], 'feedback_disabled')
        self.assertIs(row['first_pass'], False)
        self.assertIs(row['final_pass'], False)
        self.assertEqual(row['attempts'], 1)

    def test_generation_and_grading_have_current_case_marker_before_work(self):
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '0', '--output', str(self.output)])
        backend = FakeBackend(self.runtime, 'ar', 'fixture')
        original = backend.generate
        observed = []
        def generated(*a, **kw):
            state = json.loads((self.output / 'current_case.json').read_text())
            observed.append(state['stage'])
            self.assertEqual(state['state'], 'running')
            self.assertEqual(state['task_id'], self.tasks[0].task_id)
            return original(*a, **kw)
        def graded(*a, **kw):
            state = json.loads((self.output / 'current_case.json').read_text())
            observed.append(state['stage'])
            self.assertEqual(state['state'], 'running')
            return self.simulate_test(*a, **kw)
        with patch.object(backend, 'generate', side_effect=generated), \
             patch.object(bench, 'grade_final', side_effect=graded), contextlib.redirect_stdout(io.StringIO()):
            row = bench._run_case(backend, self.tasks[0], 'ar', 0, 42, args, bench._dard_config(args),
                                  io.StringIO(), io.StringIO(), {'phase_id': 0})
        backend.close()
        self.assertEqual(observed, ['generation', 'sealed_final_grading'])
        self.assertTrue(row['case_completed'])
        state = json.loads((self.output / 'current_case.json').read_text())
        self.assertEqual(state['state'], 'awaiting_result_persist')
        self.assertEqual(state['pending_result'], row)  # A kill during the final GPU check retains the grade.

    def test_grading_interrupt_keeps_attempt_and_stage_without_accepting_result(self):
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '0', '--output', str(self.output)])
        backend = FakeBackend(self.runtime, 'ar', 'fixture')
        traces, errors = io.StringIO(), io.StringIO()
        with patch.object(bench, 'grade_final', side_effect=KeyboardInterrupt()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                bench._run_case(backend, self.tasks[0], 'ar', 0, 42, args, bench._dard_config(args),
                                traces, errors, {'phase_id': 0})
        backend.close()
        self.assertEqual(len(traces.getvalue().splitlines()), 1)
        failure = json.loads(errors.getvalue())
        self.assertEqual(failure['stage'], 'sealed_final_grading')
        self.assertEqual(failure['error_type'], 'KeyboardInterrupt')
        self.assertEqual(json.loads((self.output / 'current_case.json').read_text())['state'], 'interrupted')

    def test_returned_grading_infrastructure_error_gets_explicit_error_record(self):
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '0'])
        backend = FakeBackend(self.runtime, 'ar', 'fixture')
        errors = io.StringIO()
        verdict = {'passed': None, 'status': 'infra_error', 'feedback': '', 'test_wall_s': .1,
                   'sealed_diagnostics': 'worker exited', 'execution': {'returncode': -9, 'signal': 9}}
        with patch.object(bench, 'grade_final', return_value=verdict), contextlib.redirect_stdout(io.StringIO()):
            row = bench._run_case(backend, self.tasks[0], 'ar', 0, 42, args, bench._dard_config(args),
                                  io.StringIO(), errors, {'phase_id': 0})
        backend.close()
        self.assertFalse(row['case_completed'])
        failure = json.loads(errors.getvalue())
        self.assertEqual(failure['error_type'], 'GradingInfrastructureError')
        self.assertEqual(failure['grading']['final']['execution']['signal'], 9)

    def test_backend_cleanup_failure_is_nonzero_even_after_all_cases_complete(self):
        self.tasks = [fixture_task()]
        original = FakeBackend.close
        def failing_close(backend):
            original(backend)
            raise RuntimeError('fixture cleanup failure')
        with patch.object(FakeBackend, 'close', failing_close):
            self.assertEqual(self.run_bench(['--backends', 'ar', '--repeats', '1', '--repairs', '0', '--warmup', '0']), 1)
        self.assertTrue(latest_cases(self.output / 'case_results.jsonl')[0]['case_completed'])
        errors = [json.loads(line) for line in (self.output / 'errors.jsonl').read_text().splitlines()]
        self.assertTrue(any(error['stage'] == 'model_cleanup' for error in errors))
        self.assertEqual(json.loads((self.output / 'worker_status.json').read_text())['exit_code'], 1)

    def test_jsonl_append_fsyncs_and_atomic_status_fsyncs_before_replacement(self):
        self.output.mkdir()
        path = self.output / 'fixture.jsonl'
        with patch.object(bench.os, 'fsync', wraps=os.fsync) as sync:
            with path.open('a') as stream:
                bench._append(stream, {'value': 1})
            self.assertEqual(sync.call_count, 1)
            bench._status(self.output, 'generation', case=True, task_id='fixture')
            self.assertEqual(sync.call_count, 3)  # JSON file and its directory.
        self.assertEqual(json.loads(path.read_text()), {'value': 1})
        self.assertFalse((self.output / 'current_case.json.tmp').exists())

    def test_exact_manifest_single_repeat_runs_each_selected_task_once_per_backend(self):
        self.tasks = [fixture_task(index) for index in range(7)]
        path = Path(self.temp.name) / 'selection.json'
        keys = [['humaneval', 'py', f'HumanEval/{index}'] for index in (1, 4, 6)]
        path.write_text(json.dumps({'schema_version': 1, 'task_keys': keys, 'selection_rule': 'public_ids'}))
        options = ['--limit', '0', '--selection-manifest', str(path), '--backends', 'ar,dard,standard',
                   '--repeats', '1', '--repairs', '0', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 0)
        self.assertEqual(self.runtime['loads'], ['ar', 'dard', 'standard'])
        self.assertEqual(len(self.runtime['calls']), 9)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual(len(rows), 9)
        for decoder in ('ar', 'dard', 'standard'):
            selected = [row for row in rows if row['decoder'] == decoder]
            self.assertEqual({row['task_id'] for row in selected}, {key[2] for key in keys})
            self.assertTrue(all(row['repeat'] == 0 and row['attempts'] == 1 for row in selected))
        settings = json.loads((self.output / 'config.json').read_text())['settings']
        self.assertEqual(settings['selection_manifest']['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(settings['sharding']['full_selection_count'], 3)
        self.assertEqual(settings['feedback_protocol'], 'disabled_single_generation')
        self.assertEqual(self.run_bench(options), 0)
        self.assertEqual(len(self.runtime['calls']), 9)
        # Even changed selection provenance, with identical identities, must be acknowledged in a new run.
        path.write_text(json.dumps({'schema_version': 1, 'task_keys': keys, 'selection_rule': 'changed'}))
        with self.assertRaisesRegex(RuntimeError, 'selection_manifest'):
            self.run_bench(options)

    def test_worker_rejects_manifest_changed_since_coordinator_planned_it(self):
        path = Path(self.temp.name) / 'selection.json'
        path.write_text(json.dumps({'schema_version': 1, 'task_keys': [['humaneval', 'py', 'HumanEval/0']]}))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text(json.dumps({'schema_version': 1, 'task_keys': [['humaneval', 'py', 'HumanEval/1']]}))
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            self.run_bench(['--limit', '0', '--selection-manifest', str(path),
                            '--selection-manifest-sha256', expected])
        self.assertEqual(self.runtime['loads'], [])
        self.assertFalse(self.output.exists())

    def test_resume_rejects_changed_task_or_budget(self):
        options = ['--backends', 'ar', '--repeats', '1', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 0)
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            self.run_bench([*options, '--tokens', '32'])
        self.tasks = [fixture_task(50)]
        with self.assertRaisesRegex(RuntimeError, 'task_manifest_sha256'):
            self.run_bench(options)

    def test_shard_keeps_all_backends_repeats_and_freezes_full_selection(self):
        self.tasks = [fixture_task(i) for i in range(7)]
        options = ['--num-shards', '4', '--shard-index', '2']
        self.assertEqual(self.run_bench(options), 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertEqual({row['task_id'] for row in rows}, {'HumanEval/2', 'HumanEval/6'})
        self.assertEqual(len(rows), 2 * 3 * 3)
        self.assertEqual(len(self.runtime['loads']), 9)
        config = json.loads((self.output / 'config.json').read_text())
        sharding = config['settings']['sharding']
        self.assertEqual(sharding['full_selection_count'], 7)
        self.assertEqual(sharding['shard_index'], 2)
        self.assertNotEqual(sharding['full_selection_sha256'], config['settings']['task_manifest_sha256'])
        self.assertEqual(self.run_bench(options), 0)
        self.assertEqual(len(self.runtime['loads']), 9)
        with self.assertRaisesRegex(RuntimeError, 'shard_index'):
            self.run_bench(['--num-shards', '4', '--shard-index', '1'])
        # Even a change outside this shard alters the full-selection provenance.
        self.tasks[0] = fixture_task(50)
        with self.assertRaisesRegex(RuntimeError, 'sharding'):
            self.run_bench(options)

    def test_empty_shard_fails_before_model_or_local_work(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.run_bench(['--num-shards', '4', '--shard-index', '3'])
        self.assertEqual(self.runtime['loads'], [])
        self.assertFalse(self.output.exists())

    def test_generate_only_has_no_accuracy_or_repairs(self):
        options = ['--runner', 'none', '--repairs', '0', '--repeats', '1', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertTrue(all(r['attempts'] == 1 and r['case_completed'] and r['final_pass'] is None for r in rows))
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertIsNone(report['groups']['ar/humaneval/py']['final_pass_rate'])
        self.assertFalse(report['paired_comparisons']['dard_vs_ar/humaneval/py']['eligible_for_controlled_comparison'])

    def test_standard_and_dard_must_use_same_diffusion_checkpoint(self):
        self.tasks = [fixture_task()]
        self.runtime['revisions'] = {'standard': 'commit-A', 'dard': 'commit-B'}
        self.assertEqual(self.run_bench(['--backends', 'standard,dard', '--repeats', '1']), 1)
        config = json.loads((self.output / 'config.json').read_text())
        self.assertIn('identity changed', config['phases'][-1]['error'])
        self.assertEqual(self.runtime['active'], 0)
        self.assertEqual(len(latest_cases(self.output / 'case_results.jsonl')), 1)

    def test_resume_rejects_altered_saved_task_manifest(self):
        options = ['--backends', 'ar', '--repeats', '1', '--warmup', '0']
        self.assertEqual(self.run_bench(options), 0)
        (self.output / 'tasks.json').write_text('[]')
        with self.assertRaisesRegex(RuntimeError, 'tasks.json'):
            self.run_bench(options)

    def test_unknown_gpu_uuid_cannot_report_uncontended(self):
        from types import SimpleNamespace
        with patch.object(bench.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='GPU-other\n')) as run:
            result = bench._gpu_processes('GPU-nonexistent')
        self.assertFalse(result['available'])
        self.assertEqual(run.call_count, 1)

    def test_cli_rejects_invalid_comparison_setup(self):
        for args in (['--backends', 'ar,ar'], ['--device', 'auto'], ['--seeds', 'oops'], ['--repeats', '0'],
                     ['--num-shards', '0'], ['--num-shards', '4', '--shard-index', '4'],
                     ['--selection-manifest', '/fixture/selection.json'],
                     ['--limit', '0', '--start', '1', '--selection-manifest', '/fixture/selection.json'],
                     ['--selection-manifest-sha256', 'fixture']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                bench.cli(args)
        args = bench.cli(['--decoders', 'ar,dard', '--seeds', '7,11'])
        self.assertEqual(args.trial_seeds, [7, 11])
        self.assertEqual(args.quant, 'bf16')
        self.assertEqual(args.temperature, 0)

    def test_final_grader_failure_cannot_trigger_repairs_or_leak_feedback(self):
        self.tasks = [fixture_task()]
        events = []
        def feedback(task, code, runner, timeout):
            events.append('feedback')
            return {'passed': True, 'status': 'feedback_passed', 'feedback': '', 'test_wall_s': .01}
        def final(task, code, runner, timeout):
            events.append('final')
            return {'passed': False, 'status': 'test_failed', 'feedback': 'SEALED_EXPECTED_VALUE', 'test_wall_s': .01}
        backend = FakeBackend(self.runtime, 'ar', 'fixture')
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '2'])
        with patch.object(bench, 'test_task', side_effect=feedback), patch.object(bench, 'grade_final', side_effect=final):
            row = bench._run_case(backend, self.tasks[0], 'ar', 0, 42, args, bench._dard_config(args),
                                  io.StringIO(), io.StringIO(), {'phase_id': 0})
        backend.close()
        self.assertEqual(events, ['feedback', 'final'])
        self.assertEqual(row['attempts'], 1)
        self.assertIs(row['final_pass'], False)
        self.assertEqual(row['feedback_last_error_category'], 'feedback_passed')
        self.assertEqual(row['last_error_category'], 'runtime_or_logic_error')
        self.assertNotIn('SEALED_EXPECTED_VALUE', repr(self.runtime['calls']))

    def test_generation_failure_keeps_partial_counters_and_phase(self):
        exc = RuntimeError('mock CUDA failure')
        exc.generation_diagnostics = {'wall_s': .7, 'model_forward_passes': 9,
            'diagnostics': {'complete': False, 'phase': 'verification', 'decoder': {'blocks': [{'step': 4}]}}}
        backend = FakeBackend(self.runtime, 'dard', 'fixture')
        args = bench.cli(['--suites', 'humaneval', '--tokens', '16', '--repairs', '0'])
        traces, errors = io.StringIO(), io.StringIO()
        with patch.object(backend, 'generate', side_effect=exc):
            row = bench._run_case(backend, self.tasks[0], 'dard', 0, 42, args, bench._dard_config(args),
                                  traces, errors, {'phase_id': 0})
        backend.close()
        self.assertFalse(row['measurement_complete'])
        self.assertEqual(row['model_forward_passes'], 9)
        self.assertEqual(json.loads(traces.getvalue())['generation']['diagnostics']['phase'], 'verification')
        self.assertEqual(json.loads(errors.getvalue())['generation_diagnostics']['model_forward_passes'], 9)

    def test_profile_run_records_instrumentation_and_excludes_controlled_speed(self):
        self.assertEqual(self.run_bench(['--diagnostics', 'profile', '--max-step-records', '2']), 0)
        rows = latest_cases(self.output / 'case_results.jsonl')
        self.assertTrue(all(r['diagnostics_mode'] == 'profile' and r['instrumentation_affects_timing'] for r in rows))
        report = json.loads((self.output / 'summary.json').read_text())
        comparison = report['paired_comparisons']['dard_vs_ar/humaneval/py']
        self.assertFalse(comparison['eligible_for_controlled_comparison'])
        self.assertIn('diagnostic_instrumentation_affects_timing', comparison['comparison_issues'])


if __name__ == '__main__':
    unittest.main()
