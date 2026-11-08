"""End-to-end planning from only code-generation data, with no optional suites.

These tests use local fixture data and fake GPU identities. They never install,
download, send notifications, load a model, or query an actual GPU.
"""
import builtins
from contextlib import redirect_stderr, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import datasets_code as datasets
import evaluation
import function_benchmarks
from local_execution import CommandResult
import main
from scripts import all_experiments, four_gpu
from scripts.experiment_pipeline import build_stages


def fixture_datasets(root):
    cache = root / 'datasets/function-cache'; cache.mkdir(parents=True)
    humaneval = [{'task_id': f'HumanEval/{i}', 'entry_point': 'f',
                 'prompt': "def f(x):\n    '''Return x.'''\n",
                 'test': 'def check(candidate):\n    assert candidate(1) == 1\n'} for i in range(164)]
    (cache / 'HumanEval.jsonl.gz').write_bytes(gzip.compress(
        ('\n'.join(json.dumps(row) for row in humaneval) + '\n').encode()))
    mbpp = [{'task_id': i, 'text': 'Return x.', 'code': 'def f(x):\n    return x\n',
             'test_list': ['assert f(1) == 1']} for i in range(11, 511)]
    (cache / 'mbpp.jsonl').write_text('\n'.join(json.dumps(row) for row in mbpp) + '\n')
    for language in ('rs', 'js', 'cpp'):
        prompt = {'rs': 'fn f(x: isize) -> isize {\n', 'cpp': 'long f(long x) {\n',
                  'js': 'function f(x) {\n'}[language]
        tests = {'rs': '}\nfn main() { let candidate = f; assert_eq!(candidate(1), 1); }\n',
                 'cpp': '}\nint main() { auto candidate = f; assert(candidate(1) == 1); }\n',
                 'js': "const assert = require('node:assert'); let candidate = f; assert.equal(candidate(1), 1);\n"}[language]
        for family, number in (('humaneval', 0), ('mbpp', 11)):
            suite = 'multiple_' + ('mbpp_' if family == 'mbpp' else '') + language
            row = {'name': f"{'HumanEval' if family == 'humaneval' else 'mbpp'}_{number}_f",
                   'language': language, 'prompt': prompt, 'tests': tests}
            (cache / (suite + '_0.json')).write_text(json.dumps({'num_rows_total': 1, 'partial': False,
                'rows': [{'row_idx': 0, 'row': row, 'truncated_cells': []}]}))
    official_paths = {}
    for suite in ('humaneval_plus', 'mbpp_plus'):
        mbpp = suite == 'mbpp_plus'
        row = {'task_id': ('Mbpp' if mbpp else 'HumanEval') + '/0', 'entry_point': 'f',
               'prompt': '"""Return x."""\n' if mbpp else "def f(x):\n    '''Return x.'''\n",
               'canonical_solution': 'def f(x): return x\n' if mbpp else '    return x\n',
               'contract': '', 'base_input': [[1]], 'plus_input': [[2]], 'atol': 0}
        path = root / 'datasets/functions/raw' / (suite + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row) + '\n')
        official_paths[suite] = path
    function_benchmarks.write_official_manifest(root, official_paths)


class CodegenIndependenceTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_selection_share_the_codegen_contract(self):
        suites = datasets.CODE_GENERATION_SUITES
        self.assertIsInstance(suites, tuple)
        self.assertEqual(len(suites), 10)
        self.assertEqual(main.DEFAULT_SUITES.split(','), list(suites))
        self.assertEqual(main.cli([]).suites, list(suites))
        self.assertEqual({datasets.SUITES[name]['language'] for name in suites}, {'py', 'cpp', 'rs', 'js'})
        self.assertNotIn('bigcodebench_instruct', suites)
        # Retained historical adapter metadata must not permit accidental use
        # through the current code-generation CLI.
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.cli(['--suites', 'bigcodebench_instruct'])

    def test_default_pipeline_only_prepares_and_executes_codegen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('subprocess.run', side_effect=AssertionError('No external actions while planning')):
                stages = build_stages(repo=root / 'repo', data_root=root / 'data', output=root / 'results', prepare=True)
            self.assertEqual([stage['name'] for stage in stages], [
                'gpu_idle_check', 'data_verify', 'setup_functions', 'download_models',
                'function_smoke', 'functions_full', 'functions_analysis'])
            commands = json.dumps([stage['command'] for stage in stages])
            for excluded in ('bigcodebench', 'swebench', 'prepare_repositories', 'system_dependencies', 'repository_runner'):
                self.assertNotIn(excluded, commands)
            for stage in stages:
                if '--execute' in stage['command']:
                    argv = stage['command']
                    self.assertEqual(argv[argv.index('--suites') + 1].split(','), list(datasets.CODE_GENERATION_SUITES))
            self.assertFalse((root / 'data').exists())

    def test_function_preparation_artifacts_require_only_three_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); locks = root / 'envs/locks'; locks.mkdir(parents=True)
            for name in ('inference', 'function', 'evalplus'):
                for extension in ('.freeze.txt', '.conda.txt'):
                    (locks / (name + extension)).write_text('fixture inventory\n')
            hashes = all_experiments.artifact_hashes(
                {'name': 'setup_functions', 'command': ['bash', 'scripts/setup_env.sh', 'functions']}, root)
            self.assertEqual(len(hashes), 6)
            self.assertFalse((locks / 'system.json').exists())
            self.assertFalse((root / 'envs/swebench').exists())
            self.assertFalse((root / 'envs/bigcodebench').exists())

    def test_ten_suite_four_gpu_plan_works_without_optional_data_or_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture_datasets(root)
            args = four_gpu.parser().parse_args(['functions', '--data-root', str(root),
                '--output', str(root / 'results'), '--limit', '1', '--repeats', '3'])
            gpus = [{'uuid': f'GPU-fixture-{i}', 'name': 'Fixture GPU'} for i in range(4)]
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.split('.')[0] in {'repository_agent', 'repository_runner', 'local_repository_preparation',
                                          'local_swe_worker', 'swebench', 'bigcodebench'}:
                    raise AssertionError('Code generation imported an inactive evaluator: ' + name)
                return original_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded_import), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('No dataset download')), \
                 patch('subprocess.run', side_effect=AssertionError('No GPU or environment command')), \
                 redirect_stdout(io.StringIO()):
                plan, tasks = four_gpu.make_plan(args, gpus)
            self.assertEqual({task.dataset for task in tasks}, set(datasets.CODE_GENERATION_SUITES))
            self.assertEqual(plan['task_count'], 10)
            self.assertEqual(plan['expected_cases'], 90)
            self.assertEqual(sorted(worker['task_count'] for worker in plan['workers']), [2, 2, 3, 3])
            self.assertEqual(plan['sources'], [])
            self.assertFalse((root / 'datasets/swebench_verified').exists())
            self.assertFalse((root / 'datasets/swebench_multilingual').exists())
            self.assertFalse((root / 'datasets/functions/raw/bigcodebench_instruct.jsonl').exists())
            self.assertTrue(all(task.final_evaluator in ('legacy_assert', 'evalplus') for task in tasks))

    def test_native_codegen_preflight_needs_no_optional_evaluator_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture_datasets(root)
            with redirect_stdout(io.StringIO()):
                tasks = datasets.load_tasks(list(datasets.CODE_GENERATION_SUITES),
                    root / 'datasets/function-cache', 1, data_root=root)
            for name in ('python', 'rustc', 'node', 'g++'):
                tool = root / 'envs/function/bin' / name
                tool.parent.mkdir(parents=True, exist_ok=True); tool.write_text('fixture')
                tool.chmod(0o755)
            official_python = root / 'envs/evalplus/bin/python'
            official_python.parent.mkdir(parents=True); official_python.write_text('fixture')
            official_python.chmod(0o755)
            def probe(argv, **kwargs):
                self.assertNotIn('bigcodebench', argv[0])
                self.assertNotIn('swebench', argv[0])
                if argv[0].endswith('/python'):
                    return CommandResult(0, json.dumps({'executable': argv[0], 'python': 'fixture',
                        'packages': [('evalplus', '0.3.1')] if '/evalplus/' in argv[0] else []}))
                return CommandResult(0, 'fixture version\n')
            with patch.object(evaluation, 'run_bounded_command', side_effect=probe), \
                 patch.dict(evaluation._EXECUTABLES, {}, clear=True), \
                 patch.dict(evaluation._OFFICIAL_PYTHONS, {}, clear=True), \
                 patch.object(evaluation, '_LOCAL_DATA_ROOT', root):
                identity = evaluation.preflight_local({'py', 'rs', 'cpp', 'js'}, tasks=tasks, data_root=root)
                self.assertEqual(set(evaluation._OFFICIAL_PYTHONS), {'evalplus'})
            self.assertEqual(set(identity), {'py', 'rs', 'cpp', 'js', 'official:evalplus'})
            self.assertFalse((root / 'envs/bigcodebench').exists())
            self.assertFalse((root / 'envs/swebench').exists())


if __name__ == '__main__':
    unittest.main()
