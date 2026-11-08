"""Native preparation/grading tests: tiny local trees, no installs or downloads."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import local_repository_preparation as preparation
import local_swe_worker as worker
import native_toolchains
from local_execution import environment_fingerprint, run_bounded_command


class RecipeTests(unittest.TestCase):
    def test_every_pinned_recipe_has_real_native_setup(self):
        catalog = json.loads(preparation.RECIPES.read_text())
        self.assertEqual(len(catalog['recipes']), 380)
        families = set()
        for key, recipe in catalog['recipes'].items():
            families.add(key.split('/')[0])
            packages, system, commands = preparation.setup_commands(recipe)
            self.assertTrue(commands or recipe.get('toolchain'), key)
            self.assertTrue(preparation.conda_packages(recipe))
            self.assertNotIn('apt-get', '\n'.join(commands))
            self.assertEqual(len(recipe['upstream_revision']), 40)
        self.assertEqual(families, {'swebench_verified', 'swebench_multilingual'})

    def test_unknown_repository_never_silently_skips(self):
        with self.assertRaisesRegex(ValueError, 'No pinned'):
            preparation.recipe_for({'repo': 'unknown/repo', 'version': '1'}, 'swebench_verified')

    def test_historical_native_versions_have_checksummed_source_bootstrap(self):
        catalog = json.loads(preparation.RECIPES.read_text())
        for recipe in catalog['recipes'].values():
            packages = preparation.conda_packages(recipe)
            self.assertFalse(any(p.startswith(('php=', 'ruby=', 'go=')) or p == 'composer' for p in packages))
            for family, hashes in [('go', native_toolchains.GO_SHA256),
                                   ('ruby', native_toolchains.RUBY_SHA256),
                                   ('php', native_toolchains.PHP_SHA256)]:
                version = recipe.get('toolchain', {}).get(family + '_version')
                if version:
                    self.assertEqual(len(hashes[version]), 64)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'archive'; artifact.write_bytes(b'fixture')
            digest = hashlib.sha256(b'fixture').hexdigest()
            with patch.object(native_toolchains.urllib.request, 'urlopen', side_effect=AssertionError('must not download')):
                self.assertEqual(native_toolchains.download_verified('https://fixture.invalid', artifact, digest), artifact)
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    native_toolchains.download_verified('https://fixture.invalid', artifact, '0' * 64)
            binary = Path(directory) / 'bin'; binary.mkdir()
            for tool in ('cc', 'c++', 'ar', 'ranlib'):
                (binary / ('x86_64-conda-linux-gnu-' + tool)).touch()
            compilers = native_toolchains.compiler_environment(directory)
            self.assertEqual(set(compilers), {'CC', 'CXX', 'AR', 'RANLIB'})
            self.assertTrue(compilers['CC'].endswith('/x86_64-conda-linux-gnu-cc'))

    def test_system_commands_are_converted_not_executed(self):
        extra, system, commands = preparation.setup_commands({
            'pre_install': ['apt-get update && apt-get install -y graphviz imagemagick',
                            "echo 'en_US UTF-8' > /etc/locale.gen", 'python -m pip install -e .'],
            'apt-pkgs': ['fixture-library-dev']})
        self.assertEqual(set(extra), {'graphviz', 'imagemagick'})
        self.assertEqual(set(system), {'fixture-library-dev', 'locales'})
        self.assertIn('locale -a', '\n'.join(commands))
        self.assertNotIn('apt-get', '\n'.join(commands))

    def test_tex_extensions_remain_explicit_system_prerequisites(self):
        packages = ['dvipng', 'texlive-luatex', 'texlive-latex-extra', 'texlive-fonts-recommended']
        extra, system, _ = preparation.setup_commands({'apt-pkgs': packages})
        self.assertEqual(system, packages)
        self.assertNotIn('texlive-core', extra)

    def test_shared_toolchain_rejects_stale_installer_before_any_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = root / 'toolchains/ruby-3.1.7'
            prefix = parent / 'prefix'; prefix.mkdir(parents=True)
            preparation.write_json(parent / 'manifest.json', {'installer_sha256': 'stale',
                'archive_sha256': native_toolchains.RUBY_SHA256['3.1.7'],
                'sha256': environment_fingerprint(prefix)})
            with self.assertRaisesRegex(ValueError, 'recipe changed'):
                native_toolchains.prepare_toolchain('ruby', '3.1.7', root, conda='not-executed',
                    run=lambda *args, **kwargs: self.fail('No install should run'))

    def test_original_control_requires_all_expected_failures_and_no_regressions(self):
        row = {'FAIL_TO_PASS': ['a', 'b'], 'PASS_TO_PASS': ['c'], 'eval_type': 'pass_and_fail'}
        controls = {'reference_patch': {'resolved': True, 'status': 'native_official_parser_report', 'parser_output_valid': True},
                    'original_base': {'resolved': False, 'status': 'native_official_parser_report',
                        'parser_output_valid': True,
                        'observed_expected_tests': {'a': 'FAILED', 'b': 'FAILED', 'c': 'PASSED'},
                        'official_report': {'tests_status': {'PASS_TO_PASS': {'failure': []}}}}}
        preparation.validate_controls(row, controls)
        controls['original_base']['observed_expected_tests']['b'] = 'PASSED'
        with self.assertRaisesRegex(RuntimeError, 'every FAIL_TO_PASS'):
            preparation.validate_controls(row, controls)
        controls['original_base']['observed_expected_tests']['b'] = None
        with self.assertRaisesRegex(RuntimeError, 'every expected'):
            preparation.validate_controls(row, controls)
        controls['original_base']['observed_expected_tests']['b'] = 'FAILED'
        controls['original_base']['official_report']['tests_status']['PASS_TO_PASS']['failure'] = ['c']
        with self.assertRaisesRegex(RuntimeError, 'regressed PASS_TO_PASS'):
            preparation.validate_controls(row, controls)

    def test_prefix_fingerprints_detect_environment_and_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = root / 'prefix'; source = root / 'source'
            prefix.mkdir(); source.mkdir()
            (prefix / 'package').write_text('original')
            (source / 'main.py').write_text('original')
            catalog = json.loads(preparation.RECIPES.read_text())
            key, recipe = next(iter(catalog['recipes'].items()))
            recipe_hash = preparation.canonical_hash(recipe)
            record = {'environment_path': str(prefix), 'source_path': str(source),
                      'base_commit': 'a' * 40, 'recipe_sha256': recipe_hash,
                      'environment_sha256': environment_fingerprint(prefix),
                      'source_sha256': environment_fingerprint(source),
                      'environment_fingerprint_kind': 'metadata_v1'}
            record['environment_id'] = preparation.canonical_hash({key: record[key] for key in
                ('environment_sha256', 'source_sha256', 'recipe_sha256')})
            preparation.write_json(root / 'preparation.json', {'status': 'ready', 'base_commit': 'a' * 40,
                'preparation_source_sha256': preparation.preparation_source_hashes(),
                'plan': {'dataset': key.split('/')[0], 'recipe': recipe, 'recipe_sha256': recipe_hash}})
            record['preparation_sha256'] = preparation.file_hash(root / 'preparation.json')
            preparation.verify_prepared_environment(record)
            with patch.object(preparation, 'preparation_source_hashes', return_value={'stale': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'adapter code changed'):
                    preparation.verify_prepared_environment(record)
            with patch.object(preparation, 'recipe_for', return_value={**recipe, 'install': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'recipe changed'):
                    preparation.verify_prepared_environment(record)
            (source / 'main.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                preparation.verify_prepared_environment(record)
            (prefix / 'package').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'environment changed'):
                preparation.verify_prepared_environment(record)


class ScriptTests(unittest.TestCase):
    def test_heredoc_test_data_remains_byte_identical(self):
        test_patch = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+'/testbed /root /tmp'\n"
        script = ("#!/bin/bash\nset -uxo pipefail\nsource /opt/miniconda3/bin/activate\n"
                  "conda activate testbed\ncd /testbed\ngit apply - <<'PATCH'\n" + test_patch +
                  "PATCH\n: '>>>>> Start Test Output'\nfalse\n: '>>>>> End Test Output'\n")
        adapted, changes = worker.adapt_eval_script(script, Path('/data/case/tree'),
                                                    Path('/data/case/prefix'), Path('/data/case/runtime'))
        self.assertIn("<<'PATCH'\n" + test_patch + "PATCH\n", adapted)
        self.assertIn('cd /data/case/tree', adapted)
        self.assertNotIn('conda activate', adapted)
        self.assertTrue(changes)
        with tempfile.TemporaryDirectory() as directory:
            script = "set -uxo pipefail\n: '>>>>> Start Test Output'\nfalse\n: '>>>>> End Test Output'\n"
            adapted, _ = worker.adapt_eval_script(script, Path(directory), Path(directory), Path(directory))
            result = run_bounded_command(['bash', '-c', adapted], timeout=10)
            self.assertEqual(result.returncode, 0)
            self.assertIn('>>>>> Test Exit Code: 1', result.output)
            self.assertEqual(result.output.count('>>>>> Start Test Output'), 1)
            self.assertEqual(result.output.count('>>>>> End Test Output'), 1)

    def test_unrecognized_script_or_shell_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'structure'):
            worker.adapt_eval_script('echo nothing\n', Path('/tree'), Path('/env'), Path('/tmp'))
        with self.assertRaisesRegex(ValueError, 'metacharacters'):
            worker.adapt_eval_script('', Path('/path with spaces'), Path('/env'), Path('/tmp'))

    def test_real_native_patch_and_test_with_isolated_fixture_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=source, text=True).strip()
            git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@local')
            (source / 'answer.txt').write_text('old\n')
            git('add', 'answer.txt'); git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            prefix = root / 'prefix'; (prefix / 'bin').mkdir(parents=True)
            row = {'instance_id': 'org__repo-1', 'repo': 'org/repo', 'version': '1', 'base_commit': base,
                   'eval_script': "set -uxo pipefail\ncd /testbed\n: '>>>>> Start Test Output'\n"
                                  "if grep -qx new answer.txt; then echo PASS; else echo FAIL; false; fi\n"
                                  ": '>>>>> End Test Output'\n"}
            patch_text = 'diff --git a/answer.txt b/answer.txt\n--- a/answer.txt\n+++ b/answer.txt\n@@ -1 +1 @@\n-old\n+new\n'
            def parser(row, log_path, patch):
                return {'resolved': '\nPASS\n' in log_path.read_text(), 'status': 'native_official_parser_report'}
            def clone(original, target, *, timeout):
                shutil.copytree(original, target)
            request = {'task': row, 'prepared': {'source_path': str(source), 'environment_path': str(prefix),
                       'environment_id': 'e' * 64}, 'patch': patch_text, 'verify_prepared': False, 'timeout': 20}
            report = worker.run_case(request, root / 'grade', environment_cloner=clone, parser=parser)
            self.assertIs(report['resolved'], True)
            self.assertEqual((source / 'answer.txt').read_text(), 'old\n')
            self.assertFalse((root / 'grade/workspace').exists())
            self.assertFalse((root / 'grade/environment').exists())
            self.assertEqual(report['patch_sha256'], hashlib.sha256(patch_text.encode()).hexdigest())
            self.assertNotEqual(report['original_eval_script_sha256'], report['adapted_eval_script_sha256'])
            request['patch'] = ''
            baseline = worker.run_case(request, root / 'baseline', environment_cloner=clone, parser=parser)
            self.assertIs(baseline['resolved'], False)

    def test_worker_failure_remains_unknown_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {'task': {'instance_id': 'x', 'base_commit': 'a' * 40, 'eval_script': ''},
                       'prepared': {'source_path': str(root / 'missing'), 'environment_path': str(root / 'env'),
                                    'environment_id': 'e' * 64}, 'verify_prepared': False}
            report = worker.run_case(request, root / 'case')
            self.assertIsNone(report['resolved'])
            self.assertEqual(report['status'], 'native_infrastructure_error')
            self.assertTrue((root / 'case/report.json').exists())


if __name__ == '__main__':
    unittest.main()
