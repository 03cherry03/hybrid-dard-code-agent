"""Identity-only disjoint multilingual selection contracts, without downloads."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from scripts.focused_selection import build_selection, selection_manifest, select_tasks


def tasks_fixture():
    tasks = []
    for family, prefix in [('humaneval', 'HumanEval'), ('mbpp', 'MBPP')]:
        for source in range(10):
            for language in ('py', 'rs', 'js', 'cpp'):
                suite = (('humaneval_plus' if family == 'humaneval' else 'mbpp') if language == 'py'
                         else ('multiple_' if family == 'humaneval' else 'multiple_mbpp_')+language)
                # Omit a different translation in each family: no partial source
                # may enter a paired cohort even if its Python variant exists.
                if source == 9 and language == ('rs' if family == 'humaneval' else 'cpp'):
                    continue
                tasks.append(SimpleNamespace(dataset=suite, language=language, family=family,
                             task_id=f'{suite}-{source}', source_id=f'{prefix}/{source}',
                             prompt='not inspected', test_source='not inspected'))
            # Duplicate Python benchmark variants must never be generated too.
            tasks.append(SimpleNamespace(dataset='humaneval' if family == 'humaneval' else 'mbpp_plus',
                         language='py', family=family, task_id=f'duplicate-{family}-{source}',
                         source_id=f'{prefix}/{source}', prompt='excluded', test_source='excluded'))
    return tasks


class FocusedSelectionTests(unittest.TestCase):
    def test_common_sources_keep_exactly_four_languages_and_one_python_variant(self):
        tasks = tasks_fixture()
        catalog = build_selection(tasks)
        self.assertEqual(catalog['families']['humaneval']['common_source_count'], 9)
        self.assertEqual(catalog['families']['mbpp']['common_source_count'], 9)
        self.assertEqual(catalog['families']['humaneval']['evaluation_source_count'], 7)
        self.assertEqual(catalog['suites'], ['humaneval_plus', 'multiple_rs', 'multiple_js', 'multiple_cpp',
                                          'mbpp', 'multiple_mbpp_rs', 'multiple_mbpp_js', 'multiple_mbpp_cpp'])
        evaluation = selection_manifest(catalog)
        self.assertEqual(evaluation['task_count'], 56)
        selected = select_tasks(tasks, evaluation)
        groups = {}
        for task in selected:
            groups.setdefault((task.family, task.source_id), []).append(task.language)
        self.assertTrue(all(languages == ['py', 'rs', 'js', 'cpp'] for languages in groups.values()))
        self.assertTrue(all(task.dataset not in ('humaneval', 'mbpp_plus') for task in selected))
        self.assertTrue(all(not task.source_id.endswith('/9') for task in selected))
        self.assertEqual(evaluation['generation_repeats'], 1)

    def test_calibration_and_evaluation_disjoint_at_source_and_variant_levels(self):
        catalog = build_selection(tasks_fixture())
        calibration = selection_manifest(catalog, split='calibration')
        evaluation = selection_manifest(catalog)
        self.assertEqual(calibration['source_count'], 4)
        self.assertEqual(calibration['task_count'], 16)
        for family in ('humaneval', 'mbpp'):
            self.assertEqual(len(calibration['source_ids_by_family'][family]), 2)
            self.assertTrue(set(calibration['source_ids_by_family'][family]).isdisjoint(
                            evaluation['source_ids_by_family'][family]))
        self.assertTrue({tuple(k) for k in calibration['task_keys']}.isdisjoint(
                        tuple(k) for k in evaluation['task_keys']))

    def test_all_nested_limits_keep_both_families_and_all_languages(self):
        tasks = tasks_fixture()
        catalog = build_selection(tasks)
        previous = set()
        for limit in range(2, 15):
            manifest = selection_manifest(catalog, source_limit=limit)
            selected = select_tasks(tasks, manifest)
            keys = {tuple(k) for k in manifest['task_keys']}
            self.assertTrue(previous.issubset(keys))
            self.assertEqual(len(keys), limit*4)
            self.assertEqual({(t.family, t.language) for t in selected},
                             {(f, l) for f in ('humaneval', 'mbpp') for l in ('py', 'rs', 'js', 'cpp')})
            counts = [len(v) for v in manifest['source_ids_by_family'].values()]
            self.assertLessEqual(max(counts)-min(counts), 1)
            previous = keys

    def test_input_order_prompt_reference_and_outcomes_cannot_change_selection(self):
        tasks = tasks_fixture()
        expected = build_selection(tasks)
        for task in tasks:
            task.prompt = 'changed prompt'
            task.test_source = 'changed reference/test'
            task.final_pass = True
            task.agent_wall_s = .000001
        self.assertEqual(build_selection(list(reversed(tasks))), expected)
        self.assertNotEqual(build_selection(tasks, seed=43)['evaluation_sources'], expected['evaluation_sources'])

    def test_manifest_and_catalog_tampering_or_missing_data_is_rejected(self):
        tasks = tasks_fixture()
        catalog = build_selection(tasks)
        changed = deepcopy(catalog)
        changed['evaluation_sources'].pop()
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            selection_manifest(changed)
        manifest = selection_manifest(catalog)
        changed = deepcopy(manifest)
        changed['task_keys'].pop()
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            select_tasks(tasks, changed)
        selected = select_tasks(tasks, manifest)
        with self.assertRaisesRegex(ValueError, 'lacks 1'):
            select_tasks(selected[1:], manifest)
        with self.assertRaisesRegex(ValueError, 'duplicate selected'):
            select_tasks([*selected, selected[0]], manifest)

    def test_invalid_limits_and_non_disjoint_small_pools_are_rejected(self):
        tasks = tasks_fixture()
        catalog = build_selection(tasks)
        for limit in (0, 1, 15, True, 2.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                selection_manifest(catalog, source_limit=limit)
        with self.assertRaises(ValueError):
            selection_manifest(catalog, split='calibration', source_limit=2)
        with self.assertRaisesRegex(ValueError, 'enough four-language'):
            build_selection(tasks, calibration_per_family=9)
        with self.assertRaises(ValueError):
            build_selection(tasks, calibration_per_family=0)
        with self.assertRaisesRegex(ValueError, 'Duplicate selected'):
            build_selection([*tasks, tasks[0]])

    def test_python_policy_is_explicit_and_alternative_variants_are_not_combined(self):
        tasks = tasks_fixture()
        alternative = build_selection(tasks, python_suites={'humaneval': 'humaneval', 'mbpp': 'mbpp_plus'})
        manifest = selection_manifest(alternative)
        selected = select_tasks(tasks, manifest)
        self.assertEqual({t.dataset for t in selected if t.language == 'py'}, {'humaneval', 'mbpp_plus'})
        with self.assertRaises(ValueError):
            build_selection(tasks, python_suites={'humaneval': 'humaneval_plus'})
        with self.assertRaises(ValueError):
            build_selection(tasks, python_suites={'humaneval': 'humaneval_plus', 'mbpp': 'bigcodebench_instruct'})


if __name__ == '__main__':
    unittest.main()
