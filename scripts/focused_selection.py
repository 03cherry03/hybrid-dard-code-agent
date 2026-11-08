"""Identity-only multilingual cohorts with disjoint calibration and nested subsets.

Every selected source problem retains Python, Rust, JavaScript and C++ variants.
Python is generated once: HumanEval+ supplies the official expanded Python grade
without shrinking HumanEval translation coverage; base MBPP retains more common
translated problems than MBPP+. These are family/language-stratified cohorts, not
claims that translated and expanded Python test suites have identical difficulty.
No model output, correctness result, prompt length or reference code is inspected.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json


FAMILIES = ('humaneval', 'mbpp')
LANGUAGES = ('py', 'rs', 'js', 'cpp')
DEFAULT_PYTHON_SUITES = {'humaneval': 'humaneval_plus', 'mbpp': 'mbpp'}
ALLOWED_PYTHON_SUITES = {'humaneval': ('humaneval', 'humaneval_plus'),
                         'mbpp': ('mbpp', 'mbpp_plus')}
ALGORITHM = 'identity_hash_stratified_nested_sources_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _suite_map(python_suites):
    python_suites = dict(DEFAULT_PYTHON_SUITES if python_suites is None else python_suites)
    if set(python_suites) != set(FAMILIES):
        raise ValueError('Python suite policy must name both humaneval and mbpp families')
    if any(python_suites[f] not in ALLOWED_PYTHON_SUITES[f] for f in FAMILIES):
        raise ValueError('Choose exactly one supported Python suite for each family')
    return {family: {'py': python_suites[family], **{
            language: ('multiple_' if family == 'humaneval' else 'multiple_mbpp_') + language
            for language in LANGUAGES if language != 'py'}} for family in FAMILIES}


def _stratified_order(groups):
    """Nested proportional order, with both families present from prefix two."""
    selected = Counter()
    ordered = []
    # Always retain both task families even at the smallest valid evaluation.
    for family in FAMILIES:
        ordered.append(groups[family][0])
        selected[family] += 1
    while any(selected[f] < len(groups[f]) for f in FAMILIES):
        family = min((f for f in FAMILIES if selected[f] < len(groups[f])),
                     key=lambda f: (selected[f] / len(groups[f]), FAMILIES.index(f)))
        ordered.append(groups[family][selected[family]])
        selected[family] += 1
    return ordered


def build_selection(tasks, *, seed=42, calibration_per_family=2, python_suites=None):
    """Build a serializable selection catalog from already prepared Task objects.

    Calibration reserves the first identity-hashed sources within each family;
    evaluation uses the remaining sources, with all four language variants kept
    together. ``calibration_per_family`` must leave at least one evaluation source
    in each family. Extra/duplicate Python benchmark variants are deliberately
    excluded by the explicit suite policy rather than generated again.
    """
    _integer(seed, 'seed', 0)
    _integer(calibration_per_family, 'calibration_per_family', 1)
    suite_map = _suite_map(python_suites)
    lookup = {f: {language: {} for language in LANGUAGES} for f in FAMILIES}
    all_keys = set()
    for task in tasks:
        family, language = task.family, task.language
        if family not in lookup or language not in LANGUAGES or task.dataset != suite_map[family][language]:
            continue
        source_id = task.source_id
        prefix = 'HumanEval/' if family == 'humaneval' else 'MBPP/'
        if (not isinstance(source_id, str) or not source_id.startswith(prefix)
                or not source_id[len(prefix):].isdigit()):
            raise ValueError(f'Invalid canonical source identity for {family}: {source_id!r}')
        key = (task.dataset, task.language, task.task_id)
        if key in all_keys or source_id in lookup[family][language]:
            raise ValueError('Duplicate selected task or source/language identity')
        all_keys.add(key)
        lookup[family][language][source_id] = list(key)
    families = {}
    calibration, evaluation_by_family = [], {}
    for family in FAMILIES:
        common = set.intersection(*(set(lookup[family][language]) for language in LANGUAGES))
        if len(common) <= calibration_per_family:
            raise ValueError(f'{family} lacks enough four-language common sources for disjoint calibration and evaluation')
        ordered = sorted(common, key=lambda source: (_digest([ALGORITHM, seed, family, source]), source))
        groups = [{'family': family, 'source_id': source,
                   'task_keys': [lookup[family][language][source] for language in LANGUAGES]}
                  for source in ordered]
        calibration.extend(groups[:calibration_per_family])
        evaluation_by_family[family] = groups[calibration_per_family:]
        families[family] = {
            'available_sources_by_language': {language: len(lookup[family][language]) for language in LANGUAGES},
            'common_source_count': len(common), 'calibration_source_count': calibration_per_family,
            'evaluation_source_count': len(common)-calibration_per_family,
            'excluded_unmatched_by_language': {language: len(lookup[family][language])-len(common) for language in LANGUAGES},
        }
    result = {'schema_version': 1, 'algorithm': ALGORITHM, 'seed': seed,
              'languages': list(LANGUAGES), 'families': families, 'suite_map': suite_map,
              'suites': [suite_map[f][language] for f in FAMILIES for language in LANGUAGES],
              'calibration_per_family': calibration_per_family,
              'calibration_sources': calibration,
              'evaluation_sources': _stratified_order(evaluation_by_family),
              'selection_uses_only_identities': True,
              'calibration_disjoint_from_evaluation_by_source_id': True,
              'generation_repeats': 1,
              'notes': ['Every source retains all four language variants and all selected backends.',
                        'One Python suite is generated per family; duplicate base/plus generation is excluded.',
                        'Nested evaluation prefixes are fixed before observing model outcomes.',
                        'Results describe the selected common-source cohort, not all original benchmark tasks.',
                        'Expanded Python and translated test harnesses differ; report family/language groups separately.']}
    result['catalog_sha256'] = _digest(result)
    return result


def selection_manifest(catalog, *, split='evaluation', source_limit=None):
    """Return a frozen JSON task-key manifest for a nested evaluation prefix.

    Calibration always uses its complete reserved set. Evaluation source_limit
    ranges from two to its maximum; every permitted prefix retains both families
    and four variants per source. Omitting the limit selects all available sources.
    """
    if split not in ('calibration', 'evaluation'):
        raise ValueError('split must be calibration or evaluation')
    body = {key: value for key, value in catalog.items() if key != 'catalog_sha256'}
    if catalog.get('algorithm') != ALGORITHM or catalog.get('catalog_sha256') != _digest(body):
        raise ValueError('Selection catalog changed or has an unsupported algorithm')
    sources = catalog[f'{split}_sources']
    if source_limit is None:
        source_limit = len(sources)
    _integer(source_limit, 'source_limit', 2)
    if source_limit > len(sources) or (split == 'calibration' and source_limit != len(sources)):
        raise ValueError('Source limit exceeds the evaluation pool or changes reserved calibration')
    selected = sources[:source_limit]
    keys = [list(key) for source in selected for key in source['task_keys']]
    by_family = {family: [source['source_id'] for source in selected if source['family'] == family]
                 for family in FAMILIES}
    if not all(by_family.values()):
        raise ValueError('Every selected cohort must retain both families')
    result = {'schema_version': 1, 'algorithm': ALGORITHM, 'split': split,
              'catalog_sha256': catalog['catalog_sha256'], 'seed': catalog['seed'],
              'suites': list(catalog['suites']), 'languages': list(LANGUAGES),
              'source_count': len(selected), 'source_ids_by_family': by_family,
              'task_count': len(keys), 'task_keys': keys,
              'by_dataset_language': dict(Counter(f'{key[0]}/{key[1]}' for key in keys)),
              'generation_repeats': 1, 'selection_uses_only_identities': True,
              'calibration_disjoint_from_evaluation_by_source_id': True}
    result['task_keys_sha256'] = _digest(keys)
    result['manifest_sha256'] = _digest(result)
    return result


def select_tasks(tasks, manifest):
    """Resolve a verified manifest against prepared tasks, preserving its order."""
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if (manifest.get('algorithm') != ALGORITHM or manifest.get('manifest_sha256') != _digest(body)
            or manifest.get('task_keys_sha256') != _digest(manifest.get('task_keys'))):
        raise ValueError('Task selection manifest changed or has an unsupported algorithm')
    keys = [tuple(key) for key in manifest['task_keys']]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Selection manifest must contain unique nonempty task keys')
    requested = set(keys)
    lookup = {}
    for task in tasks:
        key = (task.dataset, task.language, task.task_id)
        if key in requested:
            if key in lookup:
                raise ValueError('Prepared data contains duplicate selected task identity')
            lookup[key] = task
    missing = requested - lookup.keys()
    if missing:
        raise ValueError(f'Prepared data lacks {len(missing)} selected task identities: {sorted(missing)[:3]}')
    return [lookup[key] for key in keys]
