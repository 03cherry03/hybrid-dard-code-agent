"""Conservative measured runtime estimates; no I/O, downloads or scheduling.

Calibration must use the target generation budget and provide completed, graded
case rows plus complete phase timings. Estimates use each suite's slowest observed
case across all calibration GPUs, and each decoder's slowest measured load/warmup.
They are planning estimates, never a guarantee that unseen tasks finish in time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math


BACKENDS = frozenset(('ar', 'standard', 'dard'))
COMMON_FLAGS = ('--tokens', '--quant', '--temperature', '--top-p', '--repairs',
                '--diagnostics', '--warmup')
BACKEND_FLAGS = {'ar': ('--ar-model', '--ar-revision'),
                 'standard': ('--standard-steps', '--model', '--revision'),
                 'dard': ('--block-size', '--dard-steps', '--tau-c', '--tau-u',
                          '--transfer-limit', '--model', '--revision')}


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} number')
    return float(value)


def _backend_list(backends):
    values = list(backends)
    if not values or len(set(values)) != len(values) or any(b not in BACKENDS for b in values):
        raise ValueError('backends must contain distinct supported decoder names')
    return values


def _signature(worker, backend):
    command = worker.get('command', [])
    if not isinstance(command, list) or not command:
        raise ValueError('Every calibration and target worker must provide its frozen command')
    values = {}
    for flag in COMMON_FLAGS + BACKEND_FLAGS[backend]:
        if command.count(flag) > 1:
            raise ValueError(f'Duplicate calibration/target option: {flag}')
        if flag in command:
            index = command.index(flag)
            if index + 1 == len(command):
                raise ValueError(f'Missing calibration/target option value: {flag}')
            values[flag] = str(command[index+1])
        else:
            values[flag] = None
    if values['--tokens'] is None:
        raise ValueError('Full-budget calibration requires an explicit --tokens value')
    return values


def _counts(worker):
    keys = worker.get('task_keys')
    if not isinstance(keys, list) or not keys:
        raise ValueError('Each target worker must have nonempty task_keys')
    if any(not isinstance(k, (list, tuple)) or len(k) != 3 for k in keys):
        raise ValueError('task_keys must contain dataset/language/task identity triples')
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError('Duplicate task identity within worker')
    counts = Counter(f'{dataset}/{language}' for dataset, language, _ in keys)
    if 'by_dataset_language' in worker and dict(counts) != worker['by_dataset_language']:
        raise ValueError('Target suite counts disagree with frozen task_keys')
    return counts


def _case_key(row):
    return (row.get('dataset'), row.get('language'), row.get('task_id'),
            row.get('decoder'), row.get('repeat'), row.get('seed'))


def estimate_seconds(calibration_plan, calibration_shards, target_workers, backends,
                     repeats, safety_factor=1.5):
    """Estimate parallel wall seconds for a complete new target campaign.

    ``calibration_shards`` is the four_gpu audit shape: a list of dictionaries
    containing ``rows`` and ``phases``, in calibration worker order. Target
    workers use the frozen four_gpu plan shape (commands and task_keys).
    All calibration rows for requested decoders must be present and graded.
    Missing suite measurements are rejected rather than borrowing another
    suite's runtime. In-progress/resumed work requires its remaining task keys.
    """
    backends = _backend_list(backends)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 3:
        raise ValueError('repeats must be an integer between 1 and 3')
    safety_factor = _number(safety_factor, 'safety_factor', positive=True)
    if safety_factor < 1:
        raise ValueError('safety_factor must be at least 1')
    calibration_workers = calibration_plan.get('workers', [])
    if not calibration_workers or len(calibration_workers) != len(calibration_shards):
        raise ValueError('Calibration shard count must match its frozen worker plan')
    if len(target_workers) != len(calibration_workers):
        raise ValueError('Calibration and target must use the same number of concurrent workers')
    signatures = {b: _signature(calibration_workers[0], b) for b in backends}
    for worker in [*calibration_workers, *target_workers]:
        for backend in backends:
            if _signature(worker, backend) != signatures[backend]:
                raise ValueError(f'Calibration generation budget/settings differ from target for {backend}')
    all_target_keys = [tuple(k) for w in target_workers for k in w.get('task_keys', [])]
    if len(set(all_target_keys)) != len(all_target_keys):
        raise ValueError('Target workers overlap in task identities')
    per_case = defaultdict(list)
    overhead = defaultdict(list)
    calibration_count = 0
    for worker, shard in zip(calibration_workers, calibration_shards, strict=True):
        _counts(worker)
        seeds = worker.get('task_seeds')
        if not isinstance(seeds, list) or len(seeds) != len(worker['task_keys']):
            raise ValueError('Calibration worker lacks frozen per-task seeds')
        expected = {(*key, backend, repeat, seed)
                    for key, task_seeds in zip(worker['task_keys'], seeds, strict=True)
                    for repeat, seed in enumerate(task_seeds) for backend in backends}
        if not expected:
            raise ValueError('Calibration contains no planned samples')
        rows = [r for r in shard.get('rows', []) if r.get('decoder') in backends]
        actual = [_case_key(row) for row in rows]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError('Calibration must have exact completed coverage, without duplicate attempts')
        phase_rows = defaultdict(list)
        for row in rows:
            if (row.get('case_completed') is not True or type(row.get('final_pass')) is not bool
                    or row.get('measurement_complete') is not True or row.get('failure_kind')):
                raise ValueError('Calibration contains failed infrastructure, ungraded or incomplete measurements')
            seconds = (_number(row.get('agent_wall_s'), 'agent_wall_s')
                       + _number(row.get('offline_grading_wall_s'), 'offline_grading_wall_s'))
            if seconds <= 0:
                raise ValueError('Calibration case work must be positive')
            phase_rows[(row['decoder'], row.get('phase_id'))].append((row, seconds))
            calibration_count += 1
        seen_phases = set()
        for phase in shard.get('phases', []):
            backend = phase.get('decoder')
            if backend not in backends:
                continue
            key = (backend, phase.get('phase_id'))
            if key in seen_phases or key not in phase_rows or phase.get('completed') is not True:
                raise ValueError('Calibration phase is duplicated, empty or incomplete')
            seen_phases.add(key)
            load = _number(phase.get('model_load_s'), 'model_load_s')
            warmup = _number(phase.get('warmup_s'), 'warmup_s')
            phase_wall = _number(phase.get('phase_wall_s'), 'phase_wall_s')
            observed = phase_rows[key]
            overhead[backend].append(load + warmup)
            # Resource polling/reporting around cases is outside agent/grader
            # clocks. Scale this observed residual per case rather than treating
            # a tiny calibration phase's entire overhead as one fixed charge.
            residual_per_case = max(0., phase_wall-load-warmup-sum(s for _, s in observed))/len(observed)
            for row, seconds in observed:
                group = f'{backend}/{row["dataset"]}/{row["language"]}'
                per_case[group].append(seconds + residual_per_case)
        if seen_phases != set(phase_rows):
            raise ValueError('Calibration rows lack complete phase timing records')
    maxima = {group: max(values) for group, values in per_case.items()}
    phase_overhead = {backend: max(overhead[backend]) for backend in backends}
    workers = []
    for index, worker in enumerate(target_workers):
        counts = _counts(worker)
        by_backend = {}
        for backend in backends:
            missing = [group for group in counts if f'{backend}/{group}' not in maxima]
            if missing:
                raise ValueError(f'Unmeasured calibration suite(s) for {backend}: {", ".join(missing)}')
            work = sum(count * maxima[f'{backend}/{group}'] for group, count in counts.items())
            by_backend[backend] = repeats * (work + phase_overhead[backend])
        base = sum(by_backend.values())
        workers.append({'index': worker.get('index', index), 'suite_task_counts': dict(counts),
                        'base_by_backend_seconds': by_backend, 'base_seconds': base,
                        'estimated_seconds': base * safety_factor})
    base = max(worker['base_seconds'] for worker in workers)
    return {'schema_version': 1, 'backends': backends, 'repeats': repeats,
            'safety_factor': safety_factor, 'base_wall_seconds': base,
            'estimated_wall_seconds': base * safety_factor,
            'workers': workers, 'max_observed_case_seconds': maxima,
            'max_load_warmup_seconds_per_phase': phase_overhead,
            'calibration_cases': calibration_count,
            'timing_basis': 'agent_wall_s + offline_grading_wall_s + observed phase residual per case',
            'estimate_is_guarantee': False,
            'limitations': ['Unseen tasks and repairs can exceed observed maxima.',
                           'External final analysis and startup require a separate time reserve.',
                           'A hard absolute deadline must be enforced independently.']}


def select_repeats(calibration_plan, calibration_shards, target_workers, backends,
                   available_seconds, max_repeats=3, safety_factor=1.5):
    """Select 1..3 complete repeats, or report zero if even one does not fit.

    The caller subtracts analysis/cleanup reserves from available_seconds.
    Returning zero does not authorize reducing task coverage or dropping a
    required backend. It signals that the configured full campaign is infeasible.
    """
    available = _number(available_seconds, 'available_seconds')
    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int) or not 1 <= max_repeats <= 3:
        raise ValueError('max_repeats must be an integer between 1 and 3')
    estimates = []
    for repeats in range(max_repeats, 0, -1):
        estimate = estimate_seconds(calibration_plan, calibration_shards, target_workers,
                                    backends, repeats, safety_factor)
        estimates.append(estimate)
        if estimate['estimated_wall_seconds'] <= available:
            return {'selected_repeats': repeats, 'fits_estimate': True,
                    'available_seconds': available, 'estimate': estimate,
                    'estimate_is_guarantee': False}
    return {'selected_repeats': 0, 'fits_estimate': False,
            'available_seconds': available, 'estimate': estimates[-1],
            'estimate_is_guarantee': False,
            'reason': 'Even one complete repeat exceeds the measured conservative time estimate'}


def resolve_deadline(*, now_unix, hours=40, existing=None):
    """Create/preserve absolute budget state; restarting never resets the clock."""
    now = _number(now_unix, 'now_unix')
    hours = _number(hours, 'hours', positive=True)
    if existing is None:
        started, deadline = now, now + hours * 3600
    else:
        started = _number(existing.get('started_unix'), 'started_unix')
        deadline = _number(existing.get('deadline_unix'), 'deadline_unix')
        if deadline <= started:
            raise ValueError('Existing deadline must be after its original start')
        # A smaller requested budget may shorten a deadline; no resume argument
        # can silently extend the previously persisted absolute limit.
        deadline = min(deadline, started + hours * 3600)
    return {'schema_version': 1, 'started_unix': started, 'deadline_unix': deadline,
            'remaining_seconds': max(0., deadline-now), 'expired': now >= deadline}
