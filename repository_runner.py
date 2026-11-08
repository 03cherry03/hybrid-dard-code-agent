"""Offline repository-agent runs and a separate, pinned SWE-bench grading stage.

Examples are available with ``python repository_runner.py --help``. Model
generation runs in a disposable worker so a GPU forward cannot overrun the
task deadline indefinitely. No package, model, or repository is downloaded here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import importlib.metadata
import math
import multiprocessing
import os
import platform
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

from dard_decoder import DARDConfig
from experiment_shards import SHARDING_ALGORITHM, select_shard
from models import DEFAULT_AR_MODEL, DEFAULT_DIFFUSION_MODEL
from repository_agent import (AgentConfig, LocalSandbox, RepoTask, canonical_hash,
    file_hash, load_repo_tasks, run_bounded_command, run_repository_task)

HARNESS_VERSION = '5.0.1'
HARNESS_COMMIT = '87ab1f6ced28f75ba73ca899dc759b019310944a'
VALID_SUBMISSIONS = {'submitted', 'step_budget', 'tool_budget', 'deadline'}


def _backend_worker(connection, decoder, kwargs):
    backend = None
    try:
        from models import create_backend
        backend = create_backend(decoder, **kwargs)
        connection.send({'kind':'ready', 'metadata':backend.metadata(), 'pid':os.getpid()})
        while True:
            request = connection.recv()
            if request['kind']=='close':
                break
            try:
                value = backend.generate_messages(request['messages'], request['decoder'], **request['kwargs'])
                connection.send({'kind':'result','value':value})
            except Exception as exc:
                connection.send({'kind':'error','timeout':isinstance(exc,TimeoutError),
                                 'error':type(exc).__name__+': '+str(exc),
                                 'generation_diagnostics':getattr(exc,'generation_diagnostics',None)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({'kind':'error','error':type(exc).__name__+': '+str(exc)})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if backend is not None:
            backend.close()
        connection.close()


class ResidentBackend:
    """One serial model resident in a killable process, including CUDA state."""
    def __init__(self, decoder, *, load_timeout=600, worker_target=_backend_worker, **kwargs):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_target, args=(child,decoder,kwargs), daemon=True)
        self.closed = False
        self.process.start()
        child.close()
        try:
            ready = self._receive(load_timeout)
            if ready.get('kind')!='ready':
                raise RuntimeError(ready.get('error','Model worker did not initialize'))
            self._metadata, self.worker_pid = ready['metadata'], ready['pid']
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        if timeout <= 0 or not self.connection.poll(timeout):
            self.close(force=True)
            exc = TimeoutError('Model worker exceeded the hard wall-clock deadline and was terminated')
            exc.generation_diagnostics = {'measurement_complete':False,
                'diagnostics':{'complete':False,'status':'unavailable_worker_terminated'}}
            raise exc
        try:
            return self.connection.recv()
        except EOFError as exc:
            self.close(force=True)
            error = RuntimeError('Model worker exited without a response')
            error.generation_diagnostics = {'measurement_complete':False,
                'diagnostics':{'complete':False,'status':'unavailable_worker_exited'}}
            raise error from exc

    def generate_messages(self, messages, decoder, **kwargs):
        if self.closed:
            raise RuntimeError('Model worker is closed')
        timeout = kwargs.get('max_time') or 600
        if kwargs.get('deadline') is not None:
            timeout = min(timeout,kwargs['deadline']-time.monotonic())
        if timeout<=0:
            raise TimeoutError('Global deadline reached before generation')
        self.connection.send({'kind':'generate','messages':messages,'decoder':decoder,'kwargs':kwargs})
        response = self._receive(timeout)
        if response['kind']=='error':
            exc = (TimeoutError if response.get('timeout') else RuntimeError)(response['error'])
            if response.get('generation_diagnostics') is not None:
                exc.generation_diagnostics = response['generation_diagnostics']
            raise exc
        return response['value']

    def metadata(self):
        return dict(self._metadata)

    def close(self, force=False):
        if self.closed and not self.process.is_alive():
            return
        self.closed = True
        if self.process.is_alive() and not force:
            try:
                self.connection.send({'kind':'close'})
            except (OSError, EOFError):
                pass
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()
        if self.process.is_alive():
            raise RuntimeError('Model worker could not be stopped; refusing another model residency')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def freeze_text(path: Path, text: str):
    """Create once, or verify the existing immutable artifact byte-for-byte."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:
            raise ValueError(f'Refusing to modify frozen artifact: {path}')
        return
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8')
    temporary.chmod(0o444)
    temporary.replace(path)


def row_key(row):
    return (row['dataset'],row['instance_id'],row['decoder'],row['repeat'])


def gpu_snapshot(metadata, worker_pid):
    """Record other compute users on the selected GPU without invoking a shell."""
    gpu_uuid = metadata.get('gpu_uuid')
    try:
        devices = run_bounded_command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],
                                      timeout=5,max_output_bytes=10000)
        if (devices.returncode or devices.timed_out or devices.truncated or
                gpu_uuid not in {line.strip() for line in devices.output.splitlines()}):
            return {'verified':False,'other_pids':[]}
        probe = run_bounded_command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], timeout=5, max_output_bytes=10000)
        if probe.returncode or probe.timed_out or probe.truncated or not gpu_uuid:
            return {'verified':False,'other_pids':[]}
        users=[]
        for line in probe.output.splitlines():
            fields=[value.strip() for value in line.split(',')]
            if len(fields)==2 and fields[0]==gpu_uuid and fields[1].isdigit():
                if int(fields[1])!=worker_pid:
                    users.append(int(fields[1]))
        return {'verified':True,'other_pids':sorted(set(users))}
    except OSError:
        return {'verified':False,'other_pids':[]}


def runtime_fingerprint():
    versions={}
    for name in ('torch','transformers','accelerate','bitsandbytes','safetensors','huggingface_hub'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions,
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}


def freeze_backend_identity(output, decoder, metadata):
    keys=('model_name','resolved_revision','tokenizer_revision','tokenizer_vocab_sha256',
          'chat_template_sha256','model_config_sha256','parameter_count','quant','compute_dtype',
          'attention_implementation')
    identity={key:metadata.get(key) for key in keys}
    family='ar' if decoder=='ar' else 'diffusion'
    freeze_text(output/'identities'/f'{family}.json',json.dumps(identity,sort_keys=True,indent=2)+'\n')
    hardware={key:metadata.get(key) for key in ('device','gpu_uuid','gpu_name','gpu_total_gib',
                                             'gpu_capability','cuda_build','versions')}
    freeze_text(output/'identities'/'hardware.json',json.dumps(hardware,sort_keys=True,indent=2)+'\n')
    return all(identity.get(key) is not None for key in keys) and bool(hardware.get('gpu_uuid'))


def _case_seed(task, seed, repeat):
    return (int(canonical_hash([task.dataset,task.instance_id])[:8],16)+seed+repeat*1000003) % (2**31)


def _error_record(task, decoder, repeat, seed, error):
    return dict(dataset=task.dataset,instance_id=task.instance_id,language=task.language,
        repo=task.repo,base_commit=task.base_commit,
        decoder=decoder,repeat=repeat,seed=seed,task_sha256=canonical_hash(asdict(task)),
        environment_id=task.environment_id,execution_backend='native',status='error',error=error,model_patch='',
        total_wall_s=0.0,agent_wall_s=0.0,generation_wall_s=0.0,tool_wall_s=0.0,setup_wall_s=0.0,
        cleanup_wall_s=0.0,steps=0,tool_calls=0,output_tokens=0,model_forward_passes=0,
        attention_fallback_steps=0,deadline_violation=False,speed_eligible=False,
        measurement_complete=False,diagnostics={'complete':False,'status':'generation_not_started'},trace=[])


def run_experiment(args, *, backend_factory=ResidentBackend, sandbox_factory=LocalSandbox,
                   gpu_probe=gpu_snapshot):
    tasks,sources=load_repo_tasks([Path(p) for p in args.tasks],limit=args.limit,
        instance_ids=set(args.instance_ids) if args.instance_ids else None)
    if not tasks:
        raise ValueError('No repository tasks selected')
    full_selection_sha256=canonical_hash([asdict(task) for task in tasks])
    full_selection_count=len(tasks)
    tasks=select_shard(tasks,args.num_shards,args.shard_index)
    decoders=args.decoders.split(',')
    if len(set(decoders))!=len(decoders) or any(d not in ('ar','standard','dard') for d in decoders):
        raise ValueError('decoders must be a comma-separated subset of ar,standard,dard')
    if args.repeats<1 or args.warmup<1 or args.steps<1:
        raise ValueError('repeats, warmup and steps must be positive')
    if any(not math.isfinite(value) or value<=0 for value in (args.model_load_timeout,args.warmup_timeout)):
        raise ValueError('Model load and warmup timeouts must be positive and finite')
    config=AgentConfig(max_steps=args.max_steps,max_tool_calls=args.max_tool_calls,
        max_context_chars=args.max_context_chars,max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,max_tool_output_chars=args.max_tool_output_chars,
        task_timeout_s=args.task_timeout,tool_timeout_s=args.tool_timeout,
        memory_bytes=args.memory_bytes,cpu_seconds=args.cpu_seconds,max_processes=args.max_processes,work_root=args.work_root)
    config.validate()
    dard=DARDConfig(block_size=args.block_size,steps_per_block=args.dard_steps,
        temperature=args.temperature,top_p=args.top_p,allow_attention_fallback=False,
        diagnostics=args.diagnostics,max_step_records=args.max_step_records)
    dard.validate()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    task_set_hash=canonical_hash([asdict(task) for task in tasks])
    run_config={'schema_version':1,'task_set_sha256':task_set_hash,'sources':sources,
        'num_shards':args.num_shards,'shard_index':args.shard_index,
        'sharding':{'algorithm':SHARDING_ALGORITHM,'num_shards':args.num_shards,
                    'shard_index':args.shard_index,'full_selection_sha256':full_selection_sha256,
                    'full_selection_count':full_selection_count},
        'tasks':[asdict(task) for task in tasks],'decoders':decoders,'repeats':args.repeats,
        'seed':args.seed,'warmup':args.warmup,'steps':args.steps,'agent_config':asdict(config),
        'dard_config':asdict(dard),'temperature':args.temperature,'top_p':args.top_p,
        'diagnostics_mode':dard.diagnostics,'instrumentation_affects_timing':dard.diagnostics!='summary',
        'device':args.device,'quant':args.quant,'ar_model':args.ar_model,
        'diffusion_model':args.diffusion_model,'ar_revision':args.ar_revision,
        'diffusion_revision':args.diffusion_revision,
        'runtime':runtime_fingerprint(),
        'source_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in
            ('repository_agent.py','repository_runner.py','models.py','dard_decoder.py','experiment_shards.py','evaluation.py','local_execution.py','local_repository_preparation.py','local_swe_worker.py','native_toolchains.py','native_repository_recipes.json')}}
    freeze_text(output/'run.json',json.dumps(run_config,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    rows=read_jsonl(output/'results.jsonl')
    done={row_key(row) for row in rows}
    if len(done)!=len(rows):
        raise ValueError('Duplicate case results; refusing ambiguous resume')
    for repeat in range(args.repeats):
        order=decoders[repeat%len(decoders):]+decoders[:repeat%len(decoders)]
        shuffled=list(tasks)
        random.Random(args.seed+repeat).shuffle(shuffled)
        for position,decoder in enumerate(order):
            remaining=[task for task in shuffled if (task.dataset,task.instance_id,decoder,repeat) not in done]
            if not remaining:
                continue
            backend=None
            metadata={}
            phase_error=None
            def load():
                load_started=time.monotonic()
                value=backend_factory(decoder,model_name=args.ar_model if decoder=='ar' else args.diffusion_model,
                    quant=args.quant,device=args.device,
                    revision=args.ar_revision if decoder=='ar' else args.diffusion_revision,
                    local_files_only=True,load_timeout=args.model_load_timeout)
                warmup=[]
                try:
                    for index in range(args.warmup):
                        warmed=value.generate_messages([
                            {'role':'system','content':'You are a coding assistant.'},
                            {'role':'user','content':'Reply with exactly {"action":"finish"}.'}],decoder,
                            seed=args.seed+index,num_new_tokens=min(config.max_output_tokens,32),
                            steps=args.steps,dard_config=dard,temperature=args.temperature,top_p=args.top_p,
                            max_time=args.warmup_timeout,max_input_tokens=config.max_input_tokens)
                        warmup.append({k:v for k,v in warmed.items() if k!='text'})
                    info=value.metadata()
                    info['identity_verified']=freeze_backend_identity(output,decoder,info)
                    append_jsonl(output/'phases.jsonl',dict(repeat=repeat,decoder=decoder,
                        order_position=position,metadata=info,warmup=warmup,
                        model_load_and_warmup_wall_s=time.monotonic()-load_started))
                    return value,info
                except BaseException:
                    value.close()
                    raise
            try:
                try:
                    backend,metadata=load()
                except Exception as exc:
                    phase_error=type(exc).__name__+': '+str(exc)
                for task in remaining:
                    seed=_case_seed(task,args.seed,repeat)
                    before={'verified':False,'other_pids':[]}
                    after=before
                    if phase_error:
                        row=_error_record(task,decoder,repeat,seed,'Model load/warmup failed: '+phase_error)
                    else:
                        if getattr(backend,'closed',False):
                            try:
                                # A prior hard timeout must release its process
                                # before any replacement model is loaded.
                                backend.close()
                                backend,metadata=load()
                            except Exception as exc:
                                phase_error=type(exc).__name__+': '+str(exc)
                        if phase_error:
                            row=_error_record(task,decoder,repeat,seed,'Model reload failed: '+phase_error)
                        else:
                            before=gpu_probe(metadata,getattr(backend,'worker_pid',None))
                            row=run_repository_task(task,backend,decoder,config=config,seed=seed,
                                steps=args.steps,dard_config=dard,temperature=args.temperature,top_p=args.top_p,
                                sandbox_factory=sandbox_factory)
                            after=gpu_probe(metadata,getattr(backend,'worker_pid',None))
                    row.update(repeat=repeat,model_metadata=metadata,task_set_sha256=task_set_hash,
                               gpu_before=before,gpu_after=after)
                    row.setdefault('diagnostics_mode',dard.diagnostics)
                    row.setdefault('instrumentation_affects_timing',dard.diagnostics!='summary')
                    row['speed_eligible'] &= all(p['verified'] and not p['other_pids'] for p in (before,after))
                    row['speed_eligible'] &= bool(metadata.get('identity_verified'))
                    trace=row.pop('trace',[])
                    append_jsonl(output/'traces.jsonl',dict(key=list(row_key(row)),trace=trace))
                    append_jsonl(output/'results.jsonl',row)
                    rows.append(row)
                    done.add(row_key(row))
                    print(json.dumps({'decoder':decoder,'repeat':repeat,'instance_id':task.instance_id,
                                      'status':row['status'],'total_wall_s':row['total_wall_s']}),flush=True)
            finally:
                if backend is not None:
                    backend.close()
    freeze_predictions(output,run_config,rows)
    report=build_report(rows,read_jsonl(output/'grades.jsonl'))
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def freeze_predictions(output, config, rows):
    by_key={row_key(row):row for row in rows}
    entries=[]
    for dataset in sorted({task['dataset'] for task in config['tasks']}):
        tasks=[task for task in config['tasks'] if task['dataset']==dataset]
        for decoder in config['decoders']:
            for repeat in range(config['repeats']):
                selected=[by_key[(dataset,task['instance_id'],decoder,repeat)] for task in tasks]
                model=config['ar_model'] if decoder=='ar' else config['diffusion_model']
                label=model.replace('/','__')+'__'+decoder
                predictions=[{'instance_id':row['instance_id'],'model_patch':row['model_patch'],
                              'model_name_or_path':label} for row in selected]
                relative=Path('predictions')/dataset/f'{decoder}.repeat{repeat}.jsonl'
                path=output/relative
                freeze_text(path,''.join(json.dumps(pred,ensure_ascii=False,sort_keys=True)+'\n' for pred in predictions))
                entries.append({'dataset':dataset,'decoder':decoder,'repeat':repeat,
                    'path':str(relative),'sha256':file_hash(path),'count':len(predictions),
                    'instance_ids':[pred['instance_id'] for pred in predictions],
                    'environments':{row['instance_id']:row.get('environment_id') for row in selected}})
    manifest={'task_set_sha256':config['task_set_sha256'],'run_sha256':file_hash(output/'run.json'),
              'results_sha256':file_hash(output/'results.jsonl'),'submissions':entries}
    freeze_text(output/'submissions.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def _percentile(values, quantile):
    if not values:
        return None
    return sorted(values)[min(len(values)-1,math.ceil(quantile*len(values))-1)]


def build_report(rows, grades):
    graded={row_key(row):row for row in grades}
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['language'],row['decoder'])].append(row)
        groups[(row['dataset'],'all',row['decoder'])].append(row)
    summaries=[]
    for (dataset,language,decoder),cases in sorted(groups.items()):
        known=[graded[row_key(row)].get('resolved') for row in cases if row_key(row) in graded]
        resolved=sum(value is True for value in known)
        unknown=len(cases)-sum(value is not None for value in known)
        elapsed=sum(row['total_wall_s'] for row in cases)
        summaries.append({'dataset':dataset,'language':language,'decoder':decoder,'cases':len(cases),
            'status_counts':dict(Counter(row['status'] for row in cases)),
            'resolved':resolved,'ungraded_or_error':unknown,
            'resolved_fraction_all_submissions':resolved/len(cases),
            'resolved_fraction_is_lower_bound':unknown>0,
            'solved_per_hour':resolved*3600/elapsed if elapsed>0 else None,
            'solved_per_hour_is_lower_bound':unknown>0,
            'median_total_wall_s':statistics.median(row['total_wall_s'] for row in cases),
            'p95_total_wall_s':_percentile([row['total_wall_s'] for row in cases],.95),
            'total_wall_s':elapsed,
            'generation_wall_s':sum(row['generation_wall_s'] for row in cases),
            'tool_wall_s':sum(row['tool_wall_s'] for row in cases),
            'output_tokens':sum(row['output_tokens'] for row in cases),
            'model_forward_passes':sum(row['model_forward_passes'] for row in cases),
            'diagnostics_modes':dict(Counter(row.get('diagnostics_mode','unknown') for row in cases)),
            'instrumented_cases':sum(bool(row.get('instrumentation_affects_timing')) for row in cases),
            'incomplete_measurement_cases':sum(row.get('measurement_complete') is False for row in cases),
            'invalid_action_count':sum(row.get('invalid_action_count',0) for row in cases),
            'tool_error_count':sum(row.get('tool_error_count',0) for row in cases),
            'context_trim_events':sum(row.get('context_trim_events',0) for row in cases),
            'attention_fallback_steps':sum(row.get('attention_fallback_steps',0) for row in cases)})
    comparisons=[]
    indexed={(r['dataset'],r['instance_id'],r['repeat'],r['decoder']):r for r in rows}
    for dataset in sorted({r['dataset'] for r in rows}):
        for baseline,candidate in (('ar','standard'),('ar','dard'),('standard','dard')):
            pairs=[]
            for key,left in indexed.items():
                if key[0]!=dataset or key[3]!=baseline:
                    continue
                right=indexed.get((*key[:3],candidate))
                if right is not None:
                    pairs.append((left,right))
            if not pairs:
                continue
            issues=set()
            joint=[]
            quality_deltas=[]
            all_ratios=[]
            for left,right in pairs:
                if left['task_sha256']!=right['task_sha256'] or left['seed']!=right['seed']:
                    issues.add('mismatched_task_or_seed')
                if left.get('initial_messages_sha256')!=right.get('initial_messages_sha256'):
                    issues.add('mismatched_initial_prompt')
                if left.get('environment_id')!=right.get('environment_id') or not left.get('environment_id'):
                    issues.add('missing_or_mismatched_prepared_environment')
                if not left.get('speed_eligible') or not right.get('speed_eligible'):
                    issues.add('generation_or_environment_not_speed_eligible')
                if any(r.get('instrumentation_affects_timing') or r.get('diagnostics_mode','summary')!='summary'
                       for r in (left,right)):
                    issues.add('diagnostic_instrumentation_affects_timing')
                if any(r.get('measurement_complete') is False for r in (left,right)):
                    issues.add('incomplete_generation_measurements')
                a=graded.get(row_key(left),{}).get('resolved')
                b=graded.get(row_key(right),{}).get('resolved')
                if a is None or b is None:
                    issues.add('missing_official_outcome')
                else:
                    quality_deltas.append(int(b)-int(a))
                if right['total_wall_s']>0:
                    all_ratios.append(left['total_wall_s']/right['total_wall_s'])
                    if a is True and b is True:
                        joint.append(left['total_wall_s']/right['total_wall_s'])
            comparisons.append({'dataset':dataset,'name':candidate+'_vs_'+baseline,
                'paired_cases':len(pairs),'issues':sorted(issues),
                'median_total_speed_ratio_all_pairs':statistics.median(all_ratios) if all_ratios else None,
                'jointly_resolved_pairs':len(joint),
                'median_total_speed_ratio_jointly_resolved':statistics.median(joint) if joint else None,
                'paired_resolve_rate_delta':statistics.mean(quality_deltas) if quality_deltas else None,
                'speed_claim_validated':False,
                'note':'Descriptive paired results; no statistical noninferiority or speed claim is inferred.'})
    return {'schema_version':1,'summaries':summaries,'comparisons':comparisons,
            'denominator_policy':'All attempted submissions retained; missing/error grades are unknown, never failures.',
            'timing_policy':'Total includes native checkout and offline environment clone, agent loop, and cleanup; model load/warmup and final grading are separate.'}


def verify_harness(python, *, command_runner=run_bounded_command):
    script=("import sys,json; sys.path.insert(0,"+repr(str(Path(__file__).resolve().parent))+"); "
            "from local_swe_worker import load_official_api; print(json.dumps(load_official_api()['identity']))")
    result=command_runner([str(python),'-c',script],timeout=30,max_output_bytes=10000)
    if result.returncode or result.timed_out:
        raise RuntimeError('Pinned SWE-bench harness Python is unavailable: '+result.output[:1000])
    identity=json.loads(result.output)
    if identity!={'version':HARNESS_VERSION,'commit':HARNESS_COMMIT}:
        raise ValueError(f'Expected SWE-bench {HARNESS_VERSION} at {HARNESS_COMMIT}; found {identity}')
    return identity


def grade_experiment(args, *, command_runner=run_bounded_command):
    from concurrent.futures import ThreadPoolExecutor
    if (args.max_workers<1 or args.test_timeout<1 or not math.isfinite(args.grading_timeout)
            or args.grading_timeout<=0):
        raise ValueError('Grading worker/time limits must be positive and finite')
    output=Path(args.output).resolve()
    manifest=json.loads((output/'submissions.json').read_text())
    if file_hash(output/'run.json')!=manifest['run_sha256'] or file_hash(output/'results.jsonl')!=manifest['results_sha256']:
        raise ValueError('Frozen run/results checksum changed')
    config=json.loads((output/'run.json').read_text())
    for name,digest in config.get('source_sha256',{}).items():
        if file_hash(Path(__file__).with_name(name))!=digest:
            raise ValueError('Experiment code changed since submissions were frozen: '+name)
    source=next((s for s in config['sources'] if s['dataset']==args.dataset),None)
    if source is None or not source.get('sealed_sha256'):
        raise ValueError('Grading requires a prepared dataset manifest with a sealed checksum')
    sealed=Path(args.sealed).resolve()
    if file_hash(sealed)!=source['sealed_sha256']:
        raise ValueError('Sealed evaluation snapshot checksum mismatch')
    identity=verify_harness(args.harness_python,command_runner=command_runner)
    sealed_rows={row['instance_id']:row for row in read_jsonl(sealed)}
    rows=read_jsonl(output/'results.jsonl')
    indexed={row_key(row):row for row in rows}
    tasks={(row['dataset'],row['instance_id']):row for row in config['tasks']}
    existing=read_jsonl(output/'grades.jsonl')
    done={row_key(row) for row in existing if isinstance(row.get('resolved'),bool)}
    all_grades=list(existing)
    deadline=time.monotonic()+args.grading_timeout
    for entry in manifest['submissions']:
        if entry['dataset']!=args.dataset:
            continue
        predictions_path=output/entry['path']
        if file_hash(predictions_path)!=entry['sha256']:
            raise ValueError('Frozen predictions checksum changed')
        predictions=read_jsonl(predictions_path)
        run_id=f"native_{args.dataset}_{entry['decoder']}_r{entry['repeat']}_{entry['sha256'][:10]}"
        def grade_one(prediction):
            key=(args.dataset,prediction['instance_id'],entry['decoder'],entry['repeat'])
            if key in done:
                return None
            row=indexed[key]
            grade={'dataset':key[0],'instance_id':key[1],'decoder':key[2],'repeat':key[3],
                   'resolved':None,'status':'missing_native_report','harness':identity,'run_id':run_id,
                   'prediction_sha256':entry['sha256'],'backend':'native',
                   'environment_id':entry['environments'].get(key[1]),'report_sha256':None}
            if row['status'] not in VALID_SUBMISSIONS:
                grade['status']='agent_error'
                return grade
            if not prediction['model_patch']:
                grade.update(resolved=False,status='deadline' if row['status']=='deadline' else 'empty_patch')
                return grade
            task=tasks[(args.dataset,key[1])]
            if not grade['environment_id'] or grade['environment_id']!=task.get('environment_id'):
                raise ValueError('Submitted task has no matching prepared native environment')
            sealed_row=sealed_rows[key[1]]
            if sealed_row['base_commit']!=task['base_commit']:
                raise ValueError('Sealed base commit differs from the agent task')
            remaining=deadline-time.monotonic()
            if remaining<=0:
                grade['status']='grading_deadline'
                return grade
            parent=output/'grading'/run_id/key[1]
            parent.mkdir(parents=True,exist_ok=True)
            attempt=1
            while (parent/f'attempt-{attempt}').exists():
                attempt+=1
            grading_dir=parent/f'attempt-{attempt}'
            grading_dir.mkdir(); grading_dir.chmod(0o700)
            request={'task':sealed_row,'prepared':task,'patch':prediction['model_patch'],
                     'timeout':min(args.test_timeout,remaining),'verify_prepared':True}
            private=grading_dir/'request.private.json'
            private.write_text(json.dumps(request,ensure_ascii=False)+'\n'); private.chmod(0o600)
            command=[str(Path(args.harness_python).resolve()),str(Path(__file__).with_name('local_swe_worker.py')),
                     '--request',str(private),'--output',str(grading_dir)]
            execution=command_runner(command,timeout=remaining,max_output_bytes=2*1024*1024,cwd=grading_dir)
            (grading_dir/'worker.log').write_text(execution.output)
            (grading_dir/'invocation.json').write_text(json.dumps({'command':command,'harness':identity,
                'backend':'native','returncode':execution.returncode,'timed_out':execution.timed_out,
                'wall_s':execution.wall_s,'output_truncated':execution.truncated,
                'evaluation_sha256':file_hash(private),'prediction_sha256':entry['sha256']},indent=2)+'\n')
            report_path=grading_dir/'report.json'
            if report_path.exists() and not execution.timed_out and not execution.truncated:
                native=json.loads(report_path.read_text())
                if (native.get('instance_id')!=key[1] or native.get('environment_id')!=grade['environment_id']
                        or native.get('patch_sha256')!=__import__('hashlib').sha256(prediction['model_patch'].encode()).hexdigest()):
                    raise ValueError('Native report provenance does not match its frozen submission')
                value=native.get('resolved')
                if value is not None and not isinstance(value,bool):
                    raise ValueError('resolved must be boolean or null')
                if execution.returncode and isinstance(value,bool):
                    value=None
                    native['status']='native_worker_abnormal_exit'
                grade.update(resolved=value,status=native.get('status','invalid_native_report'),
                             report_sha256=file_hash(report_path),native_report_path=str(report_path.relative_to(output)))
            elif execution.timed_out:
                grade['status']='native_grading_timeout'
            elif execution.truncated:
                grade['status']='native_worker_log_truncated'
            return grade
        # Only final grading is parallelized here; no model/agent can consume
        # results and change its already-frozen submission.
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for grade in pool.map(grade_one,predictions):
                if grade is not None:
                    append_jsonl(output/'grades.jsonl',grade)
                    all_grades.append(grade)
                    if isinstance(grade.get('resolved'),bool):
                        done.add(row_key(grade))
        if file_hash(predictions_path)!=entry['sha256']:
            raise ValueError('Predictions changed during native grading')
    report=build_report(rows,all_grades)
    report['evaluation_backend']='native_pinned_official_parsers'
    report['environment_equivalence']='Native adaptation; official environment equivalence is not claimed.'
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check-data',help='Read only public tasks and verify prepared checksums')
    check.add_argument('--tasks',nargs='+',required=True)
    check.add_argument('--limit',type=int,default=0)
    check.add_argument('--instance-ids',nargs='*')
    run=commands.add_parser('run',help='Run agents with cached models and prepared native repositories')
    run.add_argument('--tasks',nargs='+',required=True)
    run.add_argument('--output',required=True)
    run.add_argument('--limit',type=int,default=0)
    run.add_argument('--instance-ids',nargs='*')
    run.add_argument('--num-shards',type=int,default=1,
                     help='Split the selected tasks across independent workers before loading models')
    run.add_argument('--shard-index',type=int,default=0,
                     help='Zero-based shard index; use a distinct output and fixed GPU per shard')
    run.add_argument('--decoders',default='ar,standard,dard')
    run.add_argument('--repeats',type=int,default=3)
    run.add_argument('--seed',type=int,default=42)
    run.add_argument('--warmup',type=int,default=1)
    run.add_argument('--warmup-timeout',type=float,default=120)
    run.add_argument('--model-load-timeout',type=float,default=600)
    run.add_argument('--device',default='cuda:0')
    run.add_argument('--quant',choices=['bf16','4bit'],default='bf16')
    run.add_argument('--ar-model',default=DEFAULT_AR_MODEL)
    run.add_argument('--diffusion-model',default=DEFAULT_DIFFUSION_MODEL)
    run.add_argument('--ar-revision',default='c03e6d358207e414f1eca0bb1891e29f1db0e242')
    run.add_argument('--diffusion-revision','--revision',default='4fdd4580064ca5d11808069ce78f88d068753c96')
    run.add_argument('--local-files-only',action='store_true',default=True,
                     help='Compatibility flag; repository runs always require locally cached models')
    run.add_argument('--steps',type=int,default=128)
    run.add_argument('--block-size',type=int,default=128)
    run.add_argument('--dard-steps',type=int,default=192)
    run.add_argument('--diagnostics',choices=('summary','steps','profile'),default='summary',
                     help='Steps/profile affect timing and are excluded from speed comparisons')
    run.add_argument('--max-step-records',type=int,default=256)
    run.add_argument('--temperature',type=float,default=0)
    run.add_argument('--top-p',type=float,default=1)
    run.add_argument('--max-steps',type=int,default=30)
    run.add_argument('--max-tool-calls',type=int,default=30)
    run.add_argument('--max-context-chars',type=int,default=30000)
    run.add_argument('--max-input-tokens',type=int,default=8192)
    run.add_argument('--max-output-tokens',type=int,default=1024)
    run.add_argument('--max-tool-output-chars',type=int,default=10000)
    run.add_argument('--task-timeout',type=float,default=600)
    run.add_argument('--tool-timeout',type=float,default=90)
    run.add_argument('--memory-bytes',type=int)
    run.add_argument('--cpu-seconds',type=int)
    run.add_argument('--max-processes',type=int)
    run.add_argument('--work-root',default='/data/hybrid-dard/work')
    grade=commands.add_parser('grade',help='Native grading with pinned official SWE parsers; no agents run')
    grade.add_argument('--output',required=True)
    grade.add_argument('--dataset',required=True)
    grade.add_argument('--sealed',required=True)
    grade.add_argument('--harness-python',required=True)
    grade.add_argument('--max-workers',type=int,default=1)
    grade.add_argument('--test-timeout',type=int,default=1800)
    grade.add_argument('--grading-timeout',type=float,default=86400)
    return parser


def main(argv=None):
    args=make_parser().parse_args(argv)
    if args.command=='check-data':
        tasks,sources=load_repo_tasks([Path(p) for p in args.tasks],limit=args.limit,
            instance_ids=set(args.instance_ids) if args.instance_ids else None)
        result={'count':len(tasks),'datasets':dict(Counter(t.dataset for t in tasks)),
                'languages':dict(Counter(t.language for t in tasks)),
                'task_set_sha256':canonical_hash([asdict(t) for t in tasks]),'sources':sources}
    elif args.command=='run':
        result=run_experiment(args)
    else:
        result=grade_experiment(args)
    print(json.dumps(result,indent=2,sort_keys=True))
    if args.command=='run':
        return int(any(row['status'] not in VALID_SUBMISSIONS for row in read_jsonl(Path(args.output)/'results.jsonl')))
    if args.command=='grade':
        latest={row_key(row):row for row in read_jsonl(Path(args.output)/'grades.jsonl')}
        return int(any(row.get('resolved') is None for row in latest.values() if row['dataset']==args.dataset))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
