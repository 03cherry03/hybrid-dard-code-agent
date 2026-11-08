"""Offline orchestration integration; never loads models or installs environments."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from repository_agent import CommandResult, file_hash
import repository_runner as runner


ENVIRONMENT='e'*64
BASE='a'*40
PATCH='diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new\n'


def stalled_worker(connection, decoder, kwargs):
    connection.send({'kind':'ready','metadata':{},'pid':os.getpid()})
    connection.recv()
    time.sleep(60)


def failed_worker(connection, decoder, kwargs):
    connection.send({'kind':'ready','metadata':{},'pid':os.getpid()})
    connection.recv()
    connection.send({'kind':'error','timeout':False,'error':'late mock failure',
        'generation_diagnostics':{'model_forward_passes':11,
            'diagnostics':{'complete':False,'phase':'sampling'}}})
    connection.recv()


class FakeBackend:
    active=0
    events=[]
    messages=[]

    def __init__(self,decoder,**kwargs):
        if FakeBackend.active:
            raise AssertionError('Concurrent model residency')
        FakeBackend.active+=1
        FakeBackend.events.append(('load',decoder))
        self.decoder=decoder
        self.kwargs=kwargs
        self.closed=False
        self.worker_pid=1

    def metadata(self):
        return {'model_name':self.kwargs['model_name'],'resolved_revision':'fixed-commit',
            'tokenizer_revision':'fixed-tokenizer','tokenizer_vocab_sha256':'vocab',
            'chat_template_sha256':'template','model_config_sha256':'config',
            'parameter_count':7000000000,'quant':self.kwargs['quant'],'compute_dtype':'bfloat16',
            'attention_implementation':'sdpa','device':'cuda:0','gpu_uuid':'GPU-fixture',
            'gpu_name':'Fixture','gpu_total_gib':24,'gpu_capability':[8,9],
            'cuda_build':'fixture','versions':{'torch':'fixture'}}

    def generate_messages(self,messages,decoder,**kwargs):
        FakeBackend.messages.append(json.loads(json.dumps(messages)))
        is_warmup=not messages[1]['content'].startswith('Repository:')
        if is_warmup:
            FakeBackend.events.append(('warmup',decoder))
            action={'action':'finish'}
        else:
            assert kwargs['max_time']>0 and kwargs['deadline']>time.monotonic()
            assert kwargs['max_input_tokens']==8192
            index=(len(messages)-2)//2
            action=[{'action':'read','path':'main.py'},
                    {'action':'edit','path':'main.py','old':'old','new':'new'},
                    {'action':'run','command':'python -m unittest'},
                    {'action':'finish'}][index]
        return {'text':json.dumps(action),'output_tokens':10,'model_forward_passes':2,
                'gpu_peak_gib':1,'attention_fallback_steps':0}

    def close(self):
        if not self.closed:
            self.closed=True
            FakeBackend.active-=1
            FakeBackend.events.append(('close',self.decoder))


class FakeSandbox:
    instances=[]

    def __init__(self,task,config):
        self.task=task
        self.config=config
        self.environment_id=ENVIRONMENT
        self.closed=False
        self.actions=[]
        FakeSandbox.instances.append(self)

    def start(self,*,timeout):
        assert timeout>0

    def action(self,action,*,timeout):
        self.actions.append(action)
        return {'ok':True,'output':'old' if action['action']=='read' else 'completed'}

    def patch(self,*,timeout):
        return PATCH

    def close(self):
        self.closed=True


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.tasks=self.root/'tasks.jsonl'
        self.sealed=self.root/'sealed.jsonl'
        self.output=self.root/'run'
        public=[]
        sealed=[]
        for number in range(2):
            row={'dataset':'fixture','instance_id':f'org__repo-{number}',
                 'repo':'org/repo','base_commit':BASE,'problem_statement':'Fix the public issue.',
                 'language':'py','environment_id':ENVIRONMENT}
            public.append(row)
            sealed.append({**row,'patch':'GOLD_SECRET',
                           'test_patch':'TEST_SECRET','eval_script':'SEALED_SCRIPT'})
        self.tasks.write_text(''.join(json.dumps(row)+'\n' for row in public))
        self.sealed.write_text(''.join(json.dumps(row)+'\n' for row in sealed))
        (self.root/'manifest.json').write_text(json.dumps({'dataset':'fixture','count':2,
            'tasks_sha256':file_hash(self.tasks),'sealed_sha256':file_hash(self.sealed)}))
        FakeBackend.active=0
        FakeBackend.events=[]
        FakeBackend.messages=[]
        FakeSandbox.instances=[]

    def tearDown(self):
        self.temp.cleanup()

    def args(self,extra=()):
        return runner.make_parser().parse_args(['run','--tasks',str(self.tasks),
            '--output',str(self.output),'--repeats','1','--max-steps','5',*extra])

    def run_mock(self,args=None):
        return runner.run_experiment(args or self.args(),backend_factory=FakeBackend,
            sandbox_factory=FakeSandbox,gpu_probe=lambda *args:{'verified':True,'other_pids':[]})

    def grade_args(self):
        return runner.make_parser().parse_args(['grade','--output',str(self.output),
            '--dataset','fixture','--sealed',str(self.sealed),'--harness-python','/fixture/python'])

    def test_serial_rotated_runs_shared_seed_and_immutable_submission(self):
        report=self.run_mock(self.args(['--repeats','2']))
        self.assertEqual(FakeBackend.active,0)
        self.assertEqual([event[1] for event in FakeBackend.events if event[0]=='load'],
                         ['ar','standard','dard','standard','dard','ar'])
        self.assertEqual(sum(event[0]=='warmup' for event in FakeBackend.events),6)
        self.assertEqual(len(FakeSandbox.instances),12)
        self.assertTrue(all(box.closed for box in FakeSandbox.instances))
        rows=runner.read_jsonl(self.output/'results.jsonl')
        self.assertEqual(len(rows),12)
        self.assertTrue(all(row['speed_eligible'] for row in rows))
        for repeat in range(2):
            for iid in ('org__repo-0','org__repo-1'):
                cases=[row for row in rows if row['repeat']==repeat and row['instance_id']==iid]
                self.assertEqual(len({row['seed'] for row in cases}),1)
                self.assertEqual(len({row['initial_messages_sha256'] for row in cases}),1)
        self.assertNotIn('GOLD_SECRET',json.dumps(FakeBackend.messages))
        self.assertNotIn('TEST_SECRET',json.dumps(FakeBackend.messages))
        self.assertNotIn('SEALED_SCRIPT',json.dumps([asdict(box.task) for box in FakeSandbox.instances]))
        manifest=json.loads((self.output/'submissions.json').read_text())
        self.assertEqual(len(manifest['submissions']),6)
        for entry in manifest['submissions']:
            prediction_path=self.output/entry['path']
            self.assertEqual(prediction_path.stat().st_mode & 0o222,0)
            self.assertEqual(file_hash(prediction_path),entry['sha256'])
            self.assertEqual(set(runner.read_jsonl(prediction_path)[0]),
                             {'instance_id','model_patch','model_name_or_path'})
        self.assertIn('dard_vs_standard',{row['name'] for row in report['comparisons']})
        self.assertTrue(all(not row['speed_claim_validated'] for row in report['comparisons']))
        loads=len(FakeBackend.events)
        self.run_mock(self.args(['--repeats','2']))
        self.assertEqual(len(FakeBackend.events),loads)

    def test_family_identity_change_is_a_recorded_error(self):
        original=FakeBackend.metadata
        def metadata(backend):
            result=original(backend)
            if backend.decoder=='dard':
                result['tokenizer_vocab_sha256']='changed'
            return result
        with patch.object(FakeBackend,'metadata',metadata):
            self.run_mock()
        rows=runner.read_jsonl(self.output/'results.jsonl')
        dard=[row for row in rows if row['decoder']=='dard']
        self.assertTrue(all(row['status']=='error' for row in dard))
        self.assertTrue(all(row['base_commit']==BASE for row in dard))
        self.assertEqual(FakeBackend.active,0)

    def test_shards_cover_full_selection_without_splitting_decoder_pairs(self):
        template=json.loads(self.tasks.read_text().splitlines()[0])
        public=[{**template,'instance_id':f'org__repo-{i}','language':('py','cpp','rs')[i%3]}
                for i in range(9)]
        self.tasks.write_text(''.join(json.dumps(row)+'\n' for row in public))
        manifest=json.loads((self.root/'manifest.json').read_text())
        manifest.update(count=len(public),tasks_sha256=file_hash(self.tasks))
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        selected=[]
        full_hashes=set()
        shard_sizes=[]
        for shard in range(4):
            self.output=self.root/f'run-{shard}'
            self.run_mock(self.args(['--num-shards','4','--shard-index',str(shard)]))
            config=json.loads((self.output/'run.json').read_text())
            self.assertEqual(config['sharding']['full_selection_count'],len(public))
            self.assertEqual(config['num_shards'],4)
            self.assertEqual(config['shard_index'],shard)
            self.assertIn('experiment_shards.py',config['source_sha256'])
            full_hashes.add(config['sharding']['full_selection_sha256'])
            ids={task['instance_id'] for task in config['tasks']}
            selected.extend(ids)
            shard_sizes.append(len(ids))
            rows=runner.read_jsonl(self.output/'results.jsonl')
            for iid in ids:
                cases=[row for row in rows if row['instance_id']==iid]
                self.assertEqual({row['decoder'] for row in cases},{'ar','standard','dard'})
                self.assertEqual(len(cases),3)
                self.assertEqual(len({row['seed'] for row in cases}),1)
        self.assertEqual(len(full_hashes),1)
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual(set(selected),{row['instance_id'] for row in public})
        self.assertLessEqual(max(shard_sizes)-min(shard_sizes),1)

    def test_shard_resume_is_immutable_and_identical_resume_does_not_reload(self):
        args=self.args(['--num-shards','2','--shard-index','0'])
        self.run_mock(args)
        loads=len(FakeBackend.events)
        self.run_mock(args)
        self.assertEqual(len(FakeBackend.events),loads)
        original=(self.output/'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'frozen artifact'):
            self.run_mock(self.args(['--num-shards','2','--shard-index','1']))
        self.assertEqual((self.output/'run.json').read_bytes(),original)
        self.assertEqual(len(FakeBackend.events),loads)

    def test_invalid_or_empty_shard_fails_before_backend_creation(self):
        for count,index in ((0,0),(-1,0),(2,-1),(2,2),(4,3)):
            with self.subTest(count=count,index=index),self.assertRaises(ValueError):
                self.run_mock(self.args(['--num-shards',str(count),'--shard-index',str(index)]))
        self.assertEqual(FakeBackend.events,[])
        self.assertFalse(self.output.exists())

    def fake_grader(self,*,write_second=True):
        def command(argv,**kwargs):
            if '-c' in argv:
                return CommandResult(0,json.dumps({'version':runner.HARNESS_VERSION,'commit':runner.HARNESS_COMMIT}))
            self.assertTrue(argv[1].endswith('local_swe_worker.py'))
            private=Path(argv[argv.index('--request')+1])
            self.assertEqual(private.stat().st_mode & 0o777,0o600)
            request=json.loads(private.read_text())
            self.assertEqual(request['prepared']['environment_id'],ENVIRONMENT)
            self.assertTrue(request['verify_prepared'])
            self.assertIn('TEST_SECRET',private.read_text())
            if request['task']['instance_id'].endswith('-1') and not write_second:
                return CommandResult(1,'fixture missing report')
            report=Path(argv[argv.index('--output')+1])/'report.json'
            report.write_text(json.dumps({'instance_id':request['task']['instance_id'],
                'resolved':True,'environment_id':ENVIRONMENT,'status':'native_official_parser_report',
                'patch_sha256':__import__('hashlib').sha256(request['patch'].encode()).hexdigest()}))
            return CommandResult(0,'native fixture grading\n',wall_s=.2)
        return command

    def test_native_grade_pins_environment_and_missing_outcome_is_retryable_unknown(self):
        self.run_mock()
        prompt_count=len(FakeBackend.messages)
        report=runner.grade_experiment(self.grade_args(),command_runner=self.fake_grader(write_second=False))
        self.assertEqual(len(FakeBackend.messages),prompt_count)
        grades=runner.read_jsonl(self.output/'grades.jsonl')
        self.assertEqual(sum(row['resolved'] is True for row in grades),3)
        self.assertEqual(sum(row['resolved'] is None for row in grades),3)
        self.assertTrue(all(row['resolved_fraction_is_lower_bound'] for row in report['summaries']))
        runner.grade_experiment(self.grade_args(),command_runner=self.fake_grader())
        grades=runner.read_jsonl(self.output/'grades.jsonl')
        self.assertEqual(len(grades),9)
        self.assertTrue(all(row['resolved'] is True for row in {runner.row_key(g):g for g in grades}.values()))

    def test_locked_model_load_failure_stays_unknown_during_grading(self):
        def failed(*args,**kwargs):
            raise RuntimeError('fixture model unavailable')
        runner.run_experiment(self.args(),backend_factory=failed,sandbox_factory=FakeSandbox)
        runner.grade_experiment(self.grade_args(),command_runner=self.fake_grader())
        grades=runner.read_jsonl(self.output/'grades.jsonl')
        self.assertEqual(len(grades),6)
        self.assertTrue(all(row['status']=='agent_error' and row['resolved'] is None for row in grades))

    def test_tampered_predictions_rejected_before_harness(self):
        self.run_mock()
        manifest=json.loads((self.output/'submissions.json').read_text())
        path=self.output/manifest['submissions'][0]['path']
        path.chmod(0o644)
        path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'predictions checksum'):
            runner.grade_experiment(self.grade_args(),command_runner=self.fake_grader())

    def test_unknown_gpu_uuid_is_not_verified(self):
        with patch.object(runner,'run_bounded_command',return_value=CommandResult(0,'GPU-other\n')):
            self.assertFalse(runner.gpu_snapshot({'gpu_uuid':'GPU-fixture'},1)['verified'])

    def test_selected_gpu_only_external_users_count(self):
        responses=[CommandResult(0,'GPU-fixture\nGPU-other\n'),
                   CommandResult(0,'GPU-fixture, 1\nGPU-fixture, 2\nGPU-other, 3\n')]
        with patch.object(runner,'run_bounded_command',side_effect=responses):
            self.assertEqual(runner.gpu_snapshot({'gpu_uuid':'GPU-fixture'},1),
                             {'verified':True,'other_pids':[2]})

    def test_hard_generation_timeout_terminates_worker(self):
        backend=runner.ResidentBackend('ar',load_timeout=10,worker_target=stalled_worker)
        try:
            started=time.monotonic()
            with self.assertRaisesRegex(TimeoutError,'terminated') as caught:
                backend.generate_messages([{'role':'user','content':'fixture'}],'ar',max_time=.05)
            self.assertLess(time.monotonic()-started,5)
            self.assertTrue(backend.closed)
            self.assertFalse(backend.process.is_alive())
            self.assertFalse(caught.exception.generation_diagnostics['measurement_complete'])
            self.assertEqual(caught.exception.generation_diagnostics['diagnostics']['status'],
                             'unavailable_worker_terminated')
        finally:
            backend.close()

    def test_worker_error_preserves_partial_generation_diagnostics(self):
        backend=runner.ResidentBackend('dard',load_timeout=10,worker_target=failed_worker)
        try:
            with self.assertRaisesRegex(RuntimeError,'late mock failure') as caught:
                backend.generate_messages([{'role':'user','content':'fixture'}],'dard',max_time=3)
            self.assertEqual(caught.exception.generation_diagnostics['model_forward_passes'],11)
            self.assertEqual(caught.exception.generation_diagnostics['diagnostics']['phase'],'sampling')
        finally:
            backend.close()

    def test_profile_mode_is_frozen_and_not_speed_eligible(self):
        report=self.run_mock(self.args(['--diagnostics','profile','--max-step-records','2']))
        rows=runner.read_jsonl(self.output/'results.jsonl')
        self.assertTrue(all(row['instrumentation_affects_timing'] and not row['speed_eligible'] for row in rows))
        config=json.loads((self.output/'run.json').read_text())
        self.assertEqual(config['dard_config']['max_step_records'],2)
        self.assertTrue(all('diagnostic_instrumentation_affects_timing' in row['issues'] for row in report['comparisons']))


if __name__=='__main__':
    unittest.main()
