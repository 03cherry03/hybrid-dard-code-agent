"""Guard lifecycle checks using isolated fixture children and mocked SMTP only."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

from scripts import campaign_guard as guard


class CampaignGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.enterContext(patch.dict(os.environ, {}, clear=False))
        self.enterContext(patch.object(guard.mail, 'preflight', return_value={'authenticated': True}))
        self.notify = self.enterContext(patch.object(guard.pipeline, 'send_failure',
            return_value={'smtp_accepted': True, 'pending': False, 'receipt_persisted': True}))
        self.flush = self.enterContext(patch.object(guard.mail, 'flush_pending',
            return_value={'pending_remaining': 0, 'remaining_scan_complete': True, 'errors': []}))

    def config(self):
        return {'output': str(self.output), 'previous_output': str(self.output / 'previous'),
                'data_root': str(self.output / 'data'), 'deadline_unix': time.time() + 3600,
                'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'auxiliary_command': [sys.executable, '-c', 'pass'],
                'handoff_command': [sys.executable, '-c', 'pass']}

    def test_exit_receipt_is_durable_before_reaping_actual_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit(7)'])
        try:
            identity = guard.handoff.process(child.pid)
            until = time.monotonic() + 3
            while time.monotonic() < until and guard.handoff.process(child.pid)['state'] != 'Z':
                time.sleep(.01)
            self.assertEqual(guard.handoff.process(child.pid)['state'], 'Z')
            receipt = self.output / 'exit.json'
            original_wait = child.wait
            def inspected_wait():
                saved = json.loads(receipt.read_text())
                self.assertEqual((saved['pid'], saved['start_ticks'], saved['exit_code']),
                                 (child.pid, identity['start_ticks'], 7))
                self.assertEqual(guard.handoff.process(child.pid)['state'], 'Z')
                return original_wait()
            with patch.object(child, 'wait', side_effect=inspected_wait):
                self.assertEqual(guard.exited(child, identity, receipt), 7)
            self.assertIsNone(guard.handoff.process(child.pid))
        finally:
            child.wait(timeout=3)

    def run_real_guard_fixture(self, mode, *, prepared=False):
        """Subreaper and process-group cleanup never affect the unittest host."""
        fixture = textwrap.dedent('''
            import json, os, signal, subprocess, sys, time
            from pathlib import Path
            from unittest.mock import patch
            from scripts import campaign_guard as guard
            root, mode = Path(sys.argv[1]), sys.argv[2]
            prepared = sys.argv[3] == 'prepared'
            output = root/'guard'; output.mkdir()
            previous = root/'previous'; previous.mkdir()
            (previous/'audit_gate.json').write_text(json.dumps({'fixture': True}))
            fake = {
                -101: {'pid': -101, 'parent': -102, 'start_ticks': 'worker', 'state': 'T'},
                -102: {'pid': -102, 'parent': 0, 'start_ticks': 'parent', 'state': 'S'},
                -103: {'pid': -103, 'parent': -101, 'start_ticks': 'download', 'state': 'S'},
            }
            preparation = {'worker': fake[-101], 'supervisor': fake[-102], 'download': fake[-103],
                           'old_invocation_id': 'fixture-old'}
            aux_code = 'import time,sys; time.sleep(.1); sys.exit(' + ('7' if mode=='aux7' else '0') + ')'
            if mode == 'normal':
                main_code = ('import json,time,os; from pathlib import Path; time.sleep(.3); '
                             + f"d=Path({str(output)!r}); "
                             + "ticks=Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19]; "
                             + "(d/'supervisor.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':ticks,'invocation_id':'fixture'})); "
                             + "(d/'pipeline_state.json').write_text(json.dumps({'status':'completed','invocation_id':'fixture'}))")
            elif mode == 'stale_success':
                main_code = ('import json,time; from pathlib import Path; time.sleep(.3); '
                             + f"Path({str(output/'pipeline_state.json')!r}).write_text(json.dumps({{'status':'completed','invocation_id':'stale'}}))")
            elif mode == 'main7':
                main_code = 'import time,sys; time.sleep(.3); sys.exit(7)'
            elif mode == 'sigkill':
                main_code = 'import time,os,signal; time.sleep(.3); os.kill(os.getpid(),signal.SIGKILL)'
            else:
                main_code = 'import time; time.sleep(30)'
            config = {'output': str(output), 'previous_output': str(previous), 'data_root': str(root/'data'),
                      'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      'deadline_unix': time.time()+3600, 'preparation': preparation,
                      'auxiliary_command': [sys.executable,'-c',aux_code],
                      'handoff_command': [sys.executable,'-c',main_code]}
            if prepared:
                config.update(mode='prepared', pipeline_command=config.pop('handoff_command'),
                              prepared_model_receipt=str(root/'verified.json'),
                              prepared_model_receipt_sha256='a'*64)
                for key in ('auxiliary_command', 'preparation', 'previous_output'):
                    del config[key]
            notices = []
            def send(output, event):
                notices.append(event)
                return {'smtp_accepted': True, 'pending': False, 'receipt_persisted': True}
            actual_process = guard.handoff.process
            def process(pid):
                return fake.get(pid) if pid in fake else actual_process(pid)
            def validate(config):
                assert guard.pipeline.descendants() == {}, 'model validation must precede process launch'
                state = json.loads((output/'guard_state.json').read_text())
                assert state['stage'] == 'model_validation'
                assert state['model_validation']['status'] == 'running'
                if mode == 'invalid_models':
                    raise ValueError('changed checkpoint fixture')
                return {'status': 'passed', 'models': 2}
            with patch.object(guard.handoff,'process',side_effect=process), \
                 patch.object(guard,'cleanup_external') as cleanup, \
                 patch.object(guard,'verify_prepared',side_effect=validate) as validation, \
                 patch.object(guard.mail,'preflight',return_value={'authenticated':True}), \
                 patch.object(guard.pipeline,'send_failure',side_effect=send), \
                 patch.object(guard.mail,'flush_pending',return_value={'pending_remaining':0,'remaining_scan_complete':True,'errors':[]}):
                code = guard.run(config)
            state = json.loads((output/'guard_state.json').read_text())
            receipt_path = output/'auxiliary_download_exit.json'
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            print(json.dumps({'code':code,'state':state,'receipt':receipt,'notices':notices,
                              'events':[json.loads(line) for line in (output/'guard_events.jsonl').read_text().splitlines()],
                              'validation_calls':validation.call_count, 'cleanup_calls':cleanup.call_count,
                              'auxiliary_log_exists':(output/'auxiliary_download.log').exists(),
                              'remaining_children':guard.pipeline.descendants()}))
        ''')
        result = subprocess.run([sys.executable, '-c', fixture, str(self.output), mode,
                                 'prepared' if prepared else 'handoff'],
                                cwd=guard.ROOT, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_actual_children_complete_with_no_failure_notification(self):
        result = self.run_real_guard_fixture('normal')
        self.assertEqual(result['code'], 0)
        self.assertEqual(result['state']['status'], 'completed')
        self.assertEqual(result['receipt']['exit_code'], 0)
        self.assertEqual(result['notices'], [])
        self.assertEqual(result['remaining_children'], {})

    def test_actual_main_exit7_is_reported_and_all_owned_children_are_reaped(self):
        result = self.run_real_guard_fixture('main7')
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['exit_code'], 7)
        self.assertEqual(result['remaining_children'], {})

    def test_zero_exit_with_stale_completion_record_is_reported_as_failure(self):
        result = self.run_real_guard_fixture('stale_success')
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['stage'], 'supervisor_lost')
        self.assertEqual(result['remaining_children'], {})

    def test_actual_main_sigkill_is_detected_independently_of_inner_supervisor(self):
        result = self.run_real_guard_fixture('sigkill')
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['exit_code'], -signal.SIGKILL)
        self.assertEqual(result['remaining_children'], {})

    def test_actual_auxiliary_exit7_stops_running_main_and_persists_receipt(self):
        result = self.run_real_guard_fixture('aux7')
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(result['receipt']['exit_code'], 7)
        self.assertEqual(result['notices'][0]['stage'], 'auxiliary_download')
        self.assertEqual(result['notices'][0]['exit_code'], 7)
        self.assertEqual(result['remaining_children'], {})

    def test_prepared_local_validation_then_direct_child_completes_without_download(self):
        result = self.run_real_guard_fixture('normal', prepared=True)
        state = result['state']
        self.assertEqual((state['status'], state['stage']), ('completed', 'completed'))
        self.assertEqual(state['model_validation']['status'], 'passed')
        self.assertIn('pipeline', state)
        self.assertNotIn('handoff', state)
        self.assertNotIn('auxiliary', state)
        self.assertEqual(result['validation_calls'], 1)
        self.assertEqual(result['cleanup_calls'], 0)
        self.assertIsNone(result['receipt'])
        self.assertFalse(result['auxiliary_log_exists'])
        self.assertEqual([row['stage'] for row in result['events']],
                         ['model_validation', 'model_validation', 'direct_pipeline_launch', 'coordinator_exited'])
        self.assertEqual(result['notices'], [])
        self.assertEqual(result['remaining_children'], {})

    def test_prepared_child_failure_notifies_without_external_preparation_cleanup(self):
        result = self.run_real_guard_fixture('main7', prepared=True)
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(result['state']['model_validation']['status'], 'passed')
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['exit_code'], 7)
        self.assertTrue(result['notices'][0]['log_path'].endswith('/pipeline.log'))
        self.assertEqual(result['cleanup_calls'], 0)
        self.assertEqual(result['remaining_children'], {})

    def test_prepared_invalid_models_never_launch_and_record_validation_failure(self):
        result = self.run_real_guard_fixture('invalid_models', prepared=True)
        state = result['state']
        self.assertEqual((state['status'], state['stage']), ('failed', 'model_validation'))
        self.assertEqual(state['model_validation']['status'], 'failed')
        self.assertEqual(state['model_validation']['error'], 'changed checkpoint fixture')
        self.assertNotIn('pipeline', state)
        self.assertNotIn('auxiliary', state)
        self.assertEqual(result['validation_calls'], 1)
        self.assertEqual(result['cleanup_calls'], 0)
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['stage'], 'model_validation')
        self.assertFalse(result['auxiliary_log_exists'])
        self.assertEqual(result['remaining_children'], {})

    def test_prepared_zero_exit_with_stale_completion_record_fails(self):
        result = self.run_real_guard_fixture('stale_success', prepared=True)
        self.assertEqual(result['state']['status'], 'failed')
        self.assertEqual(len(result['notices']), 1)
        self.assertEqual(result['notices'][0]['stage'], 'supervisor_lost')
        self.assertEqual(result['cleanup_calls'], 0)
        self.assertEqual(result['remaining_children'], {})

    def test_prepared_configuration_needs_only_direct_command_and_verified_receipt(self):
        config = self.config()
        config.update(mode='prepared', pipeline_command=config.pop('handoff_command'),
                      prepared_model_receipt=str(self.output / 'verified.json'),
                      prepared_model_receipt_sha256='a' * 64)
        del config['auxiliary_command']
        del config['previous_output']
        path = self.output / 'guard_config.json'
        path.write_text(json.dumps(config))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(guard, 'run', return_value=0) as run:
            self.assertEqual(guard.main(['--config', str(path), '--config-sha256', digest]), 0)
        run.assert_called_once_with(config)
        self.notify.assert_not_called()

    def test_prepared_configuration_without_integrity_receipt_is_rejected(self):
        config = self.config()
        config.update(mode='prepared', pipeline_command=config.pop('handoff_command'))
        path = self.output / 'guard_config.json'
        path.write_text(json.dumps(config))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(guard, 'run') as run:
            self.assertEqual(guard.main(['--config', str(path), '--config-sha256', digest]), 0)
        run.assert_not_called()
        self.assertEqual(self.notify.call_args.args[1]['stage'], 'guard_configuration')

    def test_prepared_cleanup_never_reads_or_retires_external_processes(self):
        with patch.object(guard.handoff, 'process') as process, \
             patch.object(guard.handoff, 'cancel_downloads') as cancel, \
             patch.object(guard.handoff, 'retire_previous') as retire:
            guard.cleanup_external({'mode': 'prepared'})
        process.assert_not_called()
        cancel.assert_not_called()
        retire.assert_not_called()

    def test_restart_after_guard_loss_only_notifies_never_launches_duplicate_jobs(self):
        config = self.config()
        state = {'config_sha256': guard.pipeline.fingerprint(config), 'status': 'running',
                 'invocation_id': 'original-guard', 'handoff': {'pid': 111, 'start_ticks': 'original'}}
        guard.pipeline.write_json(self.output / 'guard_state.json', state)
        with patch.object(guard.subprocess, 'Popen') as launch, \
             patch.object(guard, 'cleanup_external') as cleanup, \
             patch.object(guard, 'retry_notifications') as retry:
            self.assertEqual(guard.run(config), 0)
        launch.assert_not_called()
        cleanup.assert_called_once_with(config)
        saved = json.loads((self.output / 'guard_state.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['failure_event']['stage'], 'guard_restarted')
        retry.assert_called_once_with(self.output, saved['failure_event'])

    def test_restart_of_failed_guard_retries_same_event_without_relaunch_or_new_event(self):
        config = self.config()
        event = guard.pipeline.failure_event(self.output, 'original', 'auxiliary_download', 7)
        state = {'config_sha256': guard.pipeline.fingerprint(config), 'status': 'failed',
                 'invocation_id': 'original', 'failure_event': event}
        guard.pipeline.write_json(self.output / 'guard_state.json', state)
        with patch.object(guard.subprocess, 'Popen') as launch, patch.object(guard, 'cleanup_external') as cleanup:
            self.assertEqual(guard.run(config), 0)
        launch.assert_not_called()
        cleanup.assert_not_called()
        self.notify.assert_called_once_with(self.output, event)
        self.assertEqual(json.loads((self.output / 'guard_state.json').read_text()), state)

    def test_delivery_retry_survives_pending_outbox_and_records_acceptance(self):
        event = guard.pipeline.failure_event(self.output, 'guard', 'stage', 7)
        self.notify.side_effect = [{'smtp_accepted': False, 'pending': True},
                                  {'smtp_accepted': True, 'pending': False}]
        self.flush.side_effect = [
            {'pending_remaining': 1, 'remaining_scan_complete': True, 'errors': []},
            {'pending_remaining': 0, 'remaining_scan_complete': True, 'errors': []},
        ]
        with patch.object(guard.time, 'sleep') as sleep:
            guard.retry_notifications(self.output, event, retry_seconds=.01)
        self.assertEqual(self.notify.call_count, 2)
        sleep.assert_called_once_with(.01)
        monitor = json.loads((self.output / 'notification_monitor.json').read_text())
        self.assertFalse(monitor['pending'])
        self.assertTrue(monitor['smtp_accepted'])
        self.assertEqual(monitor['event_id'], event['event_id'])

    def test_existing_fallback_is_retried_without_relaunching_any_experiment(self):
        event = guard.pipeline.failure_event(self.output, 'original', 'stage', 7)
        guard.pipeline.write_json(self.output / 'notification_fallback.json', {'event': event, 'pending': True})
        with patch.object(guard.subprocess, 'Popen') as launch:
            guard.retry_notifications(self.output)
        launch.assert_not_called()
        self.notify.assert_called_once_with(self.output, event)
        self.assertFalse(json.loads((self.output / 'notification_fallback.json').read_text())['pending'])

    def test_exact_inner_handoff_failure_event_is_reused_and_stale_identity_is_rejected(self):
        identity = {'pid': 111, 'start_ticks': 'original'}
        state = {'invocation_id': 'guard', 'handoff': identity}
        event = guard.pipeline.failure_event(self.output, 'handoff', 'audit_handoff.download_models', 7)
        transfer = {'status': 'handoff_failed', 'invocation_id': 'handoff', 'failure_event': event,
                    'handoff_pid': 111, 'handoff_start_ticks': 'original'}
        guard.pipeline.write_json(self.output / 'audit_handoff.json', transfer)
        self.assertEqual(guard.failure_for_exit(self.output, state, 1), event)
        state['handoff']['start_ticks'] = 'replacement'
        rejected = guard.failure_for_exit(self.output, state, 1)
        self.assertNotEqual(rejected['event_id'], event['event_id'])
        self.assertEqual(rejected['stage'], 'supervisor_lost')

    def test_exact_pipeline_failure_event_is_reused_and_stale_supervisor_is_rejected(self):
        identity = {'pid': 111, 'start_ticks': 'original'}
        state = {'invocation_id': 'guard', 'handoff': identity}
        event = guard.pipeline.failure_event(self.output, 'pipeline', 'functions', 7)
        guard.pipeline.write_json(self.output / 'pipeline_failure.json', event)
        guard.pipeline.write_json(self.output / 'supervisor.json', {**identity, 'invocation_id': 'pipeline'})
        self.assertEqual(guard.failure_for_exit(self.output, state, 7), event)
        guard.pipeline.write_json(self.output / 'supervisor.json', {**identity, 'invocation_id': 'stale'})
        self.assertEqual(guard.failure_for_exit(self.output, state, 7)['stage'], 'supervisor_lost')

    def test_changed_launch_configuration_is_reported_without_process_launch(self):
        path = self.output / 'guard_config.json'
        path.write_text(json.dumps(self.config()))
        with patch.object(guard.subprocess, 'Popen') as launch, patch.object(guard, 'retry_notifications') as retry:
            self.assertEqual(guard.main(['--config', str(path), '--config-sha256', '0' * 64]), 0)
        launch.assert_not_called()
        event = retry.call_args.args[1]
        self.assertEqual(event['stage'], 'guard_configuration')
        self.assertTrue((self.output / 'guard_configuration_failure.json').exists())

    def test_external_cleanup_requires_same_boot_and_frozen_process_relationships(self):
        config = self.config()
        previous = Path(config['previous_output'])
        previous.mkdir()
        rows = {101: {'pid': 101, 'parent': 102, 'start_ticks': 'worker', 'state': 'T'},
                102: {'pid': 102, 'parent': 1, 'start_ticks': 'supervisor', 'state': 'S'},
                103: {'pid': 103, 'parent': 101, 'start_ticks': 'download', 'state': 'S'}}
        config['preparation'] = {'worker': dict(rows[101]), 'supervisor': dict(rows[102]),
                                 'download': dict(rows[103]), 'old_invocation_id': 'old'}
        gate = {'paused_worker_pid': 101, 'worker_start_ticks': 'worker', 'download_pid': 103}
        state = {'invocation_id': 'old', 'stage': 'download_models', 'status': 'running',
                 'worker_pid': 101, 'worker_start_ticks': 'worker', 'stage_pid': 103}
        guard.pipeline.write_json(previous / 'audit_gate.json', gate)
        guard.pipeline.write_json(previous / 'pipeline_state.json', state)
        with patch.object(guard.handoff, 'process', side_effect=lambda pid: rows[pid]), \
             patch.object(guard.handoff, 'cancel_downloads') as cancel, \
             patch.object(guard.handoff, 'retire_previous') as retire:
            guard.cleanup_external(config)
        cancel.assert_called_once()
        retire.assert_called_once()
        self.assertTrue(retire.call_args.args[0]['planned_maintenance'])
        for changed in ('boot', 'worker', 'supervisor', 'download', 'parent', 'invocation', 'gate'):
            modified = copy.deepcopy(config)
            actual = copy.deepcopy(rows)
            local_state, local_gate = dict(state), dict(gate)
            if changed == 'boot':
                modified['host_boot_id'] = 'different-boot'
            elif changed in ('worker', 'supervisor', 'download'):
                actual[{'worker': 101, 'supervisor': 102, 'download': 103}[changed]]['start_ticks'] = 'reused-pid'
            elif changed == 'parent':
                actual[103]['parent'] = 999
            elif changed == 'invocation':
                local_state['invocation_id'] = 'other-run'
            else:
                local_gate['paused_worker_pid'] = 999
            guard.pipeline.write_json(previous / 'audit_gate.json', local_gate)
            guard.pipeline.write_json(previous / 'pipeline_state.json', local_state)
            with self.subTest(changed=changed), \
                 patch.object(guard.handoff, 'process', side_effect=lambda pid: actual[pid]), \
                 patch.object(guard.handoff, 'cancel_downloads') as cancel, \
                 patch.object(guard.handoff, 'retire_previous') as retire:
                guard.cleanup_external(modified)
            cancel.assert_not_called()
            retire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
