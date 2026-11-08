"""CPU-only identity, exit-status and continuation checks for the review gate."""
import json
import contextlib
import fcntl
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import call, patch

from scripts import resume_reviewed_experiments as handoff


class ReviewedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.dict(os.environ, {}, clear=False))
        self.preflight = self.enterContext(patch.object(handoff.pipeline.mail, 'preflight',
                                                       return_value={'authenticated': True}))
        self.notify = self.enterContext(patch.object(handoff.pipeline.mail, 'notify',
            return_value={'smtp_accepted': True, 'pending': False, 'receipt_persisted': True}))

    def test_live_process_identity_is_verified_before_signalling(self):
        actual = handoff.process(os.getpid())
        self.assertEqual(actual['parent'], os.getppid())
        with patch.object(handoff.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'absent, exited or replaced'):
                handoff.require_process(os.getpid(), 'wrong-start-time')
            kill.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'no longer stopped'):
            handoff.require_process(os.getpid(), actual['start_ticks'], stopped=True)

    def test_waiter_reads_real_child_exit_status_without_reaping(self):
        for code in (0, 7):
            child = subprocess.Popen([sys.executable, '-c', f'import sys; sys.exit({code})'])
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    row = handoff.process(child.pid)
                    if row['state'] == 'Z':
                        break
                    time.sleep(.01)
                self.assertEqual(row['state'], 'Z')
                record = {'worker': handoff.process(os.getpid()),
                          'supervisor': handoff.process(os.getppid()),
                          'download': row, 'plan': {'data_root': str(self.root)}}
                with patch.object(handoff, 'require_process'), patch.object(handoff, 'verify_models') as verify:
                    if code == 0:
                        self.assertTrue(handoff.downloader_finished(record))
                        verify.assert_called_once()
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                            handoff.downloader_finished(record)
                        verify.assert_not_called()
            finally:
                child.wait(timeout=5)

    def test_real_stopped_coordinator_retires_without_dispatching_next_stage(self):
        info = self.root / 'worker.json'
        forbidden = self.root / 'next-stage-started'
        worker_code = (
            'import os,signal,json,time; from pathlib import Path; '
            f'Path({str(info)!r}).write_text(json.dumps({{"pid":os.getpid()}})); '
            'os.kill(os.getpid(), signal.SIGSTOP); '
            f'Path({str(forbidden)!r}).write_text("unsafe")')
        parent_code = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",sys.argv[1]]); p.wait()'
        parent = subprocess.Popen([sys.executable, '-c', parent_code, worker_code])
        worker = None
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if info.exists():
                    try:
                        worker = handoff.process(json.loads(info.read_text())['pid'])
                    except json.JSONDecodeError:
                        continue
                    if worker and worker['state'] == 'T':
                        break
                time.sleep(.01)
            self.assertIsNotNone(worker)
            self.assertEqual(worker['state'], 'T')
            record = {'worker': worker, 'supervisor': handoff.process(parent.pid),
                      'previous_output': str(self.root), 'output': str(self.root / 'reviewed')}
            handoff.retire_previous(record, timeout=5)
            parent.wait(timeout=5)
            self.assertFalse(forbidden.exists())
            self.assertTrue((self.root / 'audit_replacement.json').exists())
        finally:
            if worker and handoff.process(worker['pid']):
                try:
                    os.kill(worker['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)

    def test_continuation_reuses_prepared_models_without_prepare_or_download(self):
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'data_root': str(self.root), 'gpus': '0,1,2,3'}}
        with patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'downloader_finished', return_value=True), \
             patch.object(handoff, 'retire_previous') as retire, \
             patch.object(handoff.os, 'execv') as execute:
            handoff.run(record)
        retire.assert_called_once_with(record)
        command = execute.call_args.args[1]
        self.assertIn('--execute', command)
        self.assertNotIn('--prepare', command)
        self.assertNotIn('--download', command)
        self.assertEqual(command[command.index('--scope') + 1], 'functions')
        self.assertEqual(json.loads((self.root / 'audit_handoff.json').read_text())['status'],
                         'launching_reviewed_pipeline')

    def test_priority_deadline_and_quiet_policy_survive_exec_handoff(self):
        deadline = time.time() + 3600
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'data_root': str(self.root), 'gpus': '0,1,2,3',
                           'priority_codegen': True, 'deadline_unix': deadline, 'notifications': False}}
        with patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'downloader_finished', return_value=True), \
             patch.object(handoff, 'retire_previous'), \
             patch.object(handoff.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
            handoff.run(record)
        command = execute.call_args.args[1]
        self.assertIn('--priority-codegen', command)
        self.assertIn('--no-notify', command)
        self.assertEqual(float(command[command.index('--deadline-unix') + 1]), deadline)

    def test_focused_three_way_mode_deadline_and_quiet_policy_survive_exec(self):
        deadline = time.time() + 3600
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'data_root': str(self.root), 'gpus': '0,1,2,3',
                           'focused_codegen': True, 'priority_codegen': False,
                           'deadline_unix': deadline, 'notifications': False}}
        with patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'downloader_finished', return_value=True), \
             patch.object(handoff, 'retire_previous') as retire, \
             patch.object(handoff.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
            handoff.run(record)
        retire.assert_called_once_with(record)
        command = execute.call_args.args[1]
        self.assertIn('--focused-codegen', command)
        self.assertNotIn('--priority-codegen', command)
        self.assertNotIn('--prepare', command)
        self.assertNotIn('--repeats', command)
        self.assertIn('--no-notify', command)
        self.assertEqual(float(command[command.index('--deadline-unix') + 1]), deadline)

    def test_focused_reviewed_plan_roundtrip_preserves_flag_and_rejects_tampering(self):
        repo = self.root / 'fixture-repo'
        repo.mkdir()
        (repo / 'fixture.py').write_text('# Frozen fixture source\n')
        args = handoff.pipeline.parser().parse_args(['--output', str(self.root), '--data-root', str(self.root / 'data'),
            '--focused-codegen', '--deadline-unix', '2000000000', '--no-notify'])
        with patch.object(handoff.pipeline, 'ROOT', repo), \
             patch.object(handoff.pipeline.mail, 'configuration_status', return_value={'recipient': 'fixture@example.com'}):
            plan = handoff.pipeline.make_plan(args)
            handoff.pipeline.freeze_plan(self.root, plan)
            record = {'plan': plan, 'plan_sha256': handoff.pipeline.fingerprint(plan)}
            handoff.check_reviewed_plan(record)
            (repo / 'fixture.py').write_text('# Changed source\n')
            with self.assertRaisesRegex(ValueError, 'source/settings changed'):
                handoff.check_reviewed_plan(record)

    def test_focused_cli_passes_mode_to_review_gate_without_starting_execution(self):
        previous = self.root / 'previous'
        previous.mkdir()
        deadline = time.time() + 3600
        record = {'invocation_id': 'fixture', 'status': 'prepared'}
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(handoff, 'make_record', return_value=record) as make, \
             patch.object(handoff, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            code = handoff.main(['--previous-output', str(previous), '--output', str(self.root),
                '--focused-codegen', '--deadline-unix', str(deadline), '--no-notify'])
        self.assertEqual(code, 0)
        self.assertEqual(make.call_args.kwargs, {'priority_codegen': False, 'focused_codegen': True,
                                               'deadline_unix': deadline, 'no_notify': True})
        run.assert_not_called()

    def test_budget_expiring_during_retirement_never_execs_next_pipeline(self):
        record = {'output': str(self.root), 'download': {'pid': 123}, 'plan': {}}
        with patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'downloader_finished', return_value=True), \
             patch.object(handoff, 'retire_previous') as retire, \
             patch.object(handoff.pipeline, 'remaining_seconds',
                          side_effect=[120, handoff.pipeline.DeadlineExceeded('deadline')]), \
             patch.object(handoff.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(handoff.pipeline.DeadlineExceeded):
                handoff.run(record)
        retire.assert_called_once_with(record)
        execute.assert_not_called()

    def test_cleanup_reserve_blocks_download_completion_or_late_launch(self):
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'deadline_unix': time.time() + 30}}
        with patch.object(handoff, 'downloader_finished') as completed, \
             patch.object(handoff.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(handoff.pipeline.DeadlineExceeded):
                handoff.run(record)
        completed.assert_not_called()
        execute.assert_not_called()

    def test_quiet_retirement_signals_validated_supervisor_before_worker_resume(self):
        worker = {'pid': 101, 'start_ticks': 'worker'}
        parent = {'pid': 102, 'start_ticks': 'parent'}
        record = {'worker': worker, 'supervisor': parent, 'plan': {'notifications': False}, 'planned_maintenance': True,
                  'previous_output': str(self.root), 'output': str(self.root / 'new')}
        with patch.object(handoff, 'require_process') as require, \
             patch.object(handoff, 'process', return_value=None), patch.object(handoff.os, 'kill') as kill:
            handoff.retire_previous(record)
        self.assertEqual(require.call_args_list, [call(101, 'worker', stopped=True), call(102, 'parent')])
        self.assertEqual(kill.call_args_list, [call(102, signal.SIGKILL), call(101, signal.SIGTERM),
                                               call(101, signal.SIGCONT)])

    def test_planned_retirement_suppression_is_independent_of_future_failure_email(self):
        worker = {'pid': 101, 'start_ticks': 'worker'}
        parent = {'pid': 102, 'start_ticks': 'parent'}
        record = {'worker': worker, 'supervisor': parent, 'plan': {'notifications': True},
                  'planned_maintenance': True, 'previous_output': str(self.root), 'output': str(self.root / 'new')}
        with patch.object(handoff, 'require_process'), patch.object(handoff, 'process', return_value=None), \
             patch.object(handoff.os, 'kill') as kill:
            handoff.retire_previous(record)
        self.assertEqual(kill.call_args_list, [call(102, signal.SIGKILL), call(101, signal.SIGTERM),
                                               call(101, signal.SIGCONT)])
        replacement = json.loads((self.root / 'audit_replacement.json').read_text())
        self.assertTrue(replacement['planned_maintenance'])
        self.assertEqual(replacement['supervisor'], parent)
        self.assertTrue(record['plan']['notifications'])

    def test_smtp_preflight_happens_before_wait_and_inherited_quiet_environment_is_cleared(self):
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'data_root': str(self.root), 'gpus': '0,1,2,3', 'notifications': True}}
        events = []
        self.preflight.side_effect = lambda: events.append('authenticate')
        def finished(_):
            events.append('download_check')
            self.assertEqual(os.environ['HYBRID_DARD_DISABLE_EMAIL'], '0')
            return True
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '1'}), \
             patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'downloader_finished', side_effect=finished), \
             patch.object(handoff, 'retire_previous'), \
             patch.object(handoff.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()):
            handoff.run(record)
        self.assertEqual(events, ['authenticate', 'download_check'])
        self.assertNotIn('--no-notify', execute.call_args.args[1])
        self.notify.assert_not_called()

    def test_smtp_failure_prevents_wait_retirement_and_exec(self):
        record = {'output': str(self.root), 'download': {'pid': 123}, 'plan': {'notifications': True}}
        self.preflight.side_effect = RuntimeError('fixture authentication failed')
        with patch.object(handoff, 'downloader_finished') as finished, \
             patch.object(handoff, 'retire_previous') as retire, patch.object(handoff.os, 'execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'authentication failed'):
                handoff.run(record)
        self.assertEqual(record['handoff_stage'], 'email_preflight')
        finished.assert_not_called()
        retire.assert_not_called()
        execute.assert_not_called()

    def test_quiet_real_retirement_cannot_dispatch_or_notify_and_reaps_children(self):
        # Keep subreaper ownership inside this disposable fixture process, so
        # the unittest host's unrelated children are never adopted or signalled.
        fixture = textwrap.dedent('''
            import json, os, signal, subprocess, sys, time
            from pathlib import Path
            from scripts import resume_reviewed_experiments as handoff
            root = Path(sys.argv[1])
            handoff.pipeline.enable_subreaper()
            info, unsafe, notice = root/'child.json', root/'unsafe', root/'notice'
            code = ('import os,signal,json; from pathlib import Path; '
                    + f'Path({str(info)!r}).write_text(json.dumps({{"pid":os.getpid()}})); '
                    + 'os.kill(os.getpid(), signal.SIGSTOP); '
                    + f'Path({str(unsafe)!r}).write_text("unsafe")')
            parent_code = ('import subprocess,sys; from pathlib import Path; '
                           + 'p=subprocess.Popen([sys.executable,"-c",sys.argv[1]]); p.wait(); '
                           + f'Path({str(notice)!r}).write_text("notification")')
            parent = subprocess.Popen([sys.executable,'-c',parent_code,code])
            try:
                deadline = time.monotonic()+5
                worker = None
                while time.monotonic() < deadline:
                    if info.exists():
                        try:
                            worker = handoff.process(json.loads(info.read_text())['pid'])
                        except json.JSONDecodeError:
                            pass
                        if worker and worker['state'] == 'T': break
                    time.sleep(.01)
                assert worker and worker['state'] == 'T'
                record = {'worker': worker, 'supervisor': handoff.process(parent.pid),
                          'previous_output': str(root), 'output': str(root/'new'),
                          'plan': {'notifications': False}, 'planned_maintenance': True}
                handoff.retire_previous(record, timeout=5)
                assert parent.wait(timeout=5) == -signal.SIGKILL
                assert os.waitpid(worker['pid'], 0)[0] == worker['pid']
                assert not unsafe.exists() and not notice.exists()
                assert not handoff.pipeline.descendants()
            finally:
                handoff.pipeline.signal_owned(signal.SIGKILL)
                while True:
                    try: os.waitpid(-1, 0)
                    except ChildProcessError: break
        ''')
        result = subprocess.run([sys.executable, '-c', fixture, str(self.root)], cwd=handoff.ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cancel_downloads_escalates_but_ignores_reused_pid(self):
        record = {'download': {'pid': 101, 'start_ticks': 'original'},
                  'auxiliary_downloads': [{'pid': 102, 'start_ticks': 'old'}]}
        states = {101: {'pid': 101, 'start_ticks': 'original', 'state': 'S'},
                  102: {'pid': 102, 'start_ticks': 'replacement', 'state': 'S'}}
        with patch.object(handoff, 'process', side_effect=lambda pid: states[pid]), \
             patch.object(handoff.os, 'kill') as kill:
            handoff.cancel_downloads(record, grace=0)
        self.assertEqual(kill.call_args_list, [call(101, signal.SIGTERM), call(101, signal.SIGKILL)])

    def test_auxiliary_zombie_failure_records_exact_exit_status(self):
        expected = {'pid': 301, 'parent': 300, 'start_ticks': 'original'}
        record = {'output': str(self.root), 'auxiliary_downloads': [expected]}
        zombie = {**expected, 'state': 'Z', 'wait_status': 7 << 8}
        with patch.object(handoff, 'process', return_value=zombie):
            with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                handoff.check_auxiliary_downloads(record)
        receipt = json.loads((self.root / 'auxiliary-exits/301-original.json').read_text())
        self.assertEqual(receipt['exit_code'], 7)
        self.assertEqual(record['active_child_pid'], 301)
        self.assertEqual(record['active_child_exit_code'], 7)
        self.assertEqual(record['handoff_stage'], 'auxiliary_download')

    def test_auxiliary_exit_receipt_bridges_owner_reap_race_without_guessing_success(self):
        path = self.root / 'owned-exit.json'
        expected = {'pid': 301, 'parent': 300, 'start_ticks': 'original', 'exit_status_path': str(path)}
        record = {'output': str(self.root), 'auxiliary_downloads': [expected]}
        with patch.object(handoff, 'process', return_value=None), patch.object(handoff.time, 'monotonic', return_value=100):
            self.assertFalse(handoff.check_auxiliary_downloads(record))
        path.write_text(json.dumps({'pid': 301, 'start_ticks': 'original', 'exit_code': 0, 'ended_unix': time.time()}))
        with patch.object(handoff, 'process', return_value=None), patch.object(handoff.time, 'monotonic', return_value=101):
            self.assertTrue(handoff.check_auxiliary_downloads(record))
        self.assertEqual(record['auxiliary_missing_since'], {})

    def test_auxiliary_disappearance_without_receipt_fails_after_bounded_grace(self):
        expected = {'pid': 301, 'parent': 300, 'start_ticks': 'original'}
        record = {'output': str(self.root), 'auxiliary_downloads': [expected]}
        with patch.object(handoff, 'process', return_value=None), patch.object(handoff.time, 'monotonic', return_value=100):
            self.assertFalse(handoff.check_auxiliary_downloads(record))
        with patch.object(handoff, 'process', return_value=None), patch.object(handoff.time, 'monotonic', return_value=116):
            with self.assertRaisesRegex(RuntimeError, 'without a verified exit status'):
                handoff.check_auxiliary_downloads(record)

    def test_auxiliary_receipt_rejects_reused_identity_and_nonzero_exit(self):
        path = self.root / 'exit.json'
        expected = {'pid': 301, 'start_ticks': 'original', 'exit_status_path': str(path)}
        record = {'output': str(self.root), 'auxiliary_downloads': [expected]}
        for receipt, message in (
            ({'pid': 301, 'start_ticks': 'other', 'exit_code': 0, 'ended_unix': time.time()}, 'identity'),
            ({'pid': 301, 'start_ticks': 'original', 'exit_code': -9, 'ended_unix': time.time()}, 'exit code -9'),
        ):
            path.write_text(json.dumps(receipt))
            with self.subTest(receipt=receipt), patch.object(handoff, 'process', return_value=None), \
                 self.assertRaisesRegex(RuntimeError, message):
                handoff.check_auxiliary_downloads(record)

    def test_download_complete_still_waits_for_active_auxiliary(self):
        record = {'output': str(self.root), 'download': {'pid': 123},
                  'plan': {'data_root': str(self.root), 'gpus': '0,1,2,3', 'notifications': True}}
        with patch.object(handoff, 'check_reviewed_plan'), \
             patch.object(handoff, 'check_auxiliary_downloads', side_effect=[False, True]), \
             patch.object(handoff, 'downloader_finished', return_value=True) as finished, \
             patch.object(handoff, 'process', return_value=None), \
             patch.object(handoff.time, 'sleep') as sleep, patch.object(handoff, 'retire_previous'), \
             patch.object(handoff.os, 'execv'), contextlib.redirect_stdout(io.StringIO()):
            handoff.run(record)
        self.assertEqual(finished.call_count, 2)
        sleep.assert_called_once()

    def test_real_term_ignoring_download_is_killed(self):
        ready = self.root / 'ready'
        code = ('import signal,time; from pathlib import Path; '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                f'Path({str(ready)!r}).touch(); time.sleep(30)')
        child = subprocess.Popen([sys.executable, '-c', code])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            handoff.cancel_downloads({'download': handoff.process(child.pid)}, grace=.05)
            self.assertEqual(child.wait(timeout=5), -signal.SIGKILL)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

    def test_deadline_cleanup_error_is_persisted_and_notification_stays_local(self):
        previous = self.root / 'previous'
        previous.mkdir()
        record = {'invocation_id': 'fixture', 'worker': {'pid': 101}, 'download': {'pid': 102}}
        deadline = time.time() + 3600
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '0'}), \
             patch.object(handoff, 'make_record', return_value=record) as make, \
             patch.object(handoff, 'run', side_effect=handoff.pipeline.DeadlineExceeded('budget')), \
             patch.object(handoff, 'cancel_downloads') as cancel, \
             patch.object(handoff, 'retire_previous', side_effect=RuntimeError('already exited')), \
             patch.object(handoff.pipeline.mail, 'notify') as notify, \
             contextlib.redirect_stderr(io.StringIO()):
            code = handoff.main(['--previous-output', str(previous), '--output', str(self.root),
                                 '--execute', '--priority-codegen', '--no-notify', '--deadline-unix', str(deadline)])
        self.assertEqual(code, 124)
        self.assertEqual(make.call_args.kwargs, {'priority_codegen': True, 'focused_codegen': False,
                                               'deadline_unix': deadline, 'no_notify': True})
        cancel.assert_called_once_with(record)
        notify.assert_not_called()
        saved = json.loads((self.root / 'audit_handoff.json').read_text())
        self.assertEqual(saved['status'], 'deadline_exceeded')
        self.assertIn('already exited', saved['cleanup_error'])
        self.assertTrue(json.loads((self.root / 'local_failure_notice.json').read_text())['disabled'])

    def test_failure_logs_local_traceback_and_mails_only_failure_context(self):
        previous = self.root / 'previous'
        previous.mkdir()
        private = 'fixture-private-detail-not-for-mail'
        record = {'invocation_id': 'fixture', 'handoff_stage': 'auxiliary_download',
                  'active_child_pid': 301, 'active_child_start_ticks': 'original', 'active_child_exit_code': 7,
                  'plan_sha256': 'fixture-plan-hash', 'plan': {'notifications': True, 'source_sha256': {'file.py': 'abc'}}}
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '1'}), \
             patch.object(handoff, 'make_record', return_value=record), \
             patch.object(handoff, 'run', side_effect=RuntimeError(private)), \
             contextlib.redirect_stderr(io.StringIO()):
            code = handoff.main(['--previous-output', str(previous), '--output', str(self.root), '--execute'])
        self.assertEqual(code, 1)
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual(event['stage'], 'audit_handoff.auxiliary_download')
        self.assertEqual(event['child_pid'], 301)
        self.assertEqual(event['child_exit_code'], 7)
        self.assertEqual(event['exit_code'], 7)
        self.assertNotIn(private, json.dumps(event))
        detail = json.loads((self.root / 'handoff_errors.jsonl').read_text())
        self.assertIn(private, detail['traceback'])
        self.assertEqual(detail['source_sha256'], {'file.py': 'abc'})
        self.assertEqual(detail['plan_sha256'], 'fixture-plan-hash')
        self.assertGreater(detail['timestamp_unix'], 0)
        self.assertIn('+00:00', detail['timestamp_utc'])

    def test_changed_reviewed_source_blocks_launch(self):
        record = {'output': str(self.root), 'download': {'pid': 123}}
        with patch.object(handoff, 'check_reviewed_plan', side_effect=ValueError('source changed')), \
             patch.object(handoff, 'retire_previous') as retire, \
             patch.object(handoff.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'source changed'):
                handoff.run(record)
        retire.assert_not_called()
        execute.assert_not_called()

    def test_second_waiter_cannot_clobber_live_output_record(self):
        path = self.root / 'audit_handoff.json'
        path.write_text('{"status":"waiting_for_existing_download"}')
        original = path.read_bytes()
        with (self.root / 'audit_handoff.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(handoff, 'make_record') as prepare, contextlib.redirect_stderr(io.StringIO()):
                code = handoff.main(['--previous-output', str(self.root / 'previous'), '--output', str(self.root)])
        self.assertNotEqual(code, 0)
        prepare.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_different_destinations_cannot_share_a_paused_source_run(self):
        previous = self.root / 'previous'
        previous.mkdir()
        with (previous / 'audit_handoff.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(handoff, 'make_record') as prepare, contextlib.redirect_stderr(io.StringIO()):
                code = handoff.main(['--previous-output', str(previous), '--output', str(self.root / 'another')])
        self.assertNotEqual(code, 0)
        prepare.assert_not_called()

    def test_failure_record_is_written_before_releasing_output_lock(self):
        previous = self.root / 'previous'
        previous.mkdir()
        def write_locked(path, value):
            with (self.root / 'audit_handoff.lock').open('a') as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(value['status'], 'handoff_failed')
        with patch.object(handoff, 'make_record', side_effect=ValueError('bad gate')), \
             patch.object(handoff.pipeline, 'write_json', side_effect=write_locked) as write, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(handoff.main(['--previous-output', str(previous), '--output', str(self.root)]), 0)
        write.assert_called_once()


if __name__ == '__main__':
    unittest.main()
