"""Offline pipeline supervision checks using tiny local stages and mocked mail."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

from scripts import all_experiments as pipeline


class AllExperimentsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.output = self.base / "output"
        self.output.mkdir()
        self.root = self.base / "data"
        self.events = self.base / "executed.txt"
        # These patches cover every externally callable email route used here.
        self.notify = self.enterContext(patch.object(pipeline.mail, "notify",
            return_value={"smtp_accepted": True, "pending": False, "receipt_persisted": True}))
        self.preflight = self.enterContext(patch.object(pipeline.mail, "preflight",
            return_value={"authenticated": True}))
        self.enterContext(patch.object(pipeline.mail, "configuration_status",
            return_value={"recipient": "leejwian@gmail.com", "metadata_ready": True}))
        self.enterContext(patch.object(pipeline.mail, "flush_pending",
            return_value={"pending": 0}))

    def stage(self, name, code, *extra):
        return {"name": name, "command": [sys.executable, "-c", code, *map(str, extra)],
                "cwd": str(self.base), "downloads": False}

    def record_stage(self, name, exit_code=0):
        code = ("from pathlib import Path; import sys; "
                f"p=Path({str(self.events)!r}); "
                f"p.open('a').write({name + chr(10)!r}); print({name!r}, flush=True); "
                f"sys.exit({exit_code})")
        return self.stage(name, code)

    def plan(self, stages):
        return {"schema_version": 1, "data_root": str(self.root), "output": str(self.output),
                "recipient": "leejwian@gmail.com", "stages": stages}

    def test_worker_runs_in_order_and_records_each_success(self):
        plan = self.plan([self.record_stage("alpha"), self.record_stage("beta")])
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.run_worker(plan, self.output, "invocation-one")
        self.assertEqual(code, 0)
        self.assertEqual(self.events.read_text().splitlines(), ["alpha", "beta"])
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["invocation_id"], "invocation-one")
        self.assertEqual(state["stage"], "beta")
        for stage in plan["stages"]:
            stamp = json.loads((self.output / "stage_results" / (stage["name"] + ".json")).read_text())
            self.assertEqual(stamp["exit_code"], 0)
            self.assertEqual(stamp["stage_sha256"], pipeline.fingerprint(stage))
            self.assertIn(stage["name"], (self.output / "logs" / (stage["name"] + ".log")).read_text())
        self.notify.assert_not_called()
        self.preflight.assert_not_called()

    def test_absolute_deadline_terminates_sleeping_stage_and_never_starts_next(self):
        pid_path = self.base / "sleeping-pid.txt"
        sleeping = self.stage("sleeping", "import os,time; from pathlib import Path; "
                              f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)")
        deadline = time.time() + 60.7  # Production reserves the final 60 seconds for cleanup.
        plan = {**self.plan([sleeping, self.record_stage("must_not_run")]), "deadline_unix": deadline}
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.run_worker(plan, self.output, "fixed-deadline")
        self.assertEqual(code, 124)
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(pid_path.exists(), "The real sleeping stage must have started")
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)
        self.assertFalse(self.events.exists())
        self.assertFalse((self.output / "stage_results/sleeping.json").exists())
        self.assertFalse((self.output / "logs/must_not_run.log").exists())
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual((state["status"], state["error_type"]), ("deadline_exceeded", "DeadlineExceeded"))
        self.assertEqual(plan["deadline_unix"], deadline)
        # Resume retains the absolute deadline; the cleanup reserve cannot fund new work.
        with patch.object(pipeline.subprocess, "Popen") as launch, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pipeline.run_worker(plan, self.output, "expired-resume"), 124)
        launch.assert_not_called()

    def test_expiry_during_final_artifact_accounting_is_not_reported_as_success(self):
        plan = {**self.plan([self.record_stage("finished_stage")]), "deadline_unix": 161.0}
        now = [100.0]
        def expensive_hashes(*_):
            now[0] = 162.0
            return {}
        with patch.object(pipeline.time, "time", side_effect=lambda: now[0]), \
             patch.object(pipeline, "artifact_hashes", side_effect=expensive_hashes), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.run_worker(plan, self.output, "late-accounting")
        self.assertEqual(code, 124)
        self.assertEqual(self.events.read_text().splitlines(), ["finished_stage"])
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual(state["status"], "deadline_exceeded")

    def test_expiry_during_prelaunch_checks_does_not_launch_a_late_stage(self):
        plan = {**self.plan([self.record_stage("must_not_run")]), "deadline_unix": 101.0}
        now = [100.0]
        def expensive_check(_plan):
            now[0] = 102.0
        with patch.object(pipeline.time, "time", side_effect=lambda: now[0]), \
             patch.object(pipeline, "verify_plan_sources", side_effect=expensive_check), \
             patch.object(pipeline.subprocess, "Popen", return_value=Mock(pid=123463, poll=Mock(return_value=0))) as launch, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.run_worker(plan, self.output, "late-preflight")
        self.assertEqual(code, 124)
        launch.assert_not_called()
        self.assertFalse(self.events.exists())

    def test_deadline_is_frozen_and_not_extended_by_replanning_or_resume(self):
        deadline = 2_000_000_000.0
        args = pipeline.parser().parse_args(["--data-root", str(self.root), "--output", str(self.output),
            "--deadline-unix", str(deadline), "--priority-codegen", "--no-notify"])
        first = pipeline.make_plan(args)
        pipeline.freeze_plan(self.output, first)
        with patch.object(pipeline.time, "time", return_value=deadline - 1):
            second = pipeline.make_plan(args)
        self.assertEqual(first, second)
        self.assertEqual(second["deadline_unix"], deadline)
        self.assertTrue(second["priority_codegen"])
        self.assertFalse(second["notifications"])
        args.deadline_unix += 40 * 3600
        with self.assertRaisesRegex(ValueError, "settings/source changed"):
            pipeline.freeze_plan(self.output, pipeline.make_plan(args))
        self.assertEqual(json.loads((self.output / "pipeline_plan.json").read_text())["deadline_unix"], deadline)

    def test_invalid_deadlines_are_rejected_before_execution(self):
        args = pipeline.parser().parse_args(["--data-root", str(self.root), "--output", str(self.output)])
        for deadline in (-1, 0, float("nan"), float("inf")):
            with self.subTest(deadline=deadline), self.assertRaisesRegex(ValueError, "deadline"):
                args.deadline_unix = deadline
                pipeline.make_plan(args)

    def test_focused_plan_freezes_required_three_way_mode_and_fixed_deadline(self):
        deadline = 2_000_000_000.0
        args = pipeline.parser().parse_args(["--data-root", str(self.root), "--output", str(self.output),
            "--deadline-unix", str(deadline), "--focused-codegen", "--no-notify"])
        plan = pipeline.make_plan(args)
        self.assertTrue(plan["focused_codegen"])
        self.assertFalse(plan["priority_codegen"])
        self.assertFalse(plan["notifications"])
        self.assertEqual(plan["deadline_unix"], deadline)
        self.assertEqual([stage["name"] for stage in plan["stages"]],
                         ["gpu_idle_check", "data_verify", "focused_codegen", "focused_analysis"])
        pipeline.freeze_plan(self.output, plan)
        self.assertEqual(pipeline.make_plan(args), plan)
        args.focused_codegen = False
        args.priority_codegen = True
        with self.assertRaisesRegex(ValueError, "settings/source changed"):
            pipeline.freeze_plan(self.output, pipeline.make_plan(args))

    def test_focused_analysis_artifacts_pin_both_goal_reports(self):
        paths = [self.output / "goal_report.json", self.output / "goal_report.md"]
        paths[0].write_text('{"complete": true}')
        paths[1].write_text('Matched three-way result\n')
        stage = {"name": "focused_analysis", "command": ["runner", "functions", "--output", str(self.output), "--analyze"]}
        hashes = pipeline.artifact_hashes(stage, self.root)
        self.assertEqual(set(hashes), {str(path) for path in paths})
        paths[1].write_text('Changed interpretation\n')
        self.assertNotEqual(hashes, pipeline.artifact_hashes(stage, self.root))
        paths[0].unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.artifact_hashes(stage, self.root)

    def test_no_notify_cli_records_failure_locally_without_authentication_or_mail(self):
        plan = {**self.plan([self.record_stage("failure", 19)]), "notifications": False}
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(pipeline, "make_plan", return_value=plan), \
             patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup"), \
             contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["--output", str(self.output), "--data-root", str(self.root),
                                  "--execute", "--no-notify"])
        self.assertEqual(code, 19)
        self.preflight.assert_not_called()
        self.notify.assert_not_called()
        receipt = json.loads((self.output / "local_failure_notice.json").read_text())
        self.assertTrue(receipt["disabled"])
        self.assertFalse(receipt["pending"])
        self.assertEqual(receipt["event"]["stage"], "failure")
        self.assertEqual(receipt["event"]["exit_code"], 19)

    def test_no_notify_also_suppresses_bootstrap_failure_mail(self):
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(pipeline, "make_plan", side_effect=ValueError("fixture invalid plan")), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(["--output", str(self.output), "--data-root", str(self.root),
                                  "--execute", "--no-notify"])
        self.assertNotEqual(code, 0)
        self.preflight.assert_not_called()
        self.notify.assert_not_called()
        notice = json.loads((self.output / "local_failure_notice.json").read_text())
        self.assertEqual(notice["event"]["stage"], "plan_validation")
        self.assertTrue(notice["disabled"])

    def test_supervisor_deadline_independently_cleans_worker_and_records_timeout(self):
        worker = Mock(pid=123462)
        worker.poll.return_value = None
        plan = {**self.plan([]), "deadline_unix": 1.0}
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup") as cleanup, \
             patch.object(pipeline.subprocess, "Popen", return_value=worker), \
             contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(plan, self.output, "supervisor-deadline")
        self.assertEqual(code, 124)
        cleanup.assert_called_once_with(worker)
        failure = json.loads((self.output / "pipeline_failure.json").read_text())
        self.assertEqual((failure["error_type"], failure["exit_code"]), ("DeadlineExceeded", 124))

    def test_skipped_optional_campaign_hashes_decision_without_generation_artifacts(self):
        decision = self.output / "budget_decision.json"
        decision.write_text('{"status": "skipped", "reason": "insufficient_time"}')
        stage = {"name": "functions_standard", "command": ["runner", "functions", "--output", str(self.output), "--execute"]}
        hashes = pipeline.artifact_hashes(stage, self.root)
        self.assertEqual(set(hashes), {str(decision)})
        decision.write_text('{"status": "skipped", "reason": "changed"}')
        self.assertNotEqual(hashes, pipeline.artifact_hashes(stage, self.root))

    def test_frozen_source_change_before_stage_prevents_execution(self):
        source = self.base / "fixture.py"
        source.write_text("original")
        plan = self.plan([self.record_stage("must_not_run")])
        plan["source_sha256"] = {source.name: pipeline.hashlib.sha256(source.read_bytes()).hexdigest()}
        source.write_text("changed")
        with patch.object(pipeline, "ROOT", self.base), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(pipeline.run_worker(plan, self.output, "source-changed"), 0)
        self.assertFalse(self.events.exists())

    def test_source_change_during_stage_blocks_success_stamp_and_next_stage(self):
        source = self.base / "fixture.py"
        source.write_text("original")
        mutation = self.stage("long_stage", f"from pathlib import Path; Path({str(source)!r}).write_text('changed')")
        plan = self.plan([mutation, self.record_stage("must_not_run")])
        plan["source_sha256"] = {source.name: pipeline.hashlib.sha256(source.read_bytes()).hexdigest()}
        with patch.object(pipeline, "ROOT", self.base), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertNotEqual(pipeline.run_worker(plan, self.output, "during-stage"), 0)
        self.assertFalse((self.output / "stage_results/long_stage.json").exists())
        self.assertFalse(self.events.exists())
        self.assertEqual(json.loads((self.output / "pipeline_state.json").read_text())["stage"], "long_stage")

    def test_analysis_artifacts_include_campaign_summary_when_generated(self):
        for index in range(4):
            shard = self.output / f"shard-{index}"
            shard.mkdir()
            (shard / "diagnostics.json").write_text("{}")
        summary = self.output / "campaign_summary.json"
        summary.write_text('{"cases": 12}')
        stage = {"name": "functions_analysis", "command": ["runner", "--output", str(self.output), "--analyze"]}
        hashes = pipeline.artifact_hashes(stage, self.root)
        self.assertIn(str(summary), hashes)
        summary.write_text('{"cases": 11}')
        self.assertNotEqual(hashes, pipeline.artifact_hashes(stage, self.root))

    def test_failed_stage_stops_before_later_stages_without_success_stamp(self):
        plan = self.plan([self.record_stage("alpha"), self.record_stage("broken", 23),
                          self.record_stage("must_not_run")])
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.run_worker(plan, self.output, "invocation-failure")
        self.assertEqual(code, 23)
        self.assertEqual(self.events.read_text().splitlines(), ["alpha", "broken"])
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual((state["status"], state["stage"], state["exit_code"]), ("failed", "broken", 23))
        self.assertFalse((self.output / "stage_results/broken.json").exists())
        self.assertFalse((self.output / "logs/must_not_run.log").exists())

    def test_stage_sigkill_is_preserved_as_signal_failure(self):
        killed = self.stage("killed", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)")
        with contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.run_worker(self.plan([killed, self.record_stage("must_not_run")]),
                                       self.output, "invocation-killed-stage")
        self.assertEqual(code, 128 + signal.SIGKILL)
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual(state["exit_code"], -signal.SIGKILL)
        self.assertEqual(state["stage"], "killed")
        self.assertFalse(self.events.exists())

    def test_resume_skips_completed_work_preserves_log_and_reruns_cheap_checks(self):
        plan = self.plan([self.record_stage("data_verify"), self.record_stage("expensive_work")])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.run_worker(plan, self.output, "first"), 0)
            original_log = (self.output / "logs/expensive_work.log").read_bytes()
            self.assertEqual(pipeline.run_worker(plan, self.output, "second"), 0)
        self.assertEqual(self.events.read_text().splitlines(), ["data_verify", "expensive_work", "data_verify"])
        self.assertEqual((self.output / "logs/expensive_work.log").read_bytes(), original_log)
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual((state["status"], state["invocation_id"]), ("completed", "second"))

    def test_resume_refuses_missing_completed_artifact(self):
        artifact_dir = self.base / "campaign"
        script = ("from pathlib import Path; "
                  f"d=Path({str(artifact_dir)!r}); "
                  "[(d/f'shard-{i}').mkdir(parents=True,exist_ok=True) for i in range(4)]; "
                  "[(d/f'shard-{i}'/'diagnostics.json').write_text('{}') for i in range(4)]; "
                  f"Path({str(self.events)!r}).open('a').write('analysis\\n')")
        stage = self.stage("analysis", script, "--output", artifact_dir, "--analyze")
        plan = self.plan([stage])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(pipeline.run_worker(plan, self.output, "first"), 0)
            (artifact_dir / "shard-2/diagnostics.json").unlink()
            self.assertNotEqual(pipeline.run_worker(plan, self.output, "second"), 0)
        self.assertEqual(self.events.read_text().splitlines(), ["analysis"])
        state = json.loads((self.output / "pipeline_state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stage"], "analysis")

    def test_supervisor_runs_tiny_worker_to_completion_without_notification(self):
        plan = self.plan([self.record_stage("tiny")])
        pipeline.write_json(self.output / "pipeline_plan.json", plan)
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup") as cleanup, \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(plan, self.output, "supervised-success")
        self.assertEqual(code, 0)
        self.assertEqual(self.events.read_text().splitlines(), ["tiny"])
        self.notify.assert_not_called()
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.args[0].returncode, 0)

    def test_supervisor_emails_failed_stage_metadata_without_log_contents(self):
        private_value = "fixture-private-value-never-email"
        stage = self.stage("failure", "import os,sys; print(os.environ['PIPELINE_PRIVATE_TEST_VALUE']); sys.exit(19)")
        plan = self.plan([stage, self.record_stage("must_not_run")])
        pipeline.write_json(self.output / "pipeline_plan.json", plan)
        with patch.dict(os.environ, {"PIPELINE_PRIVATE_TEST_VALUE": private_value}), \
                patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup"), \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(plan, self.output, "supervised-failure")
        self.assertEqual(code, 19)
        self.notify.assert_called_once()
        directory, event = self.notify.call_args.args
        self.assertEqual(directory, self.output)
        self.assertEqual(event["stage"], "failure")
        self.assertEqual(event["exit_code"], 19)
        self.assertNotIn(private_value, json.dumps(event))
        self.assertIn(private_value, (self.output / "logs/failure.log").read_text())
        self.assertFalse(self.events.exists())

    def test_enabled_supervisor_clears_inherited_email_disable_for_real_stage_exception(self):
        secret = 'controlled-stage-private-exception-body'
        stage = self.stage('early_setup', "import os; raise RuntimeError(os.environ['CONTROLLED_ERROR_BODY'])")
        plan = {**self.plan([stage, self.record_stage('must_not_run')]), 'notifications': True}
        pipeline.write_json(self.output/'pipeline_plan.json', plan)
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '1', 'CONTROLLED_ERROR_BODY': secret}), \
                patch.object(pipeline, 'enable_subreaper'), patch.object(pipeline, 'cleanup'), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.supervise(plan, self.output, 'real-early-exception'), 1)
            self.assertNotIn('HYBRID_DARD_DISABLE_EMAIL', os.environ)
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual(event['stage'], 'early_setup')
        self.assertEqual(event['exit_code'], 1)
        self.assertNotIn(secret, json.dumps(event))
        self.assertIn(secret, (self.output/'logs/early_setup.log').read_text())
        self.assertFalse(self.events.exists())

    def test_enabled_cli_clears_inherited_disable_even_for_plan_validation_failure(self):
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '1'}), \
                patch.object(pipeline, 'make_plan', side_effect=ValueError('controlled invalid plan')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(['--output', str(self.output), '--data-root', str(self.root), '--execute'])
            self.assertNotIn('HYBRID_DARD_DISABLE_EMAIL', os.environ)
        self.assertNotEqual(code, 0)
        self.notify.assert_called_once()
        self.assertEqual(self.notify.call_args.args[1]['stage'], 'plan_validation')
        self.assertFalse((self.output/'local_failure_notice.json').exists())

    def test_runtime_unbuffered_logging_overrides_inherited_value_in_real_child(self):
        marker = self.base/'runtime.json'
        stage = self.stage('runtime', "import json,os,sys; from pathlib import Path; "
                           f"Path({str(marker)!r}).write_text(json.dumps(dict(env=os.environ.get('PYTHONUNBUFFERED'), "
                           "write_through=sys.stdout.write_through)))")
        with patch.dict(os.environ, {'PYTHONUNBUFFERED': '0'}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.runtime_env(self.root)['PYTHONUNBUFFERED'], '1')
            self.assertEqual(pipeline.run_worker(self.plan([stage]), self.output, 'runtime-unbuffered'), 0)
        observed = json.loads(marker.read_text())
        self.assertEqual(observed, {'env': '1', 'write_through': True})

    def test_real_postprocessing_sigkill_notifies_after_preserving_generation_artifact(self):
        generated = self.base/'preserved-generation.json'
        generation = self.stage('generation', 'from pathlib import Path; '
                                f"Path({str(generated)!r}).write_text('{{\"generated\":true}}')")
        postprocess = self.stage('postprocessing', 'import os,signal; os.kill(os.getpid(),signal.SIGKILL)')
        plan = {**self.plan([generation, postprocess, self.record_stage('must_not_run')]), 'notifications': True}
        pipeline.write_json(self.output/'pipeline_plan.json', plan)
        with patch.object(pipeline, 'enable_subreaper'), patch.object(pipeline, 'cleanup'), \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(plan, self.output, 'real-postprocess-sigkill')
        self.assertEqual(code, 137)
        self.assertEqual(json.loads(generated.read_text()), {'generated': True})
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual(event['stage'], 'postprocessing')
        self.assertEqual(event['signal'], signal.SIGKILL)
        self.assertEqual(event['exit_code'], -signal.SIGKILL)
        self.assertFalse(self.events.exists())
        self.assertFalse((self.output/'stage_results/postprocessing.json').exists())

    def test_pending_unpersisted_notification_gets_durable_fallback_for_guard_retry(self):
        event = pipeline.failure_event(self.output, 'busy-outbox', 'calibration', 7)
        self.notify.return_value = {'smtp_accepted': False, 'pending': True,
                                    'receipt_persisted': False, 'error_type': 'OutboxBusy'}
        with patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop('HYBRID_DARD_DISABLE_EMAIL', None)
            receipt = pipeline.send_failure(self.output, event)
        self.assertTrue(receipt['pending'])
        fallback = json.loads((self.output/'notification_fallback.json').read_text())
        self.assertTrue(fallback['pending'])
        self.assertEqual(fallback['event']['event_id'], event['event_id'])

    def test_test_email_cli_sends_only_the_explicit_test_without_starting_experiments(self):
        with patch.dict(os.environ, {'HYBRID_DARD_DISABLE_EMAIL': '1'}), \
                patch.object(pipeline, 'make_plan') as plan, patch.object(pipeline, 'supervise') as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(['--output', str(self.output), '--test-email']), 0)
        plan.assert_not_called()
        run.assert_not_called()
        self.preflight.assert_not_called()
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual((event['stage'], event['error_type'], event['exit_code']),
                         ('notification_test', 'ExplicitTest', 0))

    def test_supervisor_handles_killed_worker_and_does_not_trust_stale_state(self):
        plan = self.plan([])
        pipeline.write_json(self.output / "pipeline_state.json",
                            {"invocation_id": "previous-invocation", "status": "completed", "stage": "stale", "exit_code": 0})
        worker = Mock(pid=123456)
        worker.poll.return_value = -signal.SIGKILL
        worker.wait.return_value = -signal.SIGKILL
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup") as cleanup, \
                patch.object(pipeline.subprocess, "Popen", return_value=worker), \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(plan, self.output, "current-invocation")
        self.assertEqual(code, 128 + signal.SIGKILL)
        cleanup.assert_called_once_with(worker)
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual(event["signal"], signal.SIGKILL)
        self.assertEqual(event["worker_exit_code"], -signal.SIGKILL)
        self.assertNotEqual(event["stage"], "stale")
        self.assertEqual(event["invocation_id"], "current-invocation")

    def test_supervisor_zero_exit_without_completion_record_is_failure(self):
        worker = Mock(pid=123457)
        worker.poll.return_value = worker.wait.return_value = 0
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup"), \
                patch.object(pipeline.subprocess, "Popen", return_value=worker), \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(self.plan([]), self.output, "no-state")
        self.assertNotEqual(code, 0)
        self.notify.assert_called_once()

    def test_plan_only_does_not_start_process_or_authenticate_smtp(self):
        plan = self.plan([self.record_stage("never_run")])
        with patch.object(pipeline, "make_plan", return_value=plan), \
                patch.object(pipeline.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.main(["--output", str(self.output), "--data-root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.output / "pipeline_plan.json").read_text()), plan)
        popen.assert_not_called()
        self.preflight.assert_not_called()
        self.notify.assert_not_called()
        self.assertFalse(self.events.exists())

    def test_bootstrap_failure_alert_contains_error_type_not_exception_message(self):
        private_value = "fixture-bootstrap-secret"
        with patch.object(pipeline, "make_plan", side_effect=ValueError(private_value)), \
                patch.object(pipeline.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = pipeline.main(["--output", str(self.output), "--data-root", str(self.root), "--execute"])
        self.assertNotEqual(code, 0)
        popen.assert_not_called()
        self.notify.assert_called_once()
        event = self.notify.call_args.args[1]
        self.assertEqual(event["stage"], "plan_validation")
        self.assertEqual(event["error_type"], "ValueError")
        self.assertNotIn(private_value, json.dumps(event) + stdout.getvalue() + stderr.getvalue())

    def test_signal_owned_checks_process_identity_and_never_signals_external_pid(self):
        # No real signals: one descendant retains its start time, the other PID
        # has been recycled and must not be targeted.
        def proc_stat(path, *args, **kwargs):
            pid = int(path.parent.name)
            start = "start-one" if pid == 111 else "recycled-start"
            return "1 (fixture) " + " ".join(["S"] + ["0"]*18 + [start])
        with patch.object(pipeline, "descendants", return_value={111: "start-one", 222: "old-start"}), \
                patch.object(Path, "read_text", proc_stat), patch.object(pipeline.os, "kill") as kill:
            pipeline.signal_owned(signal.SIGTERM)
        kill.assert_called_once_with(111, signal.SIGTERM)

    def test_ignored_hangup_remains_ignored_when_installing_and_restoring_handlers(self):
        with patch.object(pipeline.signal, "getsignal", return_value=signal.SIG_IGN), \
                patch.object(pipeline.signal, "signal", return_value=signal.SIG_DFL) as install:
            previous = pipeline.install_handlers()
            pipeline.restore_handlers(previous)
        self.assertEqual(set(previous), {signal.SIGINT, signal.SIGTERM})
        self.assertEqual([entry.args[0] for entry in install.call_args_list],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGINT, signal.SIGTERM])

    def test_unexpected_notifier_failure_keeps_original_stage_code_and_fallback_record(self):
        invocation = "notification-failed"
        pipeline.write_json(self.output / "pipeline_state.json", {
            "invocation_id": invocation, "status": "failed", "stage": "original_stage",
            "exit_code": 29, "log_path": str(self.output / "logs/original_stage.log")})
        worker = Mock(pid=123458)
        worker.poll.return_value = worker.wait.return_value = 29
        private_value = "fixture-mail-failure-secret"
        self.notify.side_effect = RuntimeError(private_value)
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup"), \
                patch.object(pipeline.subprocess, "Popen", return_value=worker), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = pipeline.supervise(self.plan([]), self.output, invocation)
        self.assertEqual(code, 29)
        self.notify.assert_called_once()
        fallback = json.loads((self.output / "notification_fallback.json").read_text())
        self.assertTrue(fallback["pending"])
        self.assertFalse(fallback["smtp_accepted"])
        self.assertEqual(fallback["error_type"], "RuntimeError")
        self.assertEqual((fallback["event"]["stage"], fallback["event"]["exit_code"]),
                         ("original_stage", 29))
        self.assertNotIn(private_value, json.dumps(fallback) + stdout.getvalue())

    def test_recovery_rejects_active_previous_worker_and_notifies_without_killing_it(self):
        state = {"status": "running", "invocation_id": "old", "worker_pid": 123459,
                 "worker_start_ticks": "same-start", "stage": "functions_full"}
        pipeline.write_json(self.output / "pipeline_state.json", state)
        with patch.object(pipeline, "process_start", return_value="same-start"), \
                patch.object(pipeline.os, "kill") as kill, \
                patch.object(pipeline.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                pipeline.recover_previous(self.output)
        self.assertEqual(json.loads((self.output / "pipeline_state.json").read_text()), state)
        event = self.notify.call_args.args[1]
        self.assertEqual(event["error_type"], "PriorWorkerStillActive")
        self.assertEqual(event["invocation_id"], "old")
        kill.assert_not_called()
        popen.assert_not_called()

    def test_recovery_records_dead_or_recycled_worker_once(self):
        for current_start in (None, "different-start"):
            with self.subTest(current_start=current_start):
                state = {"status": "running", "invocation_id": "old", "worker_pid": 123460,
                         "worker_start_ticks": "old-start", "stage": "repositories_full"}
                pipeline.write_json(self.output / "pipeline_state.json", state)
                self.notify.reset_mock()
                with patch.object(pipeline, "process_start", return_value=current_start), \
                        patch.object(pipeline.os, "kill") as kill, contextlib.redirect_stdout(io.StringIO()):
                    pipeline.recover_previous(self.output)
                    pipeline.recover_previous(self.output)
                recovered = json.loads((self.output / "pipeline_state.json").read_text())
                self.assertEqual(recovered["status"], "interrupted")
                self.assertIn("recovery_detected_unix", recovered)
                self.notify.assert_called_once()
                self.assertEqual(self.notify.call_args.args[1]["error_type"], "UnrecordedPreviousExit")
                kill.assert_not_called()

    def test_preparation_artifacts_pin_all_native_environment_lockfiles(self):
        names = ("inference", "swebench", "function", "evalplus")
        paths = [self.root / "envs/locks" / (name + suffix) for name in names
                 for suffix in (".freeze.txt", ".conda.txt")]
        paths.append(self.root / "envs/locks/system.json")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.stem + "==fixture\n")
        stage = {"name": "setup_all", "command": ["bash", "scripts/setup_env.sh", "all"]}
        first = pipeline.artifact_hashes(stage, self.root)
        self.assertEqual(set(first), {str(path) for path in paths})
        paths[-1].write_text("changed==version\n")
        self.assertNotEqual(first, pipeline.artifact_hashes(stage, self.root))
        paths[2].unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.artifact_hashes(stage, self.root)

    def test_pipeline_rejects_resume_when_dependency_recipe_changes(self):
        repo = self.base / "fixture_repo"
        (repo / "scripts").mkdir(parents=True)
        requirements = repo / "requirements.evalplus.txt"
        requirements.write_text("fixture==1.0\n")
        args = pipeline.parser().parse_args(["--data-root", str(self.root), "--output", str(self.output)])
        with patch.object(pipeline, "ROOT", repo):
            first = pipeline.make_plan(args)
            pipeline.freeze_plan(self.output, first)
            self.assertIn(requirements.name, first["source_sha256"])
            requirements.write_text("fixture==2.0\n")
            changed = pipeline.make_plan(args)
            self.assertNotEqual(first["source_sha256"], changed["source_sha256"])
            with self.assertRaisesRegex(ValueError, "settings/source changed"):
                pipeline.freeze_plan(self.output, changed)

    def test_four_gpu_plan_also_freezes_dependency_recipes(self):
        from datasets_code import Task
        from scripts import four_gpu
        repo = self.base / "fixture_repo"
        (repo / "scripts").mkdir(parents=True)
        for name in ("four_gpu.py", "run_four_gpu.sh", "runtime_env.sh"):
            (repo / "scripts" / name).write_text("# source fixture\n")
        requirements = repo / "requirements.inference.txt"
        requirements.write_text("fixture==1.0\n")
        args = four_gpu.parser().parse_args(["functions", "--data-root", str(self.root),
                                             "--output", str(self.output)])
        tasks = [Task("humaneval", "py", f"fixture/{index}", index, "prompt", "test") for index in range(4)]
        gpus = [{"uuid": f"GPU-fixture-{index}", "name": "Fixture"} for index in range(4)]
        with patch.object(four_gpu, "ROOT", repo), \
             patch.object(four_gpu, "__file__", str(repo / "scripts/four_gpu.py")), \
             patch("datasets_code.load_tasks", return_value=tasks):
            first, _ = four_gpu.make_plan(args, gpus)
            self.assertIn(requirements.name, first["source_sha256"])
            requirements.write_text("fixture==2.0\n")
            second, _ = four_gpu.make_plan(args, gpus)
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])

    def test_four_gpu_custom_root_overrides_inherited_runtime_cache_paths(self):
        from scripts import four_gpu
        root = self.root / "custom"
        worker = {"gpu": {"uuid": "GPU-fixture-0"}}
        expected = {
            "CONDA_PKGS_DIRS": "cache/conda", "PIP_CACHE_DIR": "cache/pip",
            "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
            "HF_DATASETS_CACHE": "cache/huggingface/datasets", "TORCH_HOME": "cache/torch",
            "XDG_CACHE_HOME": "cache/xdg", "EVALPLUS_CACHE_DIR": "cache/evalplus",
            "BIGCODEBENCH_CACHE_DIR": "cache/bigcodebench", "NLTK_DATA": "cache/nltk_data",
            "TRITON_CACHE_DIR": "cache/four-gpu/GPU-fixture-0/triton",
            "CUDA_CACHE_PATH": "cache/four-gpu/GPU-fixture-0/cuda",
            "HF_MODULES_CACHE": "cache/four-gpu/GPU-fixture-0/modules", "TMPDIR": "work/tmp",
        }
        with patch.dict(os.environ, {name: "/wrong/inherited/root" for name in expected}):
            env = four_gpu.worker_env({"root": str(root)}, worker)
        for name, relative in expected.items():
            with self.subTest(variable=name):
                self.assertEqual(env[name], str(root / relative))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-fixture-0")

    def test_native_repository_preparation_records_only_its_suite_lock(self):
        for stage_name, suite in (("prepare_verified_repositories", "swebench_verified"),
                                  ("prepare_multilingual_repositories", "swebench_multilingual")):
            path = self.root / "datasets" / suite / "repositories.lock.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"fixture": "environment identity"}')
            stage = {"name": stage_name, "command": [sys.executable, "prepare_repositories.py", "--prepare"]}
            self.assertEqual(set(pipeline.artifact_hashes(stage, self.root)), {str(path)})

    def test_default_plan_directory_is_distinct_for_native_protocol(self):
        args = ["--data-root", str(self.root)]
        with patch.object(pipeline, "make_plan", side_effect=lambda parsed: self.plan([])) as make_plan, \
             patch.object(pipeline.subprocess, "Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.main(args)
        self.assertEqual(result, 0)
        self.assertEqual(make_plan.call_args.args[0].output, self.root / "results/code-generation")
        popen.assert_not_called()
        self.preflight.assert_not_called()

    def test_function_preparation_artifacts_do_not_require_repository_or_library_environments(self):
        paths = []
        for name in ("inference", "function", "evalplus"):
            for suffix in (".freeze.txt", ".conda.txt"):
                path = self.root / "envs/locks" / (name + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture==1\n")
                paths.append(path)
        stage = {"name": "setup_functions", "command": ["bash", "scripts/setup_env.sh", "functions"]}
        self.assertEqual(set(pipeline.artifact_hashes(stage, self.root)), {str(path) for path in paths})
        paths[-1].unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.artifact_hashes(stage, self.root)

    def test_scope_change_cannot_reuse_a_frozen_function_pipeline(self):
        args = pipeline.parser().parse_args(["--data-root", str(self.root), "--output", str(self.output)])
        initial = pipeline.make_plan(args)
        self.assertEqual(initial["scope"], "functions")
        pipeline.freeze_plan(self.output, initial)
        args.scope = "repositories"
        changed = pipeline.make_plan(args)
        self.assertNotEqual(initial["stages"], changed["stages"])
        with self.assertRaisesRegex(ValueError, "settings/source changed"):
            pipeline.freeze_plan(self.output, changed)

    def test_supervisor_preserves_stage_signal_even_when_worker_maps_it_to_positive_exit(self):
        invocation = "stage-sigkill"
        pipeline.write_json(self.output / "pipeline_state.json", {
            "invocation_id": invocation, "status": "failed", "stage": "killed_stage",
            "exit_code": -signal.SIGKILL})
        worker = Mock(pid=123461)
        worker.poll.return_value = worker.wait.return_value = 128 + signal.SIGKILL
        with patch.object(pipeline, "enable_subreaper"), patch.object(pipeline, "cleanup"), \
                patch.object(pipeline.subprocess, "Popen", return_value=worker), \
                contextlib.redirect_stdout(io.StringIO()):
            code = pipeline.supervise(self.plan([]), self.output, invocation)
        self.assertEqual(code, 128 + signal.SIGKILL)
        event = self.notify.call_args.args[1]
        self.assertEqual(event["exit_code"], -signal.SIGKILL)
        self.assertEqual(event["worker_exit_code"], 128 + signal.SIGKILL)
        self.assertEqual(event["signal"], signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
