"""Protect the fixed reference, fair pairs and validation-only follow-up."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from iterative_tuning import AR, candidate
from scripts.tradeoff_experiments import configured, shared_context
from scripts.target_tradeoff_experiments import target_grid, finalists, execute


def reference():
    return candidate(AR, decoder="ar", tokens=512, native_steps=1,
        temperature=0, top_p=1, forward_budget=512, prefix="code_fence")


def summary(config, accuracy, latency):
    return {"candidate":config,"candidate_id":config["id"],"pass_rate":accuracy,
            "mean_generation_seconds":latency,"valid_for_selection":True}


class TargetCampaignTests(unittest.TestCase):
    def test_reference_is_preserved_and_each_dard_has_a_native_control(self):
        ar = reference()
        selection = {"winners_by_decoder":[{"candidate":ar}]}
        original = copy.deepcopy(selection)
        rows, ar_id = target_grid(selection)
        self.assertEqual(selection, original)
        self.assertEqual(next(r for r in rows if r["id"]==ar_id),ar)
        self.assertEqual(len(rows),len({r["id"] for r in rows}))
        self.assertEqual(len(rows),24)
        for dard in (r for r in rows if r["decoder"]=="dard"):
            self.assertTrue(any(n["decoder"]=="standard" and shared_context(n)==shared_context(dard)
                                for n in rows))
        native_steps = {r["standard_steps"] for r in rows if r["decoder"]=="standard"}
        self.assertTrue({2,4,8,16,128,256} <= native_steps)

    def test_shortlist_keeps_quality_leader_and_complete_native_curve(self):
        ar = reference()
        fast = configured(steps=64,tau_c=.6,tau_u=.9)
        slow = configured(steps=128,tau_c=.6,tau_u=.9)
        weak = configured(decoder="standard",steps=2)
        strong = configured(decoder="standard",steps=8)
        report = {"summaries":[summary(ar,.75,2),summary(fast,.71,1),summary(slow,.73,3),
                              summary(weak,.20,.1),summary(strong,.68,.8)]}
        selected = finalists({"ar_reference_id":ar["id"]},report)
        ids = {c["id"] for c in selected}
        self.assertTrue({ar["id"],fast["id"],slow["id"],strong["id"]} <= ids)
        self.assertIn(weak["id"],ids)

    def test_feasible_lower_quality_dard_survives_shortlisting(self):
        ar = reference()
        leader = configured(steps=64,tau_c=.6,tau_u=.9)
        feasible = configured(steps=64,tau_c=.6,tau_u=.95)
        native = configured(decoder="standard",steps=8)
        report = {"summaries":[summary(ar,.75,2),summary(leader,.74,.1),
                              summary(feasible,.72,1),summary(native,.65,.4)]}
        assessment = {"feasible_triads":[{"dard_id":feasible["id"],"standard_id":native["id"]}]}
        selected = finalists({"ar_reference_id":ar["id"]},report,assessment)
        self.assertIn(feasible["id"],{c["id"] for c in selected})

    def test_campaign_repeats_frozen_validation_choices_and_does_not_use_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ar = reference()
            dard = configured(tau_c=.6,tau_u=.9)
            native = configured(decoder="standard",steps=8)
            configs = [ar,dard,native]
            plan = {"output":tmp,"data":tmp,"deadline_unix":time.time()+3600,
                    "initial_candidates":configs,"ar_reference_id":ar["id"],
                    "seed":1,"repeat_seeds":[2,3],"target":{"max_accuracy_gap":.05}}
            (output/"iteration_plan.json").write_text(json.dumps(plan))
            calls=[]
            report={"summaries":[summary(ar,.75,2),summary(dard,.72,1),summary(native,.5,.4)]}
            def fake_eval(p,label,cs,split="validation",**kwargs):
                calls.append((label,p["seed"],split))
                if label.startswith("validation_seed_"):
                    saved=json.loads((output/"shortlist.json").read_text())
                    self.assertEqual(saved["candidates"],cs)
                phase=output/label
                phase.mkdir(exist_ok=True)
                (phase/"results.json").write_text('{}')
            with patch("scripts.target_tradeoff_experiments.verify_plan"), \
                 patch("scripts.target_tradeoff_experiments.telemetry"), \
                 patch("scripts.target_tradeoff_experiments.evaluate",side_effect=fake_eval), \
                 patch("scripts.target_tradeoff_experiments.analyze",return_value=(report,{"status":"feasible"})):
                execute(plan)
            self.assertEqual(calls,[("smoke",1,"smoke"),("validation_grid",1,"validation"),
                ("validation_seed_2",2,"validation"),("validation_seed_3",3,"validation")])
            self.assertEqual(json.loads((output/"target_result.json").read_text())["assessment"]["status"],"feasible")
            self.assertEqual(json.loads((output/"iteration_state.json").read_text())["status"],"completed")

    def test_worker_failure_is_recorded_without_completion_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan={"output":tmp,"data":tmp,"deadline_unix":time.time()+3600,"initial_candidates":[]}
            with patch("scripts.target_tradeoff_experiments.verify_plan"), \
                 patch("scripts.target_tradeoff_experiments.telemetry"), \
                 patch("scripts.target_tradeoff_experiments.evaluate",side_effect=RuntimeError("worker failed")):
                with self.assertRaisesRegex(RuntimeError,"worker failed"):
                    execute(plan)
            self.assertFalse((Path(tmp)/"target_result.json").exists())
            self.assertEqual(json.loads((Path(tmp)/"iteration_state.json").read_text())["status"],"failed")


if __name__ == "__main__":
    unittest.main()
