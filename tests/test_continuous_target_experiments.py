"""CPU-only contract and lifecycle checks for continuous target optimization."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import csv
import json
import xml.etree.ElementTree as ET

import pytest

from iterative_tuning import AR, CPGRPO, INSTRUCT, DREAM_CODER, candidate, digest, file_sha, write_json
from scripts.tradeoff_experiments import configured, shared_context
from scripts import continuous_target_experiments as controller


def ar_reference():
    return candidate(AR, decoder="ar", tokens=512, native_steps=1,
                     temperature=0, top_p=1, forward_budget=512, prefix="code_fence")

def model_config(model=CPGRPO, *, decoder="dard", steps=64, temperature=0.):
    value = configured(decoder=decoder, steps=steps, temperature=temperature,
                       top_p=1., tau_c=.6, tau_u=.9)
    value.update(model=model[0], revision=model[1])
    value.pop("id")
    value["id"] = decoder + "-" + digest(value)[:16]
    return value


def summary(config, accuracy=.7, seconds=.8):
    return {"candidate":config, "candidate_id":config["id"], "valid_for_selection":True,
            "pass_rate":accuracy, "mean_generation_seconds":seconds}


def point(row, *, rank=.01):
    return {**deepcopy(row), "rank_vector":[rank,0,0,0,0,0], "accuracy_gap_pp":6.,
            "distances":{"ar_accuracy_gap_excess":rank,"native_accuracy_gain_shortfall":0.,
                "ar_accuracy_order_shortfall":0.,"native_over_dard_latency_excess":0.,
                "dard_over_ar_latency_excess":0.}}


def plan_fixture(tmp_path, rounds=3, schedule=None):
    data = tmp_path / "data"
    data.mkdir()
    (data / "validation_tasks.jsonl").write_text('{"task_id":"validation"}\n')
    (data / "smoke_tasks.jsonl").write_text('{"task_id":"smoke"}\n')
    ar = ar_reference()
    configs = [ar, model_config(), model_config(decoder="standard", steps=8)]
    plan = {"output":str(tmp_path), "data":str(data), "deadline_unix":controller.DEADLINE,
        "final_reserve_seconds":14400, "max_rounds":rounds, "seed":20261105,
        "repeat_seeds":[20261108,20261109,20261110], "initial_candidates":configs,
        "ar_reference_id":ar["id"], "target":{"max_accuracy_gap":.05},
        "model_schedule":[list(x) for x in (schedule or [CPGRPO, INSTRUCT, DREAM_CODER])],
        "model_receipts":[], "candidate_receipt":str(tmp_path / "dream_receipt.json"),
        "gpus":[{"uuid":f"gpu-{i}","name":"fixture"} for i in range(4)]}
    write_json(tmp_path / "iteration_plan.json", plan)
    return plan


def smoke_rows(configs):
    return [{"candidate":c, "cases":16, "passed":0, "pass_rate":0.,
             "mean_generation_seconds":1., "by_language":{
                 language:{"cases":4,"passed":0,"pass_rate":0.} for language in ("py","js","cpp","rs")}}
            for c in configs]


def mock_lifecycle(monkeypatch, plan, *, assessment_status="no_feasible", fail_smoke_model=None):
    calls, analyses, phases, progress = [], [], {}, []
    monkeypatch.setattr(controller, "verify_plan", lambda p: None)
    monkeypatch.setattr(controller, "telemetry", lambda p, stop: None)
    monkeypatch.setattr(controller, "should_search", lambda p, now, estimate: True)
    monkeypatch.setattr(controller, "save_progress", lambda p,h,f,s: progress.append((s,len(h),deepcopy(f))))

    def propose(rows, **kwargs):
        model = kwargs["model_override"]
        configs = [model_config(model, steps=32+kwargs["round_index"]*16),
                   model_config(model, decoder="standard", steps=4),
                   model_config(model, decoder="standard", steps=8)]
        return {"candidates":configs, "reason":"CPU fixture"}

    monkeypatch.setattr(controller, "propose_next_candidates", propose)

    def evaluate(active, label, configs, split="validation", **kwargs):
        calls.append({"label":label,"seed":active["seed"],"split":split,
                      "configs":deepcopy(configs),"deadline":kwargs.get("deadline")})
        assert any(c["id"] == plan["ar_reference_id"] and c == plan["initial_candidates"][0]
                   for c in configs), "The fixed AR must be remeasured in every phase"
        assert split != "holdout"
        if split == "smoke" and any(c["model"] == fail_smoke_model for c in configs):
            raise RuntimeError("GPU smoke failed")
        path = Path(plan["output"]) / label
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "results.json", {"fixture":True})
        phases[label] = deepcopy(configs)
        return smoke_rows(configs) if split == "smoke" else []

    monkeypatch.setattr(controller, "evaluate", evaluate)

    def analyze(active, phase_names, name):
        analyses.append((list(phase_names),name))
        configs = phases[phase_names[-1]]
        rows = [summary(c, .75 if c["decoder"]=="ar" else .70 if c["decoder"]=="dard" else .5,
                        2. if c["decoder"]=="ar" else .8 if c["decoder"]=="dard" else .4)
                for c in configs]
        seeds = ([int(label.rsplit("_",1)[1]) for label in phase_names]
                 if name=="finalists" else [active["seed"]])
        for row in rows:
            row.update(expected_cases=len(seeds), observed_cases=len(seeds), seeds=seeds)
        return {"summaries":rows}, {"status":assessment_status}

    monkeypatch.setattr(controller, "analyze", analyze)
    monkeypatch.setattr(controller, "score_candidates", lambda rows,*a:
        {"ranked_candidates":[point(r) for r in rows if r["candidate"]["decoder"]=="dard"]})
    return calls, analyses, progress


def test_deadline_is_november_7_2030_kst_and_search_reserves_final_time():
    deadline = datetime.fromtimestamp(controller.DEADLINE, ZoneInfo("Asia/Seoul"))
    assert deadline.isoformat() == "2026-11-07T20:30:00+09:00"
    plan = {"deadline_unix":1000, "final_reserve_seconds":100}
    assert controller.should_search(plan,700,100)
    assert not controller.should_search(plan,800,100)


def test_save_progress_writes_readable_json_csv_and_standalone_svg(tmp_path):
    plan = plan_fixture(tmp_path)
    first = point(summary(model_config(steps=64), .69, .8), rank=.01)
    second = point(summary(model_config(steps=128), .71, 1.1), rank=0.)
    second["accuracy_gap_pp"] = 4.
    history = [{"round":index, "model":CPGRPO[0], "case_count":48,
                "elapsed_seconds":20. + index,
                "report":{"summaries":[entry]},
                "scores":{"ranked_candidates":[entry]}}
               for index, entry in enumerate((first, second))]

    controller.save_progress(plan, history, [], "adaptive_search")

    progress = json.loads((tmp_path / "search_progress.json").read_text())
    assert progress["completed_rounds"] == 2
    assert progress["search_should_continue"] is True
    assert progress["closest_observed"]["candidate_id"] == second["candidate_id"]
    assert all("report" not in row and "scores" not in row for row in progress["rounds"])
    with (tmp_path / "search_progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["round"]) for row in rows] == [0, 1]
    assert [float(row["closest_gap_pp"]) for row in rows] == [6., 4.]
    assert [int(row["cases"]) for row in rows] == [48, 48]
    svg = ET.parse(tmp_path / "search_progress.svg").getroot()
    namespace = {"svg":"http://www.w3.org/2000/svg"}
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(svg.findall(".//svg:circle", namespace)) == 2
    series = svg.find(".//svg:polyline", namespace)
    assert series is not None and len(series.attrib["points"].split()) == 2
    assert "Requested limit: 5 pp" in "".join(svg.itertext())


def test_prepare_accepts_four_distinct_gpu_records_and_preserves_ar(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    old = {"data_root":"fixture", "data":"fixture-data", "data_sha256":{}, "python":"python",
        "gpus":[{"uuid":f"gpu-{i}","name":"fixture"} for i in range(4)],
        "package_versions":{}, "model_receipts":[], "grading_timeout":300,
        "generation_timeout":120, "evaluation_options":{}}
    write_json(previous / "iteration_plan.json",old)
    ar = ar_reference()
    write_json(previous / "final_selection.json", {"winners_by_decoder":[{"candidate":ar}]})
    monkeypatch.setattr(controller,"verify_plan",lambda p:None)
    monkeypatch.setattr(controller,"source_files",lambda root:[])
    plan = controller.prepare(SimpleNamespace(output=tmp_path / "new", previous=previous))
    assert plan["gpus"] == old["gpus"]
    assert plan["deadline_unix"] == 1794051000
    assert plan["evaluation_options"]["grading_workers"] == 8
    assert plan["evaluation_options"]["grading_queue_size"] == 64
    assert next(c for c in plan["initial_candidates"] if c["id"]==ar["id"]) == ar
    assert set(plan["repeat_seeds"]).isdisjoint({20261105,20261106,20261107})


def test_optional_model_requires_exact_verified_nonempty_receipt(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path)
    original = deepcopy(plan)
    checked=[]
    monkeypatch.setattr(controller,"verify_plan",lambda p:checked.append(deepcopy(p)))
    assert controller.model_plan(plan,DREAM_CODER) is None
    path=Path(plan["candidate_receipt"])
    for receipt in [
        {"status":"pending","model":DREAM_CODER[0],"revision":DREAM_CODER[1],"files":[{}]},
        {"status":"verified","model":DREAM_CODER[0],"revision":"wrong","files":[{}]},
        {"status":"verified","model":DREAM_CODER[0],"revision":DREAM_CODER[1],"files":[]},
    ]:
        write_json(path,receipt)
        with pytest.raises(ValueError):controller.model_plan(plan,DREAM_CODER)
    write_json(path,{"status":"verified","model":DREAM_CODER[0],"revision":DREAM_CODER[1],"files":[{}]})
    active=controller.model_plan(plan,DREAM_CODER)
    assert active["model_receipts"][-1] == {"path":str(path),"sha256":file_sha(path)}
    assert checked[-1]==active
    assert plan==original
    with pytest.raises(ValueError):controller.model_plan(plan,("unapproved","revision"))


@pytest.mark.parametrize("status",["feasible","no_feasible"])
def test_continues_after_hit_or_miss_and_analyzes_rounds_separately(tmp_path,monkeypatch,status):
    plan=plan_fixture(tmp_path,rounds=3,schedule=[CPGRPO,INSTRUCT,CPGRPO])
    calls,analyses,_=mock_lifecycle(monkeypatch,plan,assessment_status=status)
    controller.execute(plan)
    rounds=[c for c in calls if c["label"].startswith("adaptive_round_") and c["split"]=="validation"]
    assert len(rounds)==3
    round_analyses=[phases for phases,name in analyses if name.startswith("adaptive_round_")]
    assert round_analyses==[[c["label"]] for c in rounds]
    final=[c for c in calls if c["label"].startswith("final_seed_")]
    assert [c["seed"] for c in final]==[20261108,20261109,20261110]
    assert all(c["deadline"]==controller.DEADLINE-180 for c in final)
    final_analyses=[phases for phases,name in analyses if name=="finalists"]
    labels=[c["label"] for c in final]
    assert final_analyses==[labels[:1],labels[:2],labels[:3]]
    assert json.loads((tmp_path/"target_result.json").read_text())["status"]=="completed"


def test_optional_model_smoke_failure_quarantines_without_admitting_validation(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=4,schedule=[CPGRPO,DREAM_CODER])
    write_json(Path(plan["candidate_receipt"]),{"status":"verified","model":DREAM_CODER[0],
        "revision":DREAM_CODER[1],"files":[{}]})
    calls,_,progress=mock_lifecycle(monkeypatch,plan,fail_smoke_model=DREAM_CODER[0])
    controller.execute(plan)
    dream_smokes=[c for c in calls if c["split"]=="smoke" and any(x["model"]==DREAM_CODER[0] for x in c["configs"])]
    assert len(dream_smokes)==1
    assert not any(c["split"]=="validation" and any(x["model"]==DREAM_CODER[0] for x in c["configs"]) for c in calls)
    assert any(f for _,_,f in progress)


def test_invalid_optional_receipt_quarantines_and_continues_with_current_checkpoint(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=2,schedule=[CPGRPO,DREAM_CODER])
    write_json(Path(plan["candidate_receipt"]),{"status":"verified","model":DREAM_CODER[0],
        "revision":"wrong","files":[{}]})
    calls,_,_=mock_lifecycle(monkeypatch,plan)
    controller.execute(plan)
    assert not any(any(x["model"]==DREAM_CODER[0] for x in c["configs"]) for c in calls)
    result=json.loads((tmp_path/"target_result.json").read_text())
    assert result["completed_search_rounds"]==2
    assert any(f.get("gate")=="integrity_receipt" for f in result["failures"])


def test_smoke_checks_execution_coverage_without_selecting_by_accuracy():
    configs=[model_config()]
    rows=smoke_rows(configs)
    controller.check_smoke(rows,configs)  # Zero pass rate still verifies execution coverage.
    incomplete=deepcopy(rows)
    incomplete[0]["cases"]=15
    with pytest.raises(RuntimeError):controller.check_smoke(incomplete,configs)
    incomplete=deepcopy(rows)
    del incomplete[0]["by_language"]["rs"]
    with pytest.raises(RuntimeError):controller.check_smoke(incomplete,configs)


def test_final_shortlist_repeats_complete_native_curves_across_rounds(tmp_path):
    plan=plan_fixture(tmp_path)
    dard=model_config()
    quality=model_config(temperature=.1)
    natives=[model_config(decoder="standard",steps=s) for s in (2,4,8,16,128)]
    other_native=model_config(decoder="standard",temperature=.1,steps=8)
    history=[{"round":0,"report":{"summaries":[summary(dard)]+[summary(c) for c in natives[:2]]},
              "scores":{"ranked_candidates":[point(summary(dard,.69,.8),rank=0.)]}},
             {"round":1,"report":{"summaries":[summary(quality)]+[summary(c) for c in natives[2:]]+[summary(other_native)]},
              "scores":{"ranked_candidates":[point(summary(quality,.73,3.),rank=.1)]}}]
    result=controller.final_candidates(plan,history)
    ids={c["id"] for c in result}
    assert {plan["ar_reference_id"],dard["id"],quality["id"],other_native["id"]} <= ids
    assert {c["id"] for c in natives} <= ids
    assert len(result)==len(ids)


def test_final_shortlist_keeps_feasible_point_when_equal_ar_point_wins_distance_tie(tmp_path):
    plan=plan_fixture(tmp_path)
    equal_ar=model_config(steps=64)
    feasible=model_config(steps=128)
    native=model_config(decoder="standard",steps=8)
    tied=point(summary(equal_ar,.75,.8),rank=0.)
    tied["target_met"]=False
    valid=point(summary(feasible,.74,.8),rank=0.)
    valid["target_met"]=True
    history=[{"round":0,"report":{"summaries":[summary(equal_ar),summary(feasible),summary(native)]},
              "scores":{"ranked_candidates":[tied,valid]}}]
    result=controller.final_candidates(plan,history)
    assert feasible["id"] in {c["id"] for c in result}


def test_final_timeout_preserves_complete_seed_evidence_without_completion_claim(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=1,schedule=[CPGRPO])
    _,analyses,_=mock_lifecycle(monkeypatch,plan)
    original=controller.evaluate
    def evaluate(active,label,*args,**kwargs):
        if label=="final_seed_20261109":raise TimeoutError("deadline fixture")
        return original(active,label,*args,**kwargs)
    monkeypatch.setattr(controller,"evaluate",evaluate)
    with pytest.raises(TimeoutError):controller.execute(plan)
    assert not (tmp_path/"target_result.json").exists()
    progress=json.loads((tmp_path/"final_repeat_progress.json").read_text())
    assert progress["completed_seeds"]==[20261108]
    assert json.loads((tmp_path/"iteration_state.json").read_text())["status"]=="failed"


def test_resume_uses_completed_checkpoint_when_no_more_search_time_remains(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=2,schedule=[CPGRPO])
    calls,_,_=mock_lifecycle(monkeypatch,plan)
    original=controller.evaluate
    def interrupted(active,label,*args,**kwargs):
        if label=="adaptive_round_001":raise RuntimeError("interrupt after a complete round")
        return original(active,label,*args,**kwargs)
    monkeypatch.setattr(controller,"evaluate",interrupted)
    with pytest.raises(RuntimeError):controller.execute(plan)
    checkpoint=json.loads((tmp_path/"controller_checkpoint.json").read_text())
    assert checkpoint["next_round"]==1 and len(checkpoint["history"])==1
    before=len(calls)
    monkeypatch.setattr(controller,"evaluate",original)
    monkeypatch.setattr(controller,"should_search",lambda *args:False)
    controller.execute(plan)
    assert [c["label"] for c in calls[before:]]==[
        "final_seed_20261108","final_seed_20261109","final_seed_20261110"]
    assert json.loads((tmp_path/"target_result.json").read_text())["completed_search_rounds"]==1


def test_final_reserve_increases_for_slow_finalists_and_accounts_for_three_seeds(tmp_path):
    plan=plan_fixture(tmp_path)
    ar=plan["initial_candidates"][0]
    dard=model_config()
    native=model_config(decoder="standard",steps=8)
    def history(seconds):
        rows=[summary(ar,.75,2.),summary(dard,.70,seconds),summary(native,.50,.5)]
        return [{"round":0,"report":{"summaries":rows},
                 "scores":{"ranked_candidates":[point(rows[1])]}}]
    fast=controller.final_reserve(plan,history(1.),273)
    slow=controller.final_reserve(plan,history(100.),273)
    assert fast>=14400
    assert slow>fast
    one_seed=controller.final_reserve({**plan,"repeat_seeds":[20261108]},history(100.),273)
    assert slow>one_seed


def test_completed_round_replays_without_search_budget_when_checkpoint_missing(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=1,schedule=[CPGRPO])
    write_json(tmp_path/"adaptive_round_000/results.json",{"completed":True})
    calls,_,_=mock_lifecycle(monkeypatch,plan)
    monkeypatch.setattr(controller,"should_search",lambda *args:False)
    controller.execute(plan)
    assert any(c["label"]=="adaptive_round_000" for c in calls)
    assert json.loads((tmp_path/"target_result.json").read_text())["completed_search_rounds"]==1


def test_resume_keeps_frozen_fallback_model_when_optional_receipt_arrives(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=2,schedule=[CPGRPO,DREAM_CODER])
    calls,_,_=mock_lifecycle(monkeypatch,plan)
    original=controller.evaluate
    def interrupted(active,label,*args,**kwargs):
        if label=="adaptive_round_001":raise RuntimeError("interrupted frozen fallback")
        return original(active,label,*args,**kwargs)
    monkeypatch.setattr(controller,"evaluate",interrupted)
    with pytest.raises(RuntimeError):controller.execute(plan)
    frozen=json.loads((tmp_path/"adaptive_round_001/proposal.json").read_text())
    assert all(c["model"]!=DREAM_CODER[0] for c in frozen["candidates"])
    write_json(Path(plan["candidate_receipt"]),{"status":"verified","model":DREAM_CODER[0],
        "revision":DREAM_CODER[1],"files":[{}]})
    monkeypatch.setattr(controller,"evaluate",original)
    before=len(calls)
    controller.execute(plan)
    resumed=[c for c in calls[before:] if c["label"]=="adaptive_round_001"]
    assert len(resumed)==1
    assert all(c["model"]!=DREAM_CODER[0] for c in resumed[0]["configs"])
    assert json.loads((tmp_path/"adaptive_round_001/proposal.json").read_text())==frozen


@pytest.mark.parametrize("damage",["invalid_reference","invalid_receipt","missing_seed","missing_candidate"])
def test_invalid_final_evidence_never_publishes_completion(tmp_path,monkeypatch,damage):
    plan=plan_fixture(tmp_path,rounds=1,schedule=[CPGRPO])
    mock_lifecycle(monkeypatch,plan)
    original=controller.analyze
    def analyze(active,phase_names,name):
        report,assessment=original(active,phase_names,name)
        if name=="finalists":
            if damage=="invalid_reference":assessment["status"]="invalid_reference"
            elif damage=="invalid_receipt":report["summaries"][0]["valid_for_selection"]=False
            elif damage=="missing_seed":report["summaries"][0]["seeds"]=[]
            elif damage=="missing_candidate":report["summaries"].pop()
        return report,assessment
    monkeypatch.setattr(controller,"analyze",analyze)
    with pytest.raises(ValueError,match="Final evidence"):
        controller.execute(plan)
    assert not (tmp_path/"target_result.json").exists()
    assert not (tmp_path/"final_repeat_progress.json").exists()
    assert json.loads((tmp_path/"iteration_state.json").read_text())["status"]=="failed"


def test_publication_after_deadline_is_not_marked_completed(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,rounds=1,schedule=[CPGRPO])
    mock_lifecycle(monkeypatch,plan)
    original=controller.analyze
    def analyze(active,phase_names,name):
        report,assessment=original(active,phase_names,name)
        if name=="finalists" and len(phase_names)==3:
            plan["deadline_unix"]=controller.time.time()-1
        return report,assessment
    monkeypatch.setattr(controller,"analyze",analyze)
    with pytest.raises(TimeoutError,match="before final publication"):
        controller.execute(plan)
    assert not (tmp_path/"target_result.json").exists()
    assert json.loads((tmp_path/"final_repeat_progress.json").read_text())["completed_seeds"]==plan["repeat_seeds"]
