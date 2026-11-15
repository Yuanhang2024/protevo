"""Offline reporting tests; every measurement and model response here is synthetic."""
from copy import deepcopy
import importlib.util
from itertools import product
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest

from audit import Auditor, Checks, load_frozen_core, read_json
from report import DEFINITION_VERSION, audit_analysis, completed_inputs, export_phase
from statistics_report import add_resource_metrics, confirmation, interval, paired_difference

PROJECT = Path(__file__).resolve().parents[2]
REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
BOOTSTRAP = {"seed": 20261113, "replicates": 400, "confidenceLevel": .95,
             "outerUnit": "paired data split", "innerUnit": "independent trajectory within model and split"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze(root):
    snapshot = root / "snapshot"
    for filename in ("__init__", "schema", "data", "models", "agent", "experiment", "prompts", "storage"):
        target = snapshot / f"protevo/{filename}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / f"protevo/{filename}.py", target)
    (snapshot / "prompts").mkdir()
    for filename in ("legacy-v1", "scientific-v2", "scientific-v3"):
        shutil.copyfile(PROJECT / f"prompts/{filename}.txt", snapshot / f"prompts/{filename}.txt")
    write(snapshot / "configs/knowledge.json", {"rules": [{"id": "synthetic", "scope": "test only", "condition": "test",
          "action": "test", "source": "synthetic test data", "explanation": "Synthetic offline rule"}]})
    return snapshot


class Designer:
    def __init__(self, core, config, candidates, model, fail_at=None):
        self.core, self.config, self.candidates, self.model, self.fail_at = core, config, candidates, model, fail_at
        self.used = set()
        self.prompt_metadata = core.prompts.load_prompt_profile("v3").metadata()

    def generate(self, context):
        variants = [variant for variant in self.candidates if variant not in self.used][:context["proposal_count"]]
        self.used.update(variants)
        if self.fail_at is not None and context["round"] >= self.fail_at:
            variants = ["XXXX"] * context["proposal_count"]
        proposals = [{"variant": variant, "hypothesis": "Synthetic hypothesis", "rationale": "Synthetic rationale", "knowledge_rule_ids": ["synthetic"]}
                     for variant in variants]
        return {"text": json.dumps({"analysis": "Synthetic offline observations", "proposals": proposals}), "complete": True,
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}, "model_requested": self.model,
                "model_returned": None, "api_mode": "dsh", "prompt_metadata": self.prompt_metadata}


def create_study(root, phase="phase1", failed=None):
    snapshot = freeze(root)
    core = load_frozen_core(snapshot)
    plan = {"studyId": "synthetic-report-only", "phase1": {"dataSeeds": [101, 202], "maximumGenerationCalls": 48},
            "phase3": {"dataSeeds": [101, 202], "maximumGenerationCalls": 48, "bootstrap": BOOTSTRAP}}
    source = root / "synthetic-source.csv"
    variants = ["".join(row) for row in product("ACDE", repeat=4)]
    pd.DataFrame({"variant": variants, "fitness": [float(sum((index + 1) * "ACDE".index(letter) for index, letter in enumerate(v))) for v in variants]}).to_csv(source, index=False)
    conditions, analysis_rows, runtime = [], [], []
    for seed in (101, 202):
        config = core.schema.ExperimentConfig(seed=seed, rounds=6, k=5, proposal_count=8, score_budget=8,
                    train_size=24, validation_size=16, test_size=16, pool_size=160, model_families=["random_forest"],
                    tree_estimators=5, tree_min_samples_leaf=[1], n_jobs=1, llm_max_output_tokens=None, llm_timeout=None,
                    save_predictors=False, llm_reasoning_effort="high", knowledge_path=str(snapshot / "configs/knowledge.json"))
        shared = root / "shared" / f"seed-{seed}"
        core.data.prepare(source, shared / "data", config, REFERENCE, "variant", "fitness")
        dataset = core.data.PublicDataset(shared / "data")
        selection = core.models.select_model(dataset, config, shared / "selection")
        write(shared / "ready.json", {"dataset": str(shared / "data"), "selection": str(shared / "selection/selection.json")})
        for index, model in enumerate(("synthetic-a", "synthetic-b")):
            key = f"{phase}-s{seed}-i0-{index}"
            condition = {"key": key, "operationId": f"independent-{key}", "profileKey": f"{model}-high-c8-k5-r6-reject",
                         "seed": seed, "repeatIndex": 0, "provider": "synthetic", "model": model, "modelKey": model,
                         "effort": "high", "candidatesPerRound": 8, "topK": 5, "rounds": 6,
                         "oversizePolicy": "reject", "maxCalls": 12, "effectiveMaxTokens": 128000}
            if phase == "phase3":
                condition["arm"] = "selected" if index == 0 else "reference"
            conditions.append(deepcopy(condition))
            run = root / phase / "runs" / key
            fail_at = 3 if failed == (seed, model) else None
            proposer = Designer(core, config, list(dataset.candidates.variant), model, fail_at)
            state = core.experiment.run_experiment(dataset, selection, config, run / "run", methods=["knowledge"], proposer=proposer, prompt_profile="v3")
            write(run / "worker.json", {"config": config.model_dump(), "condition": condition})
            method = state["methods"]["knowledge"]
            completed = [record for record in method["rounds"] if record["stage"] == "evaluated"]
            calls, public_rounds = [], []
            for record in method["rounds"]:
                for attempt in record["attempts"]:
                    number = len(calls) + 1
                    call = {"number": number, "round": record["round"], "usage": attempt["response"]["usage"],
                            "elapsedSeconds": 2., "generationSeconds": 1.5, "finishReason": "stop", "outcomeUnknown": False,
                            "effectiveConfig": {"provider": "synthetic", "model": model, "reasoningEffort": "high", "maxTokens": 128000}}
                    calls.append(call)
                    payload = core.agent.build_request_payload(attempt["context"], config, model, "chat", proposer.prompt_metadata["system_prompt"])
                    request = {"version": 1, "type": "llm_request", "id": f"request-{number}", "round": record["round"],
                               "payload": payload, "prompt_metadata": proposer.prompt_metadata}
                    write(root / phase / "requests" / key / f"{number}.json", {"request": request, "usage": call["usage"],
                           "transport_request": call["effectiveConfig"], "reply": {"version": 1, "type": "llm_response", "id": request["id"], "response": attempt["response"]}})
                public_rounds.append({"round": record["round"], "status": record["stage"], "top10": record.get("recommendations", [])})
            status = "completed" if len(completed) == 6 else "failed"
            condition.update(status=status, calls=calls, elapsedSeconds=15., completedRounds=len(completed))
            runtime.append(condition)
            write(root / phase / "results" / f"{key}.json", {"summary": {"rounds": public_rounds, "completed_rounds": len(completed), "calls": len(calls)}})
            fitness = [row["actual_fitness"] for record in completed for row in record["recommendations"]]
            analysis_rows.append({**condition, "complete": status == "completed", "meanFitness": float(np.mean(fitness)),
                                  "maxFitness": float(np.max(fitness)), "tokens": len(calls) * 30, "missingUsage": 0, "calls": len(calls)})
    profiles = []
    for group in ("selected", "reference") if phase == "phase3" else ("synthetic-a-high-c8-k5-r6-reject", "synthetic-b-high-c8-k5-r6-reject"):
        rows = [row for row in analysis_rows if (row.get("arm") or row["profileKey"]) == group]
        complete = [row for row in rows if row["complete"]]
        means = [[row["meanFitness"] for row in complete if row["seed"] == seed] for seed in (101, 202)]
        profiles.append({"profileKey": group, "total": len(rows), "completed": len(complete),
                         "equalSeedMeanFitness": float(np.mean([np.mean(values) for values in means])) if all(means) else None})
    analysis = {"trajectories": analysis_rows, "profiles": profiles, "selected": [{"model": "synthetic-a"}], "provisional": False}
    if phase == "phase3":
        specification = importlib.util.spec_from_file_location("reference_calibration_statistics", PROJECT / "tools/calibration/statistics_calibration.py")
        reference = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(reference)
        for row in analysis_rows:
            row["completionIndicator"] = int(row["complete"])
        analysis["intervals"] = {"meanFitness": reference.hierarchical_interval(analysis_rows, "meanFitness", plan[phase]),
                                 "maxFitness": reference.hierarchical_interval(analysis_rows, "maxFitness", plan[phase]),
                                 "completionRate": reference.hierarchical_interval(analysis_rows, "completionIndicator", plan[phase], False)}
        analysis["selectedArm"] = "selected"
    write(root / "plan.json", plan)
    write(root / phase / "execution.json", {"conditions": conditions, "maximumCalls": 48})
    write(root / phase / "state.json", {"phase": "completed", "conditions": runtime})
    write(root / phase / "analysis.json", analysis)
    return runtime


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "synthetic-study"
    create_study(root)
    return root


def test_export_audits_synthetic_runs_and_rebuilds_shared_initial_model_once(study):
    events = []
    metadata, output = export_phase(study, "phase1", confirmed=True, label="synthetic", formats=(), progress=events.append)
    assert [event["audited_trajectories"] for event in events] == [1, 2, 3, 4]
    assert events[-1]["check_totals"]["errors"] == 0
    assert metadata["audit_passed"], read_json(output / "audit.json")
    assert metadata["completed_trajectories"] == metadata["planned_trajectories"] == 4
    assert metadata["model_rebuild_counts"] == {"initial": 2, "feedback": 20}
    assert metadata["check_totals"]["passed"] == metadata["check_totals"]["total"] > 0
    counts = pd.read_csv(output / "audit-check-counts.csv")
    assert counts.total.sum() == metadata["check_totals"]["total"] and counts.errors.sum() == 0
    assert len(pd.read_csv(output / "recommendations.csv")) == 120
    assert len(pd.read_csv(output / "query-checkpoints.csv")) == 12
    assert pd.read_csv(output / "calls.csv").reported_tokens.sum() == 720
    assert not list((study / "snapshot").rglob("__pycache__"))
    assert not list((study / "phase1/runs").rglob("*.joblib"))
    with pytest.raises(FileExistsError):
        export_phase(study, "phase1", confirmed=True, label="synthetic", formats=())


@pytest.mark.parametrize("change, code", [("training", "training_ids_own_feedback"), ("scores", "rebuilt_predictor_scores"),
                                        ("fitness", "selected_measurements_original_source"), ("context", "model_visible_context"),
                                        ("ranking", "top_k_ranking_no_repeats"), ("source", "selected_measurements_original_source"),
                                        ("selection-label", "selection_event_has_no_measurements")])
def test_independent_audit_detects_corrupted_records(study, change, code):
    phase = read_json(study / "phase1/state.json")
    condition = phase["conditions"][0]
    path = study / "phase1/runs" / condition["key"] / "run/state.json"
    state = read_json(path)
    record = state["methods"]["knowledge"]["rounds"][1]
    if change == "training":
        record["training_variants"][0] = "WWWW"
    elif change == "scores":
        record["scores"][0]["predicted_fitness"] += 1
    elif change == "fitness":
        record["recommendations"][0]["actual_fitness"] += 1
    elif change == "context":
        record["attempts"][0]["context"]["known_data"]["best_fitness"] += 1
    elif change == "ranking":
        record["recommendations"] = list(reversed(record["recommendations"]))
    elif change == "selection-label":
        log_path = path.parent / "events.jsonl"
        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        event = next(event for event in events if event.get("kind") == "candidates_selected")
        event["recommendations"][0]["actual_fitness"] = 10
        log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    else:
        source = pd.read_csv(study / "synthetic-source.csv")
        source.loc[source.variant == record["recommendations"][0]["variant"], "fitness"] += 1
        source.to_csv(study / "synthetic-source.csv", index=False)
    write(path, state)
    checks = Checks()
    Auditor(study, checks).trajectory("phase1", condition)
    assert any(row["code"] == code for row in checks.rows), checks.rows


def test_nonfinalized_phase_is_rejected_before_reading_trajectory_files(tmp_path):
    root = tmp_path / "not-finished"
    write(root / "plan.json", {})
    write(root / "phase1/execution.json", {})
    write(root / "phase1/state.json", {"phase": "running", "conditions": [{"status": "running"}]})
    with pytest.raises(ValueError, match="not finalized"):
        completed_inputs(root, "phase1", True)
    with pytest.raises(ValueError, match="explicit confirmation"):
        completed_inputs(root, "phase1", False)


def test_failed_trajectory_retains_missing_checkpoints_without_zero_fitness(tmp_path):
    root = tmp_path / "partial"
    create_study(root, failed=(101, "synthetic-a"))
    metadata, output = export_phase(root, "phase1", confirmed=True, label="partial", formats=())
    assert metadata["audit_passed"], read_json(output / "audit.json")
    assert metadata["completed_trajectories"] == 3
    trajectories = pd.read_csv(output / "trajectories.csv")
    failed = trajectories.loc[trajectories.status == "failed"].iloc[0]
    assert failed.assays == 10 and failed.mean_fitness > 0
    points = pd.read_csv(output / "query-checkpoints.csv")
    failed_points = points[points.trajectory == failed.trajectory]
    assert failed_points.loc[failed_points.queries == 10, "mean_fitness"].notna().all()
    assert failed_points.loc[failed_points.queries > 10, "mean_fitness"].isna().all()
    profile = next(row for row in metadata["profiles"] if row["model"] == "synthetic-a")
    assert profile["mean_fitness"]["estimate"] is None
    assert profile["completion_indicator"]["estimate"] == .5


def test_phase3_statistics_reproduce_the_preset_without_choosing_defaults(tmp_path):
    root = tmp_path / "confirmation"
    create_study(root, phase="phase3")
    metadata, output = export_phase(root, "phase3", confirmed=True, label="confirmation", formats=())
    assert metadata["audit_passed"], read_json(output / "audit.json")
    assert metadata["preset_selection"]["selectedArm"] == "selected"
    assert metadata["definition_version"] == DEFINITION_VERSION
    for filename in ("summary.json", "audit.json", "manifest.json"):
        assert read_json(output / filename)["definition_version"] == DEFINITION_VERSION
    assert metadata["confirmation_intervals"]["completionRate"]["lower"] is None
    assert metadata["confirmation_intervals"]["completionRate"]["reason"]
    for metric in ("meanFitness", "maxFitness", "completionRate"):
        assert metadata["confirmation_intervals"][metric]["estimate"] == 0


def completion_case(patterns=((True,), (True,))):
    trajectories, historical_rows, profiles, historical_profiles = [], [], [], []
    settings = {"dataSeeds": [1, 2], "bootstrap": BOOTSTRAP}
    for arm, pattern in zip(("selected", "reference"), patterns):
        records = []
        for seed in settings["dataSeeds"]:
            for repeat, complete in enumerate(pattern):
                key = f"{arm}-{seed}-{repeat}"
                row = {"trajectory": key, "profile": arm, "arm": arm, "seed": seed, "complete": complete,
                       "completion_indicator": int(complete), "calls": 0, "reported_tokens_sum": 0, "missing_usage_calls": 0,
                       "mean_fitness": seed + repeat / 5 if complete else None,
                       "max_fitness": seed + repeat / 3 + 1 if complete else None}
                records.append(row)
                historical_rows.append({"key": key, "complete": complete, "calls": 0, "tokens": 0, "missingUsage": 0,
                                        "meanFitness": row["mean_fitness"], "maxFitness": row["max_fitness"]})
        completed = sum(row["complete"] for row in records)
        mean = interval(records, "mean_fitness", settings["dataSeeds"], BOOTSTRAP)
        profiles.append({"profile": arm, "completed_trajectories": completed, "planned_trajectories": len(records),
                         "mean_fitness": mean, "completion_indicator": interval(records, "completion_indicator", settings["dataSeeds"], BOOTSTRAP)})
        historical_profiles.append({"profileKey": arm, "completed": completed, "total": len(records),
                                    "completionRate": completed / len(records), "equalSeedMeanFitness": mean["estimate"]})
        trajectories.extend(records)
    current = confirmation(trajectories, settings)
    historical = deepcopy(current)
    if historical["completionRate"].get("reason"):
        historical["completionRate"]["lower"] = historical["completionRate"]["estimate"]
        historical["completionRate"]["upper"] = historical["completionRate"]["estimate"]
    historical["completionRate"].pop("definition_version", None)
    return trajectories, profiles, {"trajectories": historical_rows, "profiles": historical_profiles, "intervals": historical}, current


@pytest.mark.parametrize("patterns", [((True,), (True,)), ((False,), (False,)), ((True,), (False,))])
def test_completion_definition_distinguishes_legacy_bounds_from_v2(patterns):
    trajectories, profiles, historical, current = completion_case(patterns)
    before = deepcopy(historical)
    checks = Checks()
    audit_analysis(checks, trajectories, profiles, historical, "phase3", current)
    assert not checks.rows, checks.rows
    assert historical == before
    assert current["completionRate"]["lower"] is None and current["completionRate"]["reason"]
    passed = {row["code"] for row in checks.summary() if row["passed"]}
    assert {"historical_completion_observation_counts", "historical_completion_observed_proportion",
            "historical_completion_difference_estimate", "v2_completion_observation_counts",
            "v2_completion_observed_proportion", "v2_completion_difference_estimate",
            "v2_completion_difference_boundary_bounds", "v2_completion_difference_boundary_reason"} <= passed


@pytest.mark.parametrize("mutation, expected", [
    ("historical_count", "historical_completion_observation_counts"),
    ("historical_ratio", "historical_completion_observed_proportion"),
    ("historical_difference", "historical_completion_difference_estimate"),
    ("current_count", "v2_completion_observation_counts"),
    ("current_ratio", "v2_completion_observed_proportion"),
    ("current_difference", "v2_completion_difference_estimate"),
    ("current_bounds", "v2_completion_difference_boundary_bounds"),
    ("current_reason", "v2_completion_difference_boundary_reason"),
    ("profile_bounds", "v2_completion_boundary_bounds"),
    ("arm_count", "v2_completion_observation_counts"),
    ("quality_estimate", "preset_confirmation_interval"),
    ("quality_lower", "preset_confirmation_interval"),
])
def test_completion_validation_and_quality_checks_remain_effective(mutation, expected):
    trajectories, profiles, historical, current = completion_case()
    if mutation == "historical_count":
        historical["profiles"][0]["completed"] = 0
    elif mutation == "historical_ratio":
        historical["profiles"][0]["completionRate"] = .5
    elif mutation == "historical_difference":
        historical["intervals"]["completionRate"]["estimate"] = .2
    elif mutation == "current_count":
        profiles[0]["completion_indicator"]["completed"] = 0
    elif mutation == "current_ratio":
        profiles[0]["completion_indicator"]["observed_proportion"] = .5
    elif mutation == "current_difference":
        current["completionRate"]["estimate"] = .2
    elif mutation == "current_bounds":
        current["completionRate"].update(lower=0., upper=0.)
    elif mutation == "current_reason":
        current["completionRate"]["reason"] = None
    elif mutation == "profile_bounds":
        profiles[0]["completion_indicator"].update(lower=1., upper=1.)
    elif mutation == "arm_count":
        current["completionRate"]["binary_arm_counts"]["selected"]["planned"] = 1
    elif mutation == "quality_estimate":
        current["meanFitness"]["estimate"] += .1
    elif mutation == "quality_lower":
        current["maxFitness"]["lower"] += .1
    checks = Checks()
    audit_analysis(checks, trajectories, profiles, historical, "phase3", current)
    assert any(row["code"] == expected for row in checks.rows), checks.rows


def test_completion_definition_checks_interior_intervals_numerically():
    trajectories, profiles, historical, current = completion_case(((True, False), (True, False)))
    assert current["completionRate"]["lower"] is not None
    checks = Checks()
    audit_analysis(checks, trajectories, profiles, historical, "phase3", current)
    assert not checks.rows, checks.rows
    historical["intervals"]["completionRate"]["lower"] += .1
    checks = Checks()
    audit_analysis(checks, trajectories, profiles, historical, "phase3", current)
    assert any(row["code"] == "preset_completion_interior_interval" for row in checks.rows)


def test_equal_seed_weight_and_missing_seed_intervals():
    rows = [{"seed": seed, "complete": True, "value": value} for seed, value in ((1, 1), (1, 1), (2, 4), (3, 7))]
    result = interval(rows, "value", [1, 2, 3], BOOTSTRAP)
    assert result["estimate"] == 4
    assert interval(rows, "value", [1, 2, 3, 4], BOOTSTRAP)["estimate"] is None
    assert result["usable_per_seed"] == {"1": 2, "2": 1, "3": 1}


def test_paired_bootstrap_uses_independent_trajectories_within_paired_splits():
    rows = [{"seed": seed, "arm": arm, "complete": True, "value": seed + int(arm == "selected")}
            for seed in (1, 2, 3) for arm in ("selected", "reference") for _ in range(3)]
    result = paired_difference(rows, "value", {"dataSeeds": [1, 2, 3], "bootstrap": BOOTSTRAP})
    assert result["estimate"] == result["lower"] == result["upper"] == 1
    assert result["innerUnit"] == BOOTSTRAP["innerUnit"]


def test_nonconstant_phase3_bootstrap_matches_original_preset_implementation():
    specification = importlib.util.spec_from_file_location("preset_calibration_statistics_nonconstant", PROJECT / "tools/calibration/statistics_calibration.py")
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)
    settings = {"dataSeeds": [1, 2, 3], "bootstrap": {**BOOTSTRAP, "replicates": 10000}}
    rows = [{"seed": seed, "arm": arm, "complete": True,
             "value": seed / 10 + repeat / 5 + (0.3 + seed * repeat / 7 if arm == "selected" else 0)}
            for seed in settings["dataSeeds"] for arm in ("selected", "reference") for repeat in range(3)]
    assert paired_difference(rows, "value", settings) == reference.hierarchical_interval(rows, "value", settings)


def test_missing_token_usage_is_not_replaced_by_zero():
    trajectories = [{"trajectory": "a", "complete": True}, {"trajectory": "b", "complete": False}]
    calls = [{"trajectory": "a", "reported_tokens": 30, "generation_seconds": 2},
             {"trajectory": "a", "reported_tokens": None, "generation_seconds": None}]
    add_resource_metrics(trajectories, calls)
    assert trajectories[0]["reported_tokens_sum"] == 30 and trajectories[0]["tokens"] is None
    assert trajectories[0]["missing_usage_calls"] == 1
    assert trajectories[1]["calls"] == 0 and trajectories[1]["tokens"] == 0


def test_figures_export_editable_svg_pdf_and_png_with_sample_counts(study):
    metadata, output = export_phase(study, "phase1", confirmed=True, label="figures", formats=("svg", "pdf", "png"), dpi=100)
    assert metadata["audit_passed"]
    for metric in ("mean_fitness", "max_fitness", "cumulative_best"):
        for suffix in ("svg", "pdf", "png"):
            assert (output / "figures" / f"quality-{metric}.{suffix}").stat().st_size > 1000
    text = (output / "figures/quality-mean_fitness.svg").read_text(encoding="utf-8")
    assert "n=2/2" in text and "checkpoints" in text
