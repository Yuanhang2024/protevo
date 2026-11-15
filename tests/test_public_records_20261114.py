"""Synthetic temporary studies only; no external API or active experiment reads."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.data import HiddenOracle, PublicDataset, prepare as prepare_data
from protevo.models import select_model
from protevo.storage import read_json, write_json
from tools.supplement_20261114 import feedback_pilot as pilot
from tools.supplement_20261114 import strong_baselines as strong
from tools.supplement_20261114 import public_records as public

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def studies(tmp_path):
    project = tmp_path / "synthetic-project"
    for relative in public.SNAPSHOT_FILES:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, target)
    source = project / "artificial.csv"
    variants = ["".join(row) for row in product("ACDE", repeat=4)]
    pd.DataFrame({"Variants": variants, "Fitness": [sum((i + 1) * "ACDE".index(x) for i, x in enumerate(v)) / 4 for v in variants]}).to_csv(source, index=False)
    config = pilot.pilot_config(1201, project / "configs/knowledge.json", train_size=24, validation_size=16,
                                test_size=16, pool_size=200, model_families=["random_forest"], tree_estimators=5,
                                tree_min_samples_leaf=[1], n_jobs=1)
    data, selected = project / "prepared-data", project / "selection"
    prepare_data(source, data, config, strong.REFERENCE, "Variants", "Fitness")
    dataset = PublicDataset(data)
    selection = select_model(dataset, config, selected)
    correction = project / "correction-status.json"
    write_json(correction, {"status": "passed"})
    return SimpleNamespace(root=tmp_path, project=project, source=source, config=config, dataset=dataset,
                           selection=selection, selection_path=selected / "selection.json", correction=correction)


class FakeProposer:
    model, api_mode = "synthetic-model", "chat"

    def __init__(self, eligible, *, repair=False, unknown=False, missing_usage=False):
        self.eligible, self.repair, self.unknown = eligible, repair, unknown
        self.missing_usage = missing_usage
        self.calls = 0

    def generate(self, context):
        self.calls += 1
        if self.unknown:
            raise RuntimeError("Synthetic failure")
        count = min(context["proposal_count"], 11) if self.repair and context["stage"] == "initial_design" else context["proposal_count"]
        used = set(context.get("accepted_variants", [])) | set(context["forbidden_variants"])
        ids = [value for value in self.eligible if value not in used][:count]
        rule = context["knowledge"]["rules"][0]["id"]
        return {"text": json.dumps({"analysis": "公开分析保持原文。\nSynthetic observations.", "proposals": [
            {"variant": value, "hypothesis": "Synthetic hypothesis", "rationale": "Synthetic rationale", "knowledge_rule_ids": [rule]} for value in ids]}, ensure_ascii=False),
            "complete": True, "usage": None if self.missing_usage else {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "model_requested": self.model, "model_returned": self.model, "api_mode": self.api_mode,
            "prompt_metadata": pilot.load_prompt_profile("v3").metadata()}


def make_feedback(study, *, repair=False, unknown=False, missing_usage=False):
    root = study.project / "feedback"
    plan = pilot.prepare(root, [{"state_id": "seed-1201", "data": study.dataset.directory, "selection": study.selection_path}],
                         model_requested="synthetic-model", repeats=1, correction_status=study.correction)
    eligible = pd.read_csv(root / "states/seed-1201/eligible.csv").variant.tolist()
    proposer = FakeProposer(eligible, repair=repair, unknown=unknown, missing_usage=missing_usage)
    pilot.run(root, proposer)
    return root, plan, proposer


def make_strong(study):
    root = study.project / "strong"
    inputs = root / "inputs/seed-1201"
    inputs.mkdir(parents=True)
    for name in ("train.csv", "validation.csv", "candidates.csv", "manifest.json"):
        shutil.copyfile(study.dataset.directory / name, inputs / name)
    write_json(inputs / "selection.json", study.selection)
    entry = {"seed": 1201, "data": str(study.dataset.directory), "selection_path": str(study.selection_path),
             "config": study.config.model_dump(), "raw_selection": study.selection}
    plan = {"inputs": [entry], "arms": list(strong.ARMS), "random_namespace": strong.RANDOM_NAMESPACE}
    write_json(root / "prepared.json", {"plan": plan, "api_key": "synthetic-private-metadata"})
    results = []
    for arm in strong.ARMS:
        state = strong.run_arm(study.dataset, study.selection, study.config, arm, root / "runs/seed-1201" / arm, chunk_size=37)
        results.append({"seed": 1201, "arm": arm, "status": state["status"]})
    write_json(root / "run.json", {"status": "completed", "results": results, "llm_requests": 0})
    return root


def export(study, **options):
    destination = study.root / "public"
    manifest = public.export_records(study.project, destination, terminal_confirmed=True, **options)
    return destination, manifest


def guard_private_reads(monkeypatch):
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if "r" in mode and ("hidden" in path.parts or path.name.lower().startswith(".env")):
            raise AssertionError("Exporter/auditor opened private source input")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def test_confirmation_required_before_any_study_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A study was read without terminal confirmation")
    monkeypatch.setattr(public, "read_json", forbidden)
    with pytest.raises(ValueError, match="terminal confirmation"):
        public.export_records(tmp_path, tmp_path / "new", feedback=tmp_path / "active")


def test_export_preserves_payload_response_and_reconstructs_both_studies(studies, monkeypatch):
    feedback, plan, proposer = make_feedback(studies, repair=True)
    strong_root = make_strong(studies)
    guard_private_reads(monkeypatch)
    destination, manifest = export(studies, feedback=feedback, strong=strong_root)
    result = public.audit_records(destination, official_s1=studies.source)
    assert result["passed"], result
    assert result["maximum_prediction_absolute_error"] <= 1e-10
    assert result["counts"]["feedback"]["planned_generator_arms"] == 2
    assert result["counts"]["feedback"]["completed_selector_results"] == 4
    assert proposer.calls == 4
    assert result["counts"]["strong"]["policy_assays"] == 90
    assert manifest["scientific_text_transformations"] == []
    for episode in plan["episodes"]:
        for number in (1, 2):
            relative = Path("episodes") / episode["episode_id"] / f"attempt-{number}"
            original = read_json(feedback / relative / "request.json")
            saved = read_json(destination / "feedback" / relative / "request.json")
            assert saved["payload"] == original["payload"]
            assert saved["context"] == original["context"]
            assert read_json(destination / "feedback" / relative / "response.json") == read_json(feedback / relative / "response.json")
    assert set(pd.read_csv(destination / "feedback/states/seed-1201/eligible.csv").columns) == {"variant"}
    assert "synthetic-private-metadata" not in (destination / "strong/prepared.json").read_text(encoding="utf-8")
    assert any(row["action"] == "private_metadata_field_omitted" for row in manifest["metadata_transformations"])
    assert not list(destination.rglob("oracle.csv")) and not list(destination.rglob("test.csv"))
    for path in destination.rglob("*.json"):
        assert str(studies.root) not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        public.export_records(studies.project, destination, feedback=feedback, terminal_confirmed=True)


def test_audit_is_read_only_and_independent_of_original_records(studies, monkeypatch):
    feedback, _, _ = make_feedback(studies)
    destination, _ = export(studies, feedback=feedback)
    before = {str(path.relative_to(destination)): (path.stat().st_mtime_ns, path.read_bytes()) for path in destination.rglob("*") if path.is_file()}
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if path.is_relative_to(studies.project) or any(flag in mode for flag in "wax+") and path.is_relative_to(destination):
            raise AssertionError("Read-only audit attempted a source read or export write")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = public.audit_records(destination)
    assert result["passed"], result
    after = {str(path.relative_to(destination)): (path.stat().st_mtime_ns, path.read_bytes()) for path in destination.rglob("*") if path.is_file()}
    assert before == after
    assert not list(destination.rglob("__pycache__"))


def test_unknown_outcome_and_unstarted_arms_stay_distinct_from_selector_replays(studies):
    feedback, _, proposer = make_feedback(studies, unknown=True)
    destination, manifest = export(studies, feedback=feedback)
    result = public.audit_records(destination)
    assert result["passed"], result
    counts = manifest["counts"]["feedback"]
    assert counts["generation_calls"] == proposer.calls == 1
    assert counts["planned_generator_arms"] == 2 and counts["planned_selector_results"] == 4
    assert counts["completed_generator_arms"] == counts["completed_selector_results"] == counts["selector_policy_assays"] == 0
    assert counts["common_prefix_assays"] == 10


def test_nonterminal_and_locked_studies_cannot_export(studies):
    feedback, _, _ = make_feedback(studies)
    state = read_json(feedback / "run-state.json")
    state["status"] = "partial"
    write_json(feedback / "run-state.json", state)
    with pytest.raises(ValueError, match="not terminal"):
        export(studies, feedback=feedback)
    (feedback / ".writer.lock").write_text("synthetic", encoding="ascii")
    with pytest.raises(ValueError, match="active writer"):
        export(studies, feedback=feedback)


def test_predictor_tamper_and_saved_score_ranking_are_detected(studies):
    feedback, plan, _ = make_feedback(studies)
    destination, _ = export(studies, feedback=feedback)
    episode = plan["episodes"][0]["episode_id"]
    folder = destination / "feedback/episodes" / episode / "P0"
    score_path = folder / "all-scores.csv"
    original = score_path.read_bytes()
    scores = pd.read_csv(score_path, dtype={"predicted_fitness": str})
    scores["predicted_fitness"] = scores.predicted_fitness.map(float)
    scores.loc[0, "predicted_fitness"] += .25
    scores.to_csv(score_path, index=False)
    assert not public.audit_records(destination)["passed"]
    score_path.write_bytes(original)
    selected = read_json(folder / "selected.json")
    selected[0], selected[1] = selected[1], selected[0]
    write_json(folder / "selected.json", selected)
    result = public.audit_records(destination)
    assert not result["passed"] and "Top-k" in result["errors"][0]["message"]


def test_changed_context_or_feedback_order_is_detected(studies):
    feedback, plan, _ = make_feedback(studies)
    destination, _ = export(studies, feedback=feedback)
    context_path = destination / "feedback/states/seed-1201/C0-context.json"
    context = read_json(context_path)
    original = deepcopy(context)
    context["known_data"]["n_known"] += 10
    write_json(context_path, context)
    assert not public.audit_records(destination)["passed"]
    write_json(context_path, original)
    folder = destination / "feedback/episodes" / plan["episodes"][0]["episode_id"] / "P0"
    events = (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    (folder / "events.jsonl").write_text("\n".join(reversed(events)) + "\n", encoding="utf-8")
    result = public.audit_records(destination)
    assert not result["passed"] and "precede feedback" in result["errors"][0]["message"]


def test_scientific_text_with_private_content_is_rejected_without_rewriting(studies):
    feedback, plan, _ = make_feedback(studies)
    path = feedback / "episodes" / plan["episodes"][0]["episode_id"] / "attempt-1/response.json"
    response = read_json(path)
    response["text"] += " sk-" + "A" * 30
    write_json(path, response)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Private content"):
        export(studies, feedback=feedback)
    assert path.read_bytes() == before


def test_crash_without_response_file_retains_unknown_dispatch(studies):
    feedback = studies.project / "feedback"
    pilot.prepare(feedback, [{"state_id": "seed-1201", "data": studies.dataset.directory, "selection": studies.selection_path}],
                  model_requested="synthetic-model", repeats=1, correction_status=studies.correction)
    eligible = pd.read_csv(feedback / "states/seed-1201/eligible.csv").variant.tolist()

    class Crash(FakeProposer):
        def generate(self, context):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        pilot.run(feedback, Crash(eligible))
    untouched = FakeProposer(eligible)
    pilot.run(feedback, untouched)
    assert untouched.calls == 0
    destination, manifest = export(studies, feedback=feedback)
    result = public.audit_records(destination)
    assert result["passed"], result
    assert manifest["counts"]["feedback"]["missing_usage_calls"] == 1
    assert manifest["counts"]["feedback"]["complete_total_tokens"] is None


def test_missing_usage_stays_unknown_even_with_completed_selection(studies):
    feedback, _, proposer = make_feedback(studies, missing_usage=True)
    destination, manifest = export(studies, feedback=feedback)
    result = public.audit_records(destination)
    assert result["passed"], result
    counts = manifest["counts"]["feedback"]
    assert proposer.calls == counts["generation_calls"] == counts["missing_usage_calls"] == 1
    assert counts["reported_known_total_tokens"] == 0 and counts["complete_total_tokens"] is None
    assert counts["completed_generator_arms"] == 1 and counts["completed_selector_results"] == 2


def test_changed_usage_or_selected_scientific_text_is_detected(studies):
    feedback, plan, _ = make_feedback(studies)
    destination, _ = export(studies, feedback=feedback)
    folder = destination / "feedback/episodes" / plan["episodes"][0]["episode_id"]
    path = folder / "attempt-1/response.json"
    original = read_json(path)
    changed = deepcopy(original)
    changed["usage"]["total_tokens"] += 1
    write_json(path, changed)
    assert not public.audit_records(destination)["passed"]
    write_json(path, original)
    selected = read_json(folder / "P0/selected.json")
    selected[0]["rationale"] = "Changed after selection"
    write_json(folder / "P0/selected.json", selected)
    result = public.audit_records(destination)
    assert not result["passed"] and "raw accepted proposal" in result["errors"][0]["message"]


def test_private_table_column_is_rejected_before_copy(studies):
    feedback, _, _ = make_feedback(studies)
    path = feedback / "states/seed-1201/prefix-observations.csv"
    observations = pd.read_csv(path)
    observations["api_key"] = "synthetic-sensitive-column"
    observations.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected fields"):
        export(studies, feedback=feedback)


def test_partial_selector_failure_keeps_available_results_without_completed_generation(studies):
    feedback = studies.project / "feedback"
    plan = pilot.prepare(feedback, [{"state_id": "seed-1201", "data": studies.dataset.directory, "selection": studies.selection_path}],
                         model_requested="synthetic-model", repeats=1, correction_status=studies.correction)
    eligible = pd.read_csv(feedback / "states/seed-1201/eligible.csv").variant.tolist()

    class FailSecond(HiddenOracle):
        calls = 0

        def evaluate(self, values):
            type(self).calls += 1
            if self.calls == 2:
                raise RuntimeError("Synthetic local replay interruption")
            return super().evaluate(values)

    with pytest.raises(RuntimeError):
        pilot.run(feedback, FakeProposer(eligible), oracle_factory=FailSecond)
    state = read_json(feedback / "run-state.json")
    state["status"] = "finished_with_failures"
    for record in state["episodes"].values():
        record["status"] = "failed" if record["status"] == "running" else "cancelled"
    write_json(feedback / "run-state.json", state)
    destination, manifest = export(studies, feedback=feedback)
    result = public.audit_records(destination)
    assert result["passed"], result
    counts = manifest["counts"]["feedback"]
    assert counts["completed_generator_arms"] == 0
    assert counts["completed_selector_results"] == 1 and counts["selector_policy_assays"] == 10
    assert counts["planned_generator_arms"] == len(plan["episodes"]) == 2


def test_prediction_tolerance_does_not_change_saved_score_ranking():
    scores = pd.DataFrame({"variant": ["AAAA", "AAAC"], "predicted_fitness": [1., 1. + 1e-15]})
    assert public.ranked(scores, 1) == ["AAAC"]
    core = {"models": SimpleNamespace(fit_model=lambda *args: SimpleNamespace(predict=lambda x: np.array([1. + 2e-15, 1.]))),
            "data": SimpleNamespace(one_hot=lambda values: list(values))}
    checks = public.Checks()
    public.predictions(checks, core, {}, SimpleNamespace(k=1), pd.DataFrame(), scores, True)
    assert 0 < checks.max_prediction_error < public.TOLERANCE


def test_public_inventory_rejects_extra_private_file(studies):
    feedback, _, _ = make_feedback(studies)
    destination, _ = export(studies, feedback=feedback)
    hidden = destination / "oracle.csv"
    hidden.write_text("variant,fitness\nAAAA,999\n", encoding="utf-8")
    result = public.audit_records(destination)
    assert not result["passed"] and "inventory" in result["errors"][0]["message"]
