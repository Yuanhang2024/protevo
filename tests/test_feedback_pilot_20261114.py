"""Artificial measurements and fake proposers only; no external model dispatch."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from protevo.data import HiddenOracle, PublicDataset, prepare as prepare_data
from protevo.models import select_model
from protevo.storage import read_json, write_json
from tools.supplement_20261114 import feedback_pilot as pilot


@pytest.fixture
def state_source(tmp_path):
    reference = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    variants = ["".join(row) for row in product("ACDE", repeat=4)]
    source = tmp_path / "artificial.csv"
    pd.DataFrame({"Variants": variants, "Fitness": [float(sum((i + 1) * "ACDE".index(x) for i, x in enumerate(v))) for v in variants]}).to_csv(source, index=False)
    knowledge = tmp_path / "knowledge.json"
    write_json(knowledge, {"rules": [{"id": "test-rule", "scope": "synthetic test", "condition": "test", "action": "test", "source": "synthetic", "explanation": "Synthetic only"}]})
    config = pilot.pilot_config(1201, knowledge, train_size=24, validation_size=16, test_size=16, pool_size=160,
                                model_families=["ridge"], ridge_alphas=[1.0], tree_estimators=3, n_jobs=1)
    data = tmp_path / "data"
    prepare_data(source, data, config, reference, "Variants", "Fitness")
    select_model(PublicDataset(data), config, tmp_path / "selection")
    correction = tmp_path / "correction-status.json"
    write_json(correction, {"status": "passed"})
    sources = [{"state_id": "seed-1201", "data": data, "selection": tmp_path / "selection/selection.json"}]
    return SimpleNamespace(root=tmp_path, sources=sources, config=config, correction=correction)


def prepared(source, name="pilot", repeats=1, **options):
    output = source.root / name
    plan = pilot.prepare(output, source.sources, model_requested="fake-model", api_mode="chat", repeats=repeats,
                         correction_status=source.correction, **options)
    eligible = pd.read_csv(output / "states/seed-1201/eligible.csv").variant.tolist()
    return output, plan, eligible


class FakeProposer:
    model = "fake-model"
    api_mode = "chat"

    def __init__(self, eligible, *, short=False, missing_usage=False, unknown=False, repair=False):
        self.eligible, self.short, self.missing_usage, self.unknown, self.repair = eligible, short, missing_usage, unknown, repair
        self.contexts = []
        self.closed = False

    def generate(self, context):
        self.contexts.append(deepcopy(context))
        if self.unknown:
            error = RuntimeError("SYNTHETIC_PRIVATE_ENDPOINT_AND_KEY")
            error.status_code = 503
            raise error
        count = context["proposal_count"]
        if self.short or self.repair and context["stage"] == "initial_design":
            count = min(count, 11)
        used = set(context.get("accepted_variants", [])) | set(context["forbidden_variants"])
        selected = [v for v in self.eligible if v not in used][:count]
        return {"text": json.dumps({"analysis": "Synthetic public analysis", "proposals": [
            {"variant": v, "hypothesis": "Synthetic hypothesis", "rationale": "Synthetic rationale", "knowledge_rule_ids": ["test-rule"]} for v in selected]}),
            "complete": True, "usage": None if self.missing_usage else {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model_requested": self.model, "model_returned": "fake-model-version", "api_mode": self.api_mode,
            "request_id": f"synthetic-{len(self.contexts)}", "prompt_metadata": pilot.load_prompt_profile("v3").metadata()}

    def close(self):
        self.closed = True


def test_c0_has_no_prefix_labels_residuals_or_aggregate_feedback(state_source):
    output, _, _ = prepared(state_source)
    directory = output / "states/seed-1201"
    original = pilot.contexts_for_state(directory)
    prefix = pd.read_csv(directory / "prefix-observations.csv")
    prefix["actual_fitness"] = 123456.789
    prefix["residual"] = prefix.actual_fitness - prefix.predicted_fitness
    prefix.to_csv(directory / "prefix-observations.csv", index=False)
    changed = pilot.contexts_for_state(directory)
    assert original["C0"] == changed["C0"]
    assert original["C1"] != changed["C1"]
    assert "123456" not in json.dumps(changed["C0"])
    assert not changed["C0"]["recent_round_feedback"]
    assert changed["C0"]["feedback_summary"]["n_observed_feedback"] == 0
    assert changed["C0"]["known_data"]["n_known"] == 24
    assert changed["C1"]["known_data"]["n_known"] == 34
    assert len(changed["C1"]["recent_round_feedback"]) == 10
    assert changed["C0"]["observed_variants"] == changed["C1"]["observed_variants"]
    assert changed["C0"]["forbidden_variants"] == changed["C1"]["forbidden_variants"]
    assert len(changed["C0"]["forbidden_variants"]) == 34
    assert changed["C0"]["round"] == changed["C1"]["round"] == 2


def test_same_batch_replayed_with_both_models_and_selections_saved_before_oracle(state_source):
    output, plan, eligible = prepared(state_source)
    class CheckedOracle(HiddenOracle):
        def evaluate(self, variants):
            matches = [path for path in (output / "episodes").glob("*/P*/selected.json")
                       if [row["variant"] for row in read_json(path)] == variants]
            assert matches, "Selection must be persisted before reading its hidden measurements"
            assert self.allowed == set(variants)
            return super().evaluate(variants)
    proposer = FakeProposer(eligible)
    summary = pilot.run(output, proposer, oracle_factory=CheckedOracle)
    assert len(proposer.contexts) == 2
    assert summary["completed_generator_arms"] == summary["planned_generator_arms"] == 2
    assert summary["calls"] == 2 and summary["max_calls"] == 4
    for episode in plan["episodes"]:
        directory = output / "episodes" / episode["episode_id"]
        accepted = read_json(directory / "accepted-proposals.json")
        p0 = pd.read_csv(directory / "P0/all-scores.csv")
        p1 = pd.read_csv(directory / "P1/all-scores.csv")
        assert list(p0.variant) == list(p1.variant) == [row["variant"] for row in accepted]
        assert len(p0) == len(p1) == 40
        assert len(read_json(directory / "P0/training-ids.json")) == 24
        assert len(read_json(directory / "P1/training-ids.json")) == 34
        assert len(pd.read_csv(directory / "P0/recommendations.csv")) == 10
        assert len(pd.read_csv(directory / "P1/recommendations.csv")) == 10
    assert summary["effects"]["P0_C1_minus_C0"]["states_with_both_conditions"] == 1


def test_original_prompt_complete_payload_usage_seed_and_pairing_are_saved(state_source):
    output, plan, eligible = prepared(state_source, repeats=3)
    assert len(plan["episodes"]) == 6 and plan["max_calls"] == 12
    for repeat in range(3):
        pair = [row for row in plan["episodes"] if row["repeat"] == repeat]
        assert len({row["design_seed"] for row in pair}) == 1
    proposer = FakeProposer(eligible, repair=True)
    pilot.run(output, proposer)
    for episode in plan["episodes"]:
        directory = output / "episodes" / episode["episode_id"]
        first = read_json(directory / "attempt-1/request.json")
        repair = read_json(directory / "attempt-2/request.json")
        response = read_json(directory / "attempt-1/response.json")
        assert first["round"] == repair["round"] == 2
        assert first["parameters"]["seed"] == 1201
        assert first["parameters"]["design_seed"] == repair["parameters"]["design_seed"] == episode["design_seed"]
        assert first["payload"]["messages"][0]["content"] == plan["prompt_metadata"]["system_prompt"]
        assert json.loads(first["payload"]["messages"][1]["content"]) == first["context"]
        assert response["model_returned"] == "fake-model-version"
        assert first["operation_id"] != repair["operation_id"]
        assert repair["context"]["known_data"] == first["context"]["known_data"]
        assert repair["context"]["observed_variants"] == first["context"]["observed_variants"]
        assert set(repair["context"]["replacement_examples"]) <= set(eligible)
        assert not set(repair["context"]["replacement_examples"]) & set(first["context"]["forbidden_variants"])
    summary = pilot.summarize(output)
    assert summary["quality_basis"] == "conditional_on_completion"
    assert summary["primary_effect"]["complete"] is True
    assert summary["effects"]["P0_C1_minus_C0"]["all_repetitions_complete_state_pairs"] == 1
    state = read_json(output / "run-state.json")
    state["episodes"][plan["episodes"][0]["episode_id"]]["status"] = "failed"
    write_json(output / "run-state.json", state)
    partial = pilot.summarize(output)
    effect = partial["effects"]["P0_C1_minus_C0"]
    assert effect["state_equal_weight_mean"] is not None
    assert effect["all_repetitions_complete_state_pairs"] == 0
    assert effect["all_repetitions_complete_state_equal_weight_mean"] is None
    assert effect["all_planned_state_equal_weight_mean"] is None
    assert partial["primary_effect"]["complete"] is False
    assert partial["primary_effect"]["all_planned_state_equal_weight_mean"] is None
    counts = [partial["state_effects"][0][f"P0_{condition}"]["completed"] for condition in pilot.CONTEXTS]
    assert sorted(counts) == [2, 3]


def test_incomplete_batch_fails_after_two_responses_without_programmatic_fill(state_source):
    output, _, eligible = prepared(state_source)
    proposer = FakeProposer(eligible, short=True)
    summary = pilot.run(output, proposer)
    assert summary["completed_generator_arms"] == 0
    assert summary["calls"] == 4
    state = read_json(output / "run-state.json")
    assert all(row["status"] == "failed" and row["accepted_count"] == 22 for row in state["episodes"].values())
    assert not list((output / "episodes").glob("*/P*/selected.json"))


def test_unknown_request_is_logged_safely_and_never_resent(state_source):
    output, _, eligible = prepared(state_source)
    proposer = FakeProposer(eligible, unknown=True)
    summary = pilot.run(output, proposer)
    assert len(proposer.contexts) == 1 and summary["calls"] == 1
    assert summary["missing_usage_count"] == 1
    assert summary["complete_total_tokens"] is None
    errors = list((output / "episodes").glob("*/attempt-1/error.json"))
    assert read_json(errors[0])["error"] == {"type": "RuntimeError", "status_code": 503}
    assert "SYNTHETIC_PRIVATE" not in errors[0].read_text(encoding="utf-8")
    pilot.run(output, proposer)
    assert len(proposer.contexts) == 1


def test_missing_usage_stops_all_subsequent_dispatch_including_correction(state_source):
    output, _, eligible = prepared(state_source)
    proposer = FakeProposer(eligible, missing_usage=True, short=True)
    summary = pilot.run(output, proposer)
    assert len(proposer.contexts) == 1 and summary["calls"] == 1
    assert summary["status"] == "stopped_missing_usage"
    assert summary["complete_total_tokens"] is None
    assert summary["reported_known_total_tokens"] == 0
    pilot.run(output, proposer)
    assert len(proposer.contexts) == 1


def test_duplicate_operation_and_changed_plan_cannot_create_new_generation(state_source):
    output, plan, eligible = prepared(state_source)
    assert pilot.prepare(output, state_source.sources, model_requested="fake-model", api_mode="chat", repeats=1,
                         correction_status=state_source.correction)["study_id"] == plan["study_id"]
    proposer = FakeProposer(eligible)
    pilot.run(output, proposer)
    pilot.run(output, proposer)
    assert len(proposer.contexts) == 2
    with pytest.raises(ValueError, match="different frozen"):
        pilot.prepare(output, state_source.sources, model_requested="other", api_mode="chat", repeats=1,
                      correction_status=state_source.correction)


def test_reserved_call_persists_before_dispatch_and_crash_does_not_retry(state_source):
    output, plan, eligible = prepared(state_source)
    class Crash(FakeProposer):
        def generate(self, context):
            state = read_json(output / "run-state.json")
            started = [row for row in state["episodes"].values() if row["status"] == "running"]
            assert state["calls"] == 1 and started[0]["attempts"][0]["status"] == "started"
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        pilot.run(output, Crash(eligible))
    replacement = FakeProposer(eligible)
    summary = pilot.run(output, replacement)
    assert summary["calls"] == 1 and summary["missing_usage_count"] == 1
    assert not replacement.contexts


def test_prepare_gate_does_not_use_an_unfinished_statistical_correction(state_source):
    write_json(state_source.correction, {"status": "pending"})
    with pytest.raises(ValueError, match="correction status"):
        prepared(state_source)


def test_public_preflight_never_generates_or_returns_endpoint_or_key(state_source):
    instances = []
    class Connection:
        model = "fake-model"
        api_mode = "chat"
        base_url = "SYNTHETIC_PRIVATE_ENDPOINT"
        def __init__(self, *args, **kwargs): instances.append(self); self.closed = False
        def generate(self, context): raise AssertionError("Preflight must never generate")
        def close(self): self.closed = True
    result = pilot.public_preflight(state_source.config, "fake-model", "chat", proposer_factory=Connection)
    assert result == {"status": "ready", "model_requested": "fake-model", "api_mode": "chat", "generation_requests": 0}
    assert instances[0].closed
    assert "SYNTHETIC_PRIVATE" not in json.dumps(result)
    class Failure(Connection):
        def __init__(self, *args, **kwargs): raise RuntimeError("SYNTHETIC_PRIVATE_ENDPOINT_AND_KEY")
    result = pilot.public_preflight(state_source.config, proposer_factory=Failure)
    assert result["status"] == "failed" and "SYNTHETIC_PRIVATE" not in json.dumps(result)


def test_no_real_proposer_is_constructed_without_explicit_authorization(state_source, monkeypatch):
    output, _, _ = prepared(state_source)
    monkeypatch.setattr(pilot, "OpenAIProposer", lambda *args, **kwargs: pytest.fail("Must not instantiate a real generator"))
    with pytest.raises(ValueError, match="explicit run authorization"):
        pilot.run(output)


def test_common_prefix_selection_is_saved_before_its_only_measurement_query(state_source):
    output = state_source.root / "prefix-audit"
    class PrefixOracle(HiddenOracle):
        def evaluate(self, variants):
            selected = read_json(output / "states/seed-1201/prefix-selected.json")
            assert [row["variant"] for row in selected] == variants
            assert len(variants) == 10
            assert self.allowed == set(variants)
            assert (output / "states/seed-1201/prefix-all-scores.csv.gz").exists()
            return super().evaluate(variants)
    pilot.prepare(output, state_source.sources, model_requested="fake-model", api_mode="chat", repeats=1,
                  correction_status=state_source.correction, oracle_factory=PrefixOracle)
    saved = read_json(output / "states/seed-1201/prepared.json")
    assert saved["common_prefix_assays"] == 10
    assert saved["prefix_score_count"] == 160


def test_ordered_prefix_discards_extra_candidates_without_scoring_them(state_source):
    output, plan, eligible = prepared(state_source)
    class Oversize(FakeProposer):
        def generate(self, context):
            reply = super().generate(context)
            body = json.loads(reply["text"])
            body["proposals"].append({"variant": "YYYY", "hypothesis": "Extra", "rationale": "Extra", "knowledge_rule_ids": []})
            reply["text"] = json.dumps(body)
            return reply
    summary = pilot.run(output, Oversize(eligible))
    assert summary["completed_generator_arms"] == 2
    for episode in plan["episodes"]:
        directory = output / "episodes" / episode["episode_id"]
        validation = read_json(directory / "attempt-1/validation.json")
        assert validation["original_response_count"] == 41 and validation["discarded_after_ordered_prefix"] == 1
        assert "YYYY" not in set(pd.read_csv(directory / "P0/all-scores.csv").variant)
        assert "YYYY" not in set(pd.read_csv(directory / "P1/all-scores.csv").variant)


def test_four_distinct_states_three_repetitions_freeze_24_arms_and_48_call_limit(state_source):
    sources = []
    for seed in pilot.STATES:
        config = state_source.config.model_copy(update={"seed": seed})
        data = state_source.root / f"four-data-{seed}"
        prepare_data(state_source.root / "artificial.csv", data, config,
                     "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "Variants", "Fitness")
        selection = state_source.root / f"four-selection-{seed}"
        select_model(PublicDataset(data), config, selection)
        sources.append({"state_id": f"seed-{seed}", "data": data, "selection": selection / "selection.json"})
    output = state_source.root / "four-state-pilot"
    plan = pilot.prepare(output, sources, model_requested="fake-model", repeats=3, correction_status=state_source.correction)
    assert plan["planned_generator_arms"] == 24 and plan["planned_selector_results"] == 48 and plan["max_calls"] == 48
    assert plan["declaration"]["state_seeds"] == list(pilot.STATES)
    assert all(value["status"] == "pending" for value in read_json(output / "run-state.json")["episodes"].values())
    for source in sources:
        for repeat in range(3):
            pair = [row for row in plan["episodes"] if row["state_id"] == source["state_id"] and row["repeat"] == repeat]
            assert {row["context"] for row in pair} == {"C0", "C1"}
            assert len({row["design_seed"] for row in pair}) == 1


def test_modified_frozen_context_or_correction_gate_prevents_any_generation(state_source):
    output, _, eligible = prepared(state_source)
    path = output / "states/seed-1201/C0-context.json"
    context = read_json(path)
    context["known_data"]["fitness_mean"] = 999999.0
    write_json(path, context)
    proposer = FakeProposer(eligible)
    with pytest.raises(ValueError, match="Frozen context"):
        pilot.run(output, proposer)
    assert not proposer.contexts
    write_json(state_source.correction, {"status": "pending"})
    with pytest.raises(ValueError, match="correction status"):
        pilot.run(output, proposer)
    assert not proposer.contexts
