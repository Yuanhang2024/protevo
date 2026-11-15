"""Synthetic-only checks for independent full-pool comparators and preparation."""
from copy import deepcopy
from itertools import islice, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.data import HiddenOracle, PublicDataset, prepare
from protevo.models import select_model
from protevo.schema import AMINO_ACIDS, ExperimentConfig
from protevo.storage import read_json, write_json
from tools.supplement_20261114 import strong_baselines as strong


@pytest.fixture
def fixture_state(tmp_path):
    variants = ["".join(v) for v in islice(product(AMINO_ACIDS, repeat=4), 160)]
    table = pd.DataFrame({"Variants": variants, "Fitness": [(i*7 % 43)/5 for i in range(len(variants))]})
    source = tmp_path / "source.csv"
    table.to_csv(source, index=False)
    config = ExperimentConfig(seed=11, train_size=8, validation_size=4, test_size=4, k=10,
                              rounds=3, proposal_count=40, score_budget=40, tree_estimators=3,
                              tree_min_samples_leaf=[1], model_families=["random_forest"], n_jobs=1,
                              save_predictors=False)
    prepare(source, tmp_path / "data", config, strong.REFERENCE, "Variants", "Fitness")
    dataset = PublicDataset(tmp_path / "data")
    selection = select_model(dataset, config, tmp_path / "selection")
    return SimpleNamespace(root=tmp_path, source=source, table=table, config=config, dataset=dataset, selection=selection)


def forbid_test_labels(monkeypatch):
    original = pd.read_csv

    def guarded(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).name == "test.csv":
            raise AssertionError("Held-out test labels must never be opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)


def assert_history(state, initial):
    prior = []
    for number, record in enumerate(state["rounds"], 1):
        assert record["stage"] == "evaluated"
        assert record["known_variants_before"] == initial + prior
        assert len(record["recommendations"]) == 10
        current = [r["variant"] for r in record["recommendations"]]
        assert not set(current) & set(prior)
        assert all("actual_fitness" in r for r in record["recommendations"])
        prior.extend(current)
    assert state["total_assays"] == 30


def test_full_pool_arms_fit_only_declared_training_history(fixture_state, monkeypatch):
    fixture = fixture_state
    forbid_test_labels(monkeypatch)
    original_fit = strong.fit_model
    fit_calls = []

    def spy(spec, config, known):
        fit_calls.append(known.copy(deep=True))
        return original_fit(spec, config, known)

    monkeypatch.setattr(strong, "fit_model", spy)
    initial = list(fixture.dataset.train.variant)
    frozen = strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "full_pool_frozen", fixture.root / "frozen", chunk_size=7)
    assert len(fit_calls) == 1
    assert list(fit_calls[0].variant) == initial
    fit_calls.clear()
    updated = strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "full_pool_updated", fixture.root / "updated", chunk_size=11)
    assert [len(known) for known in fit_calls] == [8, 18, 28]
    for record, trained in zip(updated["rounds"], fit_calls):
        assert record["training_variants"] == list(trained.variant)
        assert record["training_variants"] == record["known_variants_before"]
        if record["round"] > 1:
            old = updated["rounds"][record["round"]-2]["recommendations"]
            assert trained.tail(10).fitness.tolist() == [r["actual_fitness"] for r in old]
    assert [r["training_count"] for r in frozen["rounds"]] == [8, 8, 8]
    assert [r["predictor_fitted_this_round"] for r in frozen["rounds"]] == [True, False, False]
    assert [r["predictor_fitted_this_round"] for r in updated["rounds"]] == [True, True, True]
    first_scores = pd.read_csv(frozen["rounds"][0]["scores_file"]).set_index("variant").predicted_fitness
    for record in frozen["rounds"][1:]:
        later = pd.read_csv(record["scores_file"]).set_index("variant").predicted_fitness
        np.testing.assert_array_equal(later.to_numpy(), first_scores.loc[later.index].to_numpy())
    assert_history(frozen, initial)
    assert_history(updated, initial)
    assert frozen["rounds"][0]["recommendations"] == updated["rounds"][0]["recommendations"]
    assert fixture.dataset.train.variant.tolist() == initial
    assert len(fixture.dataset.train) == 8
    assert not list(fixture.root.rglob("*.joblib"))


def test_every_remaining_prediction_is_saved_before_selection(fixture_state, monkeypatch):
    fixture = fixture_state
    root = fixture.root / "full"
    original_oracle = HiddenOracle
    oracle_calls = []

    class CheckedOracle:
        def __init__(self, directory, allowed):
            number = len(oracle_calls)+1
            record = read_json(root / f"round-{number:03d}/selection.json")
            saved = read_json(root / "state.json")["rounds"][-1]
            assert record["stage"] == saved["stage"] == "selected"
            assert not any("actual_fitness" in r for r in record["recommendations"])
            assert list(allowed) == [r["variant"] for r in record["recommendations"]]
            scores = pd.read_csv(record["scores_file"])
            assert len(scores) == len(fixture.dataset.candidates)-10*(number-1)
            assert scores.variant.nunique() == len(scores)
            assert set(scores.columns) == {"variant", "predicted_fitness"}
            oracle_calls.append(list(allowed))
            self.actual = original_oracle(directory, allowed)

        def evaluate(self, selected):
            return self.actual.evaluate(selected)

    monkeypatch.setattr(strong, "HiddenOracle", CheckedOracle)
    state = strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "full_pool_updated", root)
    count = len(fixture.dataset.candidates)
    assert [r["score_count"] for r in state["rounds"]] == [count, count-10, count-20]
    assert state["total_score_count"] == count*3-30
    assert len(oracle_calls) == 3


def test_random_arm_has_no_predictor_and_is_deterministic(fixture_state, monkeypatch):
    fixture = fixture_state
    monkeypatch.setattr(strong, "fit_model", lambda *args: pytest.fail("Random arm must not fit a predictor"))
    first = strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "uniform_random", fixture.root / "random-a")
    second = strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "uniform_random", fixture.root / "random-b")
    assert first["total_score_count"] == 0
    assert [r["recommendations"] for r in first["rounds"]] == [r["recommendations"] for r in second["rounds"]]
    assert all(r["training_variants"] == [] and r["scores_file"] is None for r in first["rounds"])
    assert_history(first, list(fixture.dataset.train.variant))


def test_id_ties_are_deterministic_and_chunking_keeps_all_ids(tmp_path):
    class Constant:
        def predict(self, features):
            return np.ones(len(features))

    table = strong.score_full_pool(Constant(), ["AAAY", "AAAC", "AAAA"], tmp_path / "scores.csv.gz", chunk_size=2)
    assert table.variant.tolist() == ["AAAA", "AAAC", "AAAY"]
    assert strong.select_from_public_scores(table, 2).variant.tolist() == ["AAAA", "AAAC"]
    with pytest.raises(ValueError, match="overwritten"):
        strong.score_full_pool(Constant(), ["AAAA"], tmp_path / "scores.csv.gz")
    with pytest.raises(ValueError, match="unlabeled"):
        strong.select_from_public_scores(table.assign(actual_fitness=1), 2)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_model_predictions_are_rejected(tmp_path, invalid):
    class Invalid:
        def predict(self, features):
            return [invalid]*len(features)

    with pytest.raises(ValueError, match="finite"):
        strong.score_full_pool(Invalid(), ["AAAA"], tmp_path / "scores.csv.gz")
    assert not (tmp_path / "scores.csv.gz").exists()


def test_repeated_arm_does_not_overwrite_existing_execution(fixture_state):
    fixture = fixture_state
    root = fixture.root / "random"
    strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "uniform_random", root)
    before = (root / "state.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "uniform_random", root)
    assert (root / "state.json").read_text(encoding="utf-8") == before


def test_oracle_failure_retains_durable_selected_state(fixture_state, monkeypatch):
    fixture = fixture_state
    root = fixture.root / "interrupted"

    class Unavailable:
        def __init__(self, *args):
            raise RuntimeError("synthetic oracle interruption")

    monkeypatch.setattr(strong, "HiddenOracle", Unavailable)
    with pytest.raises(RuntimeError):
        strong.run_arm(fixture.dataset, fixture.selection, fixture.config, "uniform_random", root)
    saved = read_json(root / "state.json")
    assert saved["status"] == "failed"
    assert saved["error_type"] == "RuntimeError"
    assert saved["rounds"][0]["stage"] == "selected"
    assert not any("actual_fitness" in r for r in saved["rounds"][0]["recommendations"])


def test_arm_rejects_mismatching_manifest_or_config(fixture_state):
    fixture = fixture_state
    selection = deepcopy(fixture.selection)
    selection["dataset_manifest"]["counts"]["train"] += 1
    with pytest.raises(ValueError, match="dataset"):
        strong.run_arm(fixture.dataset, selection, fixture.config, "uniform_random", fixture.root / "invalid")
    assert not (fixture.root / "invalid").exists()


def test_prepare_state_is_declared_before_generation_and_never_queries_oracle(fixture_state, monkeypatch):
    fixture = fixture_state
    project = fixture.root / "project"
    source = project / "data/raw/elife-16965-supp1-v4.xlsx"
    source.parent.mkdir(parents=True)
    fixture.table.to_excel(source, index=False, sheet_name="Supplementary Dataset 1")
    output = fixture.root / "new-state"
    config = fixture.config.model_copy(update={"seed": 1201, "oversize_policy": "first_n", "require_full_batch": True,
                                                 "llm_reasoning_effort": "xhigh", "llm_max_output_tokens": None, "llm_timeout": None})
    original_prepare = strong.prepare
    called = []

    def declared(*args, **kwargs):
        declaration = read_json(output / "declaration.json")
        assert declaration["seed"] == 1201
        assert declaration["config"] == config.model_dump()
        called.append(True)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(strong, "prepare", declared)
    monkeypatch.setattr(strong, "HiddenOracle", lambda *args: pytest.fail("State preparation cannot query candidate labels"))
    forbid_test_labels(monkeypatch)
    prepared = strong.prepare_state(project, output, 1201, config=config)
    assert prepared.data_dir == output / "data"
    assert prepared.selection_path == output / "selection/selection.json"
    assert prepared.selection["config"] == config.model_dump()
    assert prepared.dataset.manifest["config"] == config.model_dump()
    assert len(called) == 1
    reused = strong.prepare_state(project, output, 1201, config=config)
    assert reused.selection == prepared.selection
    assert len(called) == 1
    with pytest.raises(ValueError, match="different declaration"):
        strong.prepare_state(project, output, 1201, config=config.model_copy(update={"llm_reasoning_effort": "high"}))
    assert not list(output.rglob("*.joblib"))


def test_prepare_historical_only_snapshots_public_inputs_and_freezes_score_budget(fixture_state, monkeypatch):
    fixture = fixture_state
    monkeypatch.setattr(strong, "historical_inputs", lambda project, seed:
                        (fixture.dataset, fixture.selection, fixture.config, fixture.root / "selection/selection.json"))
    monkeypatch.setattr(strong, "fit_model", lambda *args: pytest.fail("Historical preparation must not fit models"))
    monkeypatch.setattr(strong, "HiddenOracle", lambda *args: pytest.fail("Historical preparation must not query labels"))
    output = fixture.root / "historical-prepared"
    prepared = strong.prepare_historical(fixture.root, output)
    assert prepared["planned_trajectories"] == 9
    assert prepared["planned_rounds"] == 27
    assert prepared["planned_assays"] == 270
    assert prepared["planned_predictor_scores"] == 6*(3*len(fixture.dataset.candidates)-30)
    assert prepared["llm_requests"] == 0
    assert prepared["plan"]["historical_comparison_budget"].endswith("40 sampled candidates per round")
    assert strong.prepare_historical(fixture.root, output) == prepared
    frozen = output / "inputs/seed-11/train.csv"
    frozen.write_text(frozen.read_text(encoding="utf-8")+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        strong.prepare_historical(fixture.root, output)


def test_unapproved_or_uncorrected_execution_stops_before_any_inputs(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        strong.run_prepared(tmp_path / "absent", tmp_path / "absent-status")
    write_json(tmp_path / "status.json", {"status": "pending"})
    with pytest.raises(ValueError, match="correction"):
        strong.run_prepared(tmp_path / "absent", tmp_path / "status.json", allow_offline_experiment=True)
    assert not (tmp_path / "absent").exists()
