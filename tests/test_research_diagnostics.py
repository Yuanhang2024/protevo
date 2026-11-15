"""Offline checks for frozen research settings and label-safe scientific views."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from protevo import research_diagnostics as diagnostics
from protevo import research_session as research
from protevo.data import describe_variant, one_hot
from protevo.models import fit_model
from protevo.schema import ExperimentConfig, POSITIONS
from protevo.storage import read_json, write_json
from test_research_session import REFERENCE, session, score, submit


def init_research(session, overrides=None, predictor=None, run_id="configured"):
    args = deepcopy(session["args"])
    args["research"] = overrides or {}
    if predictor is not None:
        args["predictor"] = predictor
    return session["call"]("init", f"init-{run_id}", args, run_id)


def diagnose(session, kind, operation=None, **arguments):
    return session["call"]("diagnostics", operation or f"diagnostic-{kind}", {"kind": kind, **arguments})


def test_research_settings_derive_after_original_selection_validation(session):
    manifest_before = read_json(session["dataset"] / "manifest.json")
    result = init_research(session, {"rounds": 3, "k": 3, "score_budget": 5, "diagnostic_budget": 77,
                                     "reference": REFERENCE})
    assert result["ok"], result
    assert result["result"]["roundLimit"] == 3
    assert result["result"]["scoreLimit"] == 5
    assert result["result"]["research"]["k"] == 3
    assert result["result"]["assayLimit"] == 9
    state = read_json(session["root"] / "configured" / "state.json")
    assert state["science"]["config"]["proposal_count"] == 5
    assert state["science"]["original_config"]["score_budget"] == 4
    assert read_json(session["dataset"] / "manifest.json") == manifest_before
    selection = Path(session["args"]["selection_path"])
    original = read_json(selection)
    original["config"]["seed"] = 99
    write_json(selection, original)
    assert not init_research(session, {"rounds": 3}, run_id="bad-original")["ok"]


@pytest.mark.parametrize("overrides", [
    {"k": 0}, {"k": 41}, {"score_budget": 1}, {"score_budget": True}, {"rounds": 0},
    {"allowed_positions": []}, {"allowed_positions": [1]}, {"allowed_positions": [39, 39]},
    {"allowed_amino_acids": "AC*"}, {"allowed_amino_acids": "AA"},
    {"reference": "A" * 56}, {"reference": REFERENCE[:1] + "T" + REFERENCE[2:]},
    {"diagnostic_budget": -1}, {"diagnostic_budget": 1.5}, {"seed": 22},
])
def test_invalid_research_settings_rejected(session, overrides):
    assert not init_research(session, overrides)["ok"]


def _add_public_candidates(session, variants):
    path = session["dataset"] / "candidates.csv"
    rows = pd.read_csv(path)
    for i, variant in enumerate(variants):
        rows.loc[i] = describe_variant(variant, REFERENCE)
    rows.to_csv(path, index=False)


def test_restricted_positions_and_amino_acids_are_frozen_and_enforced(session):
    _add_public_candidates(session, ["WDGV", "VDGV", "YDGV", "WDAV"])
    result = init_research(session, {"allowed_positions": [39], "allowed_amino_acids": "W", "k": 1})
    assert result["ok"], result
    assert result["result"]["research"]["allowed_positions"] == [39]
    assert result["result"]["constraintSemantics"]["version"] == "gb1-mutable-site-constraints/1"
    state = read_json(session["root"] / "configured" / "state.json")
    assert state["science"]["constraint_semantics"] == result["result"]["constraintSemantics"]
    history = session["call"]("history", "restricted-history", {"candidate_limit": 100}, "configured")
    assert history["result"]["candidates"] == ["WDGV"]
    assert session["call"]("score", "restricted-legal", {"sequences": ["WDGV"]}, "configured")["ok"]
    for variant in ("VDGV", "YDGV", "WDAV"):
        assert not session["call"]("score", f"invalid-{variant}", {"sequences": [variant]}, "configured")["ok"]
    assert not session["call"]("score", "config-change", {"research": {"k": 1}}, "configured")["ok"]


def test_empty_restricted_pool_is_reported_before_initializing_state(session):
    result = init_research(session, {"allowed_positions": [39], "allowed_amino_acids": "W", "k": 1})
    assert not result["ok"]
    assert "No eligible unmeasured candidates" in result["error"]["message"]
    assert not (session["root"] / "configured" / "state.json").exists()
    assert not (session["root"] / "configured" / "science.json").exists()


def test_restricted_heatmap_and_scoring_use_same_candidate_definition(session):
    _add_public_candidates(session, ["WDGV", "VDGV", "YDGV", "WDAV"])
    assert init_research(session, {"allowed_positions": [39], "allowed_amino_acids": "W", "k": 1})["ok"]
    result = session["call"]("diagnostics", "restricted-heatmap", {"kind": "heatmap"}, "configured")
    assert result["ok"], result
    data = result["result"]["data"]
    assert data["background"] == "VDGV" and data["backgroundConstraintAllowed"] is False
    assert len(data["cells"]) == 1
    cell = data["cells"][0]
    assert cell["variant"] == "WDGV" and cell["constraintAllowed"] and cell["eligible"]
    scored = session["call"]("score", "heatmap-candidate-score", {"sequences": [cell["variant"]]}, "configured")
    assert scored["ok"]
    assert cell["predicted_fitness"] == scored["result"]["scored"][0]["predicted_fitness"]


def test_heatmap_controls_are_distinguished_from_ineligible_multisite_candidates(session):
    _add_public_candidates(session, ["WWGV"])
    assert init_research(session, {"allowed_positions": [39, 40], "allowed_amino_acids": "W", "k": 1})["ok"]
    result = session["call"]("diagnostics", "two-sites-heatmap", {"kind": "heatmap"}, "configured")
    assert result["ok"], result
    assert {row["variant"] for row in result["result"]["data"]["cells"]} == {"WDGV", "VWGV"}
    assert all(not row["constraintAllowed"] and not row["eligible"] for row in result["result"]["data"]["cells"])
    history = session["call"]("history", "two-sites-history", {"candidate_limit": 10}, "configured")
    assert history["result"]["candidates"] == ["WWGV"]


def test_no_remaining_eligible_candidates_gives_clear_next_round_error(session, monkeypatch):
    _add_public_candidates(session, ["WDGV"])
    assert init_research(session, {"allowed_positions": [39], "allowed_amino_acids": "W", "k": 1})["ok"]
    assert session["call"]("score", "only-score", {"sequences": ["WDGV"]}, "configured")["ok"]
    monkeypatch.setattr(research.HiddenOracle, "evaluate",
                        lambda _oracle, variants: [{"variant": v, "actual_fitness": 0.8} for v in variants])
    result = session["call"]("submit", "only-submit", {"sequences": ["WDGV"], "reasons": ["Selected scored candidate"]}, "configured")
    assert result["ok"] and result["result"]["phase"] == "completed"
    revision = result["result"]["eventCursor"]
    empty = session["call"]("next_round", "empty-next", {}, "configured")
    assert not empty["ok"] and "No eligible unmeasured candidates remain" in empty["error"]["message"]
    current = session["call"]("status", "only-status", {}, "configured")
    assert current["result"]["eventCursor"] == revision and current["result"]["currentRound"] == 1


def test_status_history_and_nonpredictive_diagnostics_are_read_only(session):
    path = session["directory"] / "state.json"
    before = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    for i, command in enumerate(("status", "history", "diagnostics", "diagnostics", "diagnostics", "diagnostics")):
        args = {} if command != "diagnostics" else {"kind": ("scatter", "epistasis", "diversity", "structure")[i - 2]}
        result = session["call"](command, f"readonly-{i}", args)
        assert result["ok"], result
        assert result["result"]["readOnly"] is True
        assert result["result"]["eventCursor"] == 1
        assert not (session["directory"] / "operations" / f"readonly-{i}.json").exists()
    assert path.read_bytes() == before and path.stat().st_mtime_ns == timestamp


def test_revision_matches_committed_result_and_retry_does_not_increment(session):
    initial = session["call"]("status")["result"]["eventCursor"]
    result = score(session)
    assert result["result"]["eventCursor"] == initial + 1
    assert session["call"]("status")["result"]["eventCursor"] == initial + 1
    assert score(session) == result
    assert session["call"]("status")["result"]["eventCursor"] == initial + 1
    final = submit(session)
    assert final["result"]["eventCursor"] == initial + 2
    assert read_json(session["directory"] / "state.json")["revision"] == initial + 2


def test_wt_heatmap_has_77_unique_predictions_and_separate_persistent_budget(session):
    first = diagnose(session, "heatmap")
    assert first["ok"], first
    result = first["result"]
    assert result["data"]["background"] == "VDGV"
    assert len(result["data"]["cells"]) == 80
    assert result["data"]["uniqueVariantCount"] == 77
    assert result["budget"]["diagnostic_used"] == 77
    assert result["budget"]["scores_used"] == 0
    assert all("actual_fitness" not in cell for cell in result["data"]["cells"])
    assert all(cell["confidence_interval"] is None for cell in result["data"]["cells"])
    repeated = diagnose(session, "heatmap", "heatmap-again")
    assert repeated["ok"] and repeated["result"]["budget"]["diagnostic_used"] == 77
    assert repeated["result"]["data"] == result["data"]
    changed = diagnose(session, "heatmap", "new-background", background="ADGV")
    assert not changed["ok"] and "diagnostic budget" in changed["error"]["message"]
    assert session["call"]("status")["result"]["diagnosticBudget"]["used"] == 77
    assert score(session)["ok"]


def test_heatmap_budget_can_be_disabled_and_is_reset_each_round(session):
    result = init_research(session, {"diagnostic_budget": 0})
    assert result["ok"]
    assert not session["call"]("diagnostics", "disabled-heatmap", {"kind": "heatmap"}, "configured")["ok"]
    assert diagnose(session, "heatmap")["ok"]
    assert score(session)["ok"] and submit(session)["ok"]
    next_round = session["call"]("next_round")
    assert next_round["result"]["diagnosticBudget"]["used"] == 0
    assert diagnose(session, "heatmap", "r2-heatmap")["result"]["budget"]["diagnostic_used"] == 77


def test_diagnostics_never_read_hidden_or_validation_labels(session, monkeypatch):
    original = pd.read_csv
    def checked(path, *args, **kwargs):
        assert "hidden" not in Path(path).parts
        assert Path(path).name != "validation.csv"
        return original(path, *args, **kwargs)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Diagnostics queried the oracle")
    monkeypatch.setattr(pd, "read_csv", checked)
    monkeypatch.setattr(research.HiddenOracle, "evaluate", forbidden)
    for kind in diagnostics.KINDS:
        result = diagnose(session, kind)
        assert result["ok"], result


def test_scatter_uses_original_prequery_predictions_only(session):
    assert diagnose(session, "scatter")["result"]["data"] == {
        "points": [], "sampleCount": 0, "spearman": None,
        "scope": "Previously selected measurements with their pre-query predictions"}
    assert score(session)["ok"] and submit(session)["ok"]
    data = diagnose(session, "scatter", "after-submit-scatter")["result"]["data"]
    assert data["sampleCount"] == len(data["points"]) == 2
    assert {p["variant"] for p in data["points"]} == set(session["variants"][:2])
    assert all(p["round"] == 1 for p in data["points"])
    state = read_json(session["directory"] / "state.json")
    assert data["points"][0]["predicted_fitness"] == state["scores"][session["variants"][0]]["predicted_fitness"]


def test_epistasis_requires_four_matching_known_measurements():
    state = {"known": [{"variant": v, "fitness": f} for v, f in
                       [("VDGV", 1), ("ADGV", 2), ("VAGV", 3), ("AAGV", 8)]]}
    data = diagnostics.epistasis(state)
    pair = next(p for p in data["pairs"] if p["positions"] == [39, 40])
    assert pair == {"positions": [39, 40], "sampleCount": 1, "meanEpistasis": 4.0}
    assert all(p["meanEpistasis"] is None and p["sampleCount"] == 0 for p in data["pairs"] if p != pair)
    assert data["scale"] == "raw WT-normalized fitness"
    state["known"] = state["known"][:-1]
    assert all(p["meanEpistasis"] is None for p in diagnostics.epistasis(state)["pairs"])


def test_epistasis_labels_arbitrary_matching_backgrounds():
    state = {"known": [{"variant": v, "fitness": f} for v, f in
                       [("VDAV", 2), ("ADAV", 4), ("VAAV", 5), ("AAAV", 10)]]}
    result = diagnostics.epistasis(state)
    assert result["pairs"][0]["meanEpistasis"] == 3
    assert "other two residues are held fixed" in result["background"]


def test_pairwise_diversity_matches_direct_hamming_calculation():
    variants = ["VDGV", "ADGV", "AAGV", "AAAA"]
    data = diagnostics.diversity({"scores": dict.fromkeys(variants)})
    direct = [sum(a != b for a, b in zip(x, y)) for x, y in combinations(variants, 2)]
    assert data["sampleCount"] == 4
    assert data["meanPairwiseHamming"] == np.mean(direct)
    assert len(data["residueCounts"]) == 80
    assert diagnostics.diversity({"scores": {}})["meanPairwiseHamming"] is None


def test_rf_tree_sd_is_model_disagreement_and_not_a_confidence_interval(session):
    result = score(session)
    assert result["ok"]
    state = read_json(session["directory"] / "state.json")
    config = ExperimentConfig.model_validate(state["science"]["config"])
    model = fit_model(state["science"]["predictor"]["spec"], config, pd.DataFrame(state["training_known"]))
    assert isinstance(model, RandomForestRegressor)
    expected = np.std([tree.predict(one_hot(session["variants"][:4])) for tree in model.estimators_], axis=0, ddof=1)
    assert np.allclose([r["model_disagreement_sd"] for r in result["result"]["scored"]], expected)
    assert all(r["uncertainty_kind"] == "tree_model_disagreement" and r["confidence_interval"] is None
               for r in result["result"]["scored"])


def test_mlp_predictor_is_real_fitted_model_and_has_no_invented_interval(session):
    assert init_research(session, predictor={"kind": "mlp_onehot"})["ok"]
    result = session["call"]("score", "mlp-score", {"sequences": session["variants"][:4]}, "configured")
    assert result["ok"], result
    assert all(np.isfinite(r["predicted_fitness"]) and r["model_disagreement_sd"] is None
               and r["confidence_interval"] is None for r in result["result"]["scored"])
    assert not init_research(session, predictor={"kind": "custom"}, run_id="custom")["ok"]


def test_structure_has_real_ca_coordinates_and_explicit_reference_difference():
    data = diagnostics.structure(REFERENCE, [39, 41])
    assert len(data["residues"]) == 56
    assert [r["position"] for r in data["residues"] if r["isMutable"]] == [39, 41]
    assert data["referenceDifferences"] == [{"position": 2, "structureResidue": "T", "experimentResidue": "Q"}]
    assert all(np.isfinite([r["x"], r["y"], r["z"]]).all() for r in data["residues"])
    assert len({(r["x"], r["y"], r["z"]) for r in data["residues"]}) == 56


def test_missing_esm_heatmap_embeddings_produce_null_predictions(session, tmp_path):
    known = pd.read_csv(session["dataset"] / "train.csv", usecols=["variant"])
    ids = tmp_path / "ids.csv"
    known.to_csv(ids, index=False)
    features = tmp_path / "known-features.npy"
    np.save(features, np.random.default_rng(11).normal(size=(len(known), 5)))
    predictor = {"kind": "esm2_ridge", "alpha": 1, "variants_path": str(ids), "embeddings_path": str(features)}
    assert init_research(session, predictor=predictor)["ok"]
    result = session["call"]("diagnostics", "missing-embeddings", {"kind": "heatmap"}, "configured")
    assert result["ok"], result
    assert result["result"]["data"]["baselinePrediction"] is None
    for cell in result["result"]["data"]["cells"]:
        assert cell["predicted_fitness"] is cell["delta_from_background"] is cell["model_disagreement_sd"] is None
        assert cell["prediction_status"] == "embedding_unavailable"


def test_replayed_receipt_preserves_original_operation_after_state(session):
    assert score(session)["ok"]
    result = submit(session)
    path = session["directory"] / "operations" / "submit.json"
    before = read_json(path)
    assert session["call"]("next_round")["ok"]
    assert submit(session) == result
    assert read_json(path) == before


def test_cross_round_heatmap_retry_retains_original_basis_round(session):
    result = diagnose(session, "heatmap")
    assert result["result"]["basisRound"] == 1
    assert result["result"]["basisTrainingKnownCount"] == 8
    assert score(session)["ok"] and submit(session)["ok"]
    assert session["call"]("next_round")["ok"]
    assert diagnose(session, "heatmap") == result
    current = diagnose(session, "heatmap", "round2-heatmap")
    assert current["result"]["basisRound"] == 2
    assert current["result"]["basisTrainingKnownCount"] == 10
    assert session["call"]("status")["result"]["eventCursor"] == current["result"]["eventCursor"]


def test_cli_utf8_request_and_diagnostic_cache_survive_separate_processes(session):
    root = Path(__file__).resolve().parents[1]
    cli = [sys.executable, str(root / "tools" / "research_session_cli.py")]
    request = {"command": "score", "operationId": "utf8-score", "runId": "run-a", "root": str(session["root"]),
               "arguments": {"sequences": session["variants"][:1], "purpose": "比较当前背景中的氨基酸替换"}}
    result = subprocess.run(cli, input=json.dumps(request, ensure_ascii=False), capture_output=True,
                            encoding="utf-8", cwd=root, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]
    stored = read_json(session["directory"] / "operations" / "utf8-score.json")
    assert stored["request"]["arguments"]["purpose"] == request["arguments"]["purpose"]
    request.update(command="diagnostics", operationId="cli-heatmap", arguments={"kind": "heatmap"})
    first = subprocess.run(cli, input=json.dumps(request), capture_output=True, encoding="utf-8", cwd=root, timeout=30)
    request["operationId"] = "cli-heatmap-again"
    second = subprocess.run(cli, input=json.dumps(request), capture_output=True, encoding="utf-8", cwd=root, timeout=30)
    assert first.returncode == second.returncode == 0
    assert json.loads(first.stdout)["result"]["data"] == json.loads(second.stdout)["result"]["data"]
    assert json.loads(second.stdout)["result"]["budget"]["diagnostic_used"] == 77


def test_initial_best_is_frozen_before_new_feedback(session, monkeypatch):
    initial = session["call"]("status")["result"]
    expected = float(pd.read_csv(session["dataset"] / "train.csv").fitness.max())
    assert initial["initialBest"] == expected
    assert initial["initialKnownCount"] == 8
    assert score(session)["ok"]
    def measured(_oracle, variants):
        return [{"variant": v, "actual_fitness": 10000.0} for v in variants]
    monkeypatch.setattr(research.HiddenOracle, "evaluate", measured)
    assert submit(session)["ok"]
    assert session["call"]("next_round")["ok"]
    current = session["call"]("status")["result"]
    assert current["initialBest"] == expected < 10000.0
    assert current["initialKnownCount"] == 8
    state = read_json(session["directory"] / "state.json")
    assert max(row["fitness"] for row in state["known"]) == 10000.0
    assert state["science"]["initial_measurements"] == {"count": 8, "best_fitness": expected}


def test_legacy_initial_best_uses_initial_prefix_and_not_later_best(session, monkeypatch):
    initial_best = session["call"]("status")["result"]["initialBest"]
    assert score(session)["ok"]
    monkeypatch.setattr(research.HiddenOracle, "evaluate",
                        lambda _oracle, variants: [{"variant": v, "actual_fitness": 10000.0} for v in variants])
    assert submit(session)["ok"]
    state = read_json(session["directory"] / "state.json")
    del state["science"]["initial_measurements"]
    assert research._status(state)["initialBest"] == initial_best
    assert research._status(state)["initialKnownCount"] == 8
    state["known"] = state["known"][1:] + state["known"][:1]
    assert research._status(state)["initialBest"] is None
    assert research._status(state)["initialKnownCount"] is None


def test_diagnostic_inputs_reject_hidden_or_unknown_fields(session):
    assert not diagnose(session, "unknown")["ok"]
    assert not diagnose(session, "heatmap", background="AAAAA")["ok"]
    assert not diagnose(session, "scatter", background="VDGV")["ok"]
    assert not diagnose(session, "structure", oracle=True)["ok"]
