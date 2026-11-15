"""Offline tests for command persistence, isolation, budgets and recovery."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from protevo import research_session as research
from protevo.data import describe_variant, prepare
from protevo.schema import ExperimentConfig, variant_allowed
from protevo.storage import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


@pytest.fixture
def session(tmp_path):
    config = ExperimentConfig(seed=11, train_size=8, validation_size=4, test_size=4,
                              pool_size=24, rounds=2, k=2, score_budget=4, proposal_count=4,
                              model_families=["random_forest"], tree_estimators=5, tree_min_samples_leaf=[1], n_jobs=1)
    variants = ["".join(v) for v in product("ACDE", repeat=4)][:40]
    source = tmp_path / "source.csv"
    pd.DataFrame({"variant": variants, "fitness": np.arange(40) / 10}).to_csv(source, index=False)
    dataset_dir = tmp_path / "dataset"
    prepare(source, dataset_dir, config, REFERENCE, "variant", "fitness")
    selection = tmp_path / "selection.json"
    write_json(selection, {"config": config.model_dump(), "dataset_manifest": read_json(dataset_dir / "manifest.json"),
                           "spec": {"family": "random_forest", "parameters": {"n_estimators": 5, "min_samples_leaf": 1}}})
    args = {"dataset_dir": str(dataset_dir), "selection_path": str(selection)}
    root = tmp_path / "runs"
    def call(command, operation_id=None, arguments=None, run_id="run-a"):
        return research.execute_request({"command": command, "operationId": operation_id or command,
                                         "runId": run_id, "root": str(root),
                                         "arguments": arguments or {}})
    result = call("init", arguments=args)
    assert result["ok"], result
    return {"call": call, "root": root, "directory": root / "run-a", "dataset": dataset_dir,
            "args": args, "config": config, "variants": pd.read_csv(dataset_dir / "candidates.csv").variant.tolist()}


def score(session, variants=None, op="score"):
    return session["call"]("score", op, {"sequences": variants or session["variants"][:4],
                                         "sources": ["native-proposal"] * len(variants or session["variants"][:4]),
                                         "purpose": "Compare the proposed residue combinations"})


def submit(session, variants=None, op="submit"):
    variants = variants or session["variants"][:2]
    return session["call"]("submit", op, {"sequences": variants, "reasons": ["Predicted binding fitness"] * len(variants)})


def test_budget_exhaustion_still_allows_history_and_submission(session):
    assert score(session)["ok"]
    history = session["call"]("history")
    assert history["ok"] and len(history["result"]["scored"]) == 4
    assert history["result"]["status"] == "active"
    assert history["result"]["budget"]["scores_remaining"] == 0
    assert not session["call"]("score", "too-many", {"sequences": session["variants"][4:5]})["ok"]
    result = submit(session)
    assert result["ok"] and result["result"]["status"] == "round_submitted"
    assert result["result"]["knownCount"] == 10
    assert len(result["result"]["recommendations"]) == 2
    assert {r["source"] for r in result["result"]["recommendations"]} == {"native-proposal"}
    assert {r["source_kind"] for r in result["result"]["recommendations"]} == {"public_self_report"}


def test_no_candidate_or_validation_labels_before_submit(session, monkeypatch):
    original = pd.read_csv
    reads = []
    def checked(path, *args, **kwargs):
        reads.append(str(path))
        assert "hidden" not in Path(path).parts
        assert Path(path).name != "validation.csv"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", checked)
    assert score(session)["ok"]
    result = session["call"]("history", arguments={"candidate_limit": 20})
    assert result["ok"] and len(result["result"]["known"]) == 8
    assert all(isinstance(v, str) for v in result["result"]["candidates"])
    assert session["call"]("status")["ok"]
    assert reads


def test_init_does_not_read_validation_or_hidden_labels(session, monkeypatch):
    original = pd.read_csv
    def checked(path, *args, **kwargs):
        assert "hidden" not in Path(path).parts
        assert Path(path).name != "validation.csv"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", checked)
    assert session["call"]("init", "other-init", session["args"], run_id="run-b")["ok"]


def test_cross_round_training_and_run_isolation(session, monkeypatch):
    captured = []
    original = research.fit_model
    def fitted(spec, config, known):
        captured.append(set(known.variant))
        return original(spec, config, known)
    monkeypatch.setattr(research, "fit_model", fitted)
    assert score(session)["ok"]
    initial = set(pd.read_csv(session["dataset"] / "train.csv").variant)
    assert captured[-1] == initial
    assert submit(session)["ok"]
    assert session["call"]("next_round")["ok"]
    assert score(session, session["variants"][4:8], "score-r2")["ok"]
    assert captured[-1] == initial | set(session["variants"][:2])
    assert session["call"]("init", "init-other", session["args"], run_id="run-b")["ok"]
    other = session["call"]("history", "other-history", run_id="run-b")
    assert other["result"]["known_count"] == 8
    assert not other["result"]["previous_rounds"]


def test_idempotency_and_changed_input(session, monkeypatch):
    calls = []
    original = research._predict
    def predicted(state, variants):
        calls.append(variants)
        return original(state, variants)
    monkeypatch.setattr(research, "_predict", predicted)
    first = score(session)
    assert score(session) == first
    assert len(calls) == 1
    changed = score(session, session["variants"][4:8])
    assert not changed["ok"] and "different request" in changed["error"]["message"]
    submitted = submit(session)
    assert submit(session) == submitted
    assert not submit(session, op="duplicate-submit")["ok"]
    assert len(read_json(session["directory"] / "state.json")["rounds"]) == 1


def test_cached_predictions_do_not_consume_budget_again(session):
    assert score(session)["ok"]
    assert score(session, session["variants"][:2], "score-cached")["ok"]
    assert session["call"]("status")["result"]["scoredCount"] == 4


@pytest.mark.parametrize("arguments", [
    {"sequences": []}, {"sequences": ["AAAA", "AAAA"]}, {"sequences": ["INVALID"]},
    {"sequences": [1]}, {"sequences": "AAAA"}, {"sequences": ["AAAA"], "sources": []},
    {"sequences": ["AAAA"], "reasoning": "private"},
])
def test_invalid_score_inputs(session, arguments):
    result = session["call"]("score", "invalid", arguments)
    assert not result["ok"]
    assert session["call"]("status")["result"]["scoredCount"] == 0


def test_atomic_oversized_score_and_unscored_submit(session):
    assert not score(session, session["variants"][:5])["ok"]
    assert session["call"]("status")["result"]["scoredCount"] == 0
    assert not submit(session)["ok"]
    assert not session["call"]("next_round")["ok"]


def test_invalid_submission_never_queries_oracle(session, monkeypatch):
    assert score(session)["ok"]
    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid submission queried measurements")
    monkeypatch.setattr(research.HiddenOracle, "evaluate", forbidden)
    assert not session["call"]("submit", "bad-count", {"sequences": session["variants"][:1], "reasons": ["Reason"]})["ok"]
    assert not session["call"]("submit", "bad-reason", {"sequences": session["variants"][:2], "reasons": ["", "Reason"]})["ok"]
    assert not session["call"]("submit", "bad-duplicate", {"sequences": session["variants"][:1] * 2,
                                                             "reasons": ["Reason"] * 2})["ok"]


def test_submit_queries_only_selected_and_records_pre_query_state(session, monkeypatch):
    assert score(session)["ok"]
    original = research.HiddenOracle.evaluate
    def checked(oracle, variants):
        state = read_json(session["directory"] / "state.json")
        assert state["status"] == "submission_pending"
        assert state["pending"]["sequences"] == variants == session["variants"][:2]
        assert oracle.allowed == set(variants)
        assert len(state["known"]) == 8
        return original(oracle, variants)
    monkeypatch.setattr(research.HiddenOracle, "evaluate", checked)
    result = submit(session)
    assert result["ok"]
    record = read_json(session["directory"] / "operations" / "submit.json")
    assert record["before"]["phase"] == "active"
    assert record["after"]["phase"] == "round_submitted"
    assert record["request"]["arguments"]["sequences"] == session["variants"][:2]
    assert all(r["actual_fitness"] is not None for r in result["result"]["recommendations"])


def test_failed_or_unknown_submit_recovers_without_duplicate_measurements(session, monkeypatch):
    assert score(session)["ok"]
    original = research.HiddenOracle.evaluate
    def interrupted(oracle, variants):
        original(oracle, variants)
        raise RuntimeError("Local query result interrupted")
    monkeypatch.setattr(research.HiddenOracle, "evaluate", interrupted)
    first = submit(session)
    assert not first["ok"] and first["retryable"]
    assert session["call"]("status")["result"]["phase"] == "submission_pending"
    assert session["call"]("history")["result"]["known_count"] == 8
    assert not submit(session, session["variants"][2:4], "different-submit")["ok"]
    assert not session["call"]("next_round")["ok"]
    monkeypatch.setattr(research.HiddenOracle, "evaluate", original)
    recovered = submit(session)
    assert recovered["ok"] and recovered["result"]["known_count"] == 10
    assert submit(session) == recovered
    assert len(read_json(session["directory"] / "operations" / "submit.json")["attempts"]) == 2


def test_process_interruption_releases_pending_submission_for_resume(session, monkeypatch):
    assert score(session)["ok"]
    original = research.HiddenOracle.evaluate
    def interrupted(*_args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(research.HiddenOracle, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        submit(session)
    monkeypatch.setattr(research.HiddenOracle, "evaluate", original)
    assert submit(session)["ok"]


def test_committed_receipt_recovers_after_journal_write_failure(session, monkeypatch):
    assert score(session)["ok"]
    original = research.write_json
    once = []
    def fail_after_commit(path, value):
        if Path(path).name == "submit.json" and value.get("status") == "finished" and not once:
            once.append(True)
            raise OSError("Result channel interrupted")
        return original(path, value)
    monkeypatch.setattr(research, "write_json", fail_after_commit)
    assert not submit(session)["ok"]
    monkeypatch.setattr(research, "write_json", original)
    result = submit(session)
    assert result["ok"] and result["result"]["known_count"] == 10
    assert len(read_json(session["directory"] / "state.json")["rounds"]) == 1


def test_configuration_is_frozen_and_source_train_is_snapshotted(session):
    original = pd.read_csv(session["dataset"] / "train.csv")
    original.fitness = 999
    original.to_csv(session["dataset"] / "train.csv", index=False)
    assert session["call"]("history")["result"]["top_known"][0]["fitness"] < 999
    science_path = session["directory"] / "science.json"
    science = read_json(science_path)
    science["config"]["rounds"] = 99
    write_json(science_path, science)
    result = session["call"]("status", "new-status")
    assert not result["ok"] and "frozen" in result["error"]["message"]
    assert not session["call"]("score", "mutate-config", {"sequences": session["variants"][:1], "config": {}})["ok"]


def test_complete_two_rounds_and_metrics(session):
    assert score(session)["ok"] and submit(session)["ok"]
    assert session["call"]("next_round")["ok"]
    assert score(session, session["variants"][4:8], "score-r2")["ok"]
    last = submit(session, session["variants"][4:6], "submit-r2")
    assert last["ok"] and last["result"]["phase"] == "completed"
    assert last["result"]["assayCount"] == 4
    assert len(last["result"]["roundMetrics"]) == 2
    assert len(last["result"]["recommendations"]) == 4
    assert not session["call"]("next_round", "next-r3")["ok"]
    assert session["call"]("history", "final-history")["ok"]


def test_unique_run_directory_and_input_path_validation(session):
    assert session["call"]("init", "init", session["args"])["ok"]
    assert not session["call"]("init", "new-init", session["args"])["ok"]
    assert not session["call"]("status", run_id="../escape")["ok"]
    assert not session["call"]("status", "../escape")["ok"]
    assert not session["call"]("status", run_id="does-not-exist")["ok"]
    assert not research.execute_request([])["ok"]


def test_os_lock_blocks_concurrent_writer_and_releases_after_process_exit(session):
    code = ("from protevo.research_session import run_lock; from pathlib import Path; import sys; "
            "ctx=run_lock(Path(sys.argv[1])); ctx.__enter__(); print('READY', flush=True); sys.stdin.readline()")
    process = subprocess.Popen([sys.executable, "-c", code, str(session["directory"])], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "READY"
        result = session["call"]("status", "concurrent-status")
        assert not result["ok"] and "busy" in result["error"]["message"]
    finally:
        process.terminate()
        process.communicate(timeout=15)
    assert session["call"]("status", "concurrent-status")["ok"]


def test_cli_one_request_per_process_with_json_receipt(session):
    request = {"command": "status", "operationId": "cli-status", "runId": "run-a",
               "root": str(session["root"]), "arguments": {}}
    command = [sys.executable, str(ROOT / "tools" / "research_session_cli.py")]
    first = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    assert receipt["result"]["knownCount"] == 8
    second = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert json.loads(second.stdout) == receipt
    bad = subprocess.run(command, input="not json", capture_output=True, text=True, cwd=ROOT, timeout=30)
    assert bad.returncode == 1 and not json.loads(bad.stdout)["ok"]


def test_esm_predictor_refits_known_only(session, tmp_path):
    all_variants = pd.concat([pd.read_csv(session["dataset"] / "train.csv"),
                              pd.read_csv(session["dataset"] / "candidates.csv")], ignore_index=True)
    ids = tmp_path / "embedding-ids.csv"
    # Extra label columns in an existing embedding index are never requested.
    all_variants[["variant"]].assign(fitness=999).to_csv(ids, index=False)
    embeddings = tmp_path / "features.npy"
    np.save(embeddings, np.random.default_rng(5).normal(size=(len(all_variants), 5)))
    args = deepcopy(session["args"])
    args["predictor"] = {"kind": "esm2_ridge", "alpha": 1.0, "variants_path": str(ids),
                         "embeddings_path": str(embeddings)}
    assert session["call"]("init", "esm-init", args, "esm-run")["ok"]
    result = session["call"]("score", "esm-score", {"sequences": session["variants"][:4]}, "esm-run")
    assert result["ok"] and len(result["result"]["scored"]) == 4


def test_total_assay_cap_allows_smaller_final_submission(session):
    manifest_path = session["dataset"] / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["config"]["max_assays"] = 3
    write_json(manifest_path, manifest)
    selection_path = Path(session["args"]["selection_path"])
    selection = read_json(selection_path)
    selection.update(config=manifest["config"], dataset_manifest=manifest)
    write_json(selection_path, selection)
    assert session["call"]("init", "capped-init", session["args"], "capped")["ok"]
    call = session["call"]
    assert call("score", "capped-score-1", {"sequences": session["variants"][:4]}, "capped")["ok"]
    assert call("submit", "capped-submit-1", {"sequences": session["variants"][:2], "reasons": ["Reason"] * 2}, "capped")["ok"]
    assert call("next_round", "capped-next", {}, "capped")["ok"]
    assert call("score", "capped-score-2", {"sequences": session["variants"][4:8]}, "capped")["ok"]
    result = call("submit", "capped-submit-2", {"sequences": session["variants"][4:5], "reasons": ["Reason"]}, "capped")
    assert result["ok"] and result["result"]["phase"] == "completed"
    assert result["result"]["assayCount"] == result["result"]["assayLimit"] == 3


def test_same_run_init_is_idempotent_across_source_changes(session):
    (session["dataset"] / "manifest.json").write_text("{}", encoding="utf-8")
    assert session["call"]("init", "init", session["args"])["ok"]
    assert not session["call"]("init", "new-run-init", session["args"], "new-run")["ok"]


def test_nonfinite_request_and_unknown_fields_are_rejected(session):
    assert not session["call"]("score", "nonfinite", {"sequences": session["variants"][:1], "purpose": float("nan")})["ok"]
    request = {"command": "status", "operationId": "unknown-field", "runId": "run-a",
               "root": str(session["root"]), "arguments": {}, "reasoning": "unaccepted"}
    assert not research.execute_request(request)["ok"]


@pytest.mark.parametrize("variant, expected", [("WDGV", True), ("VDGV", False), ("YDGV", False),
                                                ("WAGV", False), ("WDAV", False), ("WDGA", False),
                                                ("WDG*", False), ("WDG", False), (None, False)])
def test_mutable_site_aa_subset_preserves_fixed_reference_residues(variant, expected):
    config = ExperimentConfig(allowed_positions=[39], allowed_amino_acids="W")
    assert research._constraint_allowed(variant, config) is expected
    assert variant_allowed("WDGV", config) is False


@pytest.mark.parametrize("positions", [[39], [39, 54], [39, 40, 41, 54]])
@pytest.mark.parametrize("minimum,maximum", [(0, 4), (1, 2), (0, 0)])
def test_full_amino_acid_default_preserves_legacy_eligibility(positions, minimum, maximum):
    config = ExperimentConfig(allowed_positions=positions, min_mutations=minimum, max_mutations=maximum)
    for residues in product("VDGW", repeat=4):
        variant = "".join(residues)
        assert research._constraint_allowed(variant, config) == variant_allowed(variant, config)


def test_real_seed_39w_candidate_is_available_and_scoreable_without_oracle(tmp_path, monkeypatch):
    dataset = ROOT / "data" / "processed" / "seed-11"
    selection = ROOT / "results" / "seed-11" / "selection" / "selection.json"
    if not (dataset / "manifest.json").is_file() or not selection.is_file():
        pytest.skip("Frozen GB1 seed-11 public inputs are not installed")
    original = pd.read_csv
    def public_only(path, *args, **kwargs):
        assert "hidden" not in Path(path).parts and Path(path).name != "validation.csv"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", public_only)
    def forbidden(*_args):
        pytest.fail("Restricted candidate checks queried measured feedback")
    monkeypatch.setattr(research.HiddenOracle, "evaluate", forbidden)
    request = {"command": "init", "operationId": "real-init", "runId": "real-39w", "root": str(tmp_path / "runs"),
               "arguments": {"dataset_dir": str(dataset), "selection_path": str(selection),
                             "research": {"allowed_positions": [39], "allowed_amino_acids": "W", "k": 1,
                                          "rounds": 1, "score_budget": 1}}}
    result = research.execute_request(request)
    assert result["ok"], result
    request.update(command="history", operationId="real-history", arguments={"candidate_limit": 10})
    history = research.execute_request(request)
    assert history["result"]["candidates"] == ["WDGV"] and history["result"]["eligible_count"] == 1
    request.update(command="score", operationId="real-score", arguments={"sequences": ["WDGV"]})
    scored = research.execute_request(request)
    assert scored["ok"] and np.isfinite(scored["result"]["scored"][0]["predicted_fitness"])
    request.update(command="diagnostics", operationId="real-heatmap", arguments={"kind": "heatmap"})
    heatmap = research.execute_request(request)
    assert heatmap["ok"], heatmap
    cell = heatmap["result"]["data"]["cells"][0]
    assert cell["variant"] == "WDGV" and cell["eligible"] and cell["constraintAllowed"]
    assert heatmap["result"]["budget"]["scores_used"] == 1
    assert heatmap["result"]["budget"]["diagnostic_used"] == 2


def test_predictor_failure_does_not_spend_budget(session, monkeypatch):
    def failed(*_args):
        raise ValueError("Predictor unavailable")
    monkeypatch.setattr(research, "_predict", failed)
    result = score(session)
    assert not result["ok"]
    assert session["call"]("status")["result"]["scoredCount"] == 0
    record = read_json(session["directory"] / "operations" / "score.json")
    assert record["result"]["error"]["message"] == "Predictor unavailable"
