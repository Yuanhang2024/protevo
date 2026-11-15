"""Durable scientific commands for an externally controlled GB1 research session.

The caller owns the LLM conversation. This module never calls a language model.
Requests use command, operationId, runId, root and arguments. IDs are safe path
components. Reusing an operationId requires the exact same request. Failed local
measurement queries retain a fixed pending submission and may be retried with
that request. Source and purpose strings are public, self-reported annotations.

Windows locking uses the standard-library nonblocking byte lock:
https://docs.python.org/3/library/msvcrt.html#msvcrt.locking
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from . import research_diagnostics
from .data import HiddenOracle, PublicDataset, describe_variant, one_hot, validate_reference
from .models import fit_model, load_selection
from .schema import AMINO_ACIDS, ExperimentConfig, POSITIONS
from .storage import read_json, write_json

COMMANDS = ("init", "history", "score", "submit", "next_round", "status", "diagnostics")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CONSTRAINT_SEMANTICS = {
    "version": "gb1-mutable-site-constraints/1",
    "referenceVariant": "VDGV",
    "fixedPositions": "Residues outside allowed_positions must equal the experimental reference.",
    "mutablePositions": "At every allowed position the residue must belong to allowed_amino_acids.",
    "mutationCount": "min_mutations and max_mutations count differences from the reference.",
    "diagnosticBackground": "The explicitly specified background is a prediction control; candidate constraints apply separately."
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _keys(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Unexpected request fields")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(f"{name} must be a nonempty string of at most 4000 characters")
    return value


@contextmanager
def run_lock(directory):
    """OS-held lock: process termination releases ownership without stale locks."""
    path = Path(directory) / ".research.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Run is busy; retry the same operationId") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _budget(state):
    config = state["science"]["config"]
    used = len(state["scores"])
    assays = sum(len(r["recommendations"]) for r in state["rounds"])
    planned = config["rounds"] * config["k"]
    cap = min(config["max_assays"], planned) if config["max_assays"] is not None else planned
    return {"score_limit": config["score_budget"], "scores_used": used,
            "scores_remaining": config["score_budget"] - used,
            "assay_limit": cap, "assays_used": assays, "assays_remaining": cap - assays,
            "diagnostic_limit": state["science"].get("diagnostic_budget", 77),
            "diagnostic_used": len(state.get("diagnostic_cache", {})),
            "diagnostic_remaining": state["science"].get("diagnostic_budget", 77) - len(state.get("diagnostic_cache", {}))}


def _initial_measurements(state):
    frozen = state["science"].get("initial_measurements")
    if frozen is not None:
        return frozen
    selected = {row["variant"] for record in state["rounds"] for row in record["recommendations"]}
    appended = sum(len(record["recommendations"]) for record in state["rounds"])
    count = len(state["known"]) - appended
    declared = state["science"].get("dataset_manifest", {}).get("counts", {}).get("train")
    if count < 1 or (declared is not None and declared != count):
        return {"count": None, "best_fitness": None}
    initial = state["known"][:count]
    if any(row["variant"] in selected for row in initial):
        return {"count": None, "best_fitness": None}
    return {"count": count, "best_fitness": max(float(row["fitness"]) for row in initial)}


def _status(state):
    budget = _budget(state)
    config = state["science"]["config"]
    initial = _initial_measurements(state)
    metrics = []
    for record in state["rounds"]:
        actual = [r["actual_fitness"] for r in record["recommendations"]]
        metrics.append({"round": record["round"], "meanFitness": sum(actual) / len(actual),
                        "maxFitness": max(actual), "assayCount": len(actual)})
    return {"runId": state["runId"], "status": state["status"], "round": state["round"],
            "phase": state["status"], "currentRound": state["round"], "roundLimit": config["rounds"],
            "scoredCount": budget["scores_used"], "scoreLimit": budget["score_limit"],
            "assayCount": budget["assays_used"], "assayLimit": budget["assay_limit"],
            "knownCount": len(state["known"]), "roundMetrics": metrics,
            "initialBest": initial["best_fitness"], "initialKnownCount": initial["count"],
            "recommendations": [r for record in state["rounds"] for r in record["recommendations"]],
            "eventCursor": state.get("revision", len(state["receipts"])),
            "research": {key: config[key] for key in ("rounds", "k", "score_budget", "allowed_positions", "allowed_amino_acids")},
            "reference": state["science"]["reference"], "scored": _ranked(state),
            "constraintSemantics": deepcopy(state["science"].get("constraint_semantics", _CONSTRAINT_SEMANTICS)),
            "diagnosticBudget": {"limit": budget["diagnostic_limit"], "used": budget["diagnostic_used"],
                                 "remaining": budget["diagnostic_remaining"], "scope": "per_round_unique_variants"},
            "known_count": len(state["known"]), "completed_rounds": len(state["rounds"]),
            "budget": budget, "pending_operationId": (state.get("pending") or {}).get("operationId"),
            "predictor": deepcopy(state["science"]["predictor"]),
            "statisticalDefinitionVersion": state["science"]["statistical_definitions"]["version"]}


def _ranked(state):
    return [{"rank": i, "variant": v, **state["scores"][v]}
            for i, v in enumerate(sorted(state["scores"], key=lambda v: (-state["scores"][v]["predicted_fitness"], v)), 1)]


def _page(args, name, default, maximum=None):
    value = args.get(name, default)
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}")
    return value


def _history(state, directory, args):
    _keys(args, ("offset", "limit", "candidate_offset", "candidate_limit"))
    offset, limit = _page(args, "offset", 0), _page(args, "limit", 100, 1000)
    co, cl = _page(args, "candidate_offset", 0), _page(args, "candidate_limit", 0, 1000)
    known = pd.DataFrame(state["known"])
    top = known.sort_values(["fitness", "variant"], ascending=[False, True]).head(15).to_dict("records")
    residues = []
    for i, position in enumerate(POSITIONS):
        frame = known.assign(residue=known.variant.str[i])
        for residue, rows in frame.groupby("residue", sort=True):
            residues.append({"position": position, "residue": residue, "n": len(rows),
                             "mean_fitness": float(rows.fitness.mean()), "max_fitness": float(rows.fitness.max())})
    candidates = _eligible(state, directory)
    return {**_status(state), "known": state["known"][offset:offset + limit], "offset": offset,
            "top_known": top, "residue_observations": residues, "previous_rounds": state["rounds"],
            "scored": _ranked(state), "eligible_count": len(candidates),
            "candidate_offset": co, "candidates": candidates[co:co + cl],
            "annotations": "Candidate sources and action purposes are public self-reports.",
            "statistics_note": "Residue summaries describe measured backgrounds; predictions are estimates."}


def _constraint_allowed(variant, config):
    """Apply the AA subset at mutable sites and the reference at fixed sites."""
    if not isinstance(variant, str) or len(variant) != 4 or set(variant) - set(AMINO_ACIDS):
        return False
    changes = 0
    for position, reference, residue in zip(POSITIONS, "VDGV", variant):
        if position in config.allowed_positions:
            if residue not in config.allowed_amino_acids:
                return False
        elif residue != reference:
            return False
        changes += residue != reference
    return config.min_mutations <= changes <= config.max_mutations


def _eligible(state, directory):
    measured = {r["variant"] for r in state["known"]}
    config = ExperimentConfig.model_validate(state["science"]["config"])
    return [v for v in pd.read_csv(directory / "candidates.csv", usecols=["variant"]).variant
            if v not in measured and _constraint_allowed(v, config)]


def _variants(args, state, directory):
    variants = args.get("sequences")
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) for v in variants):
        raise ValueError("sequences must be a nonempty list of four-letter variants")
    if len(variants) != len(set(variants)):
        raise ValueError("sequences must be unique")
    if not set(variants) <= set(_eligible(state, directory)):
        raise ValueError("sequences must be eligible, unmeasured candidate variants")
    return variants


def _sources(args, n):
    sources = args.get("sources", ["unspecified"] * n)
    if not isinstance(sources, list) or len(sources) != n:
        raise ValueError("sources must have one public self-reported marker per sequence")
    return [_text(v, "source") for v in sources]


def _predict(state, variants, allow_missing=False):
    science = state["science"]
    config = ExperimentConfig.model_validate(science["config"])
    known = pd.DataFrame(state["training_known"])
    spec = science["predictor"]
    requested = list(variants)
    if spec["kind"] in ("rf_onehot", "mlp_onehot"):
        if spec["kind"] == "rf_onehot":
            model = fit_model(spec["spec"], config, known)
        else:
            model = MLPRegressor(**spec["parameters"])
            model.fit(one_hot(known.variant), known.fitness.to_numpy(dtype=float))
        query_x = one_hot(variants)
    else:
        ids = pd.read_csv(spec["variants_path"], usecols=["variant"]).variant.tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Embedding variant IDs must be unique")
        features = np.load(spec["embeddings_path"], mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape[0] != len(ids):
            raise ValueError("Embedding rows must match the declared variant IDs")
        lookup = dict(zip(ids, range(len(ids))))
        if allow_missing:
            variants = [v for v in variants if v in lookup]
        elif any(v not in lookup for v in variants):
            raise ValueError("A requested variant has no frozen embedding")
        train_x = np.asarray(features[[lookup[v] for v in known.variant]])
        query_x = np.asarray(features[[lookup[v] for v in variants]])
        if not np.isfinite(train_x).all() or not np.isfinite(query_x).all():
            raise ValueError("Embeddings must be finite")
        model = Ridge(alpha=spec["alpha"])
        model.fit(train_x, known.fitness.to_numpy(dtype=float))
    predicted = model.predict(query_x) if len(variants) else []
    scores = [float(v) for v in predicted]
    if len(scores) != len(variants) or not all(math.isfinite(v) for v in scores):
        raise ValueError("Predictor must return one finite score per variant")
    trees = getattr(model, "estimators_", None)
    sd = np.std([tree.predict(query_x) for tree in trees], axis=0, ddof=1) if trees and len(trees) > 1 else None
    records = {v: {"predicted_fitness": value,
                   "model_disagreement_sd": float(sd[i]) if sd is not None else None,
                   "uncertainty_kind": "tree_model_disagreement" if sd is not None else "not_available",
                   "confidence_interval": None, "prediction_status": "available"}
               for i, (v, value) in enumerate(zip(variants, scores))}
    unavailable = {"predicted_fitness": None, "model_disagreement_sd": None,
                   "uncertainty_kind": "not_available", "confidence_interval": None,
                   "prediction_status": "embedding_unavailable"}
    return [records.get(v, dict(unavailable)) for v in requested]


def _initialize(args, directory, run_id):
    _keys(args, ("dataset_dir", "selection_path", "predictor", "research"))
    dataset = PublicDataset(_text(args.get("dataset_dir"), "dataset_dir"))
    original_config = ExperimentConfig.model_validate(dataset.manifest["config"])
    selection = load_selection(_text(args.get("selection_path"), "selection_path"), dataset, original_config)
    research = args.get("research", {})
    _keys(research, ("rounds", "k", "score_budget", "allowed_positions", "allowed_amino_acids", "reference", "diagnostic_budget"))
    reference = validate_reference(research.get("reference", dataset.reference))
    if reference != dataset.reference:
        raise ValueError("The explicit reference must match the frozen GB1 experimental reference")
    diagnostic_budget = research.get("diagnostic_budget", 77)
    if type(diagnostic_budget) is not int or not 0 <= diagnostic_budget <= 10000:
        raise ValueError("diagnostic_budget must be an integer from 0 through 10000 per round")
    derived = original_config.model_dump()
    derived.update({k: v for k, v in research.items() if k not in ("reference", "diagnostic_budget")})
    derived["proposal_count"] = derived["score_budget"]
    if type(derived["k"]) is not int or not 1 <= derived["k"] <= 40:
        raise ValueError("k must be an integer from 1 through 40")
    if type(derived["score_budget"]) is not int or not 1 <= derived["score_budget"] <= 10000:
        raise ValueError("score_budget must be an integer from 1 through 10000")
    config = ExperimentConfig.model_validate(derived)
    if not config.allowed_positions:
        raise ValueError("At least one of the four GB1 positions must be allowed")
    measured = set(dataset.train.variant)
    if not any(v not in measured and _constraint_allowed(v, config) for v in dataset.candidates.variant):
        raise ValueError("No eligible unmeasured candidates satisfy the configured mutable positions, amino-acid subset and mutation limits")
    predictor = args.get("predictor", {"kind": "rf_onehot"})
    if not isinstance(predictor, dict):
        raise ValueError("predictor must be an object")
    if predictor.get("kind") == "rf_onehot":
        _keys(predictor, ("kind",))
        if selection["spec"]["family"] != "random_forest":
            raise ValueError("rf_onehot requires a frozen random_forest selection")
        predictor = {"kind": "rf_onehot", "spec": selection["spec"]}
    elif predictor.get("kind") == "mlp_onehot":
        _keys(predictor, ("kind",))
        predictor = {"kind": "mlp_onehot", "parameters": {"hidden_layer_sizes": [64], "activation": "relu",
                     "solver": "adam", "alpha": 0.0001, "max_iter": 500, "early_stopping": False,
                     "random_state": config.model_seed if config.model_seed is not None else config.seed},
                     "source": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"}
    elif predictor.get("kind") == "esm2_ridge":
        _keys(predictor, ("kind", "alpha", "variants_path", "embeddings_path"))
        alpha = predictor.get("alpha")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        predictor = {**predictor, "alpha": float(alpha)}
        for key in ("variants_path", "embeddings_path"):
            predictor[key] = str(Path(_text(predictor.get(key), key)).resolve(strict=True))
    else:
        raise ValueError("Unsupported predictor kind")
    known = dataset.train.to_dict("records")
    if not known or not all(math.isfinite(float(r["fitness"])) and r["fitness"] >= 0 for r in known):
        raise ValueError("Known measurements must be finite and nonnegative")
    definitions = read_json(Path(__file__).resolve().parents[1] / "configs" / "statistical-definitions-v2.json")
    science = {"config": config.model_dump(), "predictor": predictor, "diagnostic_budget": diagnostic_budget,
               "original_config": original_config.model_dump(), "constraint_semantics": deepcopy(_CONSTRAINT_SEMANTICS),
               "initial_measurements": {"count": len(known), "best_fitness": max(float(r["fitness"]) for r in known)},
               "dataset_dir": str(dataset.directory.resolve()), "dataset_manifest": dataset.manifest,
               "reference": dataset.reference, "statistical_definitions": definitions,
               "training_policy": "refit on initial known and earlier completed rounds only"}
    dataset.candidates.to_csv(directory / "candidates.csv", index=False)
    write_json(directory / "science.json", science)
    return {"format_version": "gb1-research-session/1", "runId": run_id, "created_at": _now(),
            "science": science, "known": known, "training_known": deepcopy(known), "round": 1,
            "status": "active", "scores": {}, "rounds": [], "pending": None, "receipts": {},
            "revision": 1, "diagnostic_cache": {}, "receipt_states": {}}


def _perform(command, args, state, directory, operation_id):
    config = ExperimentConfig.model_validate(state["science"]["config"])
    if command == "status":
        _keys(args, ())
        return _status(state)
    if command == "history":
        return _history(state, directory, args)
    if command == "diagnostics":
        _keys(args, ("kind", "background"))
        kind = args.get("kind")
        if kind not in research_diagnostics.KINDS:
            raise ValueError("Unknown diagnostic kind")
        if kind != "heatmap" and "background" in args:
            raise ValueError("background is supported only for the heatmap")
        if kind == "heatmap":
            data = research_diagnostics.heatmap(state, args.get("background", "VDGV"),
                                               lambda s, variants: _predict(s, variants, allow_missing=True))
            eligible = set(_eligible(state, directory))
            for cell in data["cells"]:
                cell["constraintAllowed"] = _constraint_allowed(cell["variant"], config)
                cell["eligible"] = cell["variant"] in eligible
            data["backgroundConstraintAllowed"] = _constraint_allowed(data["background"], config)
            data["constraintSemantics"] = deepcopy(state["science"].get("constraint_semantics", _CONSTRAINT_SEMANTICS))
        elif kind == "structure":
            data = research_diagnostics.structure(state["science"]["reference"], config.allowed_positions)
        else:
            data = getattr(research_diagnostics, kind)(state)
        return {"kind": kind, "data": data, "budget": _budget(state), "basisRound": state["round"],
                "basisKnownCount": len(state["known"]), "basisTrainingKnownCount": len(state["training_known"])}
    if state["pending"] and state["pending"]["operationId"] != operation_id:
        raise ValueError("Resume the pending submission with its original operationId first")
    if command == "next_round":
        _keys(args, ("purpose",))
        if "purpose" in args:
            _text(args["purpose"], "purpose")
        if not _eligible(state, directory):
            raise ValueError("No eligible unmeasured candidates remain under the frozen candidate constraints")
        if state["status"] != "round_submitted":
            raise ValueError("next_round requires a completed submission and remaining rounds")
        state.update(round=state["round"] + 1, status="active", scores={}, diagnostic_cache={},
                     training_known=deepcopy(state["known"]))
        return _status(state)
    if state["status"] not in ("active", "submission_pending"):
        raise ValueError("This round is already submitted")
    if "purpose" in args:
        _text(args["purpose"], "purpose")
    if command == "score":
        _keys(args, ("sequences", "sources", "purpose"))
        variants = _variants(args, state, directory)
        sources = _sources(args, len(variants))
        fresh = [v for v in variants if v not in state["scores"]]
        if len(fresh) > _budget(state)["scores_remaining"]:
            raise ValueError("Request exceeds remaining score budget; no candidates were scored")
        if fresh:
            values = _predict(state, fresh)
            source_of = dict(zip(variants, sources))
            state["scores"].update({v: {**p, "source": source_of[v],
                                         "source_kind": "public_self_report"} for v, p in zip(fresh, values)})
        return {"scored": [{"variant": v, **state["scores"][v]} for v in variants], "budget": _budget(state)}
    if command != "submit":
        raise ValueError("Unknown command")
    _keys(args, ("sequences", "reasons", "sources", "purpose"))
    variants = _variants(args, state, directory)
    source_args = dict(args)
    if "sources" not in source_args:
        source_args["sources"] = [state["scores"].get(v, {}).get("source", "unspecified") for v in variants]
    sources = _sources(source_args, len(variants))
    required = min(config.k, len(_eligible(state, directory)), _budget(state)["assays_remaining"])
    if len(variants) != required or required < 1:
        raise ValueError(f"Submit exactly {required} candidates")
    if not set(variants) <= set(state["scores"]):
        raise ValueError("Submit only candidates already scored this round")
    reasons = args.get("reasons")
    if not isinstance(reasons, list) or len(reasons) != len(variants):
        raise ValueError("Provide one reason per submitted candidate")
    reasons = [_text(r, "reason") for r in reasons]
    state["pending"] = {"operationId": operation_id, "sequences": variants, "reasons": reasons,
                        "sources": sources, "selected_at": (state.get("pending") or {}).get("selected_at", _now())}
    state["status"] = "submission_pending"
    write_json(directory / "state.json", state)
    # Only this branch has access to measured candidate feedback.
    feedback = HiddenOracle(state["science"]["dataset_dir"], variants).evaluate(variants)
    measured = {r["variant"]: r["actual_fitness"] for r in feedback}
    if set(measured) != set(variants) or len(feedback) != len(variants) or any(
            not math.isfinite(float(v)) or v < 0 for v in measured.values()):
        raise ValueError("Oracle returned invalid selected measurements")
    rows = [{"round": state["round"], "rank": i, **describe_variant(v, state["science"]["reference"]), "reason": reason,
             "source": source, "source_kind": "public_self_report",
             "predicted_fitness": state["scores"][v]["predicted_fitness"],
             "model_disagreement_sd": state["scores"][v].get("model_disagreement_sd"),
             "confidence_interval": None, "actual_fitness": float(measured[v])}
            for i, (v, reason, source) in enumerate(zip(variants, reasons, sources), 1)]
    record = {"round": state["round"], "selection": deepcopy(state["pending"]), "recommendations": rows}
    state["rounds"].append(record)
    state["known"].extend({**describe_variant(v, state["science"]["reference"]), "fitness": float(measured[v])}
                          for v in variants)
    state["pending"] = None
    state["status"] = ("completed" if state["round"] >= config.rounds or _budget(state)["assays_remaining"] == 0
                       or not _eligible(state, directory) else "round_submitted")
    return {"submission": record, **_status(state)}


def execute_request(request):
    """Execute one JSON request; return a JSON-compatible success/error envelope.

    init arguments: dataset_dir, selection_path, optional predictor (rf_onehot,
    mlp_onehot, or esm2_ridge with alpha/variants_path/embeddings_path), and research
    (rounds/k/score_budget/allowed_positions/allowed_amino_acids/reference/
    diagnostic_budget). Original data and selection are validated before these
    permitted settings are derived and frozen. history supports offset/limit and
    candidate_offset/candidate_limit. score accepts sequences and optional
    sources/purpose; submit adds reasons. status takes no arguments; next_round
    accepts optional purpose. diagnostics accepts kind and heatmap-only background.
    status/history and non-heatmap diagnostics are live read-only snapshots, with
    no durable operation receipts. Scientific mutations have exact-request
    idempotency. eventCursor is the persisted state revision; a pending submission
    is a persisted state as well. All commands require IDs.
    The host must choose root and initial scientific paths; this API is not a
    filesystem security boundary for an untrusted Python process.
    """
    try:
        _keys(request, ("command", "operationId", "runId", "root", "arguments"))
        command = request.get("command")
        if command not in COMMANDS:
            raise ValueError("Unknown command")
        for key in ("operationId", "runId"):
            if not isinstance(request.get(key), str) or not _ID.fullmatch(request[key]):
                raise ValueError(f"{key} must be a safe identifier (1-128 characters)")
        args = request.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        # Reject non-JSON and nonfinite inputs before persistence.
        canonical = json.loads(json.dumps(request, allow_nan=False))
        root = Path(_text(request.get("root"), "root")).resolve()
        directory = root / request["runId"]
        if directory.is_symlink():
            raise ValueError("Run directory must not be a symbolic link")
        if command == "init":
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise ValueError("Unknown runId")
        with run_lock(directory):
            return _execute_locked(canonical, args, directory)
    except Exception as exc:
        return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)},
                "operationId": request.get("operationId") if isinstance(request, dict) else None}


def _execute_locked(request, args, directory):
    command, operation_id = request["command"], request["operationId"]
    state_path = directory / "state.json"
    state = read_json(state_path) if state_path.exists() else None
    op_path = directory / "operations" / f"{operation_id}.json"
    previous = read_json(op_path) if op_path.exists() else None
    if previous and previous["request"] != request:
        raise ValueError("operationId was already used with a different request")
    if state and read_json(directory / "science.json") != state["science"]:
        raise ValueError("Scientific configuration differs from the frozen snapshot")
    read_only = command in ("status", "history") or (command == "diagnostics" and args.get("kind") != "heatmap")
    if read_only:
        if state is None:
            raise ValueError("Run is not initialized")
        data = _perform(command, args, state, directory, operation_id)
        data.update(readOnly=True, eventCursor=state.get("revision", len(state["receipts"])))
        return {"ok": True, "operationId": operation_id, "result": data}
    if state and operation_id in state["receipts"]:
        receipt = state["receipts"][operation_id]
        if not previous:
            raise ValueError("Operation input record is missing")
        if previous.get("status") != "finished":
            previous.update(status="finished", result=receipt,
                            after=state.get("receipt_states", {}).get(operation_id, deepcopy(receipt["result"])))
            write_json(op_path, previous)
        return receipt
    if previous and previous.get("status") == "finished":
        return previous["result"]
    before = _status(state) if state else None
    record = previous or {"request": request, "created_at": _now(), "before": before, "attempts": []}
    record.update(status="running")
    record["attempts"].append({"started_at": _now()})
    write_json(op_path, record)
    try:
        if command == "init":
            if state is not None:
                raise ValueError("runId already exists; use a unique runId for a new run")
            state = _initialize(args, directory, request["runId"])
            data = _status(state)
        else:
            if state is None:
                raise ValueError("Run is not initialized")
            state["revision"] = state.get("revision", len(state["receipts"])) + 1
            data = _perform(command, args, state, directory, operation_id)
        data["eventCursor"] = state["revision"]
        result = {"ok": True, "operationId": operation_id, "result": data}
        state["receipts"][operation_id] = result
        after = _status(state)
        state.setdefault("receipt_states", {})[operation_id] = after
        write_json(state_path, state)
        record.update(status="finished", result=result, after=after)
    except Exception as exc:
        # Reload the durable pre-query state if submission has an unknown result.
        durable = read_json(state_path) if state_path.exists() else None
        retryable = bool(durable and durable.get("pending") and durable["pending"]["operationId"] == operation_id)
        result = {"ok": False, "operationId": operation_id,
                  "error": {"type": type(exc).__name__, "message": str(exc)}, "retryable": retryable}
        record.update(status="retryable" if retryable else "finished", result=result,
                      after=_status(durable) if durable else None)
    record["attempts"][-1].update(finished_at=_now(), ok=result["ok"], error=result.get("error"))
    write_json(op_path, record)
    return result
