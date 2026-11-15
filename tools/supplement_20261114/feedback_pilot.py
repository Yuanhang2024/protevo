"""Fixed-state, single-step feedback pilot with paired selector replay."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import sklearn
from protevo.agent import OpenAIProposer, build_request_payload, collect_proposals, load_knowledge, make_context
from protevo.data import HiddenOracle, PublicDataset, one_hot
from protevo.models import fit_model, load_selection
from protevo.prompts import load_prompt_profile, repair_context
from protevo.schema import ExperimentConfig, variant_allowed
from protevo.storage import event, read_json, single_writer, write_json

FORMAT = 1
CONTEXTS = ("C0", "C1")
SELECTORS = ("P0", "P1")
STATES = (1201, 1202, 1203, 1204)
RESPONSE_FIELDS = ("text", "complete", "usage", "response_id", "request_id", "model_requested", "model_returned", "api_mode", "completion_status", "prompt_metadata")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def safe_error(error):
    status = getattr(error, "status_code", None)
    return {"type": type(error).__name__, "status_code": status if type(status) is int and 100 <= status <= 599 else None}


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pilot_config(seed, knowledge_path=None, **changes):
    values = dict(seed=seed, rounds=3, k=10, proposal_count=40, score_budget=40,
                  oversize_policy="first_n", require_full_batch=True, llm_max_calls_per_round=2,
                  llm_max_output_tokens=None, llm_timeout=None, save_predictors=False,
                  llm_reasoning_effort="xhigh", knowledge_path=str(knowledge_path or PROJECT / "configs/knowledge.json"))
    values.update(changes)
    config = ExperimentConfig(**values)
    if (config.k, config.proposal_count, config.score_budget, config.llm_max_calls_per_round) != (10, 40, 40, 2):
        raise ValueError("Pilot requires C40/k10 and at most two attempts")
    if config.oversize_policy != "first_n" or not config.require_full_batch:
        raise ValueError("Pilot requires ordered-prefix validation and a full batch")
    return config


def _identity(model_requested, api_mode):
    if not isinstance(model_requested, str) or not model_requested.strip() or len(model_requested) > 200 or any(ord(x) < 32 for x in model_requested):
        raise ValueError("A declared model ID is required")
    if api_mode not in ("chat", "responses"):
        raise ValueError("Unsupported API mode")
    return {"model_requested": model_requested, "api_mode": api_mode}


def _corrected(path):
    report = read_json(path)
    if report.get("status") != "passed":
        raise ValueError("Statistical correction status must be passed before pilot preparation")


def _scores(model, variants):
    values = np.asarray(model.predict(one_hot(variants)), dtype=float)
    if values.shape != (len(variants),) or not np.isfinite(values).all():
        raise ValueError("Invalid predictor output")
    return pd.DataFrame({"variant": list(variants), "predicted_fitness": values})


def _top(scores, k=10):
    return scores.sort_values(["predicted_fitness", "variant"], ascending=[False, True]).head(k).reset_index(drop=True)


def _state_prepare(destination, source, oracle_factory=HiddenOracle):
    destination = Path(destination)
    dataset = PublicDataset(source["data"])
    selection = read_json(source["selection"])
    config = ExperimentConfig.model_validate(selection["config"])
    load_selection(source["selection"], dataset, config)
    if (config.k, config.proposal_count, config.score_budget) != (10, 40, 40):
        raise ValueError("Prepared pilot states must use C40/k10")
    source_description = {"state_id": source["state_id"], "data": str(dataset.directory.resolve()),
                          "selection": str(Path(source["selection"]).resolve()), "config": config.model_dump(),
                          "dataset_manifest": dataset.manifest, "spec": selection["spec"]}
    if destination.exists():
        saved = read_json(destination / "prepared.json")
        if saved["source"] != source_description or saved["status"] != "ready":
            raise ValueError("Existing state differs from the declared source")
        return saved
    destination.mkdir(parents=True)
    known = dataset.train.copy()
    eligible = dataset.candidates[dataset.candidates.variant.map(lambda value: variant_allowed(value, config))].copy()
    if len(eligible) < config.k + config.proposal_count:
        raise ValueError("State has insufficient candidates after the common prefix")
    known.to_csv(destination / "initial-observations.csv", index=False)
    new_json(destination / "P0-training-ids.json", list(known.variant))
    p0 = fit_model(selection["spec"], config, known)
    start = time.perf_counter()
    scores = _scores(p0, list(eligible.variant))
    scores.to_csv(destination / "prefix-all-scores.csv.gz", index=False, compression="gzip")
    selected = _top(scores, config.k)
    selected["hypothesis"] = "Common prefix selected by the initial predictor from the full eligible pool."
    new_json(destination / "prefix-selected.json", selected.to_dict(orient="records"))
    event(destination / "events.jsonl", "prefix_selected", variants=list(selected.variant), scored_count=len(scores))
    measured = oracle_factory(dataset.directory, set(selected.variant)).evaluate(list(selected.variant))
    event(destination / "events.jsonl", "prefix_feedback", variants=[row["variant"] for row in measured])
    prefix = selected.merge(pd.DataFrame(measured), on="variant", validate="one_to_one")
    prefix["residual"] = prefix.actual_fitness - prefix.predicted_fitness
    descriptions = dataset.candidates.set_index("variant")
    for key in ("sequence", "mutations", "mutation_count"):
        prefix[key] = prefix.variant.map(descriptions[key])
    prefix.to_csv(destination / "prefix-observations.csv", index=False)
    updated = pd.concat([known, prefix.rename(columns={"actual_fitness": "fitness"})[list(known.columns)]], ignore_index=True)
    fit_model(selection["spec"], config, updated)
    excluded = sorted(set(updated.variant))
    remaining = eligible[~eligible.variant.isin(set(prefix.variant))].copy()
    remaining.to_csv(destination / "eligible.csv", index=False)
    new_json(destination / "P1-training-ids.json", list(updated.variant))
    new_json(destination / "excluded-ids.json", excluded)
    saved = {"format_version": FORMAT, "status": "ready", "source": source_description, "reference": dataset.reference,
             "initial_count": len(known), "prefix_count": len(prefix), "updated_count": len(updated), "eligible_count": len(remaining),
             "prefix_score_count": len(scores), "prefix_seconds": time.perf_counter() - start,
             "common_prefix_assays": len(prefix), "created_at": stamp(),
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    new_json(destination / "prepared.json", saved)
    return saved


def _load_state(directory):
    directory = Path(directory)
    saved = read_json(directory / "prepared.json")
    initial = pd.read_csv(directory / "initial-observations.csv")
    prefix = pd.read_csv(directory / "prefix-observations.csv")
    updated = pd.concat([initial, prefix.rename(columns={"actual_fitness": "fitness"})[list(initial.columns)]], ignore_index=True)
    excluded = read_json(directory / "excluded-ids.json")
    eligible = pd.read_csv(directory / "eligible.csv")
    if excluded != sorted(set(updated.variant)) or set(eligible.variant) & set(excluded):
        raise ValueError("Prepared exclusion list differs from state observations")
    if list(initial.variant) != read_json(directory / "P0-training-ids.json") or list(updated.variant) != read_json(directory / "P1-training-ids.json"):
        raise ValueError("Prepared training IDs differ from saved observations")
    return saved, initial, prefix, updated, eligible, excluded


def contexts_for_state(directory, config=None):
    saved, initial, prefix, updated, eligible, excluded = _load_state(directory)
    config = config or ExperimentConfig.model_validate(saved["source"]["config"])
    knowledge = load_knowledge(config.knowledge_path)
    feedback = prefix[["variant", "actual_fitness", "predicted_fitness", "residual", "hypothesis"]].to_dict(orient="records")
    contexts = {}
    for condition, known, recent in (("C0", initial, ()), ("C1", updated, feedback)):
        context = make_context(known, [], config, saved["reference"], 2, knowledge, "v3", recent)
        context.update(observed_variants=excluded.copy(), forbidden_variants=excluded.copy(), observed_variants_complete=True,
                       observed_variants_scope=f"All {len(excluded)} measured IDs are excluded in both conditions; some measurement values are not displayed.")
        contexts[condition] = context
    return contexts


def prepare(output, state_sources, *, model_requested, api_mode="chat", repeats=3, order_seed=20261114,
            correction_status, oracle_factory=HiddenOracle):
    _corrected(correction_status)
    _identity(model_requested, api_mode)
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("Pilot repetitions must be declared between one and three")
    sources = [{"state_id": row["state_id"], "data": str(Path(row["data"]).resolve()), "selection": str(Path(row["selection"]).resolve())} for row in state_sources]
    if not sources or len(sources) > 4 or len({row["state_id"] for row in sources}) != len(sources):
        raise ValueError("Declare one to four unique fixed states")
    if any(not isinstance(row["state_id"], str) or not row["state_id"].replace("-", "").isalnum() for row in sources):
        raise ValueError("Invalid state identity")
    state_seeds = [ExperimentConfig.model_validate(read_json(row["selection"])["config"]).seed for row in sources]
    if len(set(state_seeds)) != len(state_seeds) or len({row["data"] for row in sources}) != len(sources):
        raise ValueError("Each declared state requires a distinct split seed and data directory")
    if type(order_seed) is not int or not 0 <= order_seed <= 2**32 - 1:
        raise ValueError("Order seed must be a declared nonnegative 32-bit integer")
    declaration = {"sources": sources, "state_seeds": state_seeds, **_identity(model_requested, api_mode), "repeats": repeats, "order_seed": order_seed,
                   "correction_status": str(Path(correction_status).resolve()),
                   "C": 40, "k": 10, "round": 2, "contexts": list(CONTEXTS), "selectors": list(SELECTORS), "max_attempts_per_arm": 2}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        previous = read_json(output / "plan.json")
        if previous["declaration"] != declaration:
            raise ValueError("Existing output has a different frozen pilot plan")
        if not (output / "run-state.json").exists():
            raise ValueError("Incomplete preparation requires a new output directory")
        for source in sources:
            directory = output / "states" / source["state_id"]
            _state_prepare(directory, source, oracle_factory)
            for condition, context in contexts_for_state(directory).items():
                if context != read_json(directory / f"{condition}-context.json"):
                    raise ValueError("Frozen context differs from the prepared scientific state")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    study_id = "feedback-pilot-" + uuid4().hex
    rng = np.random.default_rng(order_seed)
    episodes = []
    for source in sources:
        for repeat in range(repeats):
            repair_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            for condition in CONTEXTS:
                episode_id = f"{source['state_id']}-rep-{repeat}-{condition}"
                episodes.append({"episode_id": episode_id, "state_id": source["state_id"], "repeat": repeat,
                                 "context": condition, "design_seed": repair_seed})
    episodes = [episodes[int(index)] for index in rng.permutation(len(episodes))]
    plan = {"format_version": FORMAT, "study_id": study_id, "declaration": declaration, "episodes": episodes,
            "max_calls": 2 * len(episodes), "planned_generator_arms": len(episodes), "planned_selector_results": 2 * len(episodes),
            "prompt_metadata": load_prompt_profile("v3").metadata(), "created_at": stamp(),
            "interpretation": "Fixed-state, one-step feasibility pilot; selector replay is not an independent generation or a complete multi-round trajectory."}
    new_json(output / "plan.json", plan)
    for source in sources:
        _state_prepare(output / "states" / source["state_id"], source, oracle_factory)
        for condition, context in contexts_for_state(output / "states" / source["state_id"]).items():
            new_json(output / "states" / source["state_id"] / f"{condition}-context.json", context)
    state = {"format_version": FORMAT, "study_id": study_id, "status": "prepared", "calls": 0, "reported_known_total_tokens": 0,
             "missing_usage_count": 0, "episodes": {row["episode_id"]: {"status": "pending", "attempts": []} for row in episodes}}
    new_json(output / "run-state.json", state)
    return plan


def public_preflight(config, expected_model=None, expected_api_mode=None, proposer_factory=OpenAIProposer):
    proposer = None
    try:
        proposer = proposer_factory(config, prompt_profile="v3")
        result = _identity(proposer.model, proposer.api_mode)
        if expected_model is not None and result["model_requested"] != expected_model or expected_api_mode is not None and result["api_mode"] != expected_api_mode:
            return {"status": "failed", "code": "MODEL_IDENTITY_MISMATCH"}
        return {"status": "ready", **result, "generation_requests": 0}
    except Exception as error:
        return {"status": "failed", "code": "CONNECTION_UNAVAILABLE", "error": safe_error(error)}
    finally:
        if proposer is not None:
            try:
                proposer.close()
            except Exception:
                pass


def _usage(raw):
    def count(value):
        return value if type(value) is int and value >= 0 else None
    raw = raw if isinstance(raw, dict) else {}
    incoming = count(raw.get("input_tokens", raw.get("prompt_tokens")))
    outgoing = count(raw.get("output_tokens", raw.get("completion_tokens")))
    total = count(raw.get("total_tokens"))
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": total,
            "missing_usage_count": int(incoming is None or outgoing is None or total is None)}


def _public_response(response):
    if not isinstance(response, dict):
        raise ValueError("Response is not a public object")
    value = {key: deepcopy(response[key]) for key in RESPONSE_FIELDS if key in response}
    if not isinstance(value.get("text"), str) or type(value.get("complete")) is not bool:
        raise ValueError("Invalid public response fields")
    json.dumps(value, allow_nan=False)
    return value


def _replay(output, episode, prepared, proposals, oracle_factory):
    state_directory = Path(output) / "states" / episode["state_id"]
    saved, initial, prefix, updated, eligible, excluded = _load_state(state_directory)
    config = ExperimentConfig.model_validate(saved["source"]["config"])
    variants = [row["variant"] for row in proposals]
    if len(variants) != 40 or len(set(variants)) != 40 or not set(variants) <= set(eligible.variant):
        raise ValueError("Replay requires one complete, eligible batch of forty")
    results = {}
    for selector, training in (("P0", initial), ("P1", updated)):
        directory = Path(output) / "episodes" / episode["episode_id"] / selector
        directory.mkdir(parents=True, exist_ok=True)
        new_json(directory / "training-ids.json", list(training.variant))
        new_json(directory / "model-spec.json", {"spec": saved["source"]["spec"], "config": config.model_dump()})
        start = time.perf_counter()
        model = fit_model(saved["source"]["spec"], config, training)
        scores = _scores(model, variants)
        scores.to_csv(directory / "all-scores.csv", index=False)
        selected = _top(scores)
        selected = selected.merge(pd.DataFrame(proposals), on="variant", validate="one_to_one")
        new_json(directory / "selected.json", selected.to_dict(orient="records"))
        event(directory / "events.jsonl", "selected", variants=list(selected.variant), scored_count=40)
        measured = oracle_factory(saved["source"]["data"], set(selected.variant)).evaluate(list(selected.variant))
        event(directory / "events.jsonl", "feedback", variants=[row["variant"] for row in measured])
        recommendations = selected.merge(pd.DataFrame(measured), on="variant", validate="one_to_one")
        recommendations["rank"] = range(1, len(recommendations) + 1)
        descriptions = eligible.set_index("variant")
        for key in ("sequence", "mutations", "mutation_count"):
            recommendations[key] = recommendations.variant.map(descriptions[key])
        recommendations.to_csv(directory / "recommendations.csv", index=False)
        results[selector] = {"selected_ids": list(recommendations.variant), "mean_fitness": float(recommendations.actual_fitness.mean()),
                             "max_fitness": float(recommendations.actual_fitness.max()), "training_count": len(training),
                             "scored_count": 40, "assays": 10, "seconds": time.perf_counter() - start}
        new_json(directory / "result.json", results[selector])
    results["selection_overlap"] = len(set(results["P0"]["selected_ids"]) & set(results["P1"]["selected_ids"]))
    new_json(Path(output) / "episodes" / episode["episode_id"] / "selector-comparison.json", results)
    return results


def run(output, proposer=None, *, allow_api=False, max_arms=None, oracle_factory=HiddenOracle):
    output = Path(output)
    plan = read_json(output / "plan.json")
    _corrected(plan["declaration"]["correction_status"])
    if max_arms is not None and (type(max_arms) is not int or max_arms < 1):
        raise ValueError("Maximum dispatched arms must be a positive integer")
    if isinstance(OpenAIProposer, type) and isinstance(proposer, OpenAIProposer) and not allow_api:
        raise ValueError("Real generation requires explicit run authorization")
    if plan["prompt_metadata"] != load_prompt_profile("v3").metadata():
        raise ValueError("The frozen scientific prompt has changed")
    created = False
    with single_writer(output):
        state = read_json(output / "run-state.json")
        if state["study_id"] != plan["study_id"]:
            raise ValueError("Pilot operation identity differs")
        for episode in state["episodes"].values():
            if episode["status"] == "running":
                unknown = any(attempt["status"] == "started" for attempt in episode["attempts"])
                episode["status"] = "unknown" if unknown else "interrupted"
                episode["error_code"] = "UNKNOWN_REQUEST_OUTCOME" if unknown else "INTERRUPTED_ARM"
                if unknown:
                    state["missing_usage_count"] += 1
        write_json(output / "run-state.json", state)
        if state["missing_usage_count"]:
            state["status"] = "stopped_missing_usage"
            write_json(output / "run-state.json", state)
            return summarize(output)
        pending = [row for row in plan["episodes"] if state["episodes"][row["episode_id"]]["status"] == "pending"]
        if max_arms is not None:
            pending = pending[:max_arms]
        if not pending:
            state["status"] = "completed" if all(row["status"] == "completed" for row in state["episodes"].values()) else "finished_with_failures"
            write_json(output / "run-state.json", state)
            return summarize(output)
        for state_id in {row["state_id"] for row in pending}:
            directory = output / "states" / state_id
            for condition, context in contexts_for_state(directory).items():
                if context != read_json(directory / f"{condition}-context.json"):
                    raise ValueError("Frozen context differs from the prepared scientific state")
        base = read_json(output / "states" / pending[0]["state_id"] / "prepared.json")
        initial_config = ExperimentConfig.model_validate(base["source"]["config"])
        if proposer is None:
            if not allow_api:
                raise ValueError("Real generation requires explicit run authorization")
            proposer = OpenAIProposer(initial_config, prompt_profile="v3")
            created = True
        try:
            identity = _identity(proposer.model, proposer.api_mode)
            if identity != {key: plan["declaration"][key] for key in ("model_requested", "api_mode")}:
                raise ValueError("Proposer differs from the predeclared model route")
            for episode in pending:
                if state["missing_usage_count"] or state["calls"] >= plan["max_calls"]:
                    break
                directory = output / "episodes" / episode["episode_id"]
                directory.mkdir(parents=True, exist_ok=True)
                saved, initial, prefix, updated, eligible, excluded = _load_state(output / "states" / episode["state_id"])
                source_config = ExperimentConfig.model_validate(saved["source"]["config"])
                config = source_config.model_copy(update={"design_seed": episode["design_seed"], "require_full_batch": True,
                                                          "oversize_policy": "first_n", "llm_max_calls_per_round": 2})
                proposer.config = config
                base_context = deepcopy(read_json(output / "states" / episode["state_id"] / f"{episode['context']}-context.json"))
                record = state["episodes"][episode["episode_id"]]
                record.update(status="running", started_at=stamp())
                write_json(output / "run-state.json", state)
                accepted, rejected, analyses = [], [], []
                for number in (1, 2):
                    if state["missing_usage_count"] or state["calls"] >= plan["max_calls"]:
                        break
                    context = base_context if number == 1 else repair_context(base_context, pd.DataFrame({"variant": excluded}), set(eligible.variant), accepted, rejected, config, 2, number)
                    operation_id = f"{plan['study_id']}-{episode['episode_id']}-attempt-{number}"
                    attempt_directory = directory / f"attempt-{number}"
                    payload = build_request_payload(context, config, proposer.model, proposer.api_mode, plan["prompt_metadata"]["system_prompt"])
                    request = {"operation_id": operation_id, "episode_id": episode["episode_id"], "call_number": state["calls"] + 1,
                               "attempt": number, "round": 2, "context": context, "payload": payload, "prompt_metadata": plan["prompt_metadata"],
                               "parameters": config.model_dump(), "created_at": stamp()}
                    new_json(attempt_directory / "request.json", request)
                    attempt = {"operation_id": operation_id, "status": "started", "started_at": stamp(), "call_number": state["calls"] + 1}
                    record["attempts"].append(attempt); state["calls"] += 1
                    write_json(output / "run-state.json", state)
                    event(directory / "events.jsonl", "dispatch", operation_id=operation_id, call_number=state["calls"])
                    start = time.perf_counter()
                    try:
                        response = _public_response(proposer.generate(deepcopy(context)))
                    except Exception as error:
                        attempt.update(status="unknown", seconds=time.perf_counter() - start, error=safe_error(error))
                        record.update(status="unknown", error_code="UNKNOWN_REQUEST_OUTCOME")
                        state["missing_usage_count"] += 1
                        new_json(attempt_directory / "error.json", {"code": "UNKNOWN_REQUEST_OUTCOME", "error": safe_error(error)})
                        write_json(output / "run-state.json", state)
                        break
                    new_json(attempt_directory / "response.json", response)
                    usage = _usage(response.get("usage"))
                    attempt.update(status="received", seconds=time.perf_counter() - start, ended_at=stamp(), usage=usage)
                    state["reported_known_total_tokens"] += usage["total_tokens"] or 0
                    state["missing_usage_count"] += usage["missing_usage_count"]
                    if response.get("model_requested") != proposer.model or response.get("api_mode") != proposer.api_mode:
                        record.update(status="failed", error_code="MODEL_IDENTITY_MISMATCH")
                        write_json(output / "run-state.json", state)
                        break
                    try:
                        if not response["complete"]:
                            raise ValueError("Incomplete public model response")
                        analysis, additions, rejected = collect_proposals(response["text"], set(eligible.variant), config, accepted,
                            [row["id"] for row in context.get("knowledge", {}).get("rules", [])], config.proposal_count - len(accepted))
                        accepted.extend(additions); analyses.append(analysis)
                        attempt["accepted_count"] = len(additions)
                    except ValueError as error:
                        rejected = [{"variant": None, "reason": type(error).__name__}]
                        attempt["validation_error"] = type(error).__name__
                    body = None
                    try:
                        body = json.loads(response["text"])
                    except (ValueError, TypeError):
                        pass
                    original_count = len(body["proposals"]) if isinstance(body, dict) and isinstance(body.get("proposals"), list) else None
                    requested_count = request["context"]["proposal_count"]
                    new_json(attempt_directory / "validation.json", {"accepted": accepted, "rejected": rejected, "analysis": analyses,
                        "requested_count": requested_count, "original_response_count": original_count,
                        "discarded_after_ordered_prefix": max(0, original_count - requested_count) if original_count is not None else None})
                    record["accepted_count"] = len(accepted)
                    write_json(output / "run-state.json", state)
                    if len(accepted) == 40:
                        break
                if record["status"] == "running":
                    if len(accepted) != 40:
                        record.update(status="failed", error_code="FULL_BATCH_REQUIRED")
                    else:
                        new_json(directory / "accepted-proposals.json", accepted)
                        record["selectors"] = _replay(output, episode, saved, accepted, oracle_factory)
                        record.update(status="completed", completed_at=stamp())
                    write_json(output / "run-state.json", state)
            state["status"] = "stopped_missing_usage" if state["missing_usage_count"] else "completed" if all(row["status"] == "completed" for row in state["episodes"].values()) else "finished_with_failures" if all(row["status"] != "pending" for row in state["episodes"].values()) else "partial"
            write_json(output / "run-state.json", state)
            return summarize(output)
        finally:
            if created:
                proposer.close()


def summarize(output):
    output = Path(output)
    plan, state = read_json(output / "plan.json"), read_json(output / "run-state.json")
    rows, state_effects = [], []
    for episode in plan["episodes"]:
        result = state["episodes"][episode["episode_id"]]
        if result["status"] == "completed":
            for selector in SELECTORS:
                rows.append({**episode, "selector": selector, **result["selectors"][selector]})
    for source in plan["declaration"]["sources"]:
        item = {"state_id": source["state_id"]}
        for selector in SELECTORS:
            for condition in CONTEXTS:
                values = [row["mean_fitness"] for row in rows if row["state_id"] == source["state_id"] and row["selector"] == selector and row["context"] == condition]
                item[f"{selector}_{condition}"] = {"completed": len(values), "planned": plan["declaration"]["repeats"], "mean_fitness": float(np.mean(values)) if values else None,
                                                   "quality_basis": "conditional_on_completion"}
            c0, c1 = item[f"{selector}_C0"]["mean_fitness"], item[f"{selector}_C1"]["mean_fitness"]
            item[f"{selector}_C1_minus_C0"] = None if c0 is None or c1 is None else c1 - c0
        p0, p1 = item["P0_C1_minus_C0"], item["P1_C1_minus_C0"]
        item["interaction"] = None if p0 is None or p1 is None else p1 - p0
        state_effects.append(item)
    effects = {}
    for key in ("P0_C1_minus_C0", "P1_C1_minus_C0", "interaction"):
        values = [row[key] for row in state_effects if row[key] is not None]
        required = SELECTORS if key == "interaction" else (key[:2],)
        complete = [row[key] for row in state_effects if row[key] is not None and all(
            row[f"{selector}_{condition}"]["completed"] == plan["declaration"]["repeats"]
            for selector in required for condition in CONTEXTS)]
        complete_mean = float(np.mean(complete)) if complete else None
        all_complete = len(complete) == len(state_effects)
        effects[key] = {"quality_basis": "conditional_on_completion", "state_equal_weight_mean": float(np.mean(values)) if values else None,
                        "states_with_both_conditions": len(values), "planned_states": len(state_effects),
                        "all_repetitions_complete_state_pairs": len(complete),
                        "all_repetitions_complete_state_equal_weight_mean": complete_mean,
                        "all_planned_repetitions_complete": all_complete,
                        "all_planned_state_equal_weight_mean": complete_mean if all_complete else None}
    summary = {"study_id": plan["study_id"], "status": state["status"], "quality_basis": "conditional_on_completion",
               "primary_effect": {"selector": "P0", "all_planned_state_equal_weight_mean": effects["P0_C1_minus_C0"]["all_planned_state_equal_weight_mean"],
                                  "all_repetitions_complete_state_pairs": effects["P0_C1_minus_C0"]["all_repetitions_complete_state_pairs"],
                                  "planned_states": len(state_effects), "complete": effects["P0_C1_minus_C0"]["all_planned_repetitions_complete"]},
               "completed_generator_arms": sum(row["status"] == "completed" for row in state["episodes"].values()),
               "planned_generator_arms": len(plan["episodes"]), "planned_selector_results": plan["planned_selector_results"], "calls": state["calls"], "max_calls": plan["max_calls"],
               "reported_known_total_tokens": state["reported_known_total_tokens"], "missing_usage_count": state["missing_usage_count"],
               "complete_total_tokens": state["reported_known_total_tokens"] if not state["missing_usage_count"] else None,
               "arm_statuses": {key: row["status"] for key, row in state["episodes"].items()}, "state_effects": state_effects, "effects": effects,
               "common_prefix_assays": 10 * len(state_effects), "interpretation": plan["interpretation"]}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--state-input-root", type=Path)
    preparation.add_argument("--seeds", type=int, nargs="+", default=list(STATES))
    preparation.add_argument("--repeats", type=int, default=3)
    preparation.add_argument("--order-seed", type=int, default=20261114)
    preparation.add_argument("--model", required=True)
    preparation.add_argument("--api-mode", choices=["chat", "responses"], default="chat")
    preparation.add_argument("--effort", default="xhigh")
    preparation.add_argument("--correction-status", type=Path, required=True)
    preflight = sub.add_parser("public-preflight")
    preflight.add_argument("--model")
    preflight.add_argument("--api-mode", choices=["chat", "responses"])
    preflight.add_argument("--effort", default="xhigh")
    execution = sub.add_parser("run")
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--allow-api", action="store_true")
    execution.add_argument("--max-arms", type=int)
    analysis = sub.add_parser("summarize")
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "public-preflight":
            result = public_preflight(pilot_config(STATES[0], llm_reasoning_effort=args.effort), args.model, args.api_mode)
        elif args.command == "prepare":
            _corrected(args.correction_status)
            from tools.supplement_20261114.strong_baselines import prepare_state
            state_root = args.state_input_root or args.output.parent / "feedback-state-inputs"
            sources = []
            for seed in args.seeds:
                prepared = prepare_state(PROJECT, state_root / f"seed-{seed}", seed, config=pilot_config(seed, llm_reasoning_effort=args.effort))
                sources.append({"state_id": f"seed-{seed}", "data": prepared.dataset.directory, "selection": prepared.root / "selection/selection.json"})
            result = prepare(args.output, sources, model_requested=args.model, api_mode=args.api_mode, repeats=args.repeats,
                             order_seed=args.order_seed, correction_status=args.correction_status)
            result = {"status": "prepared", "study_id": result["study_id"], "planned_generator_arms": result["planned_generator_arms"], "max_calls": result["max_calls"]}
        elif args.command == "run":
            result = run(args.output, allow_api=args.allow_api, max_arms=args.max_arms)
        else:
            result = summarize(args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("status") != "failed" else 2
    except Exception as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
