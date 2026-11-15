"""Read-only verification of finalized trajectories against frozen GB1 implementations."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib
import importlib.util
from itertools import count
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Auditing must not add bytecode files to the immutable scientific snapshot.
sys.dont_write_bytecode = True
_NAMES = count(1)
IDENTITY_FIELDS = ("key", "operationId", "profileKey", "seed", "repeatIndex", "provider", "model", "effort",
                   "candidatesPerRound", "topK", "rounds", "oversizePolicy", "maxCalls")
SELECTION_FIELDS = ("seed", "train_size", "validation_size", "test_size", "pool_size", "model_families",
                    "ridge_alphas", "tree_estimators", "tree_min_samples_leaf", "tree_max_depth", "model_seed", "top_k_metric", "n_jobs")


def read_json(path):
    path = Path(path)
    if any(part.lower() == ".env" or part.lower().startswith(".env.") for part in path.parts):
        raise ValueError("Environment files are not audit inputs")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def load_frozen_core(snapshot):
    snapshot = Path(snapshot).resolve()
    name = f"_gb1_report_frozen_{next(_NAMES)}"
    spec = importlib.util.spec_from_file_location(name, snapshot / "protevo/__init__.py", submodule_search_locations=[str(snapshot / "protevo")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return SimpleNamespace(**{module: importlib.import_module(f"{name}.{module}") for module in
                             ("schema", "data", "models", "agent", "prompts", "experiment")})


class Checks:
    def __init__(self):
        self.rows = []
        self.counts = Counter()

    def require(self, condition, code, key="", number=None, detail="", severity="error"):
        self.counts[code, "passed" if condition else severity] += 1
        if not condition:
            self.rows.append({"trajectory": key, "round": number, "severity": severity, "code": code, "detail": detail})
        return bool(condition)

    def summary(self):
        return [{"code": code, "passed": self.counts[code, "passed"], "errors": self.counts[code, "error"],
                 "warnings": self.counts[code, "warning"], "total": sum(self.counts[code, status] for status in ("passed", "error", "warning"))}
                for code in sorted({code for code, _ in self.counts})]

    def totals(self):
        rows = self.summary()
        return {field: sum(row[field] for row in rows) for field in ("passed", "errors", "warnings", "total")}

    def close(self, actual, expected, code, key="", number=None, tolerance=1e-9):
        try:
            valid = np.asarray(actual).shape == np.asarray(expected).shape and bool(np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False))
        except (TypeError, ValueError):
            valid = False
        return self.require(valid, code, key, number)


class Auditor:
    """Only saved selected variants are submitted to the measurement lookup."""
    def __init__(self, study, checks=None):
        self.study = Path(study).resolve()
        self.snapshot = self.study / "snapshot"
        self.core = load_frozen_core(self.snapshot)
        self.checks = checks if checks is not None else Checks()
        self.datasets, self.selections, self.initial_models, self.sources = {}, {}, {}, {}
        self.fit_counts = {"initial": 0, "feedback": 0}

    def seed_data(self, seed):
        if seed not in self.datasets:
            ready = read_json(self.study / "shared" / f"seed-{seed}" / "ready.json")
            directory = Path(ready["dataset"]).resolve()
            selection_path = Path(ready["selection"]).resolve()
            allowed = (self.study / "shared" / f"seed-{seed}").resolve()
            if not directory.is_relative_to(allowed) or not selection_path.is_relative_to(allowed):
                raise ValueError("Shared dataset and selection must belong to the declared seed directory")
            dataset = self.core.data.PublicDataset(directory)
            selection = read_json(selection_path)
            self.checks.require(dataset.manifest == selection["dataset_manifest"], "selection_dataset_manifest", detail=f"seed={seed}")
            validation_ids = set(pd.read_csv(directory / "validation.csv", usecols=["variant"]).variant)
            self.checks.require(not validation_ids & (set(dataset.train.variant) | set(dataset.candidates.variant)), "validation_role_disjoint", detail=f"seed={seed}")
            self.datasets[seed], self.selections[seed] = dataset, selection
        return self.datasets[seed], self.selections[seed]

    def source_values(self, dataset, variants):
        """Look up selected variants in the original public S1 source after selection validation."""
        source = Path(dataset.manifest["source"]).resolve()
        if any(part.lower().startswith(".env") for part in source.parts):
            raise ValueError("A measurement source cannot be an environment file")
        if source not in self.sources:
            schema = dataset.manifest["schema"]
            options = {"header": schema["header"], "usecols": [schema["variant_column"], schema["fitness_column"]]}
            if source.suffix.lower() == ".xlsx":
                table = pd.read_excel(source, sheet_name=schema["sheet_name"], engine="openpyxl", **options)
            elif source.suffix.lower() == ".csv":
                table = pd.read_csv(source, **options)
            else:
                raise ValueError("Audit source must be the declared public CSV or XLSX")
            names = table[schema["variant_column"]].astype("string").str.strip().str.upper()
            values = pd.to_numeric(table[schema["fitness_column"]], errors="coerce")
            clean = pd.DataFrame({"variant": names, "fitness": values}).dropna()
            duplicate = clean.groupby("variant").fitness.nunique()
            if (duplicate > 1).any():
                raise ValueError("The original measurement source has conflicting duplicates")
            self.sources[source] = clean.drop_duplicates("variant").set_index("variant").fitness
        return self.sources[source].loc[list(variants)].to_numpy(dtype=float)

    def predictor(self, seed, selection, config, known, number):
        if number == 1:
            if seed not in self.initial_models:
                self.initial_models[seed] = self.core.models.fit_model(selection["spec"], config, known)
                self.fit_counts["initial"] += 1
            return self.initial_models[seed]
        self.fit_counts["feedback"] += 1
        return self.core.models.fit_model(selection["spec"], config, known)

    def audit_calls(self, phase, condition, attempts, config=None):
        key, calls = condition["key"], condition.get("calls", [])
        self.checks.require(len(calls) <= condition["maxCalls"], "trajectory_call_budget", key)
        numbers = [call.get("number") for call in calls]
        self.checks.require(numbers == list(range(1, len(calls) + 1)), "host_call_numbers", key)
        counts = Counter(call.get("round") for call in calls)
        self.checks.require(all(number in range(1, condition["rounds"] + 1) and count <= 2 for number, count in counts.items()), "host_round_call_budget", key)
        self.checks.require(len(attempts) >= len(calls), "host_calls_have_attempts", key)
        exported = []
        for index, call in enumerate(calls):
            number = call.get("round")
            effective = call.get("effectiveConfig")
            if effective:
                self.checks.require(all(effective.get(k) == condition[v] for k, v in
                                        (("provider", "provider"), ("model", "model"), ("reasoningEffort", "effort"))), "effective_model_route", key, number)
                self.checks.require(effective.get("maxTokens") == condition.get("effectiveMaxTokens"), "adapter_output_limit", key, number)
            else:
                self.checks.require(False, "effective_config_missing", key, number, severity="warning")
            if call.get("outcomeUnknown"):
                self.checks.require(index == len(calls) - 1, "unknown_outcome_not_resent", key, number)
            path = self.study / phase / "requests" / key / f'{call["number"]}.json'
            if self.checks.require(path.is_file(), "request_snapshot_missing", key, number):
                snapshot = read_json(path)
                frame = snapshot.get("request", {})
                payload = frame.get("payload", {})
                self.checks.require(frame.get("id") == f'request-{call["number"]}' and frame.get("round") == number, "request_frame_identity", key, number)
                self.checks.require(not {"max_completion_tokens", "max_output_tokens", "max_tokens"} & set(payload), "project_output_limit_absent", key, number)
                self.checks.require(payload.get("model") == condition["model"] and payload.get("reasoning_effort") == condition["effort"], "payload_model_route", key, number)
                if effective:
                    transport = snapshot.get("transport_request") or {}
                    self.checks.require(all(transport.get(field) == effective.get(field) for field in
                                            ("provider", "model", "reasoningEffort", "maxTokens")), "transport_model_route", key, number)
                if index < len(attempts):
                    round_id, attempt = attempts[index]
                    try:
                        context = json.loads(payload["messages"][1]["content"])
                    except (KeyError, IndexError, TypeError, ValueError):
                        context = None
                    self.checks.require(number == round_id and context == attempt.get("context"), "request_context_snapshot", key, number)
                    self.checks.require(frame.get("prompt_metadata") == attempt.get("prompt_metadata"), "request_prompt_snapshot", key, number)
                    if config is not None:
                        expected_payload = self.core.agent.build_request_payload(attempt["context"], config, condition["model"], "chat", attempt["prompt_metadata"]["system_prompt"])
                        self.checks.require(payload == expected_payload, "complete_public_request_payload", key, number)
                    if "response" in attempt:
                        reply = snapshot.get("reply", {}).get("response")
                        self.checks.require(reply == attempt["response"], "response_snapshot", key, number)
                self.checks.require(snapshot.get("usage") == call.get("usage"), "host_usage_snapshot", key, number)
            total_raw = (call.get("usage") or {}).get("total_tokens")
            exported.append({"phase": phase, "trajectory": key, "profile": condition.get("arm") or condition["profileKey"],
                             "seed": condition["seed"], "repeat_index": condition["repeatIndex"], "round": number,
                             "call": call["number"], "provider": condition["provider"], "model": condition["model"],
                             "effort": condition["effort"], "reported_tokens": total_raw,
                             "usage_missing": total_raw is None, "elapsed_seconds": call.get("elapsedSeconds"),
                             "generation_seconds": call.get("generationSeconds"), "finish_reason": call.get("finishReason"),
                             "outcome_unknown": bool(call.get("outcomeUnknown"))})
        return exported

    def audit_event_order(self, root, method, key):
        path = root / "run/events.jsonl"
        if not self.checks.require(path.is_file(), "scientific_events_missing", key):
            return
        with path.open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        for record in method["rounds"]:
            number = record["round"]
            indices = {kind: [index for index, event in enumerate(events) if event.get("kind") == kind and event.get("round") == number]
                       for kind in ("fitness_model_scores", "candidates_selected", "round_evaluated")}
            if record["stage"] in ("selected", "evaluated"):
                score, selected = indices["fitness_model_scores"], indices["candidates_selected"]
                valid = len(score) == len(selected) == 1 and score[0] < selected[0]
                self.checks.require(valid, "scores_saved_before_selection", key, number)
                if selected:
                    event_rows = events[selected[0]].get("recommendations", [])
                    self.checks.require(all("actual_fitness" not in row for row in event_rows)
                                        and [row["variant"] for row in event_rows] == [row["variant"] for row in record["recommendations"]],
                                        "selection_event_has_no_measurements", key, number)
                    evaluated = indices["round_evaluated"]
                    if record["stage"] == "evaluated":
                        self.checks.require(len(evaluated) == 1 and selected[0] < evaluated[0], "selected_before_measurement_feedback", key, number)

    def trajectory(self, phase, condition):
        key, seed = condition["key"], condition["seed"]
        first_issue = len(self.checks.rows)
        root = self.study / phase / "runs" / key
        state_path = root / "run/state.json"
        identity = {"phase": phase, "trajectory": key, "profile": condition.get("arm") or condition["profileKey"],
                    **{field: condition.get(field) for field in ("profileKey", "arm", "provider", "model", "effort", "seed", "repeatIndex",
                                                               "candidatesPerRound", "topK", "rounds", "oversizePolicy")}}
        calls, recommendations, rounds_out = [], [], []
        if not state_path.is_file():
            self.checks.require(condition["status"] != "completed", "completed_state_missing", key)
            calls = self.audit_calls(phase, condition, []) if condition.get("calls") else []
            return {**identity, "status": condition["status"], "complete": False, "completed_rounds": 0,
                    "assays": 0, "mean_fitness": None, "max_fitness": None, "initial_best": None,
                    "elapsed_seconds": condition.get("elapsedSeconds"), "audit_passed": not any(r["severity"] == "error" for r in self.checks.rows[first_issue:])}, recommendations, rounds_out, calls
        state = read_json(state_path)
        config = self.core.schema.ExperimentConfig.model_validate(state["config"])
        dataset, selection = self.seed_data(seed)
        self.checks.require(state["dataset_manifest"] == dataset.manifest and state["selection"] == selection, "trajectory_frozen_dataset_selection", key)
        self.checks.require(all(selection["config"][field] == getattr(config, field) for field in SELECTION_FIELDS), "shared_selection_parameters", key)
        self.checks.require((config.seed, config.rounds, config.k, config.proposal_count, config.score_budget, config.oversize_policy) ==
                            (seed, condition["rounds"], condition["topK"], condition["candidatesPerRound"], condition["candidatesPerRound"], condition["oversizePolicy"]), "trajectory_scientific_parameters", key)
        worker = read_json(root / "worker.json")
        self.checks.require(worker["config"] == state["config"], "worker_frozen_config", key)
        self.checks.require(all(worker["condition"].get(field) == condition.get(field) for field in IDENTITY_FIELDS), "worker_condition_identity", key)
        self.checks.require(config.llm_max_output_tokens is None and config.llm_timeout is None and config.llm_temperature is None
                            and config.llm_reasoning_effort == condition["effort"], "generation_settings", key)
        self.checks.require(config.save_predictors is False, "predictor_disk_saving", key)
        self.checks.require(not list((root / "run").glob("*.joblib")), "predictor_files_absent", key)
        rules_path = Path(config.knowledge_path).resolve()
        if not rules_path.is_relative_to(self.snapshot) or rules_path.suffix.lower() != ".json":
            raise ValueError("Frozen knowledge must be inside the scientific snapshot")
        knowledge = self.core.agent.load_knowledge(rules_path)
        rule_ids = [rule.id for rule in knowledge.rules]
        prompt = self.core.prompts.load_prompt_profile("v3")
        self.checks.require(state.get("prompt_metadata") == prompt.metadata(), "frozen_prompt_metadata", key)
        known, used, evaluated, attempts = dataset.train.copy(), set(), [], []
        candidate_ids = set(dataset.candidates.variant)
        method = state["methods"]["knowledge"]
        for number, record in enumerate(method["rounds"], 1):
            self.checks.require(record["round"] == number <= config.rounds, "round_sequence", key, number)
            expected_ids = list(known.variant)
            self.checks.require(record.get("training_variants") == expected_ids and record.get("training_count") == len(expected_ids), "training_ids_own_feedback", key, number)
            recent = [row for earlier in evaluated[-config.feedback_rounds:] for row in earlier["recommendations"]] if config.feedback_rounds else []
            self.checks.require(record.get("recent_round_feedback", []) == recent, "recent_feedback_own_history", key, number)
            cohort = self.core.experiment.candidate_cohort(dataset, known, config, number)
            self.checks.require(record.get("cohort") == cohort, "reproducible_unlabeled_cohort", key, number)
            base = self.core.agent.make_context(known, record.get("cohort", []), config, dataset.reference, number, knowledge, "v3", recent)
            eligible = {v for v in candidate_ids - set(known.variant) if self.core.schema.variant_allowed(v, config)}
            accepted, rejected = [], []
            self.checks.require(len(record["attempts"]) <= config.llm_max_calls_per_round, "round_request_budget", key, number)
            for attempt_index, attempt in enumerate(record["attempts"], 1):
                attempts.append((number, attempt))
                expected = {**base, "proposal_count": config.proposal_count - len(accepted)} if attempt_index == 1 else self.core.prompts.repair_context(
                    base, known, eligible, accepted, rejected, config, number, attempt_index)
                self.checks.require(attempt.get("number") == attempt_index and attempt.get("context") == expected, "model_visible_context", key, number)
                self.checks.require(attempt.get("prompt_metadata") == prompt.metadata(), "attempt_prompt_metadata", key, number)
                response = attempt.get("response")
                if response is None:
                    self.checks.require(attempt_index == len(record["attempts"]), "unknown_attempt_not_resent", key, number)
                    continue
                self.checks.require(response.get("model_requested") == condition["model"] and response.get("api_mode") == "dsh"
                                    and response.get("prompt_metadata") == prompt.metadata(), "model_response_identity", key, number)
                raw_items = None
                try:
                    body = json.loads(response["text"])
                    raw_items = body.get("proposals") if isinstance(body, dict) else None
                except (TypeError, ValueError):
                    pass
                try:
                    if not response["complete"]:
                        raise ValueError("Model response was refused or incomplete")
                    denied = {item["variant"] for item in rejected if isinstance(item.get("variant"), str)}
                    _, valid, invalid = self.core.agent.collect_proposals(response["text"], eligible - denied, config, accepted, rule_ids, expected["proposal_count"])
                    accepted.extend(valid)
                    rejected.extend(invalid)
                except (TypeError, ValueError) as error:
                    invalid = [{"variant": item.get("variant"), "reason": str(error)} for item in raw_items
                               if isinstance(item, dict) and isinstance(item.get("variant"), str)] if isinstance(raw_items, list) else []
                    rejected.extend(invalid or [{"variant": None, "reason": str(error)}])
                    valid = []
                self.checks.require(attempt.get("valid_count") == len(valid), "saved_valid_proposal_count", key, number)
            stage = record["stage"]
            scores = record.get("scores", [])
            if scores:
                self.checks.require(record.get("proposals") == accepted, "proposals_from_saved_responses", key, number)
                variants = [row["variant"] for row in scores]
                self.checks.require(variants == [row["variant"] for row in accepted] and len(set(variants)) == len(variants)
                                    and set(variants) <= eligible and len(variants) <= config.score_budget, "scored_candidate_budget_eligibility", key, number)
                self.checks.require(record.get("model_score_count") == len(variants), "saved_scoring_count", key, number)
                model = self.predictor(seed, selection, config, known, number)
                predicted = model.predict(self.core.data.one_hot(variants))
                self.checks.close([row["predicted_fitness"] for row in scores], predicted, "rebuilt_predictor_scores", key, number)
            if stage in ("selected", "evaluated"):
                selected = record["recommendations"]
                ids = [row["variant"] for row in selected]
                ranked = sorted(scores, key=lambda row: (-row["predicted_fitness"], row["variant"]))[:config.k]
                self.checks.require(ids == [row["variant"] for row in ranked] and len(ids) == len(set(ids)) == config.k
                                    and not set(ids) & (used | set(dataset.train.variant)), "top_k_ranking_no_repeats", key, number)
                lookup = {row["variant"]: row["predicted_fitness"] for row in scores}
                self.checks.close([row["predicted_fitness"] for row in selected], [lookup.get(v, math.nan) for v in ids], "selected_prediction_matches_scores", key, number)
                for rank, row in enumerate(selected, 1):
                    description = self.core.data.describe_variant(row["variant"], dataset.reference)
                    self.checks.require(row["rank"] == rank and all(row.get(field) == value for field, value in description.items()), "selected_sequence_description", key, number)
                if stage == "evaluated":
                    measured = self.core.data.HiddenOracle(dataset.directory, ids).evaluate(ids)
                    actual = [row["actual_fitness"] for row in selected]
                    self.checks.close(actual, [row["actual_fitness"] for row in measured], "selected_measurements_oracle", key, number)
                    self.checks.close(actual, self.source_values(dataset, ids), "selected_measurements_original_source", key, number)
                    previous_best = float(known.fitness.max())
                    expected_metrics = {"mean_fitness": float(np.mean(actual)), "max_fitness": float(np.max(actual)),
                                        "cumulative_best": max(previous_best, float(np.max(actual)))}
                    for field, value in expected_metrics.items():
                        self.checks.close(record["metrics"].get(field), value, f"round_metric_{field}", key, number)
                    for row in selected:
                        recommendations.append({**identity, "round": number, "query_index": len(recommendations) + 1,
                                                **{field: row.get(field) for field in ("rank", "variant", "sequence", "mutations", "mutation_count", "predicted_fitness", "actual_fitness")}})
                    rounds_out.append({**identity, "round": number, "assays": len(recommendations), "training_count": len(known),
                                       "scored_count": len(scores), "calls": len(record["attempts"]), **expected_metrics})
                    known = pd.concat([known, pd.DataFrame([{**self.core.data.describe_variant(row["variant"], dataset.reference),
                                                            "fitness": row["actual_fitness"]} for row in selected])], ignore_index=True)
                    used.update(ids)
                    evaluated.append(record)
            self.checks.require(stage == "evaluated" or number == len(method["rounds"]), "incomplete_round_is_last", key, number)
        self.audit_event_order(root, method, key)
        calls = self.audit_calls(phase, condition, attempts, config)
        complete = condition["status"] == "completed" and len(evaluated) == config.rounds
        self.checks.require(condition["status"] != "completed" or complete and method["status"] == "completed", "completed_round_count", key)
        self.checks.require(condition.get("completedRounds", len(evaluated)) == len(evaluated), "host_feedback_round_count", key)
        result_path = self.study / phase / "results" / f"{key}.json"
        if result_path.is_file():
            summary = read_json(result_path)["summary"]
            self.checks.require(summary["completed_rounds"] == len(evaluated) and summary["calls"] == len(attempts)
                                and len(summary["rounds"]) == len(method["rounds"]), "public_summary_counts", key)
            for saved, public in zip(method["rounds"], summary["rounds"]):
                self.checks.require(public["round"] == saved["round"], "public_round_number", key)
                if saved["stage"] == "evaluated":
                    self.checks.require([row["variant"] for row in public["top10"]] == [row["variant"] for row in saved["recommendations"]], "public_selected_variants", key, saved["round"])
                    self.checks.close([row["actual_fitness"] for row in public["top10"]], [row["actual_fitness"] for row in saved["recommendations"]], "public_selected_fitness", key, saved["round"])
        else:
            self.checks.require(not complete, "completed_public_summary_missing", key)
        values = [row["actual_fitness"] for row in recommendations]
        row = {**identity, "status": condition["status"], "complete": complete, "completed_rounds": len(evaluated),
               "assays": len(values), "mean_fitness": float(np.mean(values)) if values else None,
               "max_fitness": float(np.max(values)) if values else None, "initial_best": float(dataset.train.fitness.max()),
               "elapsed_seconds": condition.get("elapsedSeconds"),
               "audit_passed": not any(issue["severity"] == "error" for issue in self.checks.rows[first_issue:])}
        return row, recommendations, rounds_out, calls
