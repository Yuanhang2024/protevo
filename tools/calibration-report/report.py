"""Export independent audits, CSV tables and line figures for a finalized calibration phase."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from audit import Auditor, Checks, IDENTITY_FIELDS, read_json
from plots import render
from statistics_report import add_resource_metrics, confirmation, query_metrics, summaries
from protevo.statistical_definitions import DEFINITION_VERSION, binary_boundary_reason

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = PROJECT / "results/default-calibration-20261113"


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value {type(value).__name__}")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, default=json_value)
        handle.write("\n")


def write_csv(path, rows):
    frame = pd.json_normalize(rows, sep=".")
    for column in frame:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value)
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=["trajectory", "round", "severity", "code", "detail"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def completed_inputs(study, phase, confirmed):
    if not confirmed:
        raise ValueError("Export requires explicit confirmation that the phase has finished")
    plan = read_json(study / "plan.json")
    execution = read_json(study / phase / "execution.json")
    state = read_json(study / phase / "state.json")
    if state.get("phase") not in ("completed", "partial", "interrupted") or any(row.get("status") == "running" for row in state["conditions"]):
        raise ValueError("The phase is not finalized; report export does not monitor active work")
    analysis = read_json(study / phase / "analysis.json")
    return plan, execution, state, analysis


def completion_observations(records):
    if not records or any(not isinstance(row.get("complete"), (bool, np.bool_)) for row in records):
        raise ValueError("Completion requires planned records with explicit boolean outcomes")
    values = [int(row["complete"]) for row in records]
    per_seed = {}
    for row in records:
        per_seed.setdefault(row["seed"], []).append(int(row["complete"]))
    return {"completed": sum(values), "planned": len(values), "observed": len(values),
            "observed_proportion": sum(values) / len(values),
            "estimate": float(np.mean([np.mean(values) for values in per_seed.values()])),
            "per_seed": per_seed, "boundary_reason": binary_boundary_reason(values)}


def audit_completion_counts(checks, current, expected, label):
    checks.require(all(current.get(field) == expected[field] for field in ("completed", "planned", "observed")),
                   "v2_completion_observation_counts", label)
    checks.close(current.get("observed_proportion"), expected["observed_proportion"], "v2_completion_observed_proportion", label)


def audit_completion_profile(checks, records, profile, historical):
    label = profile["profile"]
    try:
        observed = completion_observations(records)
    except ValueError:
        checks.require(False, "completion_source_records", label)
        return
    checks.require(historical.get("completed") == observed["completed"] and historical.get("total") == observed["planned"],
                   "historical_completion_observation_counts", label)
    historical_rate = historical.get("completionRate", historical.get("completed", 0) / historical["total"] if historical.get("total") else None)
    checks.close(historical_rate, observed["observed_proportion"], "historical_completion_observed_proportion", label)
    current = profile.get("completion_indicator", {})
    checks.require(current.get("definition_version") == DEFINITION_VERSION, "v2_completion_definition_version", label)
    audit_completion_counts(checks, current, observed, label)
    checks.close(current.get("estimate"), observed["estimate"], "v2_completion_equal_seed_estimate", label)
    if observed["boundary_reason"]:
        checks.require(current.get("lower") is None and current.get("upper") is None, "v2_completion_boundary_bounds", label)
        checks.require(current.get("reason") == observed["boundary_reason"], "v2_completion_boundary_reason", label)


def audit_completion_contrast(checks, trajectories, historical, current):
    expected = {}
    try:
        for arm in ("selected", "reference"):
            expected[arm] = completion_observations([row for row in trajectories if row.get("arm") == arm])
    except ValueError:
        checks.require(False, "completion_contrast_source_records")
        return
    seeds = set(expected["selected"]["per_seed"])
    if not checks.require(seeds == set(expected["reference"]["per_seed"]), "completion_contrast_paired_seeds"):
        return
    difference = float(np.mean([np.mean(expected["selected"]["per_seed"][seed]) - np.mean(expected["reference"]["per_seed"][seed]) for seed in sorted(seeds)]))
    checks.close(historical.get("estimate"), difference, "historical_completion_difference_estimate")
    checks.close(current.get("estimate"), difference, "v2_completion_difference_estimate")
    reasons = [expected[arm]["boundary_reason"] for arm in ("selected", "reference")]
    reason = "binary_arm_boundary_empirical_bootstrap" if any(reasons) else None
    if reason is None:
        constant_arms = all(len(set(values)) == 1 for arm in expected.values() for values in arm["per_seed"].values())
        differences = [np.mean(expected["selected"]["per_seed"][seed]) - np.mean(expected["reference"]["per_seed"][seed]) for seed in seeds]
        if constant_arms and len(set(differences)) == 1:
            reason = "degenerate_binary_difference_empirical_bootstrap"
    counts = current.get("binary_arm_counts", {})
    for arm in ("selected", "reference"):
        observed = counts.get(arm, {})
        audit_completion_counts(checks, observed, expected[arm], arm)
        checks.require(observed.get("missing") == 0 and observed.get("boundary_reason") == expected[arm]["boundary_reason"],
                       "v2_completion_arm_observation_scope", arm)
    checks.require(current.get("definition_version") == DEFINITION_VERSION, "v2_completion_contrast_definition_version")
    if reason:
        checks.require(current.get("lower") is None and current.get("upper") is None, "v2_completion_difference_boundary_bounds")
        checks.require(current.get("reason") == reason, "v2_completion_difference_boundary_reason")
    else:
        for field in ("lower", "upper"):
            checks.close(historical.get(field), current.get(field), "preset_completion_interior_interval", field)
        checks.require(current.get("reason") is None, "v2_completion_interior_reason")


def audit_analysis(checks, trajectories, profiles, analysis, phase, intervals):
    recorded = {row["key"]: row for row in analysis.get("trajectories", [])}
    for row in trajectories:
        prior = recorded.get(row["trajectory"])
        if not checks.require(prior is not None, "analysis_trajectory_missing", row["trajectory"]):
            continue
        checks.require(prior.get("complete") == row["complete"], "analysis_completion", row["trajectory"])
        if row["mean_fitness"] is not None:
            checks.close(prior.get("meanFitness"), row["mean_fitness"], "analysis_trajectory_mean", row["trajectory"])
            checks.close(prior.get("maxFitness"), row["max_fitness"], "analysis_trajectory_max", row["trajectory"])
        else:
            checks.require(prior.get("meanFitness") is None and prior.get("maxFitness") is None, "analysis_missing_quality", row["trajectory"])
        checks.require(prior.get("calls") == row["calls"] and prior.get("tokens") == row["reported_tokens_sum"]
                       and prior.get("missingUsage") == row["missing_usage_calls"], "analysis_resource_totals", row["trajectory"])
    previous_profiles = {profile["profileKey"]: profile for profile in analysis.get("profiles", [])}
    for profile in profiles:
        old = previous_profiles.get(profile["profile"])
        if checks.require(old is not None, "analysis_profile_missing", detail=profile["profile"]):
            checks.require(old["completed"] == profile["completed_trajectories"] and old["total"] == profile["planned_trajectories"], "analysis_profile_counts", detail=profile["profile"])
            expected = profile["mean_fitness"]["estimate"]
            if expected is None:
                checks.require(old.get("equalSeedMeanFitness") is None, "analysis_missing_seed_mean", detail=profile["profile"])
            else:
                checks.close(old.get("equalSeedMeanFitness"), expected, "analysis_equal_seed_mean", profile["profile"])
            audit_completion_profile(checks, [row for row in trajectories if row["profile"] == profile["profile"]], profile, old)
    if phase == "phase3":
        for metric in ("meanFitness", "maxFitness"):
            checks.require(metric in intervals, "quality_confirmation_metric_present", metric)
            value = intervals.get(metric, {})
            prior = analysis.get("intervals", {}).get(metric, {})
            for field in ("estimate", "lower", "upper"):
                if value.get(field) is None:
                    checks.require(prior.get(field) is None, "confirmation_missing_value", detail=f"{metric}.{field}")
                else:
                    checks.close(prior.get(field), value[field], "preset_confirmation_interval", f"{metric}.{field}")
        checks.require("completionRate" in intervals, "completion_confirmation_metric_present")
        audit_completion_contrast(checks, trajectories, analysis.get("intervals", {}).get("completionRate", {}), intervals.get("completionRate", {}))


def export_phase(study, phase, *, confirmed=False, label=None, formats=("svg", "pdf", "png"), dpi=600, progress=None):
    study = Path(study).resolve()
    if phase not in ("phase1", "phase2", "phase3"):
        raise ValueError("Choose phase1, phase2 or phase3")
    plan, execution, state, analysis = completed_inputs(study, phase, confirmed)
    checks = Checks()
    planned = execution["conditions"]
    actual = {row["key"]: row for row in state["conditions"]}
    checks.require(len(actual) == len(state["conditions"]) and set(actual) == {row["key"] for row in planned}, "execution_condition_coverage")
    checks.require(len({row["operationId"] for row in planned}) == len(planned), "independent_trajectory_operation_ids")
    checks.require(execution["maximumCalls"] == sum(row["maxCalls"] for row in planned)
                   <= plan[phase]["maximumGenerationCalls"], "phase_maximum_request_budget")
    auditor = Auditor(study, checks)
    trajectories, recommendations, rounds, calls = [], [], [], []
    for position, scheduled in enumerate(planned, 1):
        condition = actual.get(scheduled["key"], {**scheduled, "status": "missing", "calls": []})
        checks.require(all(scheduled.get(field) == condition.get(field) for field in IDENTITY_FIELDS), "execution_condition_unchanged", scheduled["key"])
        try:
            trajectory, selected, observations, usage = auditor.trajectory(phase, condition)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            checks.require(False, "trajectory_audit_exception", condition["key"], detail=type(error).__name__)
            trajectory = {"phase": phase, "trajectory": condition["key"], "profile": condition.get("arm") or condition["profileKey"],
                          **{key: condition.get(key) for key in ("profileKey", "arm", "provider", "model", "effort", "seed", "repeatIndex",
                                                               "candidatesPerRound", "topK", "rounds", "oversizePolicy")},
                          "status": condition["status"], "complete": False, "completed_rounds": 0, "assays": 0,
                          "mean_fitness": None, "max_fitness": None, "initial_best": None,
                          "elapsed_seconds": condition.get("elapsedSeconds"), "audit_passed": False}
            selected, observations = [], []
            usage = auditor.audit_calls(phase, condition, []) if condition.get("calls") else []
        trajectories.append(trajectory)
        recommendations.extend(selected)
        rounds.extend(observations)
        calls.extend(usage)
        if progress is not None:
            progress({"phase": phase, "audited_trajectories": position, "planned_trajectories": len(planned),
                      "trajectory": scheduled["key"], "check_totals": checks.totals(),
                      "recent_issues": checks.rows[-3:]})
    checks.require(len(calls) <= execution["maximumCalls"], "phase_actual_request_budget")
    add_resource_metrics(trajectories, calls)
    checkpoints = query_metrics(trajectories, recommendations)
    profiles, points, per_seed = summaries(trajectories, checkpoints, plan[phase]["dataSeeds"], plan["phase3"]["bootstrap"])
    intervals = confirmation(trajectories, plan[phase]) if phase == "phase3" else {}
    audit_analysis(checks, trajectories, profiles, analysis, phase, intervals)
    label = label or f'{phase}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise ValueError("Report label must contain only letters, digits, underscores and hyphens")
    reports = (study / "reports").resolve()
    output = (reports / label).resolve()
    if not output.is_relative_to(reports):
        raise ValueError("Report output must remain inside the study reports directory")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"trajectories": trajectories, "recommendations": recommendations, "rounds": rounds, "calls": calls,
              "query-checkpoints": checkpoints, "profile-summaries": profiles, "query-summaries": points,
              "seed-summaries": per_seed, "audit-issues": checks.rows, "audit-check-counts": checks.summary()}
    for name, records in tables.items():
        write_csv(output / f"{name}.csv", records)
    errors = sum(row["severity"] == "error" for row in checks.rows)
    metadata = {"definition_version": DEFINITION_VERSION, "phase": phase, "created_at": datetime.now(timezone.utc).isoformat(), "study_id": plan["studyId"],
                "audit_passed": errors == 0, "audit_errors": errors, "check_totals": checks.totals(),
                "audit_warnings": sum(row["severity"] == "warning" for row in checks.rows),
                "planned_trajectories": len(trajectories), "completed_trajectories": sum(row["complete"] for row in trajectories),
                "status_counts": dict(Counter(row["status"] for row in trajectories)),
                "request_count": len(calls), "reported_tokens_sum": sum(row["reported_tokens_sum"] for row in trajectories),
                "missing_usage_calls": sum(row["missing_usage_calls"] for row in trajectories),
                "model_rebuild_counts": auditor.fit_counts, "profiles": profiles, "query_summaries": points,
                "confirmation_intervals": intervals, "preset_selection": {key: analysis.get(key) for key in ("selected", "selectedArm", "provisional")},
                "statistics": {"quality": "Completed trajectories; each data seed receives equal weight.",
                               "missing": "Missing trajectories and measurements remain missing; failed quality is not assigned zero.",
                               "bootstrap": plan["phase3"]["bootstrap"], "query_checkpoints": [10, 20, 30],
                               "observational_unit": "Independent trajectory, nested within data split; checkpoints are dependent.",
                               "selection": "The recorded selection is copied unchanged; report code does not choose defaults."}}
    write_json(output / "summary.json", metadata)
    write_json(output / "audit.json", {"definition_version": DEFINITION_VERSION, "passed": errors == 0, "check_totals": checks.totals(), "check_counts": checks.summary(),
                                        "checks": checks.rows, "predictor_rebuilds": auditor.fit_counts})
    files = render(points, checkpoints, profiles, output / "figures", formats=formats, dpi=dpi) if profiles and formats else []
    manifest = {"definition_version": DEFINITION_VERSION, "phase": phase, "audit_passed": errors == 0, "directory": str(output),
                "files": [str(path.relative_to(output)) for path in sorted(output.glob("*.csv"))] + ["summary.json", "audit.json"]
                         + [str(path.relative_to(output)) for path in files]}
    write_json(output / "manifest.json", manifest)
    return metadata, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2", "phase3"))
    parser.add_argument("--confirmed-complete", action="store_true", required=True)
    parser.add_argument("--label")
    parser.add_argument("--formats", nargs="*", choices=("svg", "pdf", "png"), default=["svg", "pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    options = parser.parse_args(argv)
    if options.dpi < 72:
        parser.error("dpi must be at least 72")
    previous_errors = -1
    def progress(value):
        nonlocal previous_errors
        errors = value["check_totals"]["errors"]
        if value["audited_trajectories"] % 10 == 0 or value["audited_trajectories"] == value["planned_trajectories"] or errors != previous_errors:
            print(json.dumps({"event": "audit_progress", **value}, ensure_ascii=False), flush=True)
        previous_errors = errors
    metadata, output = export_phase(options.study, options.phase, confirmed=options.confirmed_complete,
                                    label=options.label, formats=tuple(options.formats), dpi=options.dpi, progress=progress)
    print(json.dumps({"phase": options.phase, "directory": str(output), "audit_passed": metadata["audit_passed"],
                      "audit_errors": metadata["audit_errors"], "trajectories": metadata["planned_trajectories"],
                      "completed": metadata["completed_trajectories"]}, ensure_ascii=False))
    return 0 if metadata["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
