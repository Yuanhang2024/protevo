"""Equal-split summaries and trajectory-level hierarchical percentile intervals."""
from __future__ import annotations

from collections import defaultdict
import statistics
import numpy as np

CHECKPOINTS = (10, 20, 30)


def finite(value):
    return value is not None and isinstance(value, (int, float)) and bool(np.isfinite(value))


def interval(rows, metric, seeds, specification, complete_only=True):
    arrays = {seed: np.asarray([row[metric] for row in rows if row["seed"] == seed
                              and (row["complete"] or not complete_only) and finite(row.get(metric))], dtype=float) for seed in seeds}
    counts = {str(seed): len(values) for seed, values in arrays.items()}
    base = {"usable_per_seed": counts, "conditional_on_completion": complete_only,
            "conditional_on_reporting": metric in ("tokens", "elapsed_seconds"),
            "confidence_level": specification["confidenceLevel"], "replicates": specification["replicates"],
            "outer_unit": "data split", "inner_unit": "independent trajectory within profile and split"}
    if any(not len(values) for values in arrays.values()):
        return {**base, "estimate": None, "lower": None, "upper": None, "seed_sd": None, "reason": "missing_data_split"}
    estimates = [float(values.mean()) for values in arrays.values()]
    count = specification["replicates"]
    rng = np.random.default_rng(specification["seed"])
    sampled = rng.choice(seeds, size=(count, len(seeds)), replace=True)
    means = np.empty((count, len(seeds)))
    for column in range(len(seeds)):
        for seed, values in arrays.items():
            mask = sampled[:, column] == seed
            n = int(mask.sum())
            if n:
                means[mask, column] = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    alpha = 1 - specification["confidenceLevel"]
    lower, upper = np.quantile(means.mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return {**base, "estimate": statistics.mean(estimates), "lower": float(lower), "upper": float(upper),
            "seed_sd": statistics.stdev(estimates) if len(estimates) > 1 else None, "reason": None}


def paired_difference(rows, metric, settings, complete_only=True):
    """Reproduce the preset phase-3 paired-split, independent-arm resampling order."""
    seeds, specification = settings["dataSeeds"], settings["bootstrap"]
    arrays = {}
    for seed in seeds:
        for arm in ("selected", "reference"):
            values = [row[metric] for row in rows if row["seed"] == seed and row.get("arm") == arm
                      and (not complete_only or row["complete"]) and finite(row.get(metric))]
            if not values:
                return {"estimate": None, "lower": None, "upper": None,
                        "reason": "At least one paired split has no usable trajectories."}
            arrays[seed, arm] = np.asarray(values, dtype=float)
    count = specification["replicates"]
    rng = np.random.default_rng(specification["seed"])
    sampled = rng.choice(seeds, size=(count, len(seeds)), replace=True)
    differences = np.empty((count, len(seeds)))
    for column in range(len(seeds)):
        for seed in seeds:
            mask = sampled[:, column] == seed
            n = int(mask.sum())
            if n:
                left, right = arrays[seed, "selected"], arrays[seed, "reference"]
                means_left = rng.choice(left, size=(n, len(left)), replace=True).mean(axis=1)
                means_right = rng.choice(right, size=(n, len(right)), replace=True).mean(axis=1)
                differences[mask, column] = means_left - means_right
    alpha = 1 - specification["confidenceLevel"]
    lower, upper = np.quantile(differences.mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return {"estimate": statistics.mean(float(arrays[seed, "selected"].mean() - arrays[seed, "reference"].mean()) for seed in seeds),
            "lower": float(lower), "upper": float(upper), "confidenceLevel": specification["confidenceLevel"],
            "replicates": count, "outerUnit": specification["outerUnit"], "innerUnit": specification["innerUnit"],
            "difference": "selected minus reference", "conditionalOnCompletion": complete_only}


def add_resource_metrics(trajectories, calls):
    grouped = defaultdict(list)
    for call in calls:
        grouped[call["trajectory"]].append(call)
    for row in trajectories:
        values = grouped[row["trajectory"]]
        row["calls"] = len(values)
        row["missing_usage_calls"] = sum(not finite(call.get("reported_tokens")) for call in values)
        row["reported_tokens_sum"] = sum(call["reported_tokens"] for call in values if finite(call.get("reported_tokens")))
        row["tokens"] = row["reported_tokens_sum"] if row["missing_usage_calls"] == 0 else None
        row["model_seconds"] = sum(call["generation_seconds"] for call in values if finite(call.get("generation_seconds")))
        row["missing_model_time_calls"] = sum(not finite(call.get("generation_seconds")) for call in values)
        row["completion_indicator"] = int(row["complete"])


def query_metrics(trajectories, recommendations):
    grouped = defaultdict(list)
    for row in recommendations:
        grouped[row["trajectory"]].append(row)
    output = []
    for row in trajectories:
        measured = sorted(grouped[row["trajectory"]], key=lambda item: item["query_index"])
        for queries in CHECKPOINTS:
            values = [item["actual_fitness"] for item in measured[:queries]]
            available = len(values) == queries
            initial = row.get("initial_best")
            maximum = max(values) if available else None
            output.append({**row, "queries": queries, "checkpoint_available": available,
                           "mean_fitness": statistics.mean(values) if available else None, "max_fitness": maximum,
                           "cumulative_best": max(initial, maximum) if available and finite(initial) else None})
    return output


def summaries(trajectories, checkpoints, seeds, specification):
    groups = defaultdict(list)
    for row in trajectories:
        groups[row["profile"]].append(row)
    profiles, points, per_seed = [], [], []
    for profile, records in sorted(groups.items()):
        example = records[0]
        identity = {key: example.get(key) for key in ("phase", "profile", "profileKey", "arm", "provider", "model", "effort",
                                                    "candidatesPerRound", "topK", "rounds", "oversizePolicy")}
        complete = [row for row in records if row["complete"]]
        base = {**identity, "planned_trajectories": len(records), "completed_trajectories": len(complete),
                "status_counts": dict((status, sum(row["status"] == status for row in records)) for status in sorted({row["status"] for row in records})),
                "calls": sum(row["calls"] for row in records), "reported_tokens_sum": sum(row["reported_tokens_sum"] for row in records),
                "missing_usage_calls": sum(row["missing_usage_calls"] for row in records),
                "median_complete_tokens": statistics.median(row["tokens"] for row in complete) if complete and all(finite(row["tokens"]) for row in complete) else None,
                "median_complete_elapsed_seconds": statistics.median(row["elapsed_seconds"] for row in complete if finite(row.get("elapsed_seconds")))
                    if any(finite(row.get("elapsed_seconds")) for row in complete) else None,
                "minimum_seed_completions": min(sum(row["seed"] == seed for row in complete) for seed in seeds),
                "audit_errors": sum(not row["audit_passed"] for row in records)}
        for metric in ("mean_fitness", "max_fitness", "completion_indicator", "tokens", "elapsed_seconds"):
            base[metric] = interval(records, metric, seeds, specification, complete_only=metric != "completion_indicator")
        profiles.append(base)
        for seed in seeds:
            subset = [row for row in records if row["seed"] == seed]
            done = [row for row in subset if row["complete"]]
            per_seed.append({**identity, "seed": seed, "planned_trajectories": len(subset), "completed_trajectories": len(done),
                             "mean_fitness": statistics.mean(row["mean_fitness"] for row in done) if done else None,
                             "max_fitness": statistics.mean(row["max_fitness"] for row in done) if done else None,
                             "completion_rate": len(done) / len(subset) if subset else None})
        for queries in CHECKPOINTS:
            subset = [row for row in checkpoints if row["profile"] == profile and row["queries"] == queries]
            point = {**identity, "queries": queries, "planned_trajectories": len(records),
                     "usable_trajectories": sum(row["complete"] and row["checkpoint_available"] for row in subset),
                     "observed_checkpoints": sum(row["checkpoint_available"] for row in subset)}
            for metric in ("mean_fitness", "max_fitness", "cumulative_best"):
                value = interval(subset, metric, seeds, specification)
                for field in ("estimate", "lower", "upper", "seed_sd", "reason"):
                    point[f"{metric}_{field}"] = value[field]
                point["usable_per_seed"] = value["usable_per_seed"]
            points.append(point)
    return profiles, points, per_seed


def confirmation(trajectories, settings):
    return {"meanFitness": paired_difference(trajectories, "mean_fitness", settings),
            "maxFitness": paired_difference(trajectories, "max_fitness", settings),
            "completionRate": paired_difference(trajectories, "completion_indicator", settings, complete_only=False)}
