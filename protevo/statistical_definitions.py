"""Explicit statistical units for saved GB1 rankings, references and usage."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics
from collections.abc import Mapping

import numpy as np
from scipy.stats import kendalltau, spearmanr

DEFINITION_VERSION = "gb1-statistical-definitions/2"
BINARY_METRICS = frozenset({"completion_indicator", "completionIndicator", "completion_rate", "completionRate"})


def validate_permutation(ranked, expected):
    ranked, expected = list(ranked), list(expected)
    if not expected or any(not isinstance(value, str) for value in expected + ranked):
        raise ValueError("Ranking inputs must contain nonempty variant ID strings")
    if len(expected) != len(set(expected)):
        raise ValueError("The supplied ranking input must contain unique variants")
    if len(ranked) != len(expected) or len(ranked) != len(set(ranked)) or set(ranked) != set(expected):
        raise ValueError("Returned ranking must be an exact permutation of the supplied variants")
    return ranked, expected


def ranking_metrics(ranked, expected, fitness, k=10):
    """Best-first position increases downward; its negation is a higher-is-better score."""
    ranked, expected = validate_permutation(ranked, expected)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    actual = np.asarray([fitness[variant] for variant in ranked], dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError("Measured fitness must be finite")
    position = np.arange(len(ranked), dtype=float)
    defined = len(ranked) > 1 and np.ptp(actual) > 0
    raw_spearman = float(spearmanr(position, actual).statistic) if defined else None
    raw_kendall = float(kendalltau(position, actual).statistic) if defined else None
    count = min(k, len(ranked))
    true_order = sorted(expected, key=lambda variant: (-float(fitness[variant]), variant))
    return {"n": len(ranked), "permutation_valid": True, "position_base": 0, "higher_score_definition": "-position",
            "spearman_position_vs_fitness": raw_spearman,
            "spearman_higher_score_vs_fitness": -raw_spearman if defined else None,
            "kendall_position_vs_fitness": raw_kendall,
            "kendall_higher_score_vs_fitness": -raw_kendall if defined else None,
            "correlation_undefined_reason": None if defined else "fewer_than_two_or_constant_fitness",
            "top_k": count, "top_k_overlap": len(set(ranked[:count]) & set(true_order[:count])),
            "true_top_variants": true_order[:count], "true_top_ties": "variant ID ascending, independent of model ordering",
            "definition_version": DEFINITION_VERSION}


def threshold_prevalence(values, quantile):
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 <= quantile <= 1:
        raise ValueError("A quantile requires a nonempty finite population and probability in [0,1]")
    threshold = float(np.quantile(values, quantile, method="linear"))
    passing = int(np.count_nonzero(values >= threshold))
    return {"quantile": quantile, "threshold": threshold, "qualifying_count": passing,
            "population_size": len(values), "uniform_ge_threshold_probability": passing / len(values),
            "comparison": ">=", "quantile_method": "linear"}


def anchor_weighted_summary(pairs):
    """Give each pair one weight, and separately each candidate record one weight."""
    groups = defaultdict(list)
    seen, successes, probabilities = set(), [], []
    for row in pairs:
        group = (row["source"], row["method"], int(row["round"]), row["variant"])
        pair = (*group, row["anchor"])
        if pair in seen:
            raise ValueError("Candidate-anchor pairs must be unique within a candidate record")
        seen.add(pair)
        success, probability = row["success"], float(row["random_ge_anchor"])
        if success not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Anchor success must be binary and reference probability must be in [0,1]")
        successes.append(float(success))
        probabilities.append(probability)
        groups[group].append((float(success), probability))
    candidate_successes = [statistics.mean(value[0] for value in rows) for rows in groups.values()]
    candidate_probabilities = [statistics.mean(value[1] for value in rows) for rows in groups.values()]
    return {"candidate_anchor_pairs": len(successes), "candidate_records": len(groups),
            "candidate_group": ["source", "method", "round", "variant"],
            "pair_weighted_success": statistics.mean(successes) if successes else None,
            "candidate_weighted_mean_success": statistics.mean(candidate_successes) if candidate_successes else None,
            "pair_weighted_uniform_ge_anchor": statistics.mean(probabilities) if probabilities else None,
            "candidate_weighted_mean_uniform_ge_anchor": statistics.mean(candidate_probabilities) if candidate_probabilities else None}


def binary_boundary_reason(values):
    values = list(values)
    if not values:
        return "no_binary_observations"
    if any(value not in (0, 1) for value in values):
        raise ValueError("Binary observations must be 0 or 1")
    if all(value == 1 for value in values):
        return "all_success_empirical_bootstrap_degenerate"
    if all(value == 0 for value in values):
        return "all_failure_empirical_bootstrap_degenerate"
    return None


def binary_sample_description(values, planned=None):
    values = list(values)
    reason = binary_boundary_reason(values)
    if planned is None:
        planned = len(values)
    if type(planned) is not int or planned < len(values):
        raise ValueError("The planned count cannot be smaller than the observed count")
    return {"completed": int(sum(values)), "observed": len(values), "planned": planned,
            "missing": planned - len(values), "observed_proportion": sum(values) / len(values) if values else None,
            "population_probability_interval": None, "boundary_reason": reason}


def summarize_usage(usages):
    totals = []
    for usage in usages:
        total = None
        if isinstance(usage, Mapping):
            total = usage.get("total_tokens")
            incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
            outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
            if total is None and incoming is not None and outgoing is not None:
                total = incoming + outgoing
            if total is not None and (type(total) is not int or total < 0):
                raise ValueError("Reported token totals must be nonnegative integers")
        totals.append(total)
    known = [value for value in totals if value is not None]
    complete = len(known) == len(totals)
    return {"request_count": len(totals), "reported_tokens_sum": sum(known),
            "missing_usage_calls": len(totals) - len(known), "total_tokens": sum(known) if complete else None,
            "mean_tokens": statistics.mean(known) if known and complete else None,
            "median_tokens": statistics.median(known) if known and complete else None,
            "complete_usage": complete}
