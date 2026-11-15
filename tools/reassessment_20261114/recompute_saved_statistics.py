"""Recompute saved GB1 statistics offline without modifying experimental records."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def ranking_metrics(ranked, expected, fitness, k=10):
    """The first position is best; score=-position uses higher-is-better units."""
    ranked, expected = list(ranked), list(expected)
    if not ranked or len(ranked) != len(expected) or len(set(ranked)) != len(ranked) or set(ranked) != set(expected):
        raise ValueError("Returned ranking must be an exact permutation of the supplied variants")
    actual = np.asarray([fitness[v] for v in ranked], dtype=float)
    if not np.isfinite(actual).all() or len(actual) < 2 or np.ptp(actual) == 0:
        raise ValueError("Correlation requires finite, nonconstant measured fitness")
    position = np.arange(len(ranked), dtype=float)
    score = -position
    raw = stats.spearmanr(position, actual)
    oriented = stats.spearmanr(score, actual)
    kk = min(k, len(ranked))
    if kk < 1:
        raise ValueError("k must be positive")
    true_top = set(sorted(expected, key=lambda v: (-float(fitness[v]), v))[:kk])
    return {
        "n": len(ranked),
        "spearman_position_vs_fitness": float(raw.statistic),
        "spearman_higher_score_vs_fitness": float(oriented.statistic),
        "kendall_position_vs_fitness": float(stats.kendalltau(position, actual).statistic),
        "kendall_higher_score_vs_fitness": float(stats.kendalltau(score, actual).statistic),
        "asymptotic_two_sided_p": float(oriented.pvalue),
        "p_method": "SciPy approximation; exploratory; small tied samples require permutation calibration",
        "top_k": kk,
        "top_k_overlap": len(set(ranked[:kk]) & true_top),
        "true_top_ties": "variant ID ascending, independent of model ordering",
    }


def paired_description(differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite state-level differences are required")
    if len(values) > 15:
        raise ValueError("Exact sign enumeration is limited to 15 states")
    result = stats.ttest_1samp(values, 0.0)
    ci = result.confidence_interval(0.95)
    observed = abs(float(values.mean()))
    null = np.asarray([np.mean(values * signs) for signs in itertools.product((-1, 1), repeat=len(values))])
    return {
        "n_states": len(values), "differences": values.tolist(),
        "observed_mean_difference": float(values.mean()),
        "state_sample_sd": float(values.std(ddof=1)),
        "state_minimum": float(values.min()), "state_maximum": float(values.max()),
        "t_sensitivity_only": {
            "statistic": float(result.statistic), "degrees_of_freedom": len(values) - 1,
            "two_sided_p": float(result.pvalue), "interval95": [float(ci.low), float(ci.high)],
            "assumptions": "independent approximately normal state contrasts; this diagnostic does not verify these assumptions",
        },
        "sign_flip_sensitivity_only": {
            "assignments": len(null),
            "two_sided_p": float(np.mean(np.abs(null) >= observed - 1e-12)),
            "assumptions": "state contrast signs exchangeable under the null; no claim of randomized treatment assignment",
        },
        "equivalence_test_performed": False,
        "causal_independent_feedback_effect_identified": False,
    }


def load_s1(path):
    frame = pd.read_excel(path, sheet_name="Supplementary Dataset 1")
    out = pd.DataFrame({"variant": frame["Variants"].astype(str).str.strip().str.upper(),
                        "fitness": pd.to_numeric(frame["Fitness"], errors="raise")})
    if len(out) != 149361 or out.variant.nunique() != 149361 or not out.variant.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}").all():
        raise ValueError("Unexpected measured S1 table")
    if not np.isfinite(out.fitness).all() or (out.fitness < 0).any():
        raise ValueError("Invalid measured fitness")
    return out.sort_values("variant").reset_index(drop=True)


def recompute(root, s1_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing statistics are not overwritten")
    source = root / "results/critique-followup-20261113"
    table = load_s1(s1_path)
    fitness = table.set_index("variant").fitness.to_dict()
    rng = np.random.default_rng(20261113)
    ranked_rows, all_variants = [], []
    for index in range(1, 4):
        expected = table.sample(n=50, random_state=int(rng.integers(1 << 31))).variant.tolist()
        ranked = json.loads((source / f"zero_context/rank_batch{index}.json").read_text(encoding="utf-8"))["ranked_variants"]
        ranked_rows.append({"batch": index, **ranking_metrics(ranked, expected, fitness)})
        all_variants.extend(ranked)

    feedback = pd.read_csv(root / "reports/validation-stage/v3_recommendations.csv")
    feedback = feedback.loc[feedback.method == "knowledge"]
    frozen = pd.read_csv(source / "no_feedback/recommendations.csv")
    if len(feedback) != 90 or len(frozen) != 90:
        raise ValueError("Both saved conditions must have 90 selected records")
    pairs = []
    for seed in (11, 22, 33):
        left, right = feedback.loc[feedback.seed == seed], frozen.loc[frozen.seed == seed]
        if len(left) != 30 or len(right) != 30 or left.variant.nunique() != 30 or right.variant.nunique() != 30:
            raise ValueError("Each trajectory must have 30 unique selections")
        for arm in (left, right):
            if not np.allclose(arm.actual_fitness, arm.variant.map(fitness), rtol=1e-12, atol=1e-12):
                raise ValueError("Selected labels differ from measured S1")
        lm, rm = float(left.actual_fitness.mean()), float(right.actual_fitness.mean())
        pairs.append({"seed": seed, "adaptive_mean": lm, "frozen_mean": rm, "adaptive_minus_frozen": lm-rm})
    detail = pd.read_csv(root / "results/feedback-ablation/paired_results.csv")
    display = paired_description(detail.full_minus_withheld_mean_fitness)
    overlap = pd.read_csv(source / "first_answer_overlap.csv")
    anchors = pd.read_csv(source / "anchor_transfer.csv")
    candidate = anchors.groupby(["source", "method", "round", "variant"], sort=True).success.mean()
    report = {
        "study": "saved-data statistical reassessment", "model_requests": 0,
        "s1": {"rows": len(table), "unique_variants": table.variant.nunique(), "zeros": int((table.fitness == 0).sum())},
        "ranking": ranked_rows,
        "ranking_total_entries": len(all_variants), "ranking_distinct_variants": len(set(all_variants)),
        "adaptive_vs_frozen": paired_description([r["adaptive_minus_frozen"] for r in pairs]),
        "detailed_display": {**display, "estimand": "detailed display and extra input length, conditional on updated known data and predictor"},
        "first_answer": {"method_round_records": len(overlap), "candidate_entries": int(overlap.first_answer_n.sum()),
                         "definition": "first response before repair in every included round, not only experiment round 1"},
        "anchors": {"candidate_anchor_pairs": len(anchors), "distinct_candidate_records": len(candidate),
                    "pair_weighted_success": float(anchors.success.mean()),
                    "candidate_weighted_mean_success": float(candidate.mean()),
                    "interpretation": "descriptive selected-and-anchor-citing subset; no independent anchor-treatment effect"},
        "completion": {"observed_completed": 15, "scheduled": 15,
                       "population_success_probability_interval": None,
                       "reason": "all-success empirical bootstrap is degenerate; do not infer population certainty"},
    }
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(ranked_rows).to_csv(output / "ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(output / "trajectory_contrasts.csv", index=False, encoding="utf-8-sig")
    (output / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    s1 = args.s1 or args.project_root / "data/raw/elife-16965-supp1-v4.xlsx"
    report = recompute(args.project_root, s1, args.output)
    print(json.dumps({"output": str(args.output), "model_requests": 0,
                      "ranking_rho": [r["spearman_higher_score_vs_fitness"] for r in report["ranking"]],
                      "observed_adaptive_minus_frozen": report["adaptive_vs_frozen"]["observed_mean_difference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
