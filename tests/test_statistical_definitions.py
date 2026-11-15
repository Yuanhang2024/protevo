"""Targeted saved-data statistics tests; no model transport or real credentials are used."""
from itertools import product
import importlib.util
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import pytest

from protevo.statistical_definitions import (DEFINITION_VERSION, anchor_weighted_summary, binary_sample_description,
                                        ranking_metrics, summarize_usage, threshold_prevalence)

PROJECT = Path(__file__).resolve().parents[1]
FOLLOWUP = PROJECT / "results/critique-followup-20261113"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analyze = module("corrected_followup_statistics", FOLLOWUP / "analyze_stats.py")
zero = module("corrected_zero_context", FOLLOWUP / "zero_context.py")
calibration = module("corrected_calibration_statistics", PROJECT / "tools/calibration-report/statistics_report.py")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("Statistical correction tests cannot access a network or model service")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def test_definition_matches_frozen_public_config():
    assert json.loads((PROJECT / "configs/statistical-definitions-v2.json").read_text(encoding="utf-8"))["version"] == DEFINITION_VERSION


def test_best_first_raw_position_and_high_score_have_opposite_signs():
    expected = ["AAAA", "AAAC", "AAAD"]
    fitness = dict(zip(expected, [3., 2., 1.]))
    correct = ranking_metrics(expected, expected, fitness, 1)
    assert correct["spearman_position_vs_fitness"] == -1
    assert correct["spearman_higher_score_vs_fitness"] == 1
    assert correct["kendall_position_vs_fitness"] == -1 and correct["kendall_higher_score_vs_fitness"] == 1
    reversed_order = ranking_metrics(expected[::-1], expected, fitness, 1)
    assert reversed_order["spearman_higher_score_vs_fitness"] == -1 and reversed_order["top_k_overlap"] == 0
    legacy = zero.batch_report(1, expected, expected, fitness)
    assert legacy["spearman_model_rank_vs_actual"] == -1 and legacy["kendall_tau"] == -1


@pytest.mark.parametrize("ranked, expected", [(["AAAA", "AAAA"], ["AAAA", "AAAC"]),
    (["AAAA"], ["AAAA", "AAAC"]), (["AAAA", "AAAD"], ["AAAA", "AAAC"]),
    (["AAAA", "AAAC"], ["AAAA", "AAAA"]), ([None], ["AAAA"])])
def test_rankings_require_an_exact_unique_permutation(ranked, expected):
    with pytest.raises(ValueError):
        ranking_metrics(ranked, expected, {"AAAA": 1, "AAAC": 2, "AAAD": 3})


def test_true_top_ties_do_not_depend_on_model_order_and_undefined_correlations_are_null():
    expected = ["AAAA", "AAAC", "AAAD"]
    result = ranking_metrics(expected[::-1], expected, dict.fromkeys(expected, 1.), 1)
    assert result["true_top_variants"] == ["AAAA"] and result["top_k_overlap"] == 0
    assert result["spearman_position_vs_fitness"] is None and result["spearman_higher_score_vs_fitness"] is None
    assert ranking_metrics(["AAAA"], ["AAAA"], {"AAAA": 0.})["correlation_undefined_reason"]
    json.dumps(result, allow_nan=False)


def test_uniform_reference_uses_actual_threshold_prevalence_including_ties():
    value = threshold_prevalence([1, 1, 1, 2], .5)
    assert value["threshold"] == 1 and value["qualifying_count"] == 4
    assert value["uniform_ge_threshold_probability"] == 1
    fit = pd.DataFrame({"variant": ["AAAA", "AAAC", "AAAD", "AAAE"], "fitness": [0., 2., 2., 2.]})
    rows, summary = analyze.overlap_stats(fit, [("test", [{"method": "llm", "round": 2, "variants": ["AAAC"]}])])
    assert len(rows) == 1 and summary["llm"]["method_round_records"] == 1
    assert summary["llm"]["uniform_expectation_top1pct"] == .75


def test_first_response_records_include_all_rounds_and_unparsed_first_responses(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [{"kind": "llm_response", "method": "llm", "round": 1, "attempt": 1, "response": {"text": "invalid"}},
              {"kind": "llm_response", "method": "llm", "round": 2, "attempt": 1,
               "response": {"text": json.dumps({"proposals": [{"variant": "AAAA"}]})}},
              {"kind": "llm_response", "method": "llm", "round": 2, "attempt": 2,
               "response": {"text": json.dumps({"proposals": [{"variant": "AAAC"}]})}}]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    rows = analyze.first_answer_variants(path)
    assert [row["round"] for row in rows] == [1, 2] and rows[0]["raw_candidate_entries"] == 0
    assert rows[1]["variants"] == ["AAAA"]


def test_pair_and_candidate_weighting_are_distinct():
    pairs = [{"source": "s", "method": "llm", "round": 1, "variant": candidate, "anchor": anchor,
              "success": success, "random_ge_anchor": probability}
             for candidate, anchor, success, probability in (("AAAA", "AAAC", 0, .1), ("AAAA", "AAAD", 1, .3), ("AAAE", "AAAF", 1, .9))]
    summary = anchor_weighted_summary(pairs)
    assert summary["candidate_anchor_pairs"] == 3 and summary["candidate_records"] == 2
    assert summary["pair_weighted_success"] == pytest.approx(2 / 3)
    assert summary["candidate_weighted_mean_success"] == .75
    assert summary["candidate_weighted_mean_uniform_ge_anchor"] == pytest.approx(.55)
    with pytest.raises(ValueError):
        anchor_weighted_summary([pairs[0], pairs[0]])


def test_known_is_updated_only_after_the_complete_round_and_scopes_are_separate(tmp_path):
    pd.DataFrame({"variant": ["AAAA"]}).to_csv(tmp_path / "train.csv", index=False)
    fitness = pd.DataFrame({"variant": ["AAAA", "AAAC", "AAAD", "AAAE"], "fitness": [1., 2., 3., 4.]})
    recommendations = [{"method": "llm", "round": number, "rank": rank, "variant": variant, "hypothesis": anchor}
                       for number, rank, variant, anchor in ((1, 1, "AAAC", "AAAA"), (1, 2, "AAAD", "AAAC"), (2, 1, "AAAE", "AAAC"))]
    visibility = {"s": {("llm", 1): {"AAAA"}, ("llm", 2): set()}}
    pairs, summary = analyze.anchor_stats(fitness, [("s", recommendations, tmp_path)], visibility)
    assert list(zip(pairs.variant, pairs.anchor)) == [("AAAC", "AAAA"), ("AAAE", "AAAC")]
    assert list(pairs.prompt_visible) == [True, False]
    assert summary["ALL"]["eligibility_scope"] == "measurement_history_eligible"
    visible = analyze.summarize_anchors(pairs.loc[pairs.prompt_visible.eq(True)], "prompt_visible")
    assert visible["ALL"]["candidate_anchor_pairs"] == 1


def test_prompt_visibility_requires_explicit_measurements_not_observed_or_example_ids(tmp_path):
    context = {"known_data": {"top_variants": [{"variant": "AAAA", "fitness": 0}],
                             "low_fitness_examples": [{"variant": "AAAC", "fitness": 1}],
                             "near_median_examples": [{"variant": "AAAD", "fitness": 2}]},
               "recent_round_feedback": [{"variant": "AAAE", "actual_fitness": 3}],
               "observed_variants": ["CCCC"], "candidate_examples": ["DDDD"], "irrelevant": "synthetic-not-exported"}
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"kind": "llm_request", "attempt": 1, "method": "llm", "round": 2, "context": context}), encoding="utf-8")
    result = analyze.first_prompt_visibility(path)
    assert result["llm", 2] == {"AAAA", "AAAC", "AAAD", "AAAE"}
    assert "synthetic-not-exported" not in str(result)


@pytest.mark.parametrize("value, reason", [(1, "all_success"), (0, "all_failure")])
def test_binary_boundary_interval_retains_counts_but_not_probability_certainty(value, reason):
    rows = [{"seed": seed, "complete": bool(value), "completion_indicator": value} for seed in (1, 2, 3) for _ in range(3)]
    result = calibration.interval(rows, "completion_indicator", [1, 2, 3], specification())
    assert result["estimate"] == value and result["completed"] == 9 * value and result["planned"] == 9
    assert result["lower"] is result["upper"] is None and reason in result["reason"]


def specification():
    return {"seed": 20261113, "replicates": 1000, "confidenceLevel": .95,
            "outerUnit": "paired data split", "innerUnit": "independent trajectory within model and split"}


@pytest.mark.parametrize("selected, reference, estimate", [(1, 1, 0), (1, 0, 1), (0, 0, 0)])
def test_binary_boundary_contrast_has_no_degenerate_population_interval(selected, reference, estimate):
    rows = [{"seed": seed, "arm": arm, "complete": bool(value), "completion_indicator": value}
            for seed in (1, 2, 3) for arm, value in (("selected", selected), ("reference", reference)) for _ in range(3)]
    result = calibration.paired_difference(rows, "completion_indicator", {"dataSeeds": [1, 2, 3], "bootstrap": specification()})
    assert result["estimate"] == estimate and result["lower"] is result["upper"] is None and result["reason"]
    assert result["binary_arm_counts"]["selected"]["completed"] == 9 * selected


def test_degenerate_binary_difference_is_detected_even_if_each_arm_is_globally_mixed():
    rows = [{"seed": seed, "arm": arm, "complete": True, "completion_indicator": seed - 1}
            for seed in (1, 2) for arm in ("selected", "reference") for _ in range(3)]
    result = calibration.paired_difference(rows, "completion_indicator", {"dataSeeds": [1, 2], "bootstrap": specification()}, complete_only=False)
    assert result["estimate"] == 0 and result["lower"] is None and result["reason"] == "degenerate_binary_difference_empirical_bootstrap"


def test_quality_intervals_reproduce_original_arithmetic_and_resampling():
    original_path = PROJECT / "results/metric-correction-20261114/originals/tools/calibration-report/statistics_report.py"
    original = module("original_calibration_statistics", original_path)
    rows = [{"seed": seed, "arm": arm, "complete": True, "mean_fitness": seed / 10 + repeat / 5 + (seed * repeat / 7 + .3 if arm == "selected" else 0)}
            for seed in (1, 2, 3) for arm in ("selected", "reference") for repeat in range(3)]
    settings = {"dataSeeds": [1, 2, 3], "bootstrap": specification()}
    assert calibration.paired_difference(rows, "mean_fitness", settings) == original.paired_difference(rows, "mean_fitness", settings)
    old = original.interval(rows, "mean_fitness", settings["dataSeeds"], settings["bootstrap"])
    new = calibration.interval(rows, "mean_fitness", settings["dataSeeds"], settings["bootstrap"])
    assert all(new[key] == value for key, value in old.items())


def test_interval_recalculation_writes_new_version_and_preserves_saved_tables(tmp_path):
    settings = {"dataSeeds": [1, 2, 3], "bootstrap": specification()}
    rows = [{"seed": seed, "arm": arm, "complete": True, "completion_indicator": 1,
             "mean_fitness": seed + repeat / 7 + (1 if arm == "selected" else 0),
             "max_fitness": seed + repeat / 5 + (2 if arm == "selected" else 0)}
            for seed in settings["dataSeeds"] for arm in ("selected", "reference") for repeat in range(3)]
    table = tmp_path / "trajectories.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    restored = pd.read_csv(table, float_precision="round_trip").to_dict("records")
    reference = {"confirmation_intervals": calibration.confirmation(restored, settings)}
    plan, summary = tmp_path / "plan.json", tmp_path / "old-summary.json"
    plan.write_text(json.dumps({"phase3": settings}), encoding="utf-8")
    summary.write_text(json.dumps(reference), encoding="utf-8")
    before = table.read_text(encoding="utf-8")
    output = tmp_path / "new-intervals"
    result = calibration.recompute_confirmation(table, plan, output, summary)
    assert result["quality_intervals_exactly_unchanged"] and result["model_requests"] == 0
    assert result["completion_by_arm"]["selected"]["lower"] is None
    assert result["definition_version"] == DEFINITION_VERSION and table.read_text(encoding="utf-8") == before
    with pytest.raises(ValueError):
        calibration.recompute_confirmation(table, plan, output, summary)


def test_missing_usage_is_unknown_for_total_mean_and_median():
    summary = summarize_usage([None, {"total_tokens": 30}, {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 100}])
    assert summary["reported_tokens_sum"] == 60 and summary["missing_usage_calls"] == 1
    assert summary["total_tokens"] is summary["mean_tokens"] is summary["median_tokens"] is None
    rows = [{"seed": 1, "complete": True, "tokens": 30}, {"seed": 1, "complete": True, "tokens": None}]
    interval = calibration.interval(rows, "tokens", [1], specification())
    assert interval["estimate"] is None and interval["reason"] == "missing_reported_resource_usage"
    assert summarize_usage([{"total_tokens": 0}])["total_tokens"] == 0


def test_zero_context_default_and_unapproved_api_never_load_a_client(tmp_path, monkeypatch):
    monkeypatch.setattr(zero, "load_client", lambda **_: pytest.fail("No API client may be loaded"))
    with pytest.raises(SystemExit) as caught:
        zero.main([])
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        zero.main(["run-api", "--output", str(tmp_path / "new")])
    with pytest.raises(PermissionError):
        zero.call(None, "test", "test", "test", {}, [], "test", tmp_path)
    with pytest.raises(ValueError):
        zero.run_api(tmp_path, allow_api=True)
    (tmp_path / "saved.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        zero.call(None, "test", "test", "test", {}, [], "saved", tmp_path, allow_api=True)


def test_saved_zero_context_analysis_uses_no_client_and_never_overwrites(tmp_path, monkeypatch):
    table = pd.DataFrame({"variant": ["".join(row) for row in product("ACDE", repeat=4)], "fitness": np.arange(256, dtype=float)})
    source = tmp_path / "saved"
    source.mkdir()
    fitness = table.set_index("variant").fitness.to_dict()
    for number, expected in enumerate(zero.samples(table), 1):
        ranked = sorted(expected, key=lambda variant: (-fitness[variant], variant))
        (source / f"rank_batch{number}.json").write_text(json.dumps({"ranked_variants": ranked}), encoding="utf-8")
    (source / "free_recall.json").write_text(json.dumps({"top_variants": list(table.tail(10).variant)}), encoding="utf-8")
    (source / "usage.json").write_text(json.dumps([{"usage": None}, {"usage": {"total_tokens": 30}}]), encoding="utf-8")
    monkeypatch.setattr(zero, "load_client", lambda **_: pytest.fail("Saved analysis cannot load an API client"))
    output = tmp_path / "new-statistics"
    result = zero.analyze_saved(source, output, table)
    assert result["model_requests"] == 0 and all(row["spearman_position_vs_fitness"] == pytest.approx(-1) for row in result["batches"])
    assert all(row["spearman_higher_score_vs_fitness"] == pytest.approx(1) for row in result["batches"])
    assert result["usage"]["total_tokens"] is None
    before = (output / "summary.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        zero.analyze_saved(source, output, table)
    assert (output / "summary.json").read_text(encoding="utf-8") == before
