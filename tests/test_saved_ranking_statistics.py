"""Mathematical regression checks for offline reassessment of saved rankings."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/reassessment_20261114/recompute_saved_statistics.py"
SPEC = spec_from_file_location("saved_statistics_reassessment", SOURCE)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_best_first_perfect_order_has_positive_high_score_correlation():
    measured = {"A": 4.0, "B": 2.0, "C": 1.0}
    result = MODULE.ranking_metrics(["A", "B", "C"], list(measured), measured, k=1)
    assert result["spearman_position_vs_fitness"] == pytest.approx(-1.0)
    assert result["spearman_higher_score_vs_fitness"] == pytest.approx(1.0)
    assert result["top_k_overlap"] == 1


def test_reversed_order_has_negative_high_score_correlation():
    measured = {"A": 4.0, "B": 2.0, "C": 1.0}
    result = MODULE.ranking_metrics(["C", "B", "A"], list(measured), measured, k=1)
    assert result["spearman_higher_score_vs_fitness"] == pytest.approx(-1.0)
    assert result["top_k_overlap"] == 0


def test_truth_tie_breaking_is_independent_of_model_position():
    measured = {"A": 3.0, "B": 2.0, "C": 2.0, "D": 0.0}
    result = MODULE.ranking_metrics(["A", "C", "B", "D"], list(measured), measured, k=2)
    assert result["top_k_overlap"] == 1
    assert result["spearman_higher_score_vs_fitness"] > 0


@pytest.mark.parametrize("ranking", [["A", "A", "C"], ["A", "B"], ["A", "B", "X"]])
def test_ranking_must_be_complete_unique_and_from_supplied_batch(ranking):
    with pytest.raises(ValueError, match="exact permutation"):
        MODULE.ranking_metrics(ranking, ["A", "B", "C"], {"A": 3, "B": 2, "C": 1})


@pytest.mark.parametrize("fitness", [{"A": 0, "B": 0}, {"A": np.nan, "B": 1}])
def test_undefined_correlations_are_rejected(fitness):
    with pytest.raises(ValueError, match="finite, nonconstant"):
        MODULE.ranking_metrics(["A", "B"], ["A", "B"], fitness)


def test_near_zero_observation_does_not_assert_equivalence_or_causality():
    result = MODULE.paired_description([0.184640496389, 0.120577837476, -0.328965645995])
    assert result["observed_mean_difference"] == pytest.approx(-0.007915770710)
    assert result["t_sensitivity_only"]["interval95"][0] < 0 < result["t_sensitivity_only"]["interval95"][1]
    assert result["equivalence_test_performed"] is False
    assert result["causal_independent_feedback_effect_identified"] is False


def test_three_positive_states_do_not_supply_small_exact_two_sided_p():
    result = MODULE.paired_description([1, 2, 3])
    assert result["sign_flip_sensitivity_only"]["assignments"] == 8
    assert result["sign_flip_sensitivity_only"]["two_sided_p"] == pytest.approx(0.25)


def test_paired_description_requires_multiple_finite_states():
    with pytest.raises(ValueError):
        MODULE.paired_description([1])
    with pytest.raises(ValueError):
        MODULE.paired_description([1, np.nan])


def test_sklearn_gaussian_process_can_fit_a_small_cpu_example():
    from sklearn.gaussian_process import GaussianProcessRegressor
    x = np.arange(6, dtype=float).reshape(-1, 1)
    model = GaussianProcessRegressor(optimizer=None, alpha=0.01).fit(x, np.sin(x[:, 0]))
    mean, deviation = model.predict([[1.5]], return_std=True)
    assert np.isfinite(mean).all() and np.isfinite(deviation).all()


def test_historical_frozen_workflow_entry_never_constructs_model_client(monkeypatch, tmp_path):
    import runpy
    import protevo.agent
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/critique-followup-20261113/no_feedback.py'
    historical = tmp_path / 'project/results/critique-followup-20261113/no_feedback.py'
    historical.parent.mkdir(parents=True)
    historical.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    records = historical.parent / 'no_feedback/recommendations.csv'
    records.parent.mkdir()
    records.write_bytes(b'preserved experimental record\n')
    before = records.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical entry attempted a new model connection')
    monkeypatch.setattr(protevo.agent, 'OpenAIProposer', forbidden)
    with pytest.raises(SystemExit, match='does not dispatch requests'):
        runpy.run_path(str(historical), run_name='__main__')
    assert records.read_bytes() == before
