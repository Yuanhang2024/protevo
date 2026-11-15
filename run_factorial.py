"""2x2 factorial experiment: predictor (RF+one-hot vs ESM-2 L33) x agent mode (prompt vs tool-calling).

Groups (3 seeds each: 11, 22, 33; 3 rounds, k=10, score budget 40/round):
  G1 = prompt architecture + RF one-hot  -> reused from results/seed-{s}/run (method "llm"), summarized only.
  G2 = prompt architecture + ESM-2 L33   -> protevo.experiment.run_experiment with patched fit_model/one_hot.
  G3 = tool-calling agent (agent_v2) + RF one-hot (per-seed spec identical to G1).
  G4 = tool-calling agent (agent_v2) + ESM-2 L33.

Predictor policy (leakage control): results/predictor_v2 joblib bundles were fit on seed-11 train
only; reusing them on seeds 22/33 would leak ~1.9k true labels into those candidate pools. Hyper-
parameters are therefore read from the bundles (ESM-2: L33 pooled + Ridge alpha=1.0; RF: each
seed's frozen selection.json spec, identical to G1), and predictors are REFIT at every round on the
method's own known set (train split + accumulated feedback), mirroring the G1 architecture.

Outputs: results/v2_experiments/{G1_summary.json, G2/, G3/, G4/, factorial_summary.csv}
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import protevo.experiment as experiment_module
from protevo.agent import OpenAIProposer
from protevo.agent_v2 import run_agent_round
from protevo.data import HiddenOracle, PublicDataset, describe_variant, one_hot
from protevo.experiment import finalize, run_experiment
from protevo.models import fit_model as fit_rf_model
from protevo.models import load_selection, regression_metrics
from protevo.schema import ExperimentConfig, variant_allowed
from protevo.storage import read_json, write_json
from protevo.tools import AgentBudget

SEEDS = (11, 22, 33)
V2_DIR = ROOT / "results" / "v2_experiments"
EMB_DIR = ROOT / "data" / "esm2_embeddings"
PRED_DIR = ROOT / "results" / "predictor_v2"
PROMPT_PATH = ROOT / "prompts" / "scientist_v2.txt"
BUDGET = {"max_api_calls": 15, "max_total_tokens": 50000, "max_score_calls": 40}
GROUP_INFO = {
    "G1": {"predictor": "rf_onehot", "agent": "prompt"},
    "G2": {"predictor": "esm2_l33_ridge", "agent": "prompt"},
    "G3": {"predictor": "rf_onehot", "agent": "tool_calling"},
    "G4": {"predictor": "esm2_l33_ridge", "agent": "tool_calling"},
}


class EsmEmbeddings:
    """Row-gather over the frozen ESM-2 mmap; rows follow data/processed/all_variants.csv order."""

    def __init__(self, layer_file):
        all_variants = pd.read_csv(ROOT / "data" / "processed" / "all_variants.csv")
        if not all_variants.variant.is_unique:
            raise ValueError("all_variants.csv must contain unique variants")
        self.row_of = {v: i for i, v in enumerate(all_variants.variant)}
        self.mmap = np.load(EMB_DIR / layer_file, mmap_mode="r")
        self.layer_file = layer_file

    def rows(self, variants):
        variants = list(variants)
        index = np.fromiter((self.row_of[v] for v in variants), dtype=np.int64, count=len(variants))
        return np.asarray(self.mmap[index])


def load_esm_config():
    """Hyperparameters from the frozen predictor_v2 bundle (refit locally per seed/round)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bundle was pickled under an older sklearn; only read params
        bundle = joblib.load(PRED_DIR / "esm2_best_model.joblib")
    return {"layer": bundle["layer"], "layer_file": bundle["layer_file"],
            "alpha": float(bundle["estimator"].alpha), "model_name": bundle["model_name"]}


def esm_fit(known, embeddings, alpha):
    model = Ridge(alpha=alpha)
    model.fit(embeddings.rows(known.variant), known.fitness.to_numpy(dtype=float))
    return model


@contextmanager
def patched_esm_predictor(embeddings, alpha):
    """Redirect protevo.experiment's fit_model/one_hot to ESM-2 embeddings for the G2 run."""
    original_fit, original_one_hot = experiment_module.fit_model, experiment_module.one_hot

    def esm_fit_model(spec, config, known):
        return esm_fit(known, embeddings, alpha)

    experiment_module.fit_model = esm_fit_model
    experiment_module.one_hot = lambda variants: embeddings.rows(variants)
    try:
        yield
    finally:
        experiment_module.fit_model, experiment_module.one_hot = original_fit, original_one_hot


def seed_context(seed):
    dataset = PublicDataset(ROOT / "data" / "processed" / f"seed-{seed}")
    config = ExperimentConfig.model_validate(dataset.manifest["config"])
    selection = load_selection(ROOT / "results" / f"seed-{seed}" / "selection" / "selection.json",
                               dataset, config)
    return dataset, config, selection


# ---------------------------------------------------------------- G2: old architecture + ESM-2
def run_g2(seed):
    dataset, config, selection = seed_context(seed)
    output = V2_DIR / "G2" / f"seed-{seed}" / "run"
    esm_cfg = load_esm_config()
    embeddings = EsmEmbeddings(esm_cfg["layer_file"])
    proposer = OpenAIProposer(config, prompt_profile="v1")
    try:
        with patched_esm_predictor(embeddings, esm_cfg["alpha"]):
            state = run_experiment(dataset, selection, config, output, methods=["llm"],
                                   proposer=proposer, prompt_profile="v1")
            if state["methods"]["llm"]["status"] == "completed" and not (output / "final_test_metrics.json").exists():
                finalize(dataset, selection, config, output, ["llm"])
    finally:
        proposer.close()
    return state["methods"]["llm"]["status"]


# ------------------------------------------------- G3/G4: tool-calling agent (agent_v2)
def known_for_state(dataset, state):
    observations = []
    for record in state["rounds"]:
        for row in record["recommendations"]:
            observations.append({**describe_variant(row["variant"], dataset.reference),
                                 "fitness": row["actual_fitness"]})
    if not observations:
        return dataset.train.copy()
    return pd.concat([dataset.train, pd.DataFrame(observations)], ignore_index=True)


def fit_group_predictor(kind, known, selection, config, embeddings, esm_cfg):
    if kind == "rf_onehot":
        return fit_rf_model(selection["spec"], config, known)
    return esm_fit(known, embeddings, esm_cfg["alpha"])


def predict_fn_for(kind, model, embeddings):
    if kind == "rf_onehot":
        return lambda sequences: [float(v) for v in model.predict(one_hot(sequences))]
    return lambda sequences: [float(v) for v in model.predict(embeddings.rows(sequences))]


def tool_round_metrics(record, feedback, known, threshold):
    submission = record["submission"]
    sequences, predicted = submission["sequences"], submission["predicted_fitness"]
    actual = np.array([feedback[v] for v in sequences], dtype=float)
    prior_best = float(known.fitness.max())
    scored = [i for i, p in enumerate(predicted) if p is not None]
    if scored:
        pm = regression_metrics(actual[scored], [predicted[i] for i in scored], len(scored))
        pm.pop("top_k")
        pm.pop("top_k_hit_rate")
    else:
        pm = {"n": 0, "spearman": None, "pearson": None, "mse": None}
    return {"mean_fitness": float(actual.mean()), "max_fitness": float(actual.max()),
            "cumulative_best": max(prior_best, float(actual.max())), "previous_best": prior_best,
            "improvement_over_previous_best": max(0.0, float(actual.max()) - prior_best),
            "new_high_fitness_count": int((actual >= threshold).sum()),
            "high_fitness_threshold": threshold, "assays": len(sequences),
            "scored_recommendations": len(scored),
            "prediction_metrics_on_selected": pm,
            "status": record["status"], "auto_submitted": submission["auto"],
            "stop_reason": submission["stop_reason"], "api_calls_used": record["api_calls_used"],
            "total_tokens_used": record["total_tokens_used"],
            "score_calls_used": record["score_calls_used"],
            "tool_call_sequence": record["tool_call_sequence"]}


def run_tool_group(group, seed):
    kind = GROUP_INFO[group]["predictor"]
    dataset, config, selection = seed_context(seed)
    output = V2_DIR / group / f"seed-{seed}"
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    threshold = float(dataset.train.fitness.quantile(config.high_fitness_quantile))
    esm_cfg = load_esm_config() if kind != "rf_onehot" else None
    embeddings = EsmEmbeddings(esm_cfg["layer_file"]) if esm_cfg else None

    state_path = output / "state.json"
    state = read_json(state_path) if state_path.exists() else {
        "format_version": 1, "group": group, "seed": seed, "predictor": kind,
        "agent_mode": "tool_calling", "config": config.model_dump(),
        "system_prompt_path": str(PROMPT_PATH),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "predictor_hyperparameters": {"source": "per-seed selection.json spec" if kind == "rf_onehot"
                                      else {"source": str(PRED_DIR / "esm2_best_model.joblib"), **esm_cfg},
                                      "refit": "every round on known (train + accumulated feedback)"},
        "budget": dict(BUDGET), "rounds": []}

    known = known_for_state(dataset, state)
    feedback_history = [record["feedback_entry"] for record in state["rounds"]]
    eligible = {v for v in set(dataset.candidates.variant) - set(known.variant) if variant_allowed(v, config)}
    for round_number in range(len(state["rounds"]) + 1, config.rounds + 1):
        model = fit_group_predictor(kind, known, selection, config, embeddings, esm_cfg)
        predict_fn = predict_fn_for(kind, model, embeddings)
        started = time.perf_counter()
        record, toolbox = run_agent_round(round_number, known, eligible, predict_fn, config,
                                          knowledge=None, system_prompt=system_prompt,
                                          feedback_history=feedback_history,
                                          log_path=output / "agent_log.jsonl",
                                          budget=AgentBudget(**BUDGET))
        submission = record["submission"]
        oracle = HiddenOracle(dataset.directory, set(dataset.candidates.variant))
        feedback = {row["variant"]: row["actual_fitness"] for row in oracle.evaluate(submission["sequences"])}
        rows = [{"rank": rank, **describe_variant(v, dataset.reference), "reason": reason,
                 "predicted_fitness": predicted, "actual_fitness": feedback[v]}
                for rank, (v, reason, predicted) in enumerate(zip(submission["sequences"],
                                                                  submission["reasons"],
                                                                  submission["predicted_fitness"]), 1)]
        round_entry = {"round": round_number, "status": record["status"], "record": record,
                       "recommendations": rows, "elapsed_seconds": round(time.perf_counter() - started, 1),
                       "metrics": tool_round_metrics(record, feedback, known, threshold),
                       "feedback_entry": {"round": round_number, "recommendations": [
                           {"variant": r["variant"], "reason": r["reason"],
                            "predicted_fitness": r["predicted_fitness"], "actual_fitness": r["actual_fitness"]}
                           for r in rows]}}
        state["rounds"].append(round_entry)
        write_json(state_path, state)
        print(f"    [{group} seed {seed}] round {round_number}: {record['status']} "
              f"max={round_entry['metrics']['max_fitness']:.4f} "
              f"best={round_entry['metrics']['cumulative_best']:.4f} "
              f"api={record['api_calls_used']} tokens={record['total_tokens_used']}", flush=True)
        known = known_for_state(dataset, state)
        feedback_history.append(round_entry["feedback_entry"])
        eligible -= set(submission["sequences"])

    if "final_test_metrics" not in state:
        test = pd.read_csv(dataset.directory / "hidden" / "test.csv")
        model = fit_group_predictor(kind, known, selection, config, embeddings, esm_cfg)
        features = one_hot(test.variant) if kind == "rf_onehot" else embeddings.rows(test.variant)
        state["final_test_metrics"] = {**regression_metrics(test.fitness, model.predict(features),
                                                            config.top_k_metric),
                                       "training_count": len(known)}
        write_json(state_path, state)
    return "completed"


# ------------------------------------------------------------------ summaries
def _usage_totals(usage_csv, method):
    frame = pd.read_csv(usage_csv)
    rows = frame[frame.method == method] if not frame.empty else frame
    tokens = 0
    for raw in rows.get("usage", pd.Series(dtype=str)):
        try:
            tokens += json.loads(raw).get("total_tokens") or 0
        except (ValueError, TypeError):
            pass
    return {"api_calls": int(len(rows)), "total_tokens": int(tokens)}


def seed_metrics_prompt(run_dir, method="llm"):
    metrics = pd.read_csv(run_dir / "round_metrics.csv")
    rows = metrics[metrics.method == method].sort_values("round")
    usage = _usage_totals(run_dir / "usage.csv", method)
    final = read_json(run_dir / "final_test_metrics.json")[method] if (run_dir / "final_test_metrics.json").exists() else None
    return {"rounds": [{"round": int(r.round), "mean_fitness": float(r.mean_fitness),
                        "max_fitness": float(r.max_fitness), "cumulative_best": float(r.cumulative_best),
                        "new_high_fitness_count": int(r.new_high_fitness_count)}
                       for r in rows.itertuples()],
            "final_cumulative_best": float(rows.cumulative_best.max()),
            "mean_round_fitness": float(rows.mean_fitness.mean()),
            "total_new_high_fitness_count": int(rows.new_high_fitness_count.sum()),
            "api_calls": usage["api_calls"], "total_tokens": usage["total_tokens"],
            "test_metrics": final, "status": "completed"}


def seed_metrics_tool(group, seed):
    state = read_json(V2_DIR / group / f"seed-{seed}" / "state.json")
    rounds = [{"round": r["round"], "mean_fitness": r["metrics"]["mean_fitness"],
               "max_fitness": r["metrics"]["max_fitness"],
               "cumulative_best": r["metrics"]["cumulative_best"],
               "new_high_fitness_count": r["metrics"]["new_high_fitness_count"]}
              for r in state["rounds"]]
    usage = {"api_calls": sum(r["metrics"]["api_calls_used"] for r in state["rounds"]),
             "total_tokens": sum(r["metrics"]["total_tokens_used"] for r in state["rounds"])}
    complete = len(state["rounds"]) >= state["config"]["rounds"]
    return {"rounds": rounds,
            "final_cumulative_best": max((r["cumulative_best"] for r in rounds), default=None),
            "mean_round_fitness": float(np.mean([r["mean_fitness"] for r in rounds])) if rounds else None,
            "total_new_high_fitness_count": sum(r["new_high_fitness_count"] for r in rounds),
            "api_calls": usage["api_calls"], "total_tokens": usage["total_tokens"],
            "test_metrics": state.get("final_test_metrics"),
            "status": "completed" if complete else "incomplete"}


def collect_seed_metrics(group, seed):
    if group == "G1":
        return seed_metrics_prompt(ROOT / "results" / f"seed-{seed}" / "run")
    if group == "G2":
        return seed_metrics_prompt(V2_DIR / "G2" / f"seed-{seed}" / "run")
    return seed_metrics_tool(group, seed)


def aggregate(per_seed):
    def stats(key):
        values = [per_seed[s][key] for s in per_seed if per_seed[s].get(key) is not None]
        if not values:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    def test_stat(key):
        values = [per_seed[s]["test_metrics"][key] for s in per_seed
                  if per_seed[s].get("test_metrics") and per_seed[s]["test_metrics"].get(key) is not None]
        if not values:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    return {"n_seeds": len(per_seed),
            "statuses": {s: per_seed[s]["status"] for s in per_seed},
            "final_cumulative_best": stats("final_cumulative_best"),
            "mean_round_fitness": stats("mean_round_fitness"),
            "total_new_high_fitness_count": stats("total_new_high_fitness_count"),
            "api_calls": stats("api_calls"), "total_tokens": stats("total_tokens"),
            "test_spearman": test_stat("spearman"), "test_mse": test_stat("mse"),
            "test_top10_hit_rate": test_stat("top_k_hit_rate")}


def summarize():
    available = {}
    for group in GROUP_INFO:
        per_seed = {}
        for seed in SEEDS:
            try:
                per_seed[seed] = collect_seed_metrics(group, seed)
            except (OSError, ValueError, KeyError) as exc:
                print(f"  [summary] {group} seed {seed} unavailable: {exc}")
        if per_seed:
            available[group] = {"info": GROUP_INFO[group], "per_seed": per_seed,
                                "aggregate": aggregate(per_seed)}
    write_json(V2_DIR / "G1_summary.json", available.get("G1", {"error": "G1 data not found"}))
    for group, payload in available.items():
        write_json(V2_DIR / f"{group}_summary.json", payload)

    effects = {}
    def effect(name, a, b, key="final_cumulative_best"):
        if a in available and b in available:
            ma, mb = available[a]["aggregate"][key]["mean"], available[b]["aggregate"][key]["mean"]
            effects[name] = None if ma is None or mb is None else round(ma - mb, 6)
    effect("esm2_effect_prompt_G2_minus_G1", "G2", "G1")
    effect("esm2_effect_tool_G4_minus_G3", "G4", "G3")
    effect("tool_effect_rf_G3_minus_G1", "G3", "G1")
    effect("tool_effect_esm_G4_minus_G2", "G4", "G2")
    effect("esm2_effect_prompt_mean_round_fitness", "G2", "G1", "mean_round_fitness")
    effect("esm2_effect_tool_mean_round_fitness", "G4", "G3", "mean_round_fitness")
    effect("tool_effect_rf_mean_round_fitness", "G3", "G1", "mean_round_fitness")
    effect("tool_effect_esm_mean_round_fitness", "G4", "G2", "mean_round_fitness")

    rows = []
    for group, payload in available.items():
        agg = payload["aggregate"]
        for seed, data in payload["per_seed"].items():
            test = data.get("test_metrics") or {}
            rows.append({"group": group, "level": "seed", "seed": seed,
                         "predictor": payload["info"]["predictor"], "agent_mode": payload["info"]["agent"],
                         "status": data["status"], "final_cumulative_best": data["final_cumulative_best"],
                         "mean_round_fitness": data["mean_round_fitness"],
                         "total_new_high_fitness_count": data["total_new_high_fitness_count"],
                         "api_calls": data["api_calls"], "total_tokens": data["total_tokens"],
                         "test_spearman": test.get("spearman"), "test_mse": test.get("mse"),
                         "test_top10_hit_rate": test.get("top_k_hit_rate")})
        rows.append({"group": group, "level": "mean_over_seeds", "seed": None,
                     "predictor": payload["info"]["predictor"], "agent_mode": payload["info"]["agent"],
                     "status": f"n={agg['n_seeds']}",
                     "final_cumulative_best": agg["final_cumulative_best"]["mean"],
                     "mean_round_fitness": agg["mean_round_fitness"]["mean"],
                     "total_new_high_fitness_count": agg["total_new_high_fitness_count"]["mean"],
                     "api_calls": agg["api_calls"]["mean"], "total_tokens": agg["total_tokens"]["mean"],
                     "test_spearman": agg["test_spearman"]["mean"], "test_mse": agg["test_mse"]["mean"],
                     "test_top10_hit_rate": agg["test_top10_hit_rate"]["mean"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(V2_DIR / "factorial_summary.csv", index=False)
    write_json(V2_DIR / "factorial_effects.json", {"metric": "final_cumulative_best unless suffixed",
                                                   "effects": effects})
    print(f"Wrote {V2_DIR / 'factorial_summary.csv'} and per-group summaries")
    for name, value in effects.items():
        print(f"  {name}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["G2", "G3", "G4"], choices=["G2", "G3", "G4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    V2_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        failures = []
        for group in args.groups:
            for seed in args.seeds:
                print(f"[{group} seed {seed}] starting", flush=True)
                started = time.perf_counter()
                try:
                    status = run_g2(seed) if group == "G2" else run_tool_group(group, seed)
                    print(f"[{group} seed {seed}] {status} in {time.perf_counter() - started:.0f}s", flush=True)
                except Exception as exc:
                    failures.append((group, seed, f"{type(exc).__name__}: {exc}"))
                    print(f"[{group} seed {seed}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
        if failures:
            for group, seed, message in failures:
                print(f"FAILED {group} seed {seed}: {message}")
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
