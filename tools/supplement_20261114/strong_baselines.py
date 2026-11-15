"""Independent full-pool GB1 comparators and public state preparation.

The experiment uses the historical split and selected model specification. All
new candidate labels are queried only after the selected IDs are saved. The CLI
separates preparation from explicitly authorized offline experiment execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from protevo.data import HiddenOracle, PublicDataset, describe_variant, one_hot, prepare
from protevo.models import fit_model, load_selection, model_specs, select_model
from protevo.schema import ExperimentConfig, compatible_config, variant_allowed
from protevo.storage import read_json, single_writer, write_json

ARMS = ("uniform_random", "full_pool_frozen", "full_pool_updated")
HISTORICAL_SEEDS = (11, 22, 33)
RANDOM_NAMESPACE = 20261114
REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
DEFAULT_OUTPUT = PROJECT / "results/supplement-20261114/strong-baselines"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def versions():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}


def public_file(project, relative):
    project = Path(project).resolve()
    path = (project / relative).resolve()
    if not path.is_relative_to(project) or not path.is_file() or path.name.lower().startswith(".env"):
        raise ValueError("Expected an existing public project file")
    return path


def reserve_empty(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory must be empty; existing results are retained")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class PreparedState:
    root: Path
    dataset: PublicDataset
    selection: dict
    config: ExperimentConfig

    @property
    def data_dir(self):
        return self.dataset.directory

    @property
    def selection_path(self):
        return self.root / "selection/selection.json"


def validate_public_roles(dataset, config):
    """Check public roles without opening hidden test or oracle tables."""
    if not compatible_config(dataset.manifest["config"], config):
        raise ValueError("Dataset manifest configuration differs from the requested preparation")
    validation = pd.read_csv(dataset.directory / "validation.csv")
    roles = (set(dataset.train.variant), set(validation.variant), set(dataset.candidates.variant))
    if any(roles[i] & roles[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Public data roles overlap")
    if len(validation) != len(roles[1]):
        raise ValueError("Validation variants must be unique")
    expected = dataset.manifest["counts"]
    if (len(dataset.train), len(validation), len(dataset.candidates)) != (expected["train"], expected["validation"], expected["pool"]):
        raise ValueError("Public data counts differ from their manifest")
    if len(dataset.train) != config.train_size or len(validation) != config.validation_size or expected["test"] != config.test_size:
        raise ValueError("Prepared data sizes differ from configuration")
    return dataset


def prepare_state(project_root, output_root, seed, *, config=None):
    """Prepare one declared split and validation-selected model, without oracle access.

    Callers own ``output_root`` and may pass a complete ExperimentConfig. An
    existing completed preparation is reused only for the same seed and config.
    Incomplete directories are retained for inspection and cannot be overwritten.
    """
    project, root = Path(project_root).resolve(), Path(output_root).resolve()
    if type(seed) is not int or seed < 0 or seed > 2**32-1:
        raise ValueError("Data seed must be a nonnegative 32-bit integer")
    config = config or ExperimentConfig(seed=seed, save_predictors=False)
    if not isinstance(config, ExperimentConfig) or config.seed != seed:
        raise ValueError("Preparation seed and configuration must agree")
    source = public_file(project, "data/raw/elife-16965-supp1-v4.xlsx")
    declaration = {"format_version": 1, "kind": "public-state-preparation", "seed": seed,
                   "source": str(source), "config": config.model_dump(), "reference": REFERENCE,
                   "selection_rule": "historical validation Spearman, top-k hit rate, lower MSE, declared order",
                   "state_sampling": "fixed declared seed; no fitness-based state selection"}
    ready = root / "prepared.json"
    if ready.exists():
        saved = read_json(ready)
        if saved["declaration"] != declaration or read_json(root / "declaration.json") != declaration:
            raise ValueError("Existing state preparation uses a different declaration")
        dataset = validate_public_roles(PublicDataset(root / "data"), config)
        selection = load_selection(root / "selection/selection.json", dataset, config)
        if saved["selection"] != selection or saved["dataset_manifest"] != dataset.manifest:
            raise ValueError("Existing prepared state differs from its saved metadata")
        return PreparedState(root, dataset, selection, config)
    reserve_empty(root)
    write_json(root / "declaration.json", declaration)
    prepare(source, root / "data", config, REFERENCE, "Variants", "Fitness", "Supplementary Dataset 1", 0)
    dataset = validate_public_roles(PublicDataset(root / "data"), config)
    selection = select_model(dataset, config, root / "selection")
    write_json(ready, {"prepared_at": stamp(), "declaration": declaration,
                       "dataset_manifest": dataset.manifest, "selection": selection,
                       "data": str(dataset.directory), "selection_path": str(root / "selection/selection.json")})
    return PreparedState(root, dataset, selection, config)


def historical_inputs(project_root, seed):
    project = Path(project_root).resolve()
    dataset_dir = project / "data/processed" / f"seed-{seed}"
    selection_path = public_file(project, f"results/seed-{seed}/selection/selection.json")
    selection = read_json(selection_path)
    config = ExperimentConfig.model_validate(selection["config"])
    if config.seed != seed or (config.rounds, config.k, config.train_size, config.validation_size, config.test_size) != (3, 10, 2000, 1000, 5000):
        raise ValueError("Historical controls require the declared seed and 2000/1000/5000 split with 3 rounds of 10")
    dataset = validate_public_roles(PublicDataset(dataset_dir), config)
    load_selection(selection_path, dataset, config)
    if selection["spec"] not in model_specs(config) or selection["spec"]["family"] != "random_forest":
        raise ValueError("Historical selected Random Forest specification is invalid")
    if selection["versions"] != versions():
        raise ValueError("Historical predictor package versions differ; review before model reconstruction")
    return dataset, selection, config, selection_path


def prepare_historical(project_root, output_root, *, seeds=HISTORICAL_SEEDS):
    """Freeze public historical inputs and the new experiment plan, without scoring."""
    project, root = Path(project_root).resolve(), Path(output_root).resolve()
    if tuple(seeds) != HISTORICAL_SEEDS:
        raise ValueError("This comparator study declares historical seeds 11, 22, and 33")
    inputs = []
    for seed in seeds:
        dataset, selection, config, selection_path = historical_inputs(project, seed)
        inputs.append({"seed": seed, "data": str(dataset.directory), "selection_path": str(selection_path),
                       "config": config.model_dump(), "raw_selection": selection,
                       "dataset_manifest": dataset.manifest, "initial_training_variants": list(dataset.train.variant),
                       "candidate_count": len(dataset.candidates)})
    plan = {"format_version": 1, "study_id": "gb1-strong-comparators-20261114", "seeds": list(seeds),
            "arms": list(ARMS), "rounds": 3, "k": 10, "assays_per_trajectory": 30,
            "random_namespace": RANDOM_NAMESPACE, "tie_break": "descending predicted fitness, ascending variant ID",
            "candidate_eligibility": "full original candidate pool minus this arm's previous selected IDs",
            "prediction_budget": "unrestricted full remaining pool for each model round; uniform_random has zero predictor scores",
            "historical_comparison_budget": "historical model arm scored 40 sampled candidates per round",
            "versions": versions(), "inputs": inputs}
    path = root / "prepared.json"
    if path.exists():
        if read_json(path)["plan"] != plan:
            raise ValueError("Historical preparation differs from its frozen plan")
        verify_public_snapshots(root, plan)
        return read_json(path)
    reserve_empty(root)
    # Public snapshots permit exact content checks without opening hidden labels.
    for row in inputs:
        destination = root / "inputs" / f'seed-{row["seed"]}'
        destination.mkdir(parents=True)
        for name in ("train.csv", "validation.csv", "candidates.csv", "manifest.json"):
            shutil.copyfile(Path(row["data"]) / name, destination / name)
        write_json(destination / "selection.json", row["raw_selection"])
    prepared = {"prepared_at": stamp(), "status": "prepared", "plan": plan,
                "planned_trajectories": len(seeds)*len(ARMS), "planned_rounds": len(seeds)*len(ARMS)*3,
                "planned_assays": len(seeds)*len(ARMS)*30,
                "planned_predictor_scores": sum(2*sum(row["candidate_count"]-10*r for r in range(3)) for row in inputs),
                "llm_requests": 0}
    write_json(path, prepared)
    return prepared


def verify_public_snapshots(root, plan):
    if plan["versions"] != versions():
        raise ValueError("Predictor package versions changed after preparation")
    for row in plan["inputs"]:
        frozen = Path(root) / "inputs" / f'seed-{row["seed"]}'
        for name in ("train.csv", "validation.csv", "candidates.csv", "manifest.json"):
            if (Path(row["data"]) / name).read_bytes() != (frozen / name).read_bytes():
                raise ValueError("Historical public inputs changed after preparation")
        if read_json(row["selection_path"]) != row["raw_selection"] or read_json(frozen / "selection.json") != row["raw_selection"]:
            raise ValueError("Historical selected predictor changed after preparation")


def score_full_pool(model, variants, output_path, *, chunk_size=8192):
    """Score an unlabeled ID list and save every value before Top-k selection."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Prediction chunk size must be a positive integer")
    ordered = sorted(variants)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Prediction candidates must be unique")
    parts = []
    for offset in range(0, len(ordered), chunk_size):
        batch = ordered[offset:offset+chunk_size]
        scores = np.asarray(model.predict(one_hot(batch)), dtype=float)
        if scores.shape != (len(batch),) or not np.isfinite(scores).all():
            raise ValueError("Predictor must return one finite score per candidate")
        parts.append(scores)
    values = np.concatenate(parts) if parts else np.array([], dtype=float)
    frame = pd.DataFrame({"variant": ordered, "predicted_fitness": values})
    path = Path(output_path)
    if path.exists():
        raise ValueError("A saved full-pool scoring table cannot be overwritten")
    frame.to_csv(path, index=False, compression="gzip", float_format="%.17g")
    return frame


def select_from_public_scores(scores, k):
    if len(scores) < k or scores.variant.duplicated().any() or "actual_fitness" in scores or "fitness" in scores:
        raise ValueError("Top-k selection requires unique, unlabeled candidate predictions")
    if not np.isfinite(scores.predicted_fitness.to_numpy(dtype=float)).all():
        raise ValueError("Candidate scores must be finite")
    return scores.sort_values(["predicted_fitness", "variant"], ascending=[False, True], kind="stable").head(k)


def _run_arm(dataset, selection, config, arm, output_root, *, chunk_size=8192):
    """Execute one independent arm; synthetic datasets are supported for tests."""
    if arm not in ARMS:
        raise ValueError("Unknown comparator arm")
    if selection["dataset_manifest"] != dataset.manifest or not compatible_config(selection["config"], config):
        raise ValueError("Arm must retain its dataset and selected predictor configuration")
    root = reserve_empty(output_root)
    initial = dataset.train.copy(deep=True)
    known = initial.copy(deep=True)
    initial_ids = list(initial.variant)
    used = set()
    frozen_model = None
    records = []
    metadata = {"format_version": 1, "arm": arm, "seed": config.seed, "status": "running",
                "config": config.model_dump(), "selection": deepcopy(selection), "versions": versions(),
                "initial_training_variants": initial_ids, "rounds": records, "started_at": stamp(),
                "prediction_scope": "none" if arm == "uniform_random" else "full_remaining_pool",
                "score_timing_scope": "prediction, scoring-table persistence and Top-k selection; sampling for uniform_random",
                "llm_requests": 0}
    write_json(root / "state.json", metadata)
    for number in range(1, config.rounds+1):
        directory = root / f"round-{number:03d}"
        directory.mkdir()
        public_ids = sorted(v for v in set(dataset.candidates.variant)-used if variant_allowed(v, config))
        if len(public_ids) < config.k:
            raise ValueError("The remaining eligible pool cannot satisfy this round's assay budget")
        train = initial if arm == "full_pool_frozen" else known
        trained_ids = list(train.variant) if arm != "uniform_random" else []
        fit_start_wall, fit_start_cpu = time.perf_counter(), time.process_time()
        fitted = False
        if arm != "uniform_random":
            if arm != "full_pool_frozen" or frozen_model is None:
                model = fit_model(selection["spec"], config, train)
                fitted = True
                if arm == "full_pool_frozen":
                    frozen_model = model
            else:
                model = frozen_model
        fit_wall, fit_cpu = time.perf_counter()-fit_start_wall, time.process_time()-fit_start_cpu
        score_start_wall, score_start_cpu = time.perf_counter(), time.process_time()
        score_path = None
        if arm == "uniform_random":
            design_seed = config.design_seed if config.design_seed is not None else config.seed
            rng = np.random.default_rng(np.random.SeedSequence([design_seed, number, RANDOM_NAMESPACE]))
            selected_ids = rng.choice(public_ids, size=config.k, replace=False).tolist()
            selected = [{"variant": v, "predicted_fitness": None} for v in selected_ids]
            score_count = 0
        else:
            score_path = directory / "full-pool-scores.csv.gz"
            scored = score_full_pool(model, public_ids, score_path, chunk_size=chunk_size)
            selected = select_from_public_scores(scored, config.k).to_dict(orient="records")
            selected_ids = [row["variant"] for row in selected]
            score_count = len(scored)
        score_wall, score_cpu = time.perf_counter()-score_start_wall, time.process_time()-score_start_cpu
        recommendations = [{**describe_variant(row["variant"], dataset.reference), **row, "rank": rank}
                           for rank, row in enumerate(selected, 1)]
        record = {"round": number, "stage": "selected", "operation_id": str(uuid4()),
                  "known_variants_before": list(known.variant), "known_count_before": len(known),
                  "training_variants": trained_ids, "training_count": len(trained_ids),
                  "predictor_fitted_this_round": fitted, "available_count": len(public_ids),
                  "score_count": score_count, "scores_file": str(score_path) if score_path else None,
                  "fit_wall_seconds": fit_wall, "fit_cpu_seconds": fit_cpu,
                  "score_wall_seconds": score_wall, "score_cpu_seconds": score_cpu,
                  "selected_at": stamp(), "recommendations": recommendations}
        # Both selection files are durable before the oracle is created or opened.
        write_json(directory / "selection.json", record)
        records.append(record)
        write_json(root / "state.json", metadata)
        feedback = HiddenOracle(dataset.directory, selected_ids).evaluate(selected_ids)
        lookup = {row["variant"]: row["actual_fitness"] for row in feedback}
        if set(lookup) != set(selected_ids) or any(not np.isfinite(v) or v < 0 for v in lookup.values()):
            raise ValueError("Oracle returned invalid measured feedback")
        for row in recommendations:
            row["actual_fitness"] = float(lookup[row["variant"]])
        actual = [r["actual_fitness"] for r in recommendations]
        previous = float(known.fitness.max())
        record.update(stage="evaluated", evaluated_at=stamp(), metrics={"mean_fitness": float(np.mean(actual)),
                      "max_fitness": max(actual), "previous_best": previous,
                      "cumulative_best": max(previous, max(actual)), "assays": len(actual)})
        write_json(directory / "evaluated.json", record)
        write_json(root / "state.json", metadata)
        observations = [{**describe_variant(v, dataset.reference), "fitness": lookup[v]} for v in selected_ids]
        known = pd.concat([known, pd.DataFrame(observations)], ignore_index=True)
        used.update(selected_ids)
        if known.variant.duplicated().any():
            raise ValueError("An arm cannot query or train on a repeated variant")
    metadata.update(status="completed", finished_at=stamp(), total_assays=len(used),
                    total_score_count=sum(r["score_count"] for r in records))
    write_json(root / "state.json", metadata)
    return metadata


def run_arm(dataset, selection, config, arm, output_root, *, chunk_size=8192):
    """Keep partial selected or evaluated records when an arm cannot finish."""
    path = Path(output_root) / "state.json"
    existed = path.exists()
    try:
        return _run_arm(dataset, selection, config, arm, output_root, chunk_size=chunk_size)
    except BaseException as error:
        if not existed and path.exists():
            state = read_json(path)
            state.update(status="failed", finished_at=stamp(), error_type=type(error).__name__)
            write_json(path, state)
        raise


def run_prepared(output_root, correction_status, *, allow_offline_experiment=False, chunk_size=8192):
    if not allow_offline_experiment:
        raise ValueError("Offline study execution requires explicit authorization")
    correction = read_json(correction_status)
    if correction.get("status") != "passed":
        raise ValueError("Statistical correction must pass before the supplemental experiment")
    root = Path(output_root).resolve()
    prepared = read_json(root / "prepared.json")
    plan = prepared["plan"]
    verify_public_snapshots(root, plan)
    if plan["study_id"] != "gb1-strong-comparators-20261114" or plan["seeds"] != list(HISTORICAL_SEEDS) or plan["arms"] != list(ARMS):
        raise ValueError("Unknown frozen comparator plan")
    if (root / "run.json").exists() or (root / "runs").exists():
        raise ValueError("An experiment with existing execution records cannot be started again")
    with single_writer(root):
        result = {"study_id": plan["study_id"], "status": "running", "started_at": stamp(),
                  "correction_status": str(Path(correction_status).resolve()), "correction_record": correction,
                  "versions": versions(), "results": [], "llm_requests": 0}
        write_json(root / "run.json", result)
        try:
            for row in plan["inputs"]:
                dataset = PublicDataset(row["data"])
                config = ExperimentConfig.model_validate(row["config"])
                selection = load_selection(row["selection_path"], dataset, config)
                for arm in ARMS:
                    trajectory = run_arm(dataset, selection, config, arm, root / "runs" / f'seed-{row["seed"]}' / arm,
                                         chunk_size=chunk_size)
                    result["results"].append({"seed": row["seed"], "arm": arm,
                        "status": trajectory["status"], "assays": trajectory["total_assays"],
                        "score_count": trajectory["total_score_count"],
                        "mean_fitness": float(np.mean([r["metrics"]["mean_fitness"] for r in trajectory["rounds"]])),
                        "max_fitness": max(r["metrics"]["max_fitness"] for r in trajectory["rounds"])})
                    write_json(root / "run.json", result)
            result.update(status="completed", finished_at=stamp())
            write_json(root / "run.json", result)
        except BaseException as error:
            result.update(status="failed", finished_at=stamp(), error_type=type(error).__name__)
            write_json(root / "run.json", result)
            raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "prepare-state"))
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--correction-status", type=Path, default=PROJECT / "results/metric-correction-20261114/correction-status.json")
    parser.add_argument("--allow-offline-experiment", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8192)
    options = parser.parse_args()
    if options.action == "prepare":
        prepared = prepare_historical(options.project_root, options.output)
        print(json.dumps({key: prepared[key] for key in ("status", "planned_trajectories", "planned_rounds", "planned_assays", "planned_predictor_scores", "llm_requests")}, ensure_ascii=False))
    elif options.action == "prepare-state":
        if options.seed is None:
            parser.error("prepare-state requires --seed")
        config = ExperimentConfig.model_validate(read_json(options.config)) if options.config else None
        state = prepare_state(options.project_root, options.output, options.seed, config=config)
        print(json.dumps({"state_id": f"seed-{state.config.seed}", "data": str(state.data_dir), "selection": str(state.selection_path)}, ensure_ascii=False))
    else:
        result = run_prepared(options.output, options.correction_status,
                              allow_offline_experiment=options.allow_offline_experiment, chunk_size=options.chunk_size)
        print(json.dumps({"status": result["status"], "trajectories": len(result["results"]), "llm_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
