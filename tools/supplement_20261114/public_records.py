"""Offline public-record export and read-only reconstruction of supplemental studies."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

FORMAT = 1
TOLERANCE = 1e-10
SNAPSHOT_FILES = (
    "protevo/__init__.py", "protevo/agent.py", "protevo/data.py", "protevo/models.py", "protevo/prompts.py", "protevo/schema.py", "protevo/storage.py",
    "prompts/legacy-v1.txt", "prompts/scientific-v2.txt", "prompts/scientific-v3.txt", "configs/knowledge.json",
    "configs/statistical-definitions-v2.json", "requirements.txt", "reports/统计定义与版本范围.md",
    "tools/supplement_20261114/strong_baselines.py", "tools/supplement_20261114/feedback_pilot.py",
    "tools/supplement_20261114/public_records.py",
)
PROTECTED = {"payload", "context", "text", "analysis", "hypothesis", "rationale", "reason", "system_prompt", "prompt_metadata", "knowledge"}
PRIVATE = {"apikey", "authorization", "headers", "baseurl", "endpoint", "credentials", "credential", "password",
           "accesstoken", "refreshtoken", "clientsecret", "reasoningcontent", "hiddenreasoning", "chainofthought", "internalthoughts"}
TOKEN = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b|\bBearer\s+[A-Za-z0-9._~+/-]{8,}", re.I)
ASSIGNMENT = re.compile(r'''(?:api[_-]?key|authorization|password|access[_-]?token)["']?\s*[:=]\s*["']?[^\s"',}]{4,}''', re.I)
PERSONAL = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:home|Users)/[^/\s]+/", re.I)
SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
TERMINAL = {"completed", "failed", "cancelled", "interrupted", "unknown"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.name.lower().split(".")[0] in {"oracle", "test"} or any(part.lower().startswith(".env") or part.lower() in {"hidden", ".venv", "profile"} for part in Path(relative).parts):
        raise ValueError("Record path leaves its public scope")
    return path


def slug(value):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ValueError("Invalid public record identity")
    return value


def absolute(value):
    return isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())


def normalized(value):
    return str(value).replace("\\", "/").rstrip("/").casefold()


def public_value(value, *, field_names=True):
    """Reject private content; scientific text is never rewritten to hide it."""
    if isinstance(value, dict):
        for key, child in value.items():
            if field_names and re.sub(r"[^a-z]", "", key.lower()) in PRIVATE:
                raise ValueError("Private field found in a public scientific record")
            public_value(child, field_names=field_names)
    elif isinstance(value, list):
        for child in value:
            public_value(child, field_names=field_names)
    elif isinstance(value, str) and (TOKEN.search(value) or ASSIGNMENT.search(value) or PERSONAL.search(value) or absolute(value)):
        raise ValueError("Private content or absolute path found in a public scientific record")


def check_csv(path):
    columns = set(pd.read_csv(path, nrows=0).columns)
    allowed = {"variant", "sequence", "mutations", "mutation_count", "fitness", "predicted_fitness", "actual_fitness", "residual", "hypothesis", "rationale", "knowledge_rule_ids", "rank"}
    if not columns <= allowed or "variant" not in columns:
        raise ValueError("Unexpected fields in a public scientific table")
    if "scores" in Path(path).name and columns != {"variant", "predicted_fitness"}:
        raise ValueError("Candidate score table contains fields beyond predictions")
    fields = columns & {"variant", "hypothesis", "rationale", "knowledge_rule_ids"}
    frame = pd.read_csv(path, usecols=list(fields)).fillna("")
    if not frame.variant.astype(str).str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}").all():
        raise ValueError("Invalid public variant ID")
    for field in fields - {"variant"}:
        for value in frame[field]:
            public_value(str(value))


class Exporter:
    def __init__(self, project, destination, roots):
        self.project, self.root = Path(project).resolve(), Path(destination).resolve()
        self.transforms = []
        self.files = []
        self.mappings = [(str(Path(source).resolve()), target) for source, target in roots]
        self.mappings.extend((str(self.project / relative), "source/" + relative) for relative in SNAPSHOT_FILES)

    def add_mapping(self, source, target):
        self.mappings.insert(0, (str(source), target))

    def clean(self, value, file, pointer="", protected=False):
        if protected:
            public_value(value)
            return deepcopy(value)
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                location = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                if re.sub(r"[^a-z]", "", key.lower()) in PRIVATE:
                    self.transforms.append({"file": file, "pointer": "#" + location, "action": "private_metadata_field_omitted"})
                    continue
                result[key] = self.clean(child, file, location, key in PROTECTED)
            return result
        if isinstance(value, list):
            return [self.clean(child, file, f"{pointer}/{index}") for index, child in enumerate(value)]
        if absolute(value):
            if PureWindowsPath(value).name.lower().startswith(".env") or Path(value).name.lower().startswith(".env"):
                raise ValueError("Environment-file reference is not public metadata")
            target = None
            for source, prefix in sorted(self.mappings, key=lambda item: len(item[0]), reverse=True):
                source_norm, value_norm = normalized(source), normalized(value)
                if value_norm == source_norm or value_norm.startswith(source_norm + "/"):
                    suffix = str(value).replace("\\", "/")[len(str(source).replace("\\", "/").rstrip("/")):].lstrip("/")
                    target = prefix + ("/" + suffix if suffix else "")
                    break
            if target is None:
                target = "external/" + PureWindowsPath(value).name if PureWindowsPath(value).is_absolute() else "external/" + Path(value).name
            self.transforms.append({"file": file, "pointer": "#" + pointer, "action": "absolute_metadata_path_rewritten", "replacement": target})
            return target
        public_value(value)
        return deepcopy(value)

    def json(self, source, relative):
        self.value(read_json(source), relative)

    def value(self, value, relative):
        cleaned = self.clean(value, relative)
        write_json(safe_path(self.root, relative), cleaned)
        self.files.append(relative)

    def copy(self, source, relative):
        source = Path(source)
        if source.name.endswith((".csv", ".csv.gz")):
            check_csv(source)
        if source.is_symlink():
            raise ValueError("Symbolic-link input is not accepted")
        path = safe_path(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, path.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        self.files.append(relative)

    def ids(self, source, relative):
        columns = pd.read_csv(source, nrows=0).columns
        if any(re.sub(r"[^a-z]", "", key.lower()) in PRIVATE for key in columns):
            raise ValueError("Candidate table contains private metadata")
        frame = pd.read_csv(source, usecols=["variant"])
        if {"fitness", "actual_fitness"} & set(columns):
            raise ValueError("Candidate eligibility table contains measurements")
        if "variant" not in frame or frame.variant.duplicated().any() or not frame.variant.astype(str).str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}").all():
            raise ValueError("Candidate IDs must be unique")
        path = safe_path(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[["variant"]].to_csv(path, index=False, mode="x")
        self.files.append(relative)
        self.transforms.append({"file": relative, "action": "unlabeled_candidate_table_reduced_to_variant_ids"})

    def events(self, source, relative):
        values = [read for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip() for read in [json.loads(line)]]
        path = safe_path(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for index, value in enumerate(values):
                stream.write(json.dumps(self.clean(value, relative, f"/{index}"), ensure_ascii=False, allow_nan=False) + "\n")
        self.files.append(relative)


def export_records(project_root, destination, *, feedback=None, strong=None, terminal_confirmed=False):
    """Export only explicitly confirmed terminal studies into an empty directory."""
    if not terminal_confirmed:
        raise ValueError("Explicit terminal confirmation is required before reading experiment records")
    if feedback is None and strong is None:
        raise ValueError("At least one study must be explicitly supplied")
    project, destination = Path(project_root).resolve(), Path(destination).resolve()
    roots = [(Path(root).resolve(), name) for root, name in ((feedback, "feedback"), (strong, "strong")) if root is not None]
    for root, _ in roots:
        if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError("Export and source directories must be separate")
        if (root / ".writer.lock").exists():
            raise ValueError("Study still has an active writer")
    feedback_plan = feedback_state = strong_plan = strong_state = None
    if feedback is not None:
        feedback = Path(feedback).resolve()
        feedback_plan, feedback_state = read_json(feedback / "plan.json"), read_json(feedback / "run-state.json")
        if feedback_state["status"] not in {"completed", "finished_with_failures", "stopped_missing_usage", "failed", "cancelled", "interrupted"}:
            raise ValueError("Feedback study is not terminal")
        if any(row["status"] == "running" for row in feedback_state["episodes"].values()):
            raise ValueError("Feedback arm is still active")
        if any(row["status"] == "pending" for row in feedback_state["episodes"].values()) and feedback_state["status"] != "stopped_missing_usage":
            raise ValueError("Feedback study still contains pending arms")
    if strong is not None:
        strong = Path(strong).resolve()
        strong_plan, strong_state = read_json(strong / "prepared.json"), read_json(strong / "run.json")
        if strong_state["status"] not in TERMINAL:
            raise ValueError("Strong-comparator study is not terminal")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Public export directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    export = Exporter(project, destination, roots)
    for relative in SNAPSHOT_FILES:
        export.copy(safe_path(project, relative), "source/" + relative)
    export.add_mapping(project / "data/raw/elife-16965-supp1-v4.xlsx", "external/official-s1.xlsx")
    counts = {}
    if feedback is not None:
        counts["feedback"] = _export_feedback(export, feedback, feedback_plan, feedback_state)
    if strong is not None:
        counts["strong"] = _export_strong(export, strong, strong_plan, strong_state)
    environment = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pydantic", "openai"):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = None
    export.value(environment, "environment.json")
    manifest = {"format_version": FORMAT, "kind": "gb1-supplement-public-records", "created_at": datetime.now(timezone.utc).isoformat(),
                "statistical_definition": "gb1-statistical-definitions/2", "counts": counts,
                "source_snapshot": {"captured_at": datetime.now(timezone.utc).isoformat(), "files": ["source/" + name for name in SNAPSHOT_FILES]},
                "metadata_transformations": export.transforms, "scientific_text_transformations": [],
                "external_sources": [{"path": "external/official-s1.xlsx", "url": "https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx", "included": False}],
                "files": sorted(export.files), "prediction_reconstruction_absolute_tolerance": TOLERANCE}
    public_value(manifest)
    write_json(destination / "manifest.json", manifest)
    return manifest


def _export_feedback(export, root, plan, state):
    for source in plan["declaration"]["sources"]:
        state_id = slug(source["state_id"])
        prefix = f"feedback/states/{state_id}"
        export.add_mapping(source["data"], prefix)
        export.add_mapping(source["selection"], prefix + "/prepared.json")
    export.value(plan, "feedback/plan.json")
    export.value(state, "feedback/run-state.json")
    if (root / "summary.json").is_file():
        export.json(root / "summary.json", "feedback/summary.json")
    prefix_queries, selected_queries, complete_selectors = [], [], 0
    for source in plan["declaration"]["sources"]:
        state_id = slug(source["state_id"])
        directory, prefix = root / "states" / state_id, f"feedback/states/{state_id}"
        for name in ("prepared.json", "P0-training-ids.json", "P1-training-ids.json", "excluded-ids.json", "prefix-selected.json", "C0-context.json", "C1-context.json"):
            export.json(directory / name, prefix + "/" + name)
        for name in ("initial-observations.csv", "prefix-observations.csv", "prefix-all-scores.csv.gz"):
            export.copy(directory / name, prefix + "/" + name)
        export.ids(directory / "eligible.csv", prefix + "/eligible.csv")
        export.events(directory / "events.jsonl", prefix + "/events.jsonl")
        prefix_queries.extend((state_id, row["variant"]) for row in read_json(directory / "prefix-selected.json"))
    for episode in plan["episodes"]:
        episode_id = slug(episode["episode_id"])
        record = state["episodes"][episode_id]
        directory, prefix = root / "episodes" / episode_id, "feedback/episodes/" + episode_id
        if (directory / "events.jsonl").is_file():
            export.events(directory / "events.jsonl", prefix + "/events.jsonl")
        for number, _ in enumerate(record["attempts"], 1):
            for name in ("request.json", "response.json", "validation.json", "error.json"):
                path = directory / f"attempt-{number}" / name
                if path.is_file():
                    export.json(path, prefix + f"/attempt-{number}/" + name)
        for name in ("accepted-proposals.json", "selector-comparison.json"):
            if (directory / name).is_file():
                export.json(directory / name, prefix + "/" + name)
        for selector in ("P0", "P1"):
            folder = directory / selector
            if not folder.is_dir():
                continue
            for name in ("training-ids.json", "model-spec.json", "selected.json", "result.json"):
                if (folder / name).is_file():
                    export.json(folder / name, prefix + f"/{selector}/" + name)
            for name in ("all-scores.csv", "recommendations.csv"):
                if (folder / name).is_file():
                    export.copy(folder / name, prefix + f"/{selector}/" + name)
            if (folder / "events.jsonl").is_file():
                export.events(folder / "events.jsonl", prefix + f"/{selector}/events.jsonl")
            if (folder / "recommendations.csv").is_file():
                selected_queries.extend((episode["state_id"], value) for value in pd.read_csv(folder / "recommendations.csv").variant)
            complete_selectors += int((folder / "result.json").is_file())
    queries = prefix_queries + selected_queries
    return {"planned_generator_arms": len(plan["episodes"]), "completed_generator_arms": sum(row["status"] == "completed" for row in state["episodes"].values()),
            "planned_selector_results": plan["planned_selector_results"], "completed_selector_results": complete_selectors,
            "generation_calls": state["calls"], "max_generation_calls": plan["max_calls"],
            "reported_known_total_tokens": state["reported_known_total_tokens"], "missing_usage_calls": state["missing_usage_count"],
            "complete_total_tokens": state["reported_known_total_tokens"] if not state["missing_usage_count"] else None,
            "common_prefix_assays": len(prefix_queries), "selector_policy_assays": len(selected_queries),
            "unique_new_variant_ids": len({variant for _, variant in queries}), "unique_new_state_variant_pairs": len(set(queries)),
            "selector_replay_is_independent_generation": False, "status": state["status"]}


def _export_strong(export, root, prepared, result):
    plan = prepared["plan"]
    for entry in plan["inputs"]:
        prefix = f"strong/inputs/seed-{int(entry['seed'])}"
        export.add_mapping(entry["data"], prefix)
        export.add_mapping(entry["selection_path"], prefix + "/selection.json")
    export.value(prepared, "strong/prepared.json")
    export.value(result, "strong/run.json")
    completed, assays, predictions, selected = 0, 0, 0, []
    for entry in plan["inputs"]:
        seed = int(entry["seed"])
        folder, prefix = root / "inputs" / f"seed-{seed}", f"strong/inputs/seed-{seed}"
        for name in ("manifest.json", "selection.json"):
            export.json(folder / name, prefix + "/" + name)
        for name in ("train.csv", "validation.csv"):
            export.copy(folder / name, prefix + "/" + name)
        export.ids(folder / "candidates.csv", prefix + "/candidates.csv")
        for arm in plan["arms"]:
            slug(arm)
            directory, target = root / "runs" / f"seed-{seed}" / arm, f"strong/runs/seed-{seed}/{arm}"
            if not (directory / "state.json").is_file():
                if result["status"] == "completed":
                    raise ValueError("Completed comparator has a missing trajectory")
                continue
            state = read_json(directory / "state.json")
            if state["status"] not in TERMINAL:
                raise ValueError("Comparator trajectory is still active")
            export.value(state, target + "/state.json")
            completed += int(state["status"] == "completed")
            for record in state["rounds"]:
                number = int(record["round"])
                round_dir, round_target = directory / f"round-{number:03d}", target + f"/round-{number:03d}"
                for name in ("selection.json", "evaluated.json"):
                    if (round_dir / name).is_file():
                        export.json(round_dir / name, round_target + "/" + name)
                if record.get("scores_file"):
                    export.copy(round_dir / "full-pool-scores.csv.gz", round_target + "/full-pool-scores.csv.gz")
                predictions += record["score_count"]
                if record["stage"] == "evaluated":
                    assays += len(record["recommendations"])
                    selected.extend((seed, row["variant"]) for row in record["recommendations"])
    return {"planned_trajectories": len(plan["inputs"]) * len(plan["arms"]), "completed_trajectories": completed,
            "policy_assays": assays, "predictor_scores": predictions, "generation_calls": 0,
            "unique_new_variant_ids": len({variant for _, variant in selected}), "unique_new_state_variant_pairs": len(set(selected)), "status": result["status"]}


class Checks:
    def __init__(self):
        self.count = 0
        self.measured = {}
        self.max_prediction_error = 0.0

    def require(self, value, label):
        self.count += 1
        if not value:
            raise ValueError(label)

    def equal(self, left, right, label):
        if isinstance(left, dict) and isinstance(right, dict):
            self.require(set(left) == set(right), label)
            for key in left:
                self.equal(left[key], right[key], label + "/" + str(key))
        elif isinstance(left, list) and isinstance(right, list):
            self.require(len(left) == len(right), label)
            for a, b in zip(left, right):
                self.equal(a, b, label)
        elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            self.require(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= TOLERANCE, label)
        else:
            self.require(left == right, label)

    def observations(self, frame, field="fitness"):
        self.require(not frame.variant.duplicated().any(), "Duplicate observation IDs")
        for row in frame.to_dict("records"):
            value = float(row[field])
            self.require(math.isfinite(value) and value >= 0, "Invalid observed fitness")
            if row["variant"] in self.measured:
                self.equal(value, self.measured[row["variant"]], "Conflicting measured feedback")
            self.measured[row["variant"]] = value


@contextmanager
def snapshot_core(root):
    name, previous = "_gb1_public_" + uuid4().hex, sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        package = Path(root) / "source/protevo"
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        yield {part: importlib.import_module(name + "." + part) for part in ("data", "schema", "models", "prompts", "agent")}
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        sys.dont_write_bytecode = previous


def table(path, *, scores=False):
    frame = pd.read_csv(path, dtype={"predicted_fitness": str} if scores else None)
    if scores and "predicted_fitness" in frame:
        frame["predicted_fitness"] = frame.predicted_fitness.map(float)
    if "variant" not in frame or frame.variant.duplicated().any():
        raise ValueError("Invalid public variant table")
    return frame


def ranked(scores, k):
    if set(scores.columns) != {"variant", "predicted_fitness"} or not np.isfinite(scores.predicted_fitness).all():
        raise ValueError("Candidate score table must have predictions only")
    return scores.sort_values(["predicted_fitness", "variant"], ascending=[False, True], kind="stable").head(k).variant.tolist()


def selected_values(checks, scores, selected, *, feedback=None, proposals=None):
    lookup = scores.set_index("variant").predicted_fitness if scores is not None else None
    proposed = {row["variant"]: row for row in proposals} if proposals is not None else {}
    observed = feedback.set_index("variant").to_dict("index") if feedback is not None else {}
    for rank, row in enumerate(selected, 1):
        if "rank" in row:
            checks.equal(row["rank"], rank, "Saved selection rank differs")
        if lookup is not None:
            checks.equal(row["predicted_fitness"], float(lookup[row["variant"]]), "Saved selection score differs")
        if row["variant"] in proposed:
            for key, value in proposed[row["variant"]].items():
                checks.equal(row[key], value, "Selected scientific text differs from the raw accepted proposal")
        if row["variant"] in observed:
            for key, value in row.items():
                checks.require(key in observed[row["variant"]] or key == "variant", "Selected scientific field missing from feedback")
                if key == "variant":
                    continue
                saved = observed[row["variant"]][key]
                if key == "knowledge_rule_ids" and isinstance(saved, str):
                    saved = ast.literal_eval(saved)
                checks.equal(saved, value, "Feedback changed a saved selection field")


def predictions(checks, core, spec, config, training, scores, recompute):
    ranked(scores, min(config.k, len(scores)))
    if not recompute:
        return
    model = core["models"].fit_model(spec, config, training)
    for start in range(0, len(scores), 8192):
        batch = scores.iloc[start:start + 8192]
        values = model.predict(core["data"].one_hot(batch.variant))
        error = float(np.max(np.abs(values - batch.predicted_fitness.to_numpy()))) if len(batch) else 0.0
        checks.max_prediction_error = max(checks.max_prediction_error, error)
        checks.require(error <= TOLERANCE, "Reconstructed predictor differs from saved scores")


def event_order(path, first, second, ids, checks):
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    earlier = [index for index, row in enumerate(events) if row["kind"] == first and row.get("variants") == ids]
    later = [index for index, row in enumerate(events) if row["kind"] == second and row.get("variants") == ids]
    checks.require(bool(earlier) and bool(later) and min(earlier) < min(later), "Selection must precede feedback in saved events")


def descriptions(frame, reference, core, checks):
    for row in frame.to_dict("records"):
        expected = core["data"].describe_variant(row["variant"], reference)
        for key in ("sequence", "mutations", "mutation_count"):
            if key in row:
                checks.equal(row[key], expected[key], "Sequence description mismatch")


def _audit_strong(root, core, checks, recompute):
    prepared, result = read_json(root / "strong/prepared.json"), read_json(root / "strong/run.json")
    plan = prepared["plan"]
    counts = {"planned_trajectories": len(plan["inputs"]) * len(plan["arms"]), "completed_trajectories": 0,
              "policy_assays": 0, "predictor_scores": 0, "generation_calls": 0}
    new_queries = []
    for entry in plan["inputs"]:
        seed = int(entry["seed"])
        directory = root / "strong/inputs" / f"seed-{seed}"
        initial, pool = table(directory / "train.csv"), table(directory / "candidates.csv")
        checks.require(set(pool.columns) == {"variant"}, "Candidate export must contain IDs only")
        validation = table(directory / "validation.csv")
        checks.require(not set(initial.variant) & set(pool.variant) and not set(validation.variant) & (set(initial.variant) | set(pool.variant)), "Public roles overlap")
        checks.observations(initial)
        selection = read_json(directory / "selection.json")
        checks.equal(selection["spec"], entry["raw_selection"]["spec"], "Frozen selected model differs")
        config = core["schema"].ExperimentConfig.model_validate(entry["config"])
        reference = read_json(directory / "manifest.json")["audit"]["reference"]
        descriptions(initial, reference, core, checks)
        for arm in plan["arms"]:
            folder = root / "strong/runs" / f"seed-{seed}" / slug(arm)
            if not (folder / "state.json").is_file():
                checks.require(result["status"] != "completed", "Completed comparator trajectory missing")
                continue
            state = read_json(folder / "state.json")
            checks.require(state["status"] in TERMINAL, "Nonterminal public comparator")
            checks.equal(state["config"], entry["config"], "Comparator config changed")
            known, used = initial.copy(), set()
            for record in state["rounds"]:
                number = record["round"]
                location = folder / f"round-{number:03d}"
                selected = read_json(location / "selection.json")
                checks.require(selected["stage"] == "selected" and not any("actual_fitness" in row for row in selected["recommendations"]), "Pre-feedback selection contains labels")
                checks.equal(selected["known_variants_before"], list(known.variant), "Comparator history is not arm-specific")
                training = initial if arm == "full_pool_frozen" else known
                training_ids = [] if arm == "uniform_random" else list(training.variant)
                checks.equal(selected["training_variants"], training_ids, "Predictor training IDs differ")
                eligible = sorted(v for v in set(pool.variant) - used if core["schema"].variant_allowed(v, config))
                ids = [row["variant"] for row in selected["recommendations"]]
                checks.require(len(ids) == len(set(ids)) == config.k and set(ids) <= set(eligible), "Invalid comparator selection")
                checks.require(selected["available_count"] == len(eligible), "Comparator pool count differs")
                for key in ("fit_wall_seconds", "fit_cpu_seconds", "score_wall_seconds", "score_cpu_seconds"):
                    checks.require(math.isfinite(selected[key]) and selected[key] >= 0, "Invalid recorded comparator time")
                if arm == "uniform_random":
                    scores = None
                    rng = np.random.default_rng(np.random.SeedSequence([config.design_seed if config.design_seed is not None else config.seed, number, plan["random_namespace"]]))
                    checks.equal(ids, rng.choice(eligible, size=config.k, replace=False).tolist(), "Uniform selection differs from declared random seed")
                    checks.require(selected["score_count"] == 0 and selected["scores_file"] is None, "Random comparator used predictor scores")
                else:
                    scores = table(location / "full-pool-scores.csv.gz", scores=True)
                    checks.equal(list(scores.variant), eligible, "Full-pool score coverage differs")
                    checks.equal(ids, ranked(scores, config.k), "Saved-score Top-k order differs")
                    checks.require(selected["score_count"] == len(scores), "Comparator prediction count differs")
                    predictions(checks, core, selection["spec"], config, training, scores, recompute)
                    lookup = scores.set_index("variant").predicted_fitness
                    for row in selected["recommendations"]:
                        checks.equal(row["predicted_fitness"], float(lookup[row["variant"]]), "Selected score differs from full scoring table")
                counts["predictor_scores"] += selected["score_count"]
                if record["stage"] != "evaluated":
                    checks.require(state["status"] != "completed", "Completed comparator has unevaluated selection")
                    continue
                evaluated = read_json(location / "evaluated.json")
                rows = pd.DataFrame(evaluated["recommendations"])
                checks.equal(list(rows.variant), ids, "Feedback was not restricted to selected IDs")
                checks.require(evaluated["evaluated_at"] >= selected["selected_at"], "Feedback timestamp precedes selection")
                checks.equal(evaluated, record, "Final state differs from persisted evaluated record")
                selected_values(checks, scores, selected["recommendations"], feedback=rows)
                checks.observations(rows, "actual_fitness")
                descriptions(rows, reference, core, checks)
                checks.equal(evaluated["metrics"]["mean_fitness"], float(rows.actual_fitness.mean()), "Comparator mean mismatch")
                checks.equal(evaluated["metrics"]["max_fitness"], float(rows.actual_fitness.max()), "Comparator maximum mismatch")
                counts["policy_assays"] += len(rows)
                new_queries.extend((seed, value) for value in ids)
                known = pd.concat([known, rows.rename(columns={"actual_fitness": "fitness"})[initial.columns]], ignore_index=True)
                used.update(ids)
            if state["status"] == "completed":
                checks.require(len(state["rounds"]) == config.rounds and len(used) == config.rounds * config.k, "Incomplete completed trajectory")
                counts["completed_trajectories"] += 1
    counts.update(unique_new_variant_ids=len({variant for _, variant in new_queries}), unique_new_state_variant_pairs=len(set(new_queries)), status=result["status"])
    return counts


def _state_contexts(directory, core, checks, recompute):
    prepared = read_json(directory / "prepared.json")
    config = core["schema"].ExperimentConfig.model_validate(prepared["source"]["config"])
    initial, prefix, eligible = table(directory / "initial-observations.csv"), table(directory / "prefix-observations.csv"), table(directory / "eligible.csv")
    checks.require(set(eligible.columns) == {"variant"}, "Eligible export must contain IDs only")
    checks.observations(initial)
    checks.observations(prefix, "actual_fitness")
    descriptions(initial, prepared["reference"], core, checks)
    descriptions(prefix, prepared["reference"], core, checks)
    updated = pd.concat([initial, prefix.rename(columns={"actual_fitness": "fitness"})[initial.columns]], ignore_index=True)
    excluded = sorted(set(updated.variant))
    checks.require(len(excluded) == len(updated) and not set(eligible.variant) & set(excluded), "Invalid state exclusions")
    for selector, training in (("P0", initial), ("P1", updated)):
        checks.equal(read_json(directory / f"{selector}-training-ids.json"), list(training.variant), "Saved state training IDs differ")
    checks.equal(read_json(directory / "excluded-ids.json"), excluded, "Saved exclusions differ")
    scores = table(directory / "prefix-all-scores.csv.gz", scores=True)
    selected = read_json(directory / "prefix-selected.json")
    ids = [row["variant"] for row in selected]
    checks.require(all("actual_fitness" not in row and "fitness" not in row for row in selected), "Common-prefix selection contains labels")
    checks.equal(ids, ranked(scores, config.k), "Common prefix not selected by saved scores")
    checks.equal(ids, list(prefix.variant), "Common prefix feedback IDs differ")
    checks.require(set(scores.variant) == set(eligible.variant) | set(ids), "Common-prefix scoring coverage differs")
    event_order(directory / "events.jsonl", "prefix_selected", "prefix_feedback", ids, checks)
    predictions(checks, core, prepared["source"]["spec"], config, initial, scores, recompute)
    selected_values(checks, scores, selected, feedback=prefix)
    for row in prefix.to_dict("records"):
        checks.equal(row["residual"], row["actual_fitness"] - row["predicted_fitness"], "Prefix residual mismatch")
    saved_contexts = {condition: read_json(directory / f"{condition}-context.json") for condition in ("C0", "C1")}
    knowledge = core["schema"].KnowledgeBase.model_validate(saved_contexts["C0"]["knowledge"])
    checks.equal(saved_contexts["C0"]["knowledge"], saved_contexts["C1"]["knowledge"], "Knowledge differs between contexts")
    feedback = prefix[["variant", "actual_fitness", "predicted_fitness", "residual", "hypothesis"]].to_dict("records")
    for condition, known, recent in (("C0", initial, ()), ("C1", updated, feedback)):
        context = core["agent"].make_context(known, [], config, prepared["reference"], 2, knowledge, "v3", recent)
        context.update(observed_variants=excluded.copy(), forbidden_variants=excluded.copy(), observed_variants_complete=True,
                       observed_variants_scope=f"All {len(excluded)} measured IDs are excluded in both conditions; some measurement values are not displayed.")
        checks.equal(saved_contexts[condition], context, "Reconstructed C0/C1 visible input differs")
    return prepared, config, initial, updated, eligible, excluded, saved_contexts


def usage(raw):
    raw = raw if isinstance(raw, dict) else {}
    count = lambda value: value if type(value) is int and value >= 0 else None
    incoming = count(raw.get("input_tokens", raw.get("prompt_tokens")))
    outgoing = count(raw.get("output_tokens", raw.get("completion_tokens")))
    total = count(raw.get("total_tokens"))
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": total,
            "missing_usage_count": int(incoming is None or outgoing is None or total is None)}


def _audit_feedback(root, core, checks, recompute):
    plan, state = read_json(root / "feedback/plan.json"), read_json(root / "feedback/run-state.json")
    checks.require(plan["study_id"] == state["study_id"], "Feedback study identity differs")
    checks.equal(plan["prompt_metadata"], core["prompts"].load_prompt_profile("v3").metadata(), "Snapshot scientific prompt differs")
    states, prefix_queries = {}, []
    for source in plan["declaration"]["sources"]:
        state_id = slug(source["state_id"])
        directory = root / "feedback/states" / state_id
        states[state_id] = _state_contexts(directory, core, checks, recompute)
        prefix_queries.extend((state_id, row["variant"]) for row in read_json(directory / "prefix-selected.json"))
    calls, complete, selectors, queries, operation_ids = 0, 0, 0, [], set()
    reported_tokens, missing_usage = 0, 0
    checks.require(plan["planned_generator_arms"] == len(plan["episodes"]) and plan["planned_selector_results"] == 2 * len(plan["episodes"]), "Generator and selector denominators differ")
    for episode in plan["episodes"]:
        episode_id = slug(episode["episode_id"])
        record = state["episodes"][episode_id]
        checks.require(record["status"] in TERMINAL | {"pending"}, "Nonterminal public generation arm")
        prepared, base_config, initial, updated, eligible, excluded, contexts = states[episode["state_id"]]
        config = base_config.model_copy(update={"design_seed": episode["design_seed"], "require_full_batch": True, "oversize_policy": "first_n", "llm_max_calls_per_round": 2})
        base_context, accepted, rejected, analyses = contexts[episode["context"]], [], [], []
        directory = root / "feedback/episodes" / episode_id
        checks.require(len(record["attempts"]) <= 2, "Generation correction budget exceeded")
        for number, attempt in enumerate(record["attempts"], 1):
            folder = directory / f"attempt-{number}"
            request = read_json(folder / "request.json")
            context = base_context if number == 1 else core["prompts"].repair_context(base_context, pd.DataFrame({"variant": excluded}), set(eligible.variant), accepted, rejected, config, 2, number)
            checks.equal(request["context"], context, "Attempt input contains unavailable or changed information")
            checks.equal(request["parameters"], config.model_dump(), "Attempt parameters differ")
            checks.equal(request["prompt_metadata"], plan["prompt_metadata"], "Attempt prompt snapshot differs")
            checks.require(request["call_number"] == attempt["call_number"] == calls + 1, "Reserved call order differs")
            payload = core["agent"].build_request_payload(request["context"], config, plan["declaration"]["model_requested"], plan["declaration"]["api_mode"], plan["prompt_metadata"]["system_prompt"])
            checks.equal(request["payload"], payload, "Saved complete payload differs from its public context")
            checks.require(request["operation_id"] == attempt["operation_id"] and request["operation_id"] not in operation_ids, "Repeated or changed dispatch identity")
            checks.require(request["created_at"] <= attempt["started_at"], "Request was not saved before dispatch")
            operation_ids.add(request["operation_id"])
            calls += 1
            if not (folder / "response.json").is_file():
                checks.require(record["status"] in {"unknown", "interrupted", "failed"} and ((folder / "error.json").is_file() or record.get("error_code") == "UNKNOWN_REQUEST_OUTCOME"), "Missing response lacks an explicit failure record")
                missing_usage += 1
                checks.require(number == len(record["attempts"]), "Unknown request outcome was retried")
                continue
            response = read_json(folder / "response.json")
            counted = usage(response.get("usage"))
            checks.equal(attempt["usage"], counted, "Response usage differs from dispatch accounting")
            reported_tokens += counted["total_tokens"] or 0
            missing_usage += counted["missing_usage_count"]
            checks.equal(response["prompt_metadata"], plan["prompt_metadata"], "Response prompt snapshot differs")
            if not (folder / "validation.json").is_file():
                checks.require(record.get("error_code") == "MODEL_IDENTITY_MISMATCH", "Response validation record missing")
                continue
            checks.equal(response["model_requested"], plan["declaration"]["model_requested"], "Response route differs")
            checks.equal(response["api_mode"], plan["declaration"]["api_mode"], "Response API mode differs")
            try:
                if not response["complete"]:
                    raise ValueError("Incomplete response")
                analysis, additions, rejected = core["agent"].collect_proposals(response["text"], set(eligible.variant), config, accepted,
                    [row["id"] for row in request["context"].get("knowledge", {}).get("rules", [])], config.proposal_count - len(accepted))
                accepted.extend(additions)
                analyses.append(analysis)
            except ValueError:
                rejected = [{"variant": None, "reason": "ValueError"}]
            saved_validation = read_json(folder / "validation.json")
            checks.equal(saved_validation["accepted"], accepted, "Raw response accepted candidates differ")
            checks.equal(saved_validation["rejected"], rejected, "Raw response rejected candidates differ")
            checks.equal(saved_validation["analysis"], analyses, "Raw public analysis differs")
            try:
                body = json.loads(response["text"])
            except (ValueError, TypeError):
                body = None
            original_count = len(body["proposals"]) if isinstance(body, dict) and isinstance(body.get("proposals"), list) else None
            checks.equal(saved_validation["requested_count"], request["context"]["proposal_count"], "Validation request budget differs")
            checks.equal(saved_validation["original_response_count"], original_count, "Raw response proposal count differs")
            discarded = max(0, original_count - request["context"]["proposal_count"]) if original_count is not None else None
            checks.equal(saved_validation["discarded_after_ordered_prefix"], discarded, "Ordered-prefix discard count differs")
        completed_arm = record["status"] == "completed"
        complete += int(completed_arm)
        if completed_arm or any((directory / selector).is_dir() for selector in ("P0", "P1")):
            checks.require(len(accepted) == len({row["variant"] for row in accepted}) == 40, "Replayed arm did not generate 40 valid candidates")
            checks.equal(read_json(directory / "accepted-proposals.json"), accepted, "Final accepted proposals differ from raw responses")
        for selector, training in (("P0", initial), ("P1", updated)):
            folder = directory / selector
            if not (folder / "model-spec.json").is_file():
                checks.require(not completed_arm, "Completed selector model record missing")
                continue
            spec = read_json(folder / "model-spec.json")
            checks.equal(read_json(folder / "training-ids.json"), list(training.variant), "Selector training data differs")
            checks.equal(spec["spec"], prepared["source"]["spec"], "Selector model family or parameters changed")
            checks.equal(spec["config"], base_config.model_dump(), "Selector model configuration changed")
            if not (folder / "all-scores.csv").is_file():
                checks.require(not completed_arm, "Completed selector scoring record missing")
                continue
            scores = table(folder / "all-scores.csv", scores=True)
            checks.equal(list(scores.variant), [row["variant"] for row in accepted], "Selector candidate batch is not shared")
            predictions(checks, core, spec["spec"], base_config, training, scores, recompute)
            if not (folder / "selected.json").is_file():
                checks.require(not completed_arm, "Completed selector selection record missing")
                continue
            selected = read_json(folder / "selected.json")
            checks.require(all("actual_fitness" not in row and "fitness" not in row for row in selected), "Selector selection contains labels")
            ids = [row["variant"] for row in selected]
            checks.equal(ids, ranked(scores, config.k), "Selector Top-k differs from saved scores")
            selected_values(checks, scores, selected, proposals=accepted)
            if not (folder / "recommendations.csv").is_file():
                checks.require(not completed_arm and not (folder / "result.json").is_file(), "Completed selector feedback record missing")
                continue
            feedback = table(folder / "recommendations.csv")
            checks.equal(list(feedback.variant), ids, "Feedback includes unselected IDs")
            event_order(folder / "events.jsonl", "selected", "feedback", ids, checks)
            selected_values(checks, scores, selected, feedback=feedback, proposals=accepted)
            checks.observations(feedback, "actual_fitness")
            descriptions(feedback, prepared["reference"], core, checks)
            queries.extend((episode["state_id"], value) for value in ids)
            if not (folder / "result.json").is_file():
                checks.require(not completed_arm, "Completed selector result missing")
                continue
            result = read_json(folder / "result.json")
            checks.equal(result["selected_ids"], ids, "Selector summary IDs differ")
            checks.require(result["training_count"] == len(training) and result["scored_count"] == len(scores) and result["assays"] == len(feedback) == config.k, "Selector accounting differs")
            if completed_arm or "selectors" in record:
                checks.equal(record["selectors"][selector], result, "Selector state summary differs")
            checks.equal(result["mean_fitness"], float(feedback.actual_fitness.mean()), "Selector feedback mean differs")
            checks.equal(result["max_fitness"], float(feedback.actual_fitness.max()), "Selector feedback maximum differs")
            selectors += 1
    checks.require(calls == state["calls"] and calls <= plan["max_calls"], "Generation-call denominator differs")
    checks.equal(reported_tokens, state["reported_known_total_tokens"], "Known token total differs")
    checks.equal(missing_usage, state["missing_usage_count"], "Missing usage denominator differs")
    all_queries = prefix_queries + queries
    return {"planned_generator_arms": len(plan["episodes"]), "completed_generator_arms": complete,
            "planned_selector_results": plan["planned_selector_results"], "completed_selector_results": selectors,
            "generation_calls": calls, "max_generation_calls": plan["max_calls"],
            "reported_known_total_tokens": reported_tokens, "missing_usage_calls": missing_usage,
            "complete_total_tokens": reported_tokens if not missing_usage else None,
            "common_prefix_assays": len(prefix_queries), "selector_policy_assays": len(queries),
            "unique_new_variant_ids": len({variant for _, variant in all_queries}), "unique_new_state_variant_pairs": len(set(all_queries)),
            "selector_replay_is_independent_generation": False, "status": state["status"]}


def audit_records(directory, *, official_s1=None, recompute_scores=True):
    """Read only the exported bundle; optionally verify already disclosed labels against S1."""
    root, checks = Path(directory).resolve(), Checks()
    try:
        manifest = read_json(root / "manifest.json")
        checks.require(manifest["format_version"] == FORMAT, "Unsupported public record format")
        expected = set(manifest["files"]) | {"manifest.json"}
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        checks.require(actual == expected, "Public file inventory differs")
        for relative in actual:
            path = safe_path(root, relative)
            checks.require(not path.is_symlink(), "Public export contains a symbolic link")
            if path.suffix == ".json":
                public_value(read_json(path), field_names=relative != "source/configs/statistical-definitions-v2.json")
            elif path.name.endswith((".csv", ".csv.gz")):
                check_csv(path)
            elif path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        public_value(json.loads(line))
        checks.require(set(manifest["source_snapshot"]["files"]) == {"source/" + value for value in SNAPSHOT_FILES}, "Source snapshot inventory differs")
        counts = {}
        with snapshot_core(root) as core:
            if "strong" in manifest["counts"]:
                counts["strong"] = _audit_strong(root, core, checks, recompute_scores)
            if "feedback" in manifest["counts"]:
                counts["feedback"] = _audit_feedback(root, core, checks, recompute_scores)
        checks.equal(counts, manifest["counts"], "Exported study counts differ from saved records")
        if official_s1 is not None:
            source = Path(official_s1)
            if source.name.lower().startswith(".env"):
                raise ValueError("Official measurement input cannot be an environment file")
            original = pd.read_excel(source, sheet_name="Supplementary Dataset 1", usecols=["Variants", "Fitness"]) if source.suffix.lower() == ".xlsx" else pd.read_csv(source, usecols=["Variants", "Fitness"])
            lookup = original.set_index("Variants").Fitness
            checks.require(not lookup.index.duplicated().any(), "Official S1 contains duplicate IDs")
            for variant, value in checks.measured.items():
                checks.equal(value, float(lookup.loc[variant]), "Disclosed measurement differs from official S1")
        return {"passed": True, "checks": checks.count, "counts": counts, "recomputed_predictions": bool(recompute_scores),
                "maximum_prediction_absolute_error": checks.max_prediction_error, "prediction_tolerance": TOLERANCE,
                "official_s1_checked": official_s1 is not None, "disclosed_measurement_ids": len(checks.measured), "errors": []}
    except (ValueError, KeyError, OSError, TypeError, AssertionError) as error:
        return {"passed": False, "checks": checks.count, "errors": [{"type": type(error).__name__, "message": str(error) if isinstance(error, ValueError) else "Public record reconstruction failed"}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("--project-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--feedback", type=Path)
    export.add_argument("--strong", type=Path)
    export.add_argument("--terminal-confirmed", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("directory", type=Path)
    audit.add_argument("--official-s1", type=Path)
    audit.add_argument("--skip-prediction-rebuild", action="store_true")
    options = parser.parse_args(argv)
    if options.action == "export":
        result = export_records(options.project_root, options.output, feedback=options.feedback, strong=options.strong, terminal_confirmed=options.terminal_confirmed)
        print(json.dumps({"exported": True, "files": len(result["files"]) + 1, "counts": result["counts"]}, ensure_ascii=False))
        return 0
    result = audit_records(options.directory, official_s1=options.official_s1, recompute_scores=not options.skip_prediction_rebuild)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
