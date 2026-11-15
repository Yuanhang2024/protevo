"""Evaluate saved zero-context rankings; new API experiments require explicit authorization."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "zero_context"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stats import load_global_fitness
from protevo.statistical_definitions import DEFINITION_VERSION, ranking_metrics, summarize_usage

RANK_SYSTEM = "You are an expert in protein engineering and deep mutational scanning data."
RANK_TASK = ("Rank the following GB1 domain four-site variants (mutation positions 39, 40, 41 and 54 "
             "relative to wild type VDGV) from highest to lowest measured IgG-Fc binding fitness "
             "(wild type = 1, higher is better). Return all variants exactly once, best first. "
             "No experimental data is provided; use your own knowledge.")
RECALL_TASK = ("List the ten four-letter GB1 variants (mutation positions 39, 40, 41 and 54, wild type "
               "VDGV) with the highest measured IgG-Fc binding fitness in the published deep mutational "
               "scanning dataset, best first.")


def rank_schema(n):
    return {"type": "json_schema", "json_schema": {
        "name": "variant_ranking", "strict": True, "schema": {
            "type": "object", "properties": {"ranked_variants": {"type": "array", "items": {"type": "string"},
                                                                       "minItems": n, "maxItems": n}},
            "required": ["ranked_variants"], "additionalProperties": False}}}


RECALL_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "variant_recall", "strict": True, "schema": {
        "type": "object", "properties": {"top_variants": {"type": "array", "items": {"type": "string"},
                                                                  "minItems": 10, "maxItems": 10}},
        "required": ["top_variants"], "additionalProperties": False}}}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def require_new_output(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; saved responses and statistics are preserved")
    return output


def samples(global_fit):
    global_fit = global_fit.sort_values("variant").reset_index(drop=True)
    rng = np.random.default_rng(20261113)
    return [global_fit.sample(n=50, random_state=int(rng.integers(1 << 31))).variant.tolist() for _ in range(3)]


def batch_report(number, ranked, expected, fitness):
    value = ranking_metrics(ranked, expected, fitness, k=10)
    raw = value["spearman_position_vs_fitness"]
    tau = value["kendall_position_vs_fitness"]
    return {"batch": number, "returned": len(ranked), "valid": len(ranked), **value,
            "spearman_model_rank_vs_actual": round(raw, 4) if raw is not None else None,
            "kendall_tau": round(tau, 4) if tau is not None else None,
            "legacy_raw_field_direction": "position versus fitness; smaller position is better",
            "top10_in_batch_hits": value["top_k_overlap"], "batch_actual_top": value["true_top_variants"][:3]}


def recall_report(recalled, global_fit):
    if not isinstance(recalled, list) or len(recalled) != 10 or any(not isinstance(value, str) for value in recalled) or len(set(recalled)) != 10:
        raise ValueError("Recall must return ten distinct variant ID strings")
    fitness = global_fit.set_index("variant").fitness.to_dict()
    order = global_fit.sort_values(["fitness", "variant"], ascending=[False, True]).variant.tolist()
    top10, top100 = set(order[:10]), set(order[:100])
    return {"recalled": recalled, "in_global_top10": [value for value in recalled if value in top10],
            "in_global_top100": [value for value in recalled if value in top100],
            "recall_fitness": {value: float(fitness[value]) if value in fitness else None for value in recalled},
            "true_top_ties": "variant ID ascending"}


def analyze_saved(source, output, global_fit=None):
    source, output = Path(source).resolve(), require_new_output(output)
    global_fit = load_global_fitness() if global_fit is None else global_fit
    fitness = global_fit.set_index("variant").fitness.to_dict()
    expected = samples(global_fit)
    batches, all_variants = [], []
    for number in range(1, 4):
        body = json.loads((source / f"rank_batch{number}.json").read_text(encoding="utf-8"))
        ranked = body["ranked_variants"]
        batches.append(batch_report(number, ranked, expected[number - 1], fitness))
        all_variants.extend(ranked)
    recalled = json.loads((source / "free_recall.json").read_text(encoding="utf-8"))["top_variants"]
    previous = json.loads((source / "summary.json").read_text(encoding="utf-8")) if (source / "summary.json").is_file() else {}
    usage = json.loads((source / "usage.json").read_text(encoding="utf-8")) if (source / "usage.json").is_file() else None
    resource = summarize_usage([row.get("usage") for row in usage]) if usage is not None else {
        "request_count": None, "reported_tokens_sum": 0, "missing_usage_calls": None, "total_tokens": None,
        "mean_tokens": None, "median_tokens": None, "complete_usage": False, "reason": "saved_usage_unavailable"}
    summary = {"definition_version": DEFINITION_VERSION, "status": "passed", "model_requests": 0,
               "model": previous.get("model"), "batches": batches, "free_recall": recall_report(recalled, global_fit),
               "ranking_total_entries": len(all_variants), "ranking_distinct_variants": len(set(all_variants)),
               "input_reconstruction": "saved original sampling seed 20261113 over variant-ID-sorted measured S1",
               "source_directory": str(source), "usage": resource}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", summary)
    pd.DataFrame(batches).drop(columns=["true_top_variants", "batch_actual_top"]).to_csv(output / "ranking.csv", index=False, encoding="utf-8-sig")
    return summary


def load_client(*, allow_api=False):
    if not allow_api:
        raise PermissionError("A new model experiment requires explicit --allow-api")
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    model = os.environ["OPENAI_MODEL"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None,
                    timeout=180.0, max_retries=0)
    return client, model


def call(client, model, system, user_text, response_format, usage_log, tag, output, *, allow_api=False):
    if not allow_api:
        raise PermissionError("A model dispatch requires explicit authorization")
    if any((Path(output) / f"{tag}{suffix}").exists() for suffix in (".json", ".request.json", ".error.json")):
        raise ValueError("This request tag already has a saved record; it cannot be dispatched again")
    operation = uuid4().hex
    request = {"operation_id": operation, "model": model, "messages": [{"role": "system", "content": system},
               {"role": "user", "content": user_text}], "response_format": response_format, "max_completion_tokens": 8000}
    write_new(Path(output) / f"{tag}.request.json", {"status": "dispatching", **request})
    try:
        response = client.chat.completions.create(model=model, messages=request["messages"],
                    response_format=response_format, max_completion_tokens=8000)
    except Exception as error:
        status = getattr(error, "status_code", None)
        public = {"tag": tag, "operation_id": operation, "status": "failed", "usage": None,
                  "error_type": type(error).__name__, "status_code": status if type(status) is int else None,
                  "outcome_unknown": status is None}
        usage_log.append(public)
        write_new(Path(output) / f"{tag}.error.json", public)
        raise RuntimeError("The model request did not complete; its outcome was preserved without retry") from None
    text = response.choices[0].message.content or ""
    usage_log.append({"tag": tag, "operation_id": operation, "model_returned": getattr(response, "model", None),
                      "finish_reason": response.choices[0].finish_reason,
                      "usage": response.usage.model_dump(mode="json") if response.usage else None})
    with (Path(output) / f"{tag}.json").open("x", encoding="utf-8") as handle:
        handle.write(text)
    if response.choices[0].finish_reason != "stop":
        raise RuntimeError("The model response was incomplete and is preserved without retry")
    return text


def run_api(output, *, allow_api=False):
    if not allow_api:
        raise PermissionError("The run-api action requires --allow-api")
    output = require_new_output(output)
    output.mkdir(parents=True, exist_ok=False)
    client, model = None, None
    usage, batches = [], []
    try:
        global_fit = load_global_fitness()
        fitness = global_fit.set_index("variant").fitness.to_dict()
        client, model = load_client(allow_api=True)
        for number, expected in enumerate(samples(global_fit), 1):
            text = call(client, model, RANK_SYSTEM, json.dumps({"task": RANK_TASK, "variants": expected}), rank_schema(len(expected)),
                        usage, f"rank_batch{number}", output, allow_api=True)
            ranked = json.loads(text)["ranked_variants"]
            batches.append(batch_report(number, ranked, expected, fitness))
        text = call(client, model, RANK_SYSTEM, json.dumps({"task": RECALL_TASK}), RECALL_FORMAT, usage, "free_recall", output, allow_api=True)
        recall = recall_report(json.loads(text)["top_variants"], global_fit)
        summary = {"definition_version": DEFINITION_VERSION, "status": "completed", "model": model, "batches": batches,
                   "free_recall": recall, "usage": summarize_usage([row.get("usage") for row in usage])}
    except Exception as error:
        summary = {"definition_version": DEFINITION_VERSION, "status": "failed", "model": model, "batches": batches,
                   "error_type": type(error).__name__, "usage": summarize_usage([row.get("usage") for row in usage])}
    finally:
        if client is not None:
            client.close()
    write_new(output / "usage.json", usage)
    write_new(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "run-api"), nargs="?", default="analyze")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-api", action="store_true")
    options = parser.parse_args(argv)
    if options.action == "run-api" and not options.allow_api:
        parser.error("run-api requires explicit --allow-api; default analysis never loads a model connection")
    summary = run_api(options.output, allow_api=True) if options.action == "run-api" else analyze_saved(options.source, options.output)
    print(json.dumps({"status": summary["status"], "output": str(options.output), "definition_version": DEFINITION_VERSION,
                      "usage": summary["usage"], "model_requests": summary.get("model_requests")}, ensure_ascii=False))
    return 0 if summary["status"] in ("passed", "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
