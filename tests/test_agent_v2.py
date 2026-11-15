"""Mocked-API tests for the v2 tool-calling agent; no credentials or network access."""
from itertools import count, product
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from protevo.agent_v2 import NO_TOOL_CALLS_NUDGE, ToolCallingAgent, run_agent_round
from protevo.data import describe_variant
from protevo.schema import ExperimentConfig, KnowledgeBase
from protevo.tools import TOOL_NAMES, TOOL_SCHEMAS, AgentBudget, ToolBox

# 合成数据与公式仅用于测试，不代表任何实验结果。
ALPHABET = "ACDE"
IDS = count()


def formula(variant):
    return float(sum(ALPHABET.index(residue) * (index + 1) for index, residue in enumerate(variant)))


@pytest.fixture
def world():
    reference = list("A" * 56)
    for position, residue in zip((39, 40, 41, 54), "VDGV"):
        reference[position - 1] = residue
    reference = "".join(reference)
    variants = ["".join(triple) for triple in product(ALPHABET, repeat=4)]
    known = pd.DataFrame([{**describe_variant(variant, reference), "fitness": formula(variant)}
                          for variant in variants[:24]])
    knowledge = KnowledgeBase.model_validate({"rules": [
        {"id": "synthetic-charge", "scope": "synthetic test", "condition": "charge-changing substitution",
         "action": "Consider lysine as positively charged under typical conditions.",
         "source": "synthetic test fixture", "explanation": "Test-only rule."},
        {"id": "synthetic-epistasis", "scope": "synthetic test", "condition": "combining substitutions",
         "action": "Combined effects need not be additive.", "source": "synthetic test fixture",
         "explanation": "Test-only rule."}]})
    config = ExperimentConfig(k=3, proposal_count=3, score_budget=3, knowledge_path=None)
    return SimpleNamespace(reference=reference, variants=variants, known=known,
                           eligible=set(variants[24:]), formula=formula,
                           predict_fn=lambda sequences: [formula(sequence) for sequence in sequences],
                           config=config, knowledge=knowledge)


def make_toolbox(world, budget=None, knowledge="default"):
    return ToolBox(known=world.known, eligible=set(world.eligible), predict_fn=world.predict_fn,
                   config=world.config, knowledge=world.knowledge if knowledge == "default" else knowledge,
                   budget=budget)


def ranked_eligible(world):
    return sorted(world.eligible, key=lambda variant: (-world.formula(variant), variant))


def _response(tool_calls=None, content=None, usage=(20, 40)):
    calls = None
    if tool_calls:
        calls = [SimpleNamespace(id=f"call_{next(IDS)}", type="function",
                                 function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
                 for name, arguments in tool_calls]
    message = SimpleNamespace(role="assistant", content=content, tool_calls=calls)
    usage_obj = SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1], total_tokens=usage[0] + usage[1])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
                           usage=usage_obj, id=f"resp-{next(IDS)}", model="mock-scientist")


class ScriptedCompletions:
    """In-process chat.completions stand-in; pops scripted responses and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("scripted client received an unexpected extra API call")
        return self.script.pop(0)


def scripted_client(script):
    completions = ScriptedCompletions(script)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tool_schemas_follow_openai_function_calling_format():
    assert tuple(schema["function"]["name"] for schema in TOOL_SCHEMAS) == TOOL_NAMES
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"].strip()
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert set(parameters.get("required", [])) <= set(parameters["properties"])
        json.dumps(schema)


def test_check_budget_reports_remaining_allowances(world):
    budget = AgentBudget(max_api_calls=15, max_total_tokens=50000, max_score_calls=40)
    budget.consume_api_call(1200)
    budget.consume_scores(7)
    result = make_toolbox(world, budget=budget).execute("check_budget", "{}")
    assert result["api_calls_remaining"] == 14
    assert result["tokens_remaining"] == 48800
    assert result["score_calls_remaining"] == 33
    assert result["submitted"] is False


def test_predict_fitness_scores_eligible_candidates_once(world):
    toolbox = make_toolbox(world)
    result = toolbox.execute("predict_fitness",
                             json.dumps({"sequences": ["DDDD", "EEEE", "DDDD", "XXXX", world.variants[0]]}))
    assert [row["variant"] for row in result["scored"]] == ["EEEE", "DDDD"]
    assert result["scored"][0]["predicted_fitness"] == 30.0 and result["scored"][0]["rank"] == 1
    assert {row["sequence"] for row in result["errors"]} == {"XXXX", "AAAA"}
    assert toolbox.budget.score_calls_used == 2
    again = toolbox.execute("predict_fitness", json.dumps({"sequences": ["DDDD"]}))
    assert again["scored"][0]["predicted_fitness"] == 20.0
    assert toolbox.budget.score_calls_used == 2  # cached rescore is free


def test_predict_fitness_enforces_hard_score_cap(world):
    toolbox = make_toolbox(world, budget=AgentBudget(max_score_calls=6))
    fresh = [variant for variant in ranked_eligible(world) if variant not in ("EEEE", "DDDD")]
    result = toolbox.execute("predict_fitness", json.dumps({"sequences": fresh[:8]}))
    assert len(result["scored"]) == 6
    assert [row["sequence"] for row in result["not_scored"]] == fresh[6:8]
    assert result["remaining_score_budget"] == 0
    assert toolbox.budget.exhausted_reason() == "max_score_calls"
    more = toolbox.execute("predict_fitness", json.dumps({"sequences": [fresh[8]]}))
    assert more["scored"] == [] and more["not_scored"][0]["sequence"] == fresh[8]
    assert toolbox.budget.score_calls_used == 6


def test_query_knowledge_matches_keywords_and_lists_ids(world):
    toolbox = make_toolbox(world)
    hit = toolbox.execute("query_knowledge", json.dumps({"query": "side chain charge changes"}))
    assert [rule["id"] for rule in hit["matched_rules"]] == ["synthetic-charge"]
    miss = toolbox.execute("query_knowledge", json.dumps({"query": "zzz qqq"}))
    assert miss["matched_rules"] == []
    assert set(miss["available_rule_ids"]) == {"synthetic-charge", "synthetic-epistasis"}
    empty = make_toolbox(world, knowledge=None)
    assert empty.execute("query_knowledge", json.dumps({"query": "charge"}))["matched_rules"] == []


def test_get_experiment_history_returns_known_summary_and_feedback(world):
    feedback = [{"round": 1, "recommendations": [{"variant": "ACCC", "actual_fitness": 1.5, "reason": "synthetic"}]}]
    toolbox = ToolBox(known=world.known, eligible=set(world.eligible), predict_fn=world.predict_fn,
                      config=world.config, feedback_history=feedback)
    result = toolbox.execute("get_experiment_history", "{}")
    assert result["known_data"]["n_known"] == 24
    assert result["known_data"]["best_fitness"] == float(world.known.fitness.max())
    assert result["previous_recommendations"] == feedback
    assert result["round_context"]["eligible_candidate_count"] == 232
    assert result["round_context"]["required_recommendations"] == 3
    assert result["scored_variants_this_round"] == []


def test_submit_recommendations_validates_and_records(world):
    toolbox = make_toolbox(world)
    assert "error" in toolbox.execute("submit_recommendations",
                                      json.dumps({"sequences": ["DDDD", "EEEE"], "reasons": ["a", "b"]}))
    rejected = toolbox.execute("submit_recommendations",
                               json.dumps({"sequences": ["XXXX", "EEEE", "DDDD"], "reasons": ["a", "b", "c"]}))
    assert "not in the eligible" in rejected["error"]
    duplicate = toolbox.execute("submit_recommendations",
                                json.dumps({"sequences": ["DDDD", "DDDD", "EEEE"], "reasons": ["a", "b", "c"]}))
    assert "unique" in duplicate["error"]
    accepted = toolbox.execute("submit_recommendations",
                               json.dumps({"sequences": ["EEEE", "EEED", "EEDC"],
                                           "reasons": ["best", "second", "third"]}))
    assert accepted["status"] == "submitted" and accepted["budget_exhausted"] is False
    assert toolbox.submission["predicted_fitness"] == [None, None, None]
    again = toolbox.execute("submit_recommendations",
                            json.dumps({"sequences": ["EEEE", "EEED", "EEDC"], "reasons": ["a", "b", "c"]}))
    assert "already submitted" in again["error"]


def test_execute_returns_errors_as_values_never_raises(world):
    toolbox = make_toolbox(world)
    assert "error" in toolbox.execute("predict_fitness", "not-json")
    assert "error" in toolbox.execute("unknown_tool", "{}")
    assert "error" in toolbox.execute("predict_fitness", json.dumps({"sequences": "AAAA"}))
    assert "error" in toolbox.execute("check_budget", json.dumps({"unexpected": 1}))


def test_agent_happy_path_submits_after_autonomous_tool_calls(world, tmp_path):
    ranked = ranked_eligible(world)
    probe, top3 = ranked[:10], ranked[:3]
    script = [
        _response(tool_calls=[("get_experiment_history", {}), ("check_budget", {})]),
        _response(tool_calls=[("predict_fitness", {"sequences": probe})]),
        _response(tool_calls=[("submit_recommendations", {"sequences": top3,
                                                          "reasons": ["reason 1", "reason 2", "reason 3"]})]),
    ]
    client, completions = scripted_client(script)
    log_path = tmp_path / "agent-v2.jsonl"
    record, toolbox = run_agent_round(2, world.known, world.eligible, world.predict_fn, world.config,
                                      client=client, model="mock-scientist", log_path=log_path)
    assert record["status"] == "submitted"
    assert record["budget_exhausted"] is False and record["stop_reason"] is None
    assert record["submission"]["sequences"] == top3
    assert record["submission"]["budget_exhausted"] is False
    assert record["api_calls_used"] == 3 and record["score_calls_used"] == 10
    assert record["tool_call_sequence"] == ["get_experiment_history", "check_budget",
                                            "predict_fitness", "submit_recommendations"]
    assert [len(request["messages"]) for request in completions.requests] == [2, 5, 7]
    for request in completions.requests:
        assert request["tools"] == TOOL_SCHEMAS and request["tool_choice"] == "auto"
        assert request["model"] == "mock-scientist"
    assert "232" in completions.requests[0]["messages"][1]["content"]
    second = completions.requests[1]["messages"]
    tool_messages = [message for message in second if message["role"] == "tool"]
    assistant = [message for message in second if message["role"] == "assistant"][0]
    assert {message["tool_call_id"] for message in tool_messages} == {call["id"] for call in assistant["tool_calls"]}
    for message in tool_messages:
        payload = json.loads(message["content"])
        assert isinstance(payload, dict) and "error" not in payload
    entries = read_log(log_path)
    assert [entry["kind"] for entry in entries] == ["round_start", "api_call", "tool_call", "tool_call",
                                                    "api_call", "tool_call", "api_call", "tool_call", "round_end"]
    required = {"timestamp", "call_id", "round", "tool_name", "arguments", "result",
                "tokens_used", "context_length", "model"}
    for entry in entries:
        assert required <= set(entry) and entry["round"] == 2 and entry["model"] == "mock-scientist"
    api_calls = [entry for entry in entries if entry["kind"] == "api_call"]
    assert [entry["context_length"] for entry in api_calls] == [2, 5, 7]
    assert [entry["tokens_used"] for entry in api_calls] == [60, 60, 60]
    tool_calls = [entry for entry in entries if entry["kind"] == "tool_call"]
    assert [entry["tool_name"] for entry in tool_calls] == ["get_experiment_history", "check_budget",
                                                            "predict_fitness", "submit_recommendations"]
    assert [row["variant"] for row in tool_calls[2]["result"]["scored"]] == probe


def test_auto_submit_when_score_budget_exhausted(world, tmp_path):
    ranked = ranked_eligible(world)
    batches = [ranked[10:14], ranked[14:18]]
    expected = sorted(batches[0] + batches[1], key=lambda v: (-world.formula(v), v))[:3]
    script = [_response(tool_calls=[("predict_fitness", {"sequences": batch})]) for batch in batches]
    client, _ = scripted_client(script)
    log_path = tmp_path / "score-budget.jsonl"
    record, _ = run_agent_round(1, world.known, world.eligible, world.predict_fn, world.config,
                                client=client, model="mock-scientist", log_path=log_path,
                                budget=AgentBudget(max_api_calls=15, max_total_tokens=50000, max_score_calls=6))
    assert record["status"] == "auto_submitted"
    assert record["budget_exhausted"] is True and record["stop_reason"] == "max_score_calls"
    assert record["api_calls_used"] == 2 and record["score_calls_used"] == 6
    assert record["submission"]["sequences"] == expected
    assert record["submission"]["budget_exhausted"] is True
    assert all(score is not None for score in record["submission"]["predicted_fitness"])
    assert all("max_score_calls" in reason for reason in record["submission"]["reasons"])
    final = [entry for entry in read_log(log_path) if entry["kind"] == "tool_call"][-1]
    assert final["tool_name"] == "submit_recommendations"
    assert final["auto"] is True and final["budget_exhausted"] is True


def test_auto_submit_when_api_calls_exhausted_after_text_only_replies(world, tmp_path):
    script = [_response(content="Still thinking.") for _ in range(3)]
    client, completions = scripted_client(script)
    log_path = tmp_path / "api-budget.jsonl"
    record, _ = run_agent_round(1, world.known, world.eligible, world.predict_fn, world.config,
                                client=client, model="mock-scientist", log_path=log_path,
                                budget=AgentBudget(max_api_calls=3, max_total_tokens=50000, max_score_calls=40))
    assert record["status"] == "auto_submitted"
    assert record["stop_reason"] == "max_api_calls"
    assert record["api_calls_used"] == 3 and record["score_calls_used"] == 0
    assert record["submission"]["sequences"] == sorted(world.eligible)[:3]
    assert record["submission"]["predicted_fitness"] == [None, None, None]
    assert all("unscored deterministic fallback" in reason for reason in record["submission"]["reasons"])
    assert [len(request["messages"]) for request in completions.requests] == [2, 4, 6]
    nudges = [message for message in completions.requests[2]["messages"]
              if message.get("content") == NO_TOOL_CALLS_NUDGE]
    assert len(nudges) == 2


def test_auto_submit_when_token_budget_exhausted(world, tmp_path):
    ranked = ranked_eligible(world)
    script = [_response(tool_calls=[("predict_fitness", {"sequences": ranked[:4]})], usage=(30, 30)),
              _response(content="Analysis pending.", usage=(30, 30))]
    client, _ = scripted_client(script)
    record, _ = run_agent_round(1, world.known, world.eligible, world.predict_fn, world.config,
                                client=client, model="mock-scientist", log_path=tmp_path / "token-budget.jsonl",
                                budget=AgentBudget(max_api_calls=15, max_total_tokens=100, max_score_calls=40))
    assert record["status"] == "auto_submitted"
    assert record["stop_reason"] == "max_total_tokens"
    assert record["api_calls_used"] == 2 and record["total_tokens_used"] == 120
    assert record["score_calls_used"] == 4
    assert record["submission"]["sequences"] == ranked[:3]


def test_injected_client_requires_model(world):
    with pytest.raises(ValueError, match="model is required"):
        ToolCallingAgent(make_toolbox(world), client=object())
