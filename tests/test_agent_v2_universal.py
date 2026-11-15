"""agent_v2 蛋白无关路径测试。

三处去 GB1 硬编码的验证：① default_task_message 配置派生分支 ② 循环 tools 用配置派生 schema
③ run_agent_round 转发 protein_config 到 ToolBox/提示/开场白；GB1 旧路径（protein_config=None）
行为与文本逐字节不变。mock 客户端模式复用 test_agent_v2 的 SimpleNamespace 方案。
"""
import json
from itertools import count
from types import SimpleNamespace

import pandas as pd

from protevo.agent_v2 import DEFAULT_SYSTEM_PROMPT, ToolCallingAgent, default_task_message, run_agent_round
from protevo.protein_config import ProteinConfig
from protevo.schema import ExperimentConfig
from protevo.tools import TOOL_SCHEMAS, ToolBox, build_tool_schemas

IDS = count()
CONFIG = ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None)


def _toolbox(protein_config=None, eligible=None):
    known = pd.DataFrame([{"variant": "CC", "fitness": 1.0, "mutations": "synthetic"}])
    return ToolBox(known=known, eligible=eligible if eligible is not None else {"CE", "DC", "DE"},
                   predict_fn=lambda sequences: [float(len(sequence)) for sequence in sequences],
                   config=CONFIG, protein_config=protein_config)


def _response(tool_calls=None):
    calls = [SimpleNamespace(id=f"call_{next(IDS)}", type="function",
                             function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
             for name, arguments in tool_calls] if tool_calls else None
    message = SimpleNamespace(role="assistant", content=None, tool_calls=calls)
    usage = SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
                           usage=usage, id=f"resp-{next(IDS)}", model="mock")


class _Completions:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


# ---------- 开场白 ----------

def test_legacy_task_message_unchanged():
    message = default_task_message(3, _toolbox())
    assert message.startswith("Directed evolution round 3: design GB1 four-site variants (positions 39,40,41,54; ")


def test_config_derived_task_message():
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1], variant_format="mutation_string")
    message = default_task_message(1, _toolbox(cfg), cfg)
    assert "GB1" not in message and "2 configured mutable positions" in message
    indel_cfg = ProteinConfig(reference_sequence="A" * 60, mutable_positions=[3, 4],
                              variant_format="indel_notation", wt_full_sequence="A" * 60)
    message2 = default_task_message(1, _toolbox(indel_cfg), indel_cfg)
    assert "GB1" not in message2 and "60-residue reference" in message2


# ---------- schema 注入与旧路径恒等 ----------

def test_agent_default_schemas_and_prompt_unchanged():
    agent = ToolCallingAgent(_toolbox(), client=SimpleNamespace(), model="mock")
    assert agent.tool_schemas == TOOL_SCHEMAS and agent.system_prompt == DEFAULT_SYSTEM_PROMPT


# ---------- 端到端：run_agent_round(protein_config=...) ----------

def test_run_agent_round_config_driven_end_to_end(tmp_path):
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1], variant_format="mutation_string")
    known = pd.DataFrame([{"variant": "CC", "fitness": 1.0, "mutations": "synthetic"}])
    completions = _Completions([
        _response(tool_calls=[("predict_fitness", {"sequences": ["CE", "DC"]})]),
        _response(tool_calls=[("submit_recommendations",
                               {"sequences": ["CE", "DE"], "reasons": ["higher polarity", "novel pair"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    record, toolbox = run_agent_round(
        1, known, {"CE", "DC", "DE"}, lambda sequences: [1.0] * len(sequences), CONFIG,
        client=client, model="mock", protein_config=cfg, log_path=tmp_path / "agent.jsonl")

    # LLM 实际收到的 tools / 系统提示 / 开场白全部蛋白化（无 GB1）
    first = completions.requests[0]
    assert "GB1" not in json.dumps(first["tools"]) and "mutation-string" in json.dumps(first["tools"])
    assert "GB1" not in first["messages"][0]["content"]      # build_system_prompt 派生
    assert "GB1" not in first["messages"][1]["content"]      # default_task_message 派生
    # 提交被配置驱动的 ToolBox 接受
    assert record["status"] == "submitted"
    assert toolbox.submission["sequences"] == ["CE", "DE"]


def test_run_agent_round_legacy_path_uses_gb1_text():
    # 不传 protein_config：LLM 收到的 tools 与旧常量逐字节一致、开场白为 GB1 文案
    known = pd.DataFrame([{"variant": "CC", "fitness": 1.0, "mutations": "synthetic"}])
    completions = _Completions([
        _response(tool_calls=[("predict_fitness", {"sequences": ["CE"]})]),
        _response(tool_calls=[("submit_recommendations",
                               {"sequences": ["CE", "DC"], "reasons": ["a", "b"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    run_agent_round(1, known, {"CE", "DC", "DE"}, lambda sequences: [1.0] * len(sequences), CONFIG,
                    client=client, model="mock")
    first = completions.requests[0]
    assert first["tools"] == TOOL_SCHEMAS
    assert "design GB1 four-site variants" in first["messages"][1]["content"]
