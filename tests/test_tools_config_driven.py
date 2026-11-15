# 合成ProteinConfig驱动的工具层测试：手工构造dataclass（不走configure_protein），无网络、无真实数据。
"""Config-driven ToolBox / tool schemas / system prompt tests over hand-built synthetic ProteinConfigs."""
from itertools import product
import json

import pandas as pd
import pytest

from protevo.protein_config import ProteinConfig
from protevo.prompt_builder import build_system_prompt
from protevo.schema import ExperimentConfig
from protevo.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolBox, build_tool_schemas

LETTERS = "ACDE"  # 合成字母表，仅用于生成2位点候选串


@pytest.fixture
def string_config():
    # 2位点突变串格式：参考AC，可变位点为0-indexed序列下标0/1
    return ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1], variant_format="mutation_string")


@pytest.fixture
def notation_config():
    # 记谱格式：可变位点为1-indexed蛋白编号1/2，野生型A/C
    return ProteinConfig(reference_sequence="AC", mutable_positions=[1, 2],
                         variant_format="mutation_notation", wt_by_position={1: "A", 2: "C"})


@pytest.fixture
def experiment_config():
    return ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None)


def make_toolbox(protein_config, experiment_config, known_variants, eligible):
    known = pd.DataFrame([{"variant": variant, "fitness": float(index), "mutations": "synthetic"}
                          for index, variant in enumerate(known_variants)])
    # 预测桩：返回与长度相关的有限分数，对任何格式的变体串都安全
    return ToolBox(known=known, eligible=set(eligible),
                   predict_fn=lambda sequences: [float(len(sequence)) for sequence in sequences],
                   config=experiment_config, protein_config=protein_config)


def test_mutation_string_accepts_legal_and_rejects_invalid(string_config, experiment_config):
    variants = ["".join(pair) for pair in product(LETTERS, repeat=2)]
    toolbox = make_toolbox(string_config, experiment_config, known_variants=variants[:4], eligible=variants[4:])
    result = toolbox.execute("predict_fitness",
                             json.dumps({"sequences": ["DD", "CE", "DD", "AC", "DDD", "ZC", 5]}))
    # 合法未测候选被评分；重复项DD只评一次
    assert {row["variant"] for row in result["scored"]} == {"DD", "CE"}
    reasons = {row["sequence"]: row["reason"] for row in result["errors"]}
    assert "already measured" in reasons["AC"]                       # 已测
    assert "length must be 2" in reasons["DDD"]                      # 错误长度
    assert "amino-acid alphabet" in reasons["ZC"]                    # 非法残基
    assert "must be a string" in reasons[5]                          # 非字符串
    assert toolbox.budget.score_calls_used == 2                      # 重复与非法项不消耗预算


def test_mutation_string_round_context_uses_config(string_config, experiment_config):
    variants = ["".join(pair) for pair in product(LETTERS, repeat=2)]
    toolbox = make_toolbox(string_config, experiment_config, known_variants=variants[:4], eligible=variants[4:])
    context = toolbox.execute("get_experiment_history", "{}")["round_context"]
    assert context["reference_variant"] == "AC" and context["positions"] == [0, 1]
    assert context["required_recommendations"] == 2


def test_submit_rejects_duplicates_under_config_path(string_config, experiment_config):
    toolbox = make_toolbox(string_config, experiment_config, known_variants=["AA"], eligible=["CC", "CD", "DC"])
    rejected = toolbox.execute("submit_recommendations",
                               json.dumps({"sequences": ["CC", "CC"], "reasons": ["a", "b"]}))
    assert "unique" in rejected["error"]


def test_notation_accepts_legal_and_rejects_undeclared_position(notation_config, experiment_config):
    toolbox = make_toolbox(notation_config, experiment_config, known_variants=["A2D"],
                           eligible=["A1C", "C2D", "A1C,C2D", "WT"])
    result = toolbox.execute("predict_fitness",
                             json.dumps({"sequences": ["A1C", "A1C,C2D", "WT", "A9C", "1C"]}))
    # "A1C"式记谱与野生型记法（WT→[]）均合法
    assert {row["variant"] for row in result["scored"]} == {"A1C", "A1C,C2D", "WT"}
    reasons = {row["sequence"]: row["reason"] for row in result["errors"]}
    assert "position 9" in reasons["A9C"]                # 未声明位点
    assert "mutation notation" in reasons["1C"]          # 记谱不可解析
    assert toolbox.budget.score_calls_used == 3


def test_build_tool_schemas_config_driven(string_config, notation_config):
    for config in (string_config, notation_config):
        schemas = build_tool_schemas(config)
        assert tuple(schema["function"]["name"] for schema in schemas) == TOOL_NAMES
        dumped = json.dumps(schemas, ensure_ascii=False)
        assert "GB1" not in dumped and "four-letter" not in dumped
    string_dump = json.dumps(build_tool_schemas(string_config), ensure_ascii=False)
    predict = build_tool_schemas(string_config)[0]["function"]
    assert "0,1" in predict["description"] and "2 mutable positions" in predict["description"]
    assert "2-letter" in predict["parameters"]["properties"]["sequences"]["description"]
    assert "0,1" in string_dump
    notation_predict = build_tool_schemas(notation_config)[0]["function"]
    assert "1,2" in notation_predict["description"] and "A1C" in notation_predict["description"]


def test_build_system_prompt_parameterized(string_config, notation_config):
    prompt = build_system_prompt(string_config, protein_name="合成肽酶", assay_description="结合活性")
    assert "GB1" not in prompt
    assert "合成肽酶" in prompt and "结合活性" in prompt
    assert "0/1" in prompt and "AC" in prompt and "2 位点" in prompt
    assert "submit_recommendations" in prompt and "科学纪律" in prompt   # 结构与scientist_v2.txt对等
    assert "目标蛋白" in build_system_prompt(string_config)              # 默认参数可用
    notation_prompt = build_system_prompt(notation_config)
    assert "GB1" not in notation_prompt and "1/2" in notation_prompt and "式突变记谱" in notation_prompt


def test_build_tool_schemas_none_matches_legacy_constant():
    assert build_tool_schemas(None) == TOOL_SCHEMAS
    # 旧常量原样保留GB1描述，保证agent_v2与run_supplementary的旧import不破
    assert "GB1" in TOOL_SCHEMAS[0]["function"]["description"]
    assert "four-letter" in TOOL_SCHEMAS[3]["function"]["parameters"]["properties"]["sequences"]["description"]
