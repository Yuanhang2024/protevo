"""Agent 提示语言（zh/en）测试：build_system_prompt 英文对等版 + agent_v2 / evolve / TUI 的 language 透传。

zh 为默认且与单语时期行为一致（显式 zh == 不传 language 的回归锚点）；en 全文无 CJK、含英文四段
首句与方向词 improving/reducing；非法 language ValueError；TUI open --lang en 后 run 的 mock
客户端收到英文 system prompt。mock 客户端复用 test_agent_v2_universal / test_tui 的脚手架模式。
"""
import json
import re
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.agent_v2 import default_task_message, run_agent_round
from protevo.prompt_builder import build_system_prompt
from protevo.protein_config import ProteinConfig
from protevo.schema import ExperimentConfig
from protevo.tui import ProtevoShell
from protevo.tools import ToolBox

PROJECT = Path(__file__).parent.parent
UBIQ_CSV = PROJECT / "data/external/ubiq_roscoe_2014/dms.csv"
CONFIG = ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None)
IDS = count()
# CJK 统一表意文字 + CJK 标点 + 全角形式（→ 为 U+2192 箭头，不属于 CJK，en 提示中保留）
_CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _cfg(higher=True):
    return ProteinConfig(reference_sequence="VDGV", mutable_positions=[0, 1, 2, 3],
                         variant_format="mutation_string", fitness_higher_is_better=higher)


def _response(tool_calls=None):
    calls = [SimpleNamespace(id=f"call_{next(IDS)}", type="function",
                             function=SimpleNamespace(name=name, arguments=json.dumps(args)))
             for name, args in tool_calls] if tool_calls else None
    message = SimpleNamespace(role="assistant", content=None, tool_calls=calls)
    usage = SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
    return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                    finish_reason="tool_calls" if calls else "stop")],
                           usage=usage, id=f"resp-{next(IDS)}", model="mock")


class _Completions:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def _toolbox(protein_config=None):
    known = pd.DataFrame([{"variant": "CC", "fitness": 1.0, "mutations": "synthetic"}])
    return ToolBox(known=known, eligible={"CE", "DC", "DE"},
                   predict_fn=lambda sequences: [1.0] * len(sequences),
                   config=CONFIG, protein_config=protein_config)


# ---------- build_system_prompt：英文版 ----------

def test_en_prompt_no_cjk_defaults_and_four_paragraphs():
    prompt = build_system_prompt(_cfg(), language="en")
    assert not _CJK.search(prompt)                          # 全文无 CJK（含标点/全角）
    assert "on the target protein" in prompt and "fitness" in prompt  # 缺省参数按语言取默认
    for first_sentence in ("You have five tools available",
                           "A typical scientific reasoning loop:",
                           "Budget awareness: API calls, tokens, and scorings are hard limits",
                           "Scientific discipline: never fabricate measurements or scores"):
        assert first_sentence in prompt                     # 英文四段齐备且首句语义对等
    assert "4-position variants (0/1/2/3; reference VDGV)" in prompt  # 位点/参考参数化


def test_en_prompt_direction_words():
    assert "for improving fitness" in build_system_prompt(_cfg(higher=True), language="en")
    lower = build_system_prompt(_cfg(higher=False), language="en")
    assert "for reducing fitness" in lower and "improving" not in lower
    assert "降低适应度" in build_system_prompt(_cfg(higher=False))   # zh 回归：方向词降低


def test_en_prompt_formats_and_parameterization():
    notation_cfg = ProteinConfig(reference_sequence="ACDE", mutable_positions=[2, 3],
                                 variant_format="mutation_notation", wt_by_position={2: "C", 3: "D"})
    prompt = build_system_prompt(notation_cfg, protein_name="TEM-1 beta-lactamase",
                                 assay_description="ampicillin resistance", language="en")
    assert "TEM-1 beta-lactamase" in prompt and "ampicillin resistance" in prompt
    assert 'mutation notation such as "C2A,D3A"' in prompt and '"WT"' in prompt
    assert not _CJK.search(prompt)
    assert "residue string over the 4 mutable positions" in build_system_prompt(_cfg(), language="en")
    full_cfg = ProteinConfig(reference_sequence="ACDEFG", mutable_positions=[0, 5], variant_format="full")
    assert "full-length amino-acid sequences (6 residues, of which 2 are the declared mutable positions)" \
        in build_system_prompt(full_cfg, language="en")
    indel_cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[3, 4],
                              variant_format="indel_notation", wt_full_sequence="ACDEFG")
    indel_prompt = build_system_prompt(indel_cfg, language="en")
    assert "insertions/deletions on the 6-aa reference sequence" in indel_prompt
    assert "covers 2 anchor positions" in indel_prompt
    assert 'insertions "D3_E4insAA" (the two anchors must be adjacent)' in indel_prompt


# ---------- build_system_prompt：zh 回归 ----------

def test_zh_prompt_explicit_equals_default():
    assert build_system_prompt(_cfg()) == build_system_prompt(_cfg(), language="zh")
    zh = build_system_prompt(_cfg())
    assert "你是 目标蛋白 蛋白定向进化研究中的自主科学家智能体" in zh
    assert "寻找提高适应度的未测候选" in zh
    assert "你有五个可用工具" in zh and "科学纪律" in zh          # 四段通用措辞沿用原文


def test_zh_prompt_parameterized_unchanged():
    notation_cfg = ProteinConfig(reference_sequence="ACDE", mutable_positions=[2, 3],
                                 variant_format="mutation_notation", wt_by_position={2: "C", 3: "D"})
    prompt = build_system_prompt(notation_cfg, protein_name="RL40A 泛素", assay_description="E1 激活结合适应度")
    assert "RL40A 泛素" in prompt and "E1 激活结合适应度" in prompt
    assert "为 2 位点变体（2/3，参考 ACDE）" in prompt
    assert '变体以 "C2A,D3A" 式突变记谱表示' in prompt


def test_invalid_language_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        build_system_prompt(_cfg(), language="fr")
    message = str(exc_info.value)
    assert "zh" in message and "en" in message                 # 报错列出合法取值


# ---------- default_task_message ----------

def test_task_message_config_path_en_and_zh():
    cfg = _cfg()
    en = default_task_message(1, _toolbox(cfg), cfg, language="en")
    zh = default_task_message(1, _toolbox(cfg), cfg, language="zh")
    assert en == zh                                            # 配置路径开场白本就是英文，zh/en 同文
    assert en.startswith("Directed evolution round 1: design variants over the 4 configured mutable positions")
    assert "to maximize fitness." in en
    assert "to minimize fitness." in default_task_message(1, _toolbox(_cfg(higher=False)),
                                                          _cfg(higher=False), language="en")
    assert default_task_message(2, _toolbox()) == default_task_message(2, _toolbox(), language="zh")


# ---------- run_agent_round 透传 ----------

def _run_one_round(protein_config, language=None, tmp_path=None):
    known = pd.DataFrame([{"variant": "CC", "fitness": 1.0, "mutations": "synthetic"}])
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": ["CE", "DC"], "reasons": ["a", "b"]})])])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    kwargs = {} if language is None else {"language": language}
    record, toolbox = run_agent_round(1, known, {"CE", "DC", "DE"},
                                      lambda sequences: [1.0] * len(sequences), CONFIG,
                                      client=client, model="mock", protein_config=protein_config,
                                      log_path=tmp_path / "agent.jsonl" if tmp_path else None, **kwargs)
    return record, toolbox, completions


def test_run_agent_round_language_en_reaches_llm(tmp_path):
    record, _toolbox_out, completions = _run_one_round(_cfg(), language="en", tmp_path=tmp_path)
    system = completions.requests[0]["messages"][0]["content"]
    assert not _CJK.search(system)
    assert "You are an autonomous scientist agent in directed-evolution research on the target protein" in system
    assert "You have five tools available" in system
    assert record["status"] == "submitted"


def test_run_agent_round_default_language_is_zh(tmp_path):
    _record, _toolbox_out, completions = _run_one_round(_cfg(), tmp_path=tmp_path)
    system = completions.requests[0]["messages"][0]["content"]
    assert "你是 目标蛋白" in system and "你有五个可用工具" in system   # 不传 language → 中文（回归）


# ---------- TUI：open --lang / lang 命令 / run 透传 ----------

def _ubiq_pool(per_round=2, initial_train=None):
    df = pd.read_csv(UBIQ_CSV)
    initial_train = initial_train if initial_train is not None else min(2000, max(1, len(df) // 2))
    idx = np.random.RandomState(42).permutation(len(df))
    known = set(df["variant"].iloc[idx[:initial_train]])
    return [v for v in df["variant"] if v not in known][:per_round]


def test_open_lang_flag_and_lang_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    assert shell.language == "zh"                             # 会话初始默认 zh
    shell.onecmd("open ubiq --lang en")
    assert shell.language == "en"                             # open 保存会话语言
    capsys.readouterr()
    shell.onecmd("open ubiq")                                 # 不传 --lang 默认回落 zh
    assert shell.language == "zh"
    shell.onecmd("lang en")                                   # 独立切换命令
    assert shell.language == "en"
    assert "Agent 提示语言已切换为 英文" in capsys.readouterr().out
    shell.onecmd("lang")
    assert "en" in capsys.readouterr().out                    # 无参查看当前语言
    shell.onecmd("lang fr")
    assert "只接受 zh | en" in capsys.readouterr().out
    shell.onecmd("open ubiq --lang fr")
    assert "--lang 只接受 zh | en" in capsys.readouterr().out
    assert shell.language == "en"                             # 非法值被拒且不改变会话语言


def _run_tui_round(shell, capsys, language_arg):
    shell.onecmd(language_arg)
    capsys.readouterr()
    batch = _ubiq_pool()
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": list(batch), "reasons": ["a", "b"]})])])
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    assert shell.last_summary is not None and shell.last_summary["rounds"][0]["submitted"] == 2
    return completions.requests[0]["messages"][0]["content"]


def test_tui_run_lang_en_sends_english_system_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    system = _run_tui_round(shell, capsys, "open ubiq --lang en")
    assert not _CJK.search(system)
    assert "autonomous scientist agent in directed-evolution research" in system
    assert "unmeasured candidates for improving" in system    # 泛素方向 higher → improving


def test_tui_run_default_sends_chinese_system_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    system = _run_tui_round(shell, capsys, "open ubiq")
    assert "你是 目标蛋白" in system and "你有五个可用工具" in system
