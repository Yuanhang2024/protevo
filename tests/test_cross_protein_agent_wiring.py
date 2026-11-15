"""跨蛋白 Agent 接线冒烟：真实 ProteinGym 泛素数据从 CSV 到配置驱动工具层全链路。

验证"换蛋白零代码"的 Agent 层闭环——真实第二蛋白（RL40A_YEAST_Roscoe_2014）经
configure_protein 配置后，ToolBox 校验/评分、工具 schema、系统提示全部由该蛋白的
ProteinConfig 驱动。无 LLM API 调用（predict_fn 为本地训练的预测器）。
"""
import json
from pathlib import Path

import pandas as pd

from protevo.protein_config import configure_protein
from protevo.prompt_builder import build_system_prompt
from protevo.schema import ExperimentConfig
from protevo.tools import ToolBox, build_tool_schemas

PROJECT = Path(__file__).parent.parent
DMS_DIR = PROJECT / "data/external/ubiq_roscoe_2014"


def _ubiq_toolbox():
    meta = json.loads((DMS_DIR / "meta.json").read_text(encoding="utf-8"))
    dms = pd.read_csv(DMS_DIR / "dms.csv")
    cfg = configure_protein(str(DMS_DIR / "dms.csv"), reference_sequence=meta["wt_sequence"],
                            auto_select_predictor=False)
    known = dms.head(50).assign(mutations="dms")
    box = ToolBox(known=known, eligible=set(dms["variant"].iloc[50:300]),
                  predict_fn=cfg.predict_fitness,
                  config=ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None),
                  protein_config=cfg)
    return cfg, box, dms


def test_ubiq_agent_wiring_end_to_end():
    cfg, box, dms = _ubiq_toolbox()

    # 记谱候选（真实库内未测变体）→ 评分成功且消耗预算
    candidates = [dms["variant"].iloc[50], dms["variant"].iloc[51]]
    result = box.execute("predict_fitness", json.dumps({"sequences": candidates}))
    assert {row["variant"] for row in result["scored"]} == set(candidates)
    assert box.budget.score_calls_used == 2

    # 位点 1 未声明 → 拒绝且不消耗预算
    bad = box.execute("predict_fitness", json.dumps({"sequences": ["A1C"]}))
    assert bad["scored"] == []
    assert "position 1 is not one of the mutable positions" in bad["errors"][0]["reason"]
    assert box.budget.score_calls_used == 2

    # round_context 来自该蛋白配置（泛素 75 位点，1-indexed 2-76）
    context = box.execute("get_experiment_history", "{}")["round_context"]
    assert context["positions"][0] == 2 and len(context["positions"]) == 75

    # 工具 schema 与系统提示不含 GB1、含该蛋白信息
    schemas_text = json.dumps(build_tool_schemas(cfg), ensure_ascii=False)
    assert "GB1" not in schemas_text
    prompt = build_system_prompt(cfg, protein_name="RL40A 泛素", assay_description="E1 激活结合适应度")
    assert "GB1" not in prompt and "RL40A 泛素" in prompt and "E1 激活结合适应度" in prompt
