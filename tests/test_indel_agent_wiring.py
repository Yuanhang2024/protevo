"""indel 库的 Agent 接线冒烟：真实 ProteinGym BLAT（TEM-1 β-内酰胺酶）单点 indel 数据全链路。

验证第四种格式（indel_notation）从 CSV 到配置驱动工具层的闭环：记谱校验、评分、schema 与
系统提示均由该蛋白配置驱动。无 LLM API 调用。
"""
import json
from pathlib import Path

import pandas as pd

from protevo.protein_config import configure_protein
from protevo.prompt_builder import build_system_prompt
from protevo.schema import ExperimentConfig
from protevo.tools import ToolBox, build_tool_schemas

PROJECT = Path(__file__).parent.parent
DMS_DIR = PROJECT / "data/external/blat_ecolx_gonzalez_2019"


def _blat_toolbox():
    meta = json.loads((DMS_DIR / "meta.json").read_text(encoding="utf-8"))
    dms = pd.read_csv(DMS_DIR / "dms.csv")
    cfg = configure_protein(str(DMS_DIR / "dms.csv"), reference_sequence=meta["wt_sequence"])
    known = dms.head(30).assign(mutations="dms")
    box = ToolBox(known=known, eligible=set(dms["variant"].iloc[30:200]),
                  predict_fn=cfg.predict_fitness,
                  config=ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None),
                  protein_config=cfg)
    return cfg, box, dms


def test_blat_indel_agent_wiring_end_to_end():
    cfg, box, dms = _blat_toolbox()

    # 配置层：indel 格式 + 序列级预测器（无 torch 环境下 composition_ridge）
    assert cfg.variant_format == "indel_notation"
    assert len(cfg.wt_full_sequence) == 286
    assert cfg.predictor_info["encoding"] == "composition+length"

    # 合法单点 indel 候选（真实库内未测变体）→ 评分成功
    candidates = [dms["variant"].iloc[30], dms["variant"].iloc[31]]
    result = box.execute("predict_fitness", json.dumps({"sequences": candidates}))
    assert {row["variant"] for row in result["scored"]} == set(candidates)

    # 三类非法记谱被拒：WT 字母不符、越界、不可解析
    bad = box.execute("predict_fitness",
                      json.dumps({"sequences": ["Z5del", "A999del", "5del"]}))
    assert bad["scored"] == []
    reasons = " | ".join(row["reason"] for row in bad["errors"])
    assert "do not match the reference" in reasons      # Z5 非该位点野生型
    assert "out of bounds" in reasons                   # 999 超出 286
    assert "无法解析" in reasons                         # 缺野生型字母

    # 工具 schema 与系统提示均含 indel 文法且无 GB1
    schemas_text = json.dumps(build_tool_schemas(cfg), ensure_ascii=False)
    assert "GB1" not in schemas_text and "insAA" in schemas_text
    prompt = build_system_prompt(cfg, protein_name="TEM-1 β-内酰胺酶", assay_description="氨苄抗性适应度")
    assert "GB1" not in prompt and "插入" in prompt and "TEM-1" in prompt
