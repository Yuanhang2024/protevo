"""净室用户验收（2026-11-15，真实 LLM + CPU 泛素数据，2 轮 × 3 推荐 × 评分额度 6）暴露缺陷的回归测试。

四组修复的行为锁定（观察材料：protevo-clean-user-20261115/observations）：
  A 提交前所有候选必须已有预测分——LLM 提交自动补分 / 预算不足拒绝 / auto_submit 回补候选先补分
  B get_experiment_history 的 round_context 提供合法未测候选示例（真实泛素数据，原生记谱）
  C protein_config 路径历史摘要完全由配置驱动（无 GB1 位点 39/40/41/54、无记谱数字当残基、无 VDGV）
  D TUI：log 的 token 数与 summary.json 对齐（不再翻倍）/ export 复制 agent_round*.jsonl / status 显示研究模型 ρ
以及用户场景复现：泛素 2 轮 × 每轮 3 条 × 预算 6，第二轮 0 有效评分触发 auto——最终 6 条全部有预测值
（用户实测修复前仅 2/6，散点 n=2、Spearman 无法计算）。

无网络、无真实 LLM：mock 客户端复用 test_agent_v2_universal 的 SimpleNamespace 模式。
"""
import json
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.agent_v2 import run_agent_round
from protevo.evolve import run_evolution
from protevo.protein_config import ProteinConfig, configure_protein
from protevo.schema import ExperimentConfig
from protevo.tools import AgentBudget, ToolBox

PROJECT = Path(__file__).parent.parent
UBIQ_DIR = PROJECT / "data/external/ubiq_roscoe_2014"
UBIQ_CSV = UBIQ_DIR / "dms.csv"
IDS = count()


# ---------- mock 客户端脚手架（同 test_evolve / test_agent_v2_universal 模式） ----------

def _response(tool_calls=None, content=None, usage=(10, 20)):
    calls = [SimpleNamespace(id=f"call_{next(IDS)}", type="function",
                             function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
             for name, arguments in tool_calls] if tool_calls else None
    message = SimpleNamespace(role="assistant", content=content, tool_calls=calls)
    usage_obj = SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                                total_tokens=usage[0] + usage[1])
    return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                    finish_reason="tool_calls" if calls else "stop")],
                           usage=usage_obj, id=f"resp-{next(IDS)}", model="mock")


class _Completions:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def _ubiq_toolbox():
    """真实泛素数据 → configure_protein 配置驱动的 ToolBox（不训练选型，快速）。"""
    meta = json.loads((UBIQ_DIR / "meta.json").read_text(encoding="utf-8"))
    dms = pd.read_csv(UBIQ_CSV)
    cfg = configure_protein(str(UBIQ_CSV), reference_sequence=meta["wt_sequence"],
                            auto_select_predictor=False)
    known = dms.head(50).assign(mutations="dms")
    eligible = set(dms["variant"].iloc[50:300])
    box = ToolBox(known=known, eligible=eligible, predict_fn=cfg.predict_fitness,
                  config=ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None),
                  protein_config=cfg)
    return cfg, box, dms


# ---------- A：提交前自动补齐评分（硬不变量：每条推荐都带 predicted_fitness） ----------

def _notation_toolbox(eligible, k=2, budget=None):
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[1, 2],
                        variant_format="mutation_notation", wt_by_position={1: "A", 2: "C"})
    known = pd.DataFrame([{"variant": "A1C", "fitness": 1.0, "mutations": "syn"}])
    calls = []

    def predict_fn(sequences):
        calls.append(list(sequences))
        return [float(len(sequence)) for sequence in sequences]

    box = ToolBox(known=known, eligible=set(eligible), predict_fn=predict_fn,
                  config=ExperimentConfig(k=k, proposal_count=k, score_budget=k, knowledge_path=None),
                  protein_config=cfg, budget=budget)
    return box, calls


def test_submit_auto_scores_unscored_variants():
    box, calls = _notation_toolbox({"A1D", "A1E", "C2E"})
    result = box.execute("submit_recommendations",
                         json.dumps({"sequences": ["A1D", "A1E"], "reasons": ["a", "b"]}))
    assert result["status"] == "submitted"
    # 每条 recommendation 均有预测分；提交记录同样无缺失
    assert all(item["predicted_fitness"] is not None for item in result["recommendations"])
    assert all(score is not None for score in box.submission["predicted_fitness"])
    # 未评分变体被批量补分（恰好一次、只含缺分序列），预算正确扣减
    assert calls == [["A1D", "A1E"]]
    assert box.budget.score_calls_used == 2


def test_submit_rejected_when_score_budget_insufficient():
    box, calls = _notation_toolbox({"A1D", "A1E", "C2E"}, k=3, budget=AgentBudget(max_score_calls=3))
    box.budget.consume_scores(2)  # 剩余 1，不足以补齐 3 条未评分
    rejected = box.execute("submit_recommendations",
                           json.dumps({"sequences": ["A1D", "A1E", "C2E"], "reasons": ["a", "b", "c"]}))
    assert rejected["status"] == "rejected"
    assert "提交前必须完成评分" in rejected["error"] and "评分额度不足" in rejected["error"]
    assert "predict_fitness" in rejected["error"]
    assert calls == [] and box.budget.score_calls_used == 2 and box.submission is None  # 拒绝不消耗预算
    # 恢复路径：predict 先消耗剩余额度评 1 条，仍含未评分变体的提交继续被拒
    scored = box.execute("predict_fitness", json.dumps({"sequences": ["A1D", "A1E"]}))
    assert [row["variant"] for row in scored["scored"]] == ["A1D"]
    assert scored["not_scored"][0]["sequence"] == "A1E"
    again = box.execute("submit_recommendations",
                        json.dumps({"sequences": ["A1D", "A1E", "C2E"], "reasons": ["a", "b", "c"]}))
    assert again["status"] == "rejected" and "评分额度不足" in again["error"]


def test_auto_submit_fill_candidates_are_scored_before_submission(tmp_path):
    # 第二轮 0 有效评分 + token 耗尽触发 auto：回补候选先补评分再进提交，全部带预测分
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1], variant_format="mutation_string")
    known = pd.DataFrame([{"variant": "AA", "fitness": 1.0, "mutations": "syn"}])
    completions = _Completions([_response(content="thinking"), _response(content="still thinking")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    record, toolbox = run_agent_round(
        1, known, {"CC", "CD", "DC", "DD"}, lambda sequences: [float(len(s)) for s in sequences],
        ExperimentConfig(k=3, proposal_count=3, score_budget=3, knowledge_path=None),
        client=client, model="mock", protein_config=cfg, log_path=tmp_path / "auto.jsonl",
        budget=AgentBudget(max_api_calls=2, max_total_tokens=50000, max_score_calls=3))
    assert record["status"] == "auto_submitted" and record["stop_reason"] == "max_api_calls"
    # 硬不变量：自动提交的每条推荐 predicted_fitness 均非 None（回补候选已先评分）
    assert all(score is not None for score in record["submission"]["predicted_fitness"])
    assert all(score is not None for score in toolbox.submission["predicted_fitness"])
    assert toolbox.budget.score_calls_used == 3  # 回补 3 条正常消耗评分预算
    entries = [json.loads(line) for line in (tmp_path / "auto.jsonl").read_text(encoding="utf-8").splitlines()]
    final = [e for e in entries if e["kind"] == "tool_call" and e["tool_name"] == "submit_recommendations"][-1]
    assert all(item["predicted_fitness"] is not None for item in final["result"]["recommendations"])
    assert "predicted fitness" in final["arguments"]["reasons"][0]


# ---------- B：round_context 提供合法未测候选示例（真实泛素数据） ----------

def test_round_context_eligible_examples_ubiq():
    _cfg, box, dms = _ubiq_toolbox()
    context = box.execute("get_experiment_history", "{}")["round_context"]
    examples = context["eligible_examples"]
    eligible = set(dms["variant"].iloc[50:300])
    known_variants = set(dms["variant"].iloc[:50])
    # 示例全部来自未测池且为数据集原生记谱（逐字符与 dms.csv 变体串一致）
    assert examples and len(examples) <= 30
    assert set(examples) <= eligible
    assert not (set(examples) & known_variants)
    assert set(examples) <= set(dms["variant"].astype(str))
    # 确定性顺序：字典序稳定取前 30；总数与说明齐备
    assert examples == sorted(eligible)[:30]
    assert context["eligible_total"] == len(eligible) == context["eligible_candidate_count"]
    assert "native variant format" in context["eligibility_note"]


# ---------- C：protein_config 路径历史摘要完全由配置驱动 ----------

def test_ubiq_known_summary_positions_and_residues():
    cfg, box, _dms = _ubiq_toolbox()
    known_data = box.execute("get_experiment_history", "{}")["known_data"]
    positions = known_data["positions"]
    # 位点列表 = 配置的 75 个可变位点（泛素 2-76），不再是 GB1 的 39/40/41/54
    assert [p["position"] for p in positions] == list(cfg.mutable_positions)
    assert positions[0]["position"] == 2 and positions[-1]["position"] == 76 and len(positions) == 75
    # 残基统计键是真实氨基酸字母——修复前记谱数字（如 "Q2C" 的 2）被当作残基
    residues = {row["residue"] for p in positions for row in p["residue_statistics"]}
    assert residues <= set("ACDEFGHIKLMNPQRSTVWY") and residues
    assert "VDGV" not in json.dumps(known_data)
    assert known_data["n_known"] == 50 and known_data["best_fitness"] > 0


def test_notation_summary_has_no_gb1_literals():
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[2, 3],
                        variant_format="mutation_notation", wt_by_position={2: "A", 3: "C"})
    known = pd.DataFrame([{"variant": "A2C", "fitness": 1.0, "mutations": "syn"},
                          {"variant": "C3D", "fitness": 2.0, "mutations": "syn"},
                          {"variant": "WT", "fitness": 3.0, "mutations": "syn"}])
    box = ToolBox(known=known, eligible={"A2D", "A2E"}, predict_fn=lambda seqs: [1.0] * len(seqs),
                  config=ExperimentConfig(k=1, proposal_count=1, score_budget=1, knowledge_path=None),
                  protein_config=cfg)
    result = box.execute("get_experiment_history", "{}")
    text = json.dumps(result, ensure_ascii=False)
    # 泛素式位点 2/3 出现；GB1 位点字面量与 VDGV 一律不出现
    for banned in ("39", "40", "41", "54", "VDGV"):
        assert banned not in text, banned
    assert [p["position"] for p in result["known_data"]["positions"]] == [2, 3]
    # 记谱残基按解析统计：位点 2 → C(来自 A2C) 与 A(来自 C3D/WT 未突变)；位点 3 → C 与 D
    stats = {p["position"]: {row["residue"] for row in p["residue_statistics"]}
             for p in result["known_data"]["positions"]}
    assert stats == {2: {"A", "C"}, 3: {"C", "D"}}


def test_mutation_string_summary_uses_config_positions():
    cfg = ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1], variant_format="mutation_string")
    known = pd.DataFrame([{"variant": "AC", "fitness": 1.0, "mutations": "syn"},
                          {"variant": "CC", "fitness": 2.0, "mutations": "syn"}])
    box = ToolBox(known=known, eligible={"DC", "DE"}, predict_fn=lambda seqs: [1.0] * len(seqs),
                  config=ExperimentConfig(k=1, proposal_count=1, score_budget=1, knowledge_path=None),
                  protein_config=cfg)
    result = box.execute("get_experiment_history", "{}")
    text = json.dumps(result, ensure_ascii=False)
    for banned in ("39", "40", "41", "54", "VDGV"):
        assert banned not in text, banned
    # 突变串按位取字符：位点 0 → A/C，位点 1 → C
    stats = {p["position"]: {row["residue"] for row in p["residue_statistics"]}
             for p in result["known_data"]["positions"]}
    assert stats == {0: {"A", "C"}, 1: {"C"}}


# ---------- D：TUI 三处用户层问题 ----------

def _write_fake_run_dir(path, api_tokens, round_total):
    """构造最小 out_dir：jsonl 含逐次 api_call 与轮末 round_end（翻倍陷阱），summary.json 对齐。"""
    records = [{"kind": "round_start", "round": 1, "tokens_used": 0},
               {"kind": "tool_call", "round": 1, "tool_name": "get_experiment_history", "tokens_used": 0}]
    records += [{"kind": "api_call", "round": 1, "tokens_used": value} for value in api_tokens]
    records.append({"kind": "round_end", "round": 1, "tokens_used": round_total})
    path.mkdir(parents=True, exist_ok=True)
    with (path / "agent_round1.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"out_dir": str(path), "rounds": [{"round": 1, "status": "submitted", "stop_reason": None,
                                                 "total_tokens_used": round_total, "api_calls_used": len(api_tokens),
                                                 "score_calls_used": 0, "max_score_budget": 6, "submitted": 3,
                                                 "measured_mean": 1.0,
                                                 "cumulative_best": {"round": 0, "variant": "Q2C",
                                                                     "fitness": 1.38}}]}
    (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return summary


def test_log_tokens_match_summary_json(tmp_path, monkeypatch, capsys):
    from protevo.tui import ProtevoShell
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    out_dir = tmp_path / "results/evolve/fake"
    summary = _write_fake_run_dir(out_dir, api_tokens=[100, 50], round_total=150)
    shell.last_out_dir, shell.last_summary = out_dir, summary
    shell.onecmd("log")
    out = capsys.readouterr().out
    # 逐次 100+50=150，与 summary.json 的 total_tokens_used 一致；旧实现把轮末总量再累加一次得 300
    assert f"{summary['rounds'][0]['total_tokens_used']} tokens" in out
    assert "300 tokens" not in out


def test_export_copies_agent_round_jsonl(tmp_path, monkeypatch, capsys):
    from protevo.tui import ProtevoShell
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    out_dir = tmp_path / "results/evolve/fake"
    summary = _write_fake_run_dir(out_dir, api_tokens=[100], round_total=100)
    (out_dir / "per_round.csv").write_text(
        "round,variant,reason,predicted_fitness,measured_fitness\n1,Q2A,r,1.0,1.0\n", encoding="utf-8")
    shell.last_out_dir, shell.last_summary = out_dir, summary
    shell.config = None  # 无会话配置：仅复制文件，不出格式适用图
    export_dir = tmp_path / "export_out"
    shell.onecmd(f"export --out {export_dir}")
    out = capsys.readouterr().out
    assert "── 导出清单" in out
    # 导出后可回放：agent_round*.jsonl 必须在导出目录（修复前缺失）
    assert (export_dir / "agent_round1.jsonl").is_file()
    assert json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))["rounds"][0]["total_tokens_used"] == 100


def test_run_and_status_show_research_predictor(tmp_path, monkeypatch, capsys):
    from protevo.tui import ProtevoShell
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    shell.onecmd("open ubiq")
    capsys.readouterr()
    df = pd.read_csv(UBIQ_CSV)
    initial_train = min(2000, max(1, len(df) // 2))
    idx = np.random.RandomState(42).permutation(len(df))
    known = set(df["variant"].iloc[idx[:initial_train]])
    pool = [v for v in df["variant"] if v not in known][:2]
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": pool, "reasons": ["a", "b"]})])])
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    out = capsys.readouterr().out
    run_rho = shell.last_summary["config"]["predictor"]["validation_spearman"]
    # run 结束后显示本次研究实际使用的预测器与 val ρ（读 summary.json 的 config.predictor）
    assert "本次研究预测器" in out and str(run_rho) in out
    shell.onecmd("status")
    out = capsys.readouterr().out
    assert "研究模型" in out and str(run_rho) in out
    session_rho = shell.config.predictor_info["validation_spearman"]
    if session_rho != run_rho:  # 两者不同则并列展示（用户实测 0.7849 vs 0.7222）
        assert str(session_rho) in out and "会话 open 时" in out


# ---------- 用户场景复现：泛素 2 轮 × 3 推荐 × 预算 6，第二轮 0 有效评分触发 auto ----------

def test_user_scenario_six_of_six_recommendations_scored(tmp_path):
    meta = json.loads((UBIQ_DIR / "meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(UBIQ_CSV)
    initial_train = min(2000, max(1, len(df) // 2))  # TUI run 的缺省划分：690
    idx = np.random.RandomState(42).permutation(len(df))
    known0 = set(df["variant"].iloc[idx[:initial_train]])
    pool = [v for v in df["variant"] if v not in known0]
    scored_two, fresh, measured_two = pool[0], pool[2], sorted(known0)[:2]
    out_of_pool = f"{pool[0]},{pool[1]}"  # 库外双突变：合法记谱但不在单替换池中 → 0 有效评分

    completions = _Completions([
        # 第 1 轮：predict 2 条合格 + 2 条已测（耗预算 2）→ 提交含 1 条未评分（Q2D 型缺陷）
        _response([("predict_fitness", {"sequences": [scored_two, pool[1], measured_two[0], measured_two[1]]})]),
        _response([("submit_recommendations", {"sequences": [pool[0], pool[1], fresh],
                                               "reasons": ["a", "b", "c"]})]),
        # 第 2 轮：请求库外双突变（全部被拒、0 有效评分）+ 巨量 token → 触发 auto_submit
        _response([("predict_fitness", {"sequences": [out_of_pool]})], usage=(30000, 25000)),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    out_dir = tmp_path / "evolve_out"
    summary = run_evolution(str(UBIQ_CSV), rounds=2, per_round=3, score_budget=6,
                            initial_train=initial_train, reference_sequence=meta["wt_sequence"],
                            seed=42, out_dir=out_dir, client=client, model="mock")

    assert [r["status"] for r in summary["rounds"]] == ["submitted", "auto_submitted"]
    assert summary["rounds"][1]["stop_reason"] == "max_total_tokens"
    # 用户实测修复前 6 条中仅 2 条有预测值；修复后 6/6 全部有预测分 → 散点/Spearman 可计算
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert len(per_round) == 6
    assert not per_round["predicted_fitness"].isna().any()
    assert not per_round["measured_fitness"].isna().any()
    # auto 提交的回补候选同样先补评分（消耗预算），提交记录每条带预测分
    assert summary["rounds"][1]["score_calls_used"] == 3
    entries = [json.loads(line) for line in (out_dir / "agent_round2.jsonl").read_text(encoding="utf-8").splitlines()]
    final = [e for e in entries if e["kind"] == "tool_call" and e["tool_name"] == "submit_recommendations"][-1]
    assert all(item["predicted_fitness"] is not None for item in final["result"]["recommendations"])
    # 第 1 轮提交里那条"未评分"变体已被提交层自动补分（预算 2+1）
    assert summary["rounds"][0]["score_calls_used"] == 3
