"""evolve 总装入口测试：两轮端到端（mock 客户端）+ oracle 回填边界 + CLI 凭据分支
+ 预测器协议 fixed / refit_per_round（retrain_each_round）。

无网络、无真实 LLM：mock 客户端复用 test_agent_v2_universal 的 _Completions 模式。
合成数据：60 aa WT 标准字母 + 记谱格式单/双点变体，fitness = 碱性残基信号 + 高斯噪声，
~300 行；初始划分用与 run_evolution 相同的种子化 RandomState 镜像，据此为 mock 编排脚本。
"""
import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.evolve import _oracle_fitness, run_evolution

PROJECT = Path(__file__).parent.parent
AA = "ACDEFGHIKLMNPQRSTVWY"
IDS = count()


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


def _synthetic_csv(path, n_rows=300, seed=7):
    """写合成记谱 CSV 并返回 60 aa WT：5 个位点（6/12/18/24/30）单/双突变，K/R 替换加分。"""
    rng = np.random.RandomState(seed)
    wt = "".join(rng.choice(list(AA), size=60, replace=True))
    rows, seen = [], set()
    while len(rows) < n_rows:
        picks = sorted(rng.choice([6, 12, 18, 24, 30], size=1 + int(rng.rand() < 0.5), replace=False))
        tokens, signal = [], 0.0
        for p in picks:
            wt_aa, mut_aa = wt[p - 1], AA[int(rng.randint(0, 20))]
            if mut_aa == wt_aa:
                continue  # 同字母替换无意义，跳过该位点
            tokens.append(f"{wt_aa}{p}{mut_aa}")
            signal += 1.0 if mut_aa in "KR" else 0.0
        if not tokens:
            continue
        variant = ",".join(tokens)
        if variant in seen:
            continue
        seen.add(variant)
        rows.append({"variant": variant, "fitness": signal + rng.normal(0, 0.05)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return wt


def _split_mirror(df, initial_train, seed=42):
    """镜像 run_evolution 的种子化划分：返回 (初始已知集合, 未测池列表)。"""
    idx = np.random.RandomState(seed).permutation(len(df))
    known = set(df["variant"].iloc[idx[:initial_train]])
    return known, [v for v in df["variant"] if v not in known]


# ---------- 端到端：两轮 mock predict+submit ----------

def test_two_round_end_to_end(tmp_path):
    csv_path = tmp_path / "syn.csv"
    wt = _synthetic_csv(csv_path)
    df = pd.read_csv(csv_path)
    known0, pool = _split_mirror(df, initial_train=200)
    round1, round2 = pool[:3], pool[3:6]
    fitness = dict(zip(df["variant"], df["fitness"]))

    completions = _Completions([
        _response([("predict_fitness", {"sequences": list(round1)})]),
        _response([("submit_recommendations", {"sequences": list(round1), "reasons": ["a", "b", "c"]})]),
        _response([("predict_fitness", {"sequences": list(round2)})]),
        _response([("submit_recommendations", {"sequences": list(round2), "reasons": ["d", "e", "f"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    out_dir = tmp_path / "evolve_out"
    summary = run_evolution(str(csv_path), rounds=2, per_round=3, score_budget=6, initial_train=200,
                            reference_sequence=wt, seed=42, out_dir=out_dir, client=client, model="mock")

    # 两轮均为 LLM 主动提交（评分预算未耗尽），脚本 4 次 API 调用全部消费
    assert [r["status"] for r in summary["rounds"]] == ["submitted", "submitted"]
    assert len(completions.requests) == 4

    # oracle 回填逐条对照原 CSV；预测分来自轮内 predict，已评分故非空
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert per_round["round"].tolist() == [1] * 3 + [2] * 3
    assert per_round["variant"].tolist() == list(round1) + list(round2)
    for _, row in per_round.iterrows():
        assert row["measured_fitness"] == pytest.approx(fitness[row["variant"]])
        assert not pd.isna(row["predicted_fitness"])

    # known 每轮增长 per_round 条、eligible 相应缩小
    assert summary["rounds"][0]["known_before"] == 200
    assert summary["rounds"][1]["known_before"] == 203
    assert summary["rounds"][0]["eligible_before"] == len(pool)
    assert summary["rounds"][1]["eligible_before"] == len(pool) - 3
    assert summary["total_measured"] == 206
    assert summary["rounds"][0]["measured_mean"] == pytest.approx(np.mean([fitness[v] for v in round1]))

    # 累计最优对照手工计算（初始已知 ∪ 两轮提交）；配置快照含格式与预测器信息
    measured_all = [fitness[v] for v in known0 | set(round1) | set(round2)]
    assert summary["cumulative_best"]["fitness"] == pytest.approx(max(measured_all))
    assert summary["config"]["variant_format"] == "mutation_notation"
    assert summary["config"]["oracle_pool"] == len(pool)
    assert summary["config"]["predictor"]["selected"]

    # summary.json 落盘且与返回值同构；初始训练子集留档
    on_disk = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["cumulative_best"] == summary["cumulative_best"]
    assert on_disk["config"]["per_round"] == 3
    assert len(pd.read_csv(out_dir / "initial_train.csv")) == 200

    # 每轮 JSONL 日志存在且记录了完整回合（round_start → submit 工具调用）
    for r in (1, 2):
        entries = [json.loads(line) for line in
                   (out_dir / f"agent_round{r}.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = [entry["kind"] for entry in entries]
        assert "round_start" in kinds and "round_end" in kinds
        assert any(entry["kind"] == "tool_call" and entry["tool_name"] == "submit_recommendations"
                   and entry["result"]["status"] == "submitted" for entry in entries)


# ---------- 边界：oracle 池外 / eligible 外提交 ----------

def test_oracle_miss_and_ineligible_submission(tmp_path):
    # ① 回填防御路径：oracle 池外变体记 None、不抛错（干净运行中提交 ⊆ eligible ⊆ oracle，不可自然到达）
    assert _oracle_fitness({"A6C": 1.25}, ["A6C", "Z99X"]) == [1.25, None]

    # ② 端到端：mock 提交未列入 eligible 的变体 → ToolBox 拒绝该次提交，agent 继续后正常完成
    csv_path = tmp_path / "syn.csv"
    wt = _synthetic_csv(csv_path, n_rows=120)
    df = pd.read_csv(csv_path)
    _, pool = _split_mirror(df, initial_train=100)
    good = list(pool[:2])
    fitness = dict(zip(df["variant"], df["fitness"]))
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": [good[0], "Z99X"], "reasons": ["a", "b"]})]),
        _response([("submit_recommendations", {"sequences": good, "reasons": ["a", "b"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    out_dir = tmp_path / "evolve_out"
    summary = run_evolution(str(csv_path), rounds=1, per_round=2, score_budget=2, initial_train=100,
                            reference_sequence=wt, seed=42, out_dir=out_dir, client=client, model="mock")

    # 被拒提交未进入结果（Z99X 不在提交里），程序未崩，实测分正确回填
    assert summary["rounds"][0]["status"] == "submitted"
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert per_round["variant"].tolist() == good
    assert per_round["measured_fitness"].tolist() == pytest.approx([fitness[v] for v in good])
    entries = [json.loads(line) for line in
               (out_dir / "agent_round1.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = [entry for entry in entries if entry["kind"] == "tool_call"
                and entry["tool_name"] == "submit_recommendations"]
    assert rejected[0]["result"]["status"] == "rejected"
    assert "Z99X" in rejected[0]["result"]["error"]


# ---------- CLI ----------

def test_cli_help():
    proc = subprocess.run([sys.executable, "-m", "protevo.evolve", "--help"], cwd=PROJECT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1"})
    assert proc.returncode == 0 and "usage" in proc.stdout


def test_cli_missing_credentials_fails_fast(tmp_path):
    csv_path = tmp_path / "syn.csv"
    _synthetic_csv(csv_path, n_rows=60)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-m", "protevo.evolve", str(csv_path), "--rounds", "1"],
                          cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env)
    # 无凭据：首轮前明确报错（exit≠0，stderr 指向 setup_assistant --doctor 的 llm_credentials 检查）
    assert proc.returncode == 2
    assert "setup_assistant" in proc.stderr and "llm_credentials" in proc.stderr
    assert "OPENAI_API_KEY" in proc.stderr


# ---------- 预测器协议：fixed（默认） / refit_per_round ----------

def _spy_configure_protein(monkeypatch, recorder):
    """包裹 protevo.evolve 命名空间内的 configure_protein：先记录训练 CSV 路径（等信息）再放行。"""
    import protevo.evolve as evolve_mod
    real_configure = evolve_mod.configure_protein

    def _spy(data_path, **kwargs):
        recorder(data_path, kwargs)
        return real_configure(data_path, **kwargs)

    monkeypatch.setattr(evolve_mod, "configure_protein", _spy)
    return real_configure


def test_default_fixed_protocol(tmp_path, monkeypatch):
    """fixed（默认回归）：configure_protein 仅初始调用一次、轮间不重训、无重训 CSV；
    顶层 predictor_protocol='fixed'，rounds 明细不含 predictor_info，note 说明保留。"""
    csv_path = tmp_path / "syn.csv"
    wt = _synthetic_csv(csv_path, n_rows=120)
    _, pool = _split_mirror(pd.read_csv(csv_path), initial_train=100)
    batch1, batch2 = list(pool[:2]), list(pool[2:4])
    completions = _Completions([
        _response([("predict_fitness", {"sequences": batch1})]),
        _response([("submit_recommendations", {"sequences": batch1, "reasons": ["a", "b"]})]),
        _response([("predict_fitness", {"sequences": batch2})]),
        _response([("submit_recommendations", {"sequences": batch2, "reasons": ["c", "d"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    calls = []
    _spy_configure_protein(monkeypatch, lambda data_path, kwargs: calls.append(data_path))

    out_dir = tmp_path / "evolve_fixed"
    summary = run_evolution(str(csv_path), rounds=2, per_round=2, score_budget=4, initial_train=100,
                            reference_sequence=wt, seed=42, out_dir=out_dir, client=client, model="mock")

    assert summary["predictor_protocol"] == "fixed"
    assert [Path(p).name for p in calls] == ["initial_train.csv"]   # 仅初始一次，轮间不重训
    assert not (out_dir / "known_before_round2.csv").exists()
    assert all("predictor_info" not in r for r in summary["rounds"])
    assert "轮间不重训" in summary["note"]


def test_retrain_each_round_two_round_end_to_end(tmp_path, monkeypatch):
    """refit_per_round：每轮 predictor_info 入 rounds 明细；第 2 轮 predict_fn 来自以
    累计 known（200 初始 + 3 回填 = 203）重训的新 config——spy 记录 configure_protein
    每次调用的训练 CSV 行数作直接证据。"""
    csv_path = tmp_path / "syn.csv"
    wt = _synthetic_csv(csv_path)
    df = pd.read_csv(csv_path)
    _, pool = _split_mirror(df, initial_train=200)
    round1, round2 = pool[:3], pool[3:6]

    completions = _Completions([
        _response([("predict_fitness", {"sequences": list(round1)})]),
        _response([("submit_recommendations", {"sequences": list(round1), "reasons": ["a", "b", "c"]})]),
        _response([("predict_fitness", {"sequences": list(round2)})]),
        _response([("submit_recommendations", {"sequences": list(round2), "reasons": ["d", "e", "f"]})]),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    fit_calls = []
    _spy_configure_protein(
        monkeypatch, lambda data_path, kwargs: fit_calls.append((Path(data_path), len(pd.read_csv(data_path)))))

    out_dir = tmp_path / "evolve_retrain"
    summary = run_evolution(str(csv_path), rounds=2, per_round=3, score_budget=6, initial_train=200,
                            reference_sequence=wt, seed=42, out_dir=out_dir, client=client,
                            model="mock", retrain_each_round=True)

    assert summary["predictor_protocol"] == "refit_per_round"
    # 调用链：初始 1 次（200 行）+ 第 2 轮前重训 1 次（累计 known=203）；重训训练集落盘留档
    assert [(p.name, n) for p, n in fit_calls] == [("initial_train.csv", 200),
                                                   ("known_before_round2.csv", 203)]
    assert fit_calls[1][0] == out_dir / "known_before_round2.csv"
    # 每轮 rounds 明细含该轮实际使用的选型与 val Spearman；第 1 轮即初始选型（random_state=seed 贯通）
    for r in summary["rounds"]:
        assert "selected" in r["predictor_info"] and "validation_spearman" in r["predictor_info"]
    assert summary["rounds"][0]["predictor_info"] == summary["config"]["predictor"]
    assert summary["rounds"][0]["predictor_info"]["random_state"] == 42
    assert summary["rounds"][1]["known_before"] == 203      # 第二轮 known 基数正确
    assert summary["total_measured"] == 206
    assert "refit_per_round" in summary["note"]
    # 端到端结果口径与 fixed 模式一致（预测分经重训后的 predict_fn 正常产出）
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert per_round["variant"].tolist() == list(round1) + list(round2)
    assert not per_round["predicted_fitness"].isna().any()
