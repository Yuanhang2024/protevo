# DSH 净室复验（release-ready-001）四个新问题的修复测试（note-003）：
# ① lower_is_better 排序方向（ToolBox ranked/scored、历史 best、auto 提交、累计最优、TUI 标记）
# ② open 的池选择透传研究阶段（run_evolution → configure_protein，summary 记实际池）
# ③ esm2_mlp_spec 反映 model_kwargs 覆盖后的实际构造参数
# ④ 中断运行的部分结果落盘（status=interrupted）+ TUI last_run_dir 可导出
# 无网络、无真实 LLM：mock 客户端复用 test_evolve 的 _Completions 模式。
# GB1 冻结路径（protein_config=None）排序方向保持降序的回归断言见本文件第 5 节。
import json
from itertools import count
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo.evolve import run_evolution
from protevo.protein_config import ProteinConfig, _ESM2_MLP_SPEC, _esm2_mlp_spec_text
from protevo.schema import ExperimentConfig
from protevo.tools import ToolBox
from protevo.tui import ProtevoShell

AA = "ACDEFGHIKLMNPQRSTVWY"
IDS = count()
CONFIG = ExperimentConfig(k=2, proposal_count=2, score_budget=2, knowledge_path=None)


# ---------- mock 客户端脚手架（同 test_evolve / test_tui 模式） ----------

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


class _FailingCompletions:
    """脚本耗尽后的 create 抛指定异常（模拟外部服务 429 中断）。"""

    def __init__(self, script, failure):
        self.script, self.failure, self.requests = list(script), failure, []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise self.failure
        return self.script.pop(0)


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _tiny_csv(path, n_rows=60, seed=7):
    """60 行记谱格式合成库（4 位点），fitness 取正态分布（正负皆有，两个方向都有区分度）。"""
    rng = np.random.RandomState(seed)
    wt = "".join(rng.choice(list(AA), size=40, replace=True))
    rows, seen = [], set()
    while len(rows) < n_rows:
        p = int(rng.choice([5, 10, 15, 20]))
        mut = AA[int(rng.randint(0, 20))]
        if mut == wt[p - 1]:
            continue
        variant = f"{wt[p - 1]}{p}{mut}"
        if variant in seen:
            continue
        seen.add(variant)
        rows.append({"variant": variant, "fitness": rng.normal(0, 1)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return wt


def _split_mirror(csv_path, initial_train, seed=42):
    """镜像 run_evolution(seed) 的种子化划分：返回 (初始已知集合, 未测池列表)。"""
    df = pd.read_csv(csv_path)
    idx = np.random.RandomState(seed).permutation(len(df))
    known = set(df["variant"].iloc[idx[:initial_train]])
    return known, [v for v in df["variant"] if v not in known]


# ============================================================
# 1. lower_is_better 排序方向
# ============================================================

def _lower_config():
    return ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1],
                         variant_format="mutation_string", fitness_higher_is_better=False)


def _higher_config():
    return ProteinConfig(reference_sequence="AC", mutable_positions=[0, 1],
                         variant_format="mutation_string", fitness_higher_is_better=True)


def _config_toolbox(protein_config, scores):
    """手工构造 ToolBox：predict_fn 按请求序返回给定分数（原始标度）。"""
    known = pd.DataFrame([{"variant": "AA", "fitness": 2.0, "mutations": "synthetic"}])
    pool = list(scores)
    return ToolBox(known=known, eligible=set(pool),
                   predict_fn=lambda sequences: [scores[s] for s in sequences],
                   config=CONFIG, protein_config=protein_config)


def test_scored_ranking_ascending_under_lower_direction():
    # DSH 复现：lower 场景 score_result 仍按数值降序（rank1=最差）。修复后 rank1=预测分最低。
    scores = {"CA": 3.0, "CC": 1.0, "DA": 2.0}
    toolbox = _config_toolbox(_lower_config(), scores)
    result = toolbox.execute("predict_fitness", json.dumps({"sequences": list(scores)}))
    assert [row["variant"] for row in result["scored"]] == ["CC", "DA", "CA"]
    assert [row["rank"] for row in result["scored"]] == [1, 2, 3]
    # ranked_scored（auto 提交的取序来源）同向：升序
    assert [row["variant"] for row in toolbox.ranked_scored()] == ["CC", "DA", "CA"]


def test_scored_ranking_descending_under_higher_direction_unchanged():
    # higher（含 GB1 语义）保持降序——config 路径与旧路径方向一致
    scores = {"CA": 3.0, "CC": 1.0, "DA": 2.0}
    toolbox = _config_toolbox(_higher_config(), scores)
    toolbox.execute("predict_fitness", json.dumps({"sequences": list(scores)}))
    assert [row["variant"] for row in toolbox.ranked_scored()] == ["CA", "DA", "CC"]


def test_known_summary_best_follows_direction():
    # 历史摘要（get_experiment_history 的 known_data）按方向取最优：lower 取 min / higher 取 max
    known = pd.DataFrame([
        {"variant": "AC", "fitness": 1.0, "mutations": "syn"},
        {"variant": "CC", "fitness": 5.0, "mutations": "syn"},
        {"variant": "GC", "fitness": 3.0, "mutations": "syn"}])
    for config, expected in ((_lower_config(), 1.0), (_higher_config(), 5.0)):
        toolbox = ToolBox(known=known, eligible={"DC"}, predict_fn=lambda seqs: [1.0] * len(seqs),
                          config=CONFIG, protein_config=config)
        known_data = toolbox.execute("get_experiment_history", "{}")["known_data"]
        assert known_data["best_fitness"] == expected
        assert known_data["top_variants"][0]["fitness"] == expected


def test_two_round_lower_run_auto_submits_lowest(tmp_path):
    """两轮 mock run（lower 方向，评分预算耗尽触发 auto 提交）：
    auto 提交选的是预测分最低候选；cumulative_best 按原始标度取 min；summary 记 direction。"""
    csv_path = tmp_path / "tiny_lower.csv"
    wt = _tiny_csv(csv_path)
    known0, pool = _split_mirror(csv_path, initial_train=30)
    batch1, batch2 = pool[:3], pool[3:6]
    completions = _Completions([
        _response([("predict_fitness", {"sequences": list(batch1)})]),
        _response([("predict_fitness", {"sequences": list(batch2)})]),
    ])
    out_dir = tmp_path / "evolve_lower"
    summary = run_evolution(str(csv_path), rounds=2, per_round=1, score_budget=3, initial_train=30,
                            reference_sequence=wt, higher_is_better=False, seed=42,
                            out_dir=out_dir, client=_client(completions), model="mock")

    # summary 记录方向；两轮均为预算耗尽 auto 提交
    assert summary["config"]["higher_is_better"] is False
    assert [r["status"] for r in summary["rounds"]] == ["auto_submitted", "auto_submitted"]
    assert summary["rounds"][0]["stop_reason"] == "max_score_calls"

    # 逐轮核对：评分列表升序，auto 提交 = 预测分最低候选，理由文本方向语义（rank 1 by predicted fitness）
    fitness = dict(zip(pd.read_csv(csv_path)["variant"], pd.read_csv(csv_path)["fitness"]))
    submitted = []
    for r, batch in ((1, batch1), (2, batch2)):
        entries = [json.loads(line) for line in
                   (out_dir / f"agent_round{r}.jsonl").read_text(encoding="utf-8").splitlines()]
        predict = next(e for e in entries if e["kind"] == "tool_call"
                       and e["tool_name"] == "predict_fitness")
        scored = predict["result"]["scored"]
        values = [row["predicted_fitness"] for row in scored]
        assert values == sorted(values)  # lower：rank1 = 预测分最低（升序）
        auto = next(e for e in entries if e["kind"] == "tool_call"
                    and e["tool_name"] == "submit_recommendations" and e.get("auto"))
        assert auto["arguments"]["sequences"] == [scored[0]["variant"]]  # 选最低而非最高
        assert "by predicted fitness" in auto["arguments"]["reasons"][0]
        submitted.append(scored[0]["variant"])

    # 累计最优按原始测量标度取 min（初始已知 ∪ 两轮提交）
    measured_all = [fitness[v] for v in known0 | set(submitted)]
    assert summary["cumulative_best"]["fitness"] == pytest.approx(min(measured_all))
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert per_round["variant"].tolist() == submitted


# ============================================================
# 2. open 的池选择透传研究阶段
# ============================================================

def _spy_configure_protein(monkeypatch, recorder):
    """包裹 protevo.evolve 命名空间内的 configure_protein：先记录 kwargs 再放行。"""
    import protevo.evolve as evolve_mod
    real_configure = evolve_mod.configure_protein

    def _spy(data_path, **kwargs):
        recorder(dict(kwargs))
        return real_configure(data_path, **kwargs)

    monkeypatch.setattr(evolve_mod, "configure_protein", _spy)


def test_run_evolution_passes_pool_to_configure_protein(tmp_path):
    csv_path = tmp_path / "tiny_pool.csv"
    wt = _tiny_csv(csv_path)
    _, pool = _split_mirror(csv_path, initial_train=30)
    batch = pool[:2]
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": list(batch), "reasons": ["a", "b"]})])])
    out_dir = tmp_path / "evolve_pool"
    summary = run_evolution(str(csv_path), rounds=1, per_round=2, score_budget=4, initial_train=30,
                            reference_sequence=wt, predictor_pool="full", seed=42,
                            out_dir=out_dir, client=_client(completions), model="mock")

    # 研究阶段 predictor_info 的池 = full，候选含 onehot_mlp；summary 记录实际采用的池
    predictor = summary["config"]["predictor"]
    assert predictor["pool"]["mode"] == "full"
    assert "onehot_mlp" in predictor["pool"]["included"]
    assert "onehot_mlp" in predictor["all_candidates"]


def test_configure_protein_receives_pool_kwarg_default_and_override(tmp_path, monkeypatch):
    csv_path = tmp_path / "tiny_spy.csv"
    wt = _tiny_csv(csv_path)
    _, pool = _split_mirror(csv_path, initial_train=30)
    kwargs_log = []
    _spy_configure_protein(monkeypatch, kwargs_log.append)

    # 默认（不传池）：configure_protein 不收 predictor_pool 键（现行为保持）
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": list(pool[:2]), "reasons": ["a", "b"]})])])
    run_evolution(str(csv_path), rounds=1, per_round=2, score_budget=4, initial_train=30,
                  reference_sequence=wt, seed=42, out_dir=tmp_path / "out_default",
                  client=_client(completions), model="mock")
    assert len(kwargs_log) == 1 and "predictor_pool" not in kwargs_log[0]

    # 传池 + --retrain：初始选型与每轮重配都收到 predictor_pool="full"
    kwargs_log.clear()
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": list(pool[:2]), "reasons": ["a", "b"]})]),
        _response([("submit_recommendations", {"sequences": list(pool[2:4]), "reasons": ["c", "d"]})])])
    run_evolution(str(csv_path), rounds=2, per_round=2, score_budget=4, initial_train=30,
                  reference_sequence=wt, predictor_pool="full", seed=42,
                  out_dir=tmp_path / "out_full", client=_client(completions), model="mock",
                  retrain_each_round=True)
    assert [kw["predictor_pool"] for kw in kwargs_log] == ["full", "full"]  # 初始 + 第 2 轮重配


def test_tui_open_pool_full_survives_into_run(tmp_path, monkeypatch, capsys):
    """DSH 场景：open --pool full 后 run——研究阶段池保持 full（修复前回落 recommended，MLP 消失）。"""
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    csv_path = tmp_path / "tiny_tui.csv"
    wt = _tiny_csv(csv_path)
    _, pool = _split_mirror(csv_path, initial_train=30)
    shell.onecmd(f"open {csv_path} --ref {wt} --pool full")
    assert shell.pool_used == "full"
    capsys.readouterr()

    batch = pool[:2]
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": list(batch), "reasons": ["a", "b"]})])])
    shell._client_for_test = (_client(completions), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    assert "第 1/1 轮" in capsys.readouterr().out

    predictor = shell.last_summary["config"]["predictor"]
    assert predictor["pool"]["mode"] == "full"
    assert "onehot_mlp" in predictor["pool"]["included"]


# ============================================================
# 3. esm2_mlp_spec 反映实际构造参数
# ============================================================

def test_esm2_mlp_spec_default_unchanged_without_override():
    # 无覆盖：与默认规格常量逐字一致（回归不变，GPU 事实里的默认描述保持）
    assert _esm2_mlp_spec_text(None) == _ESM2_MLP_SPEC
    assert _esm2_mlp_spec_text({}) == _ESM2_MLP_SPEC
    assert _esm2_mlp_spec_text({"esm2_ridge": {"alpha": 1}}) == _ESM2_MLP_SPEC
    assert _esm2_mlp_spec_text({"esm2_mlp": {}}) == _ESM2_MLP_SPEC


def test_esm2_mlp_spec_reflects_model_kwargs_override():
    # DSH GPU 场景：覆盖 (8,)/20/False 后 spec 描述实际构造参数而非默认 (256,64)/500/True
    spec = _esm2_mlp_spec_text({"esm2_mlp": {"hidden_layer_sizes": [8], "max_iter": 20,
                                             "early_stopping": False}})
    assert "hidden_layer_sizes=(8,)" in spec
    assert "max_iter=20" in spec
    assert "early_stopping=False" in spec
    assert "alpha=0.0001" in spec        # 未覆盖项保持默认值（1e-4 的 repr 渲染）
    assert "random_state=42" in spec
    assert "(256,64)" not in spec and "max_iter=500" not in spec
    # 与 model_kwargs_applied 描述同一组参数：标注覆盖来源
    assert "model_kwargs['esm2_mlp']" in spec


def test_esm2_mlp_spec_partial_override_and_tuple_render():
    spec = _esm2_mlp_spec_text({"esm2_mlp": {"hidden_layer_sizes": (128, 32)}})
    assert "hidden_layer_sizes=(128,32)" in spec
    assert "max_iter=500" in spec and "early_stopping=True" in spec  # 其余默认保留


# ============================================================
# 4. 中断运行的部分结果可用
# ============================================================

def _interrupted_setup(tmp_path):
    csv_path = tmp_path / "tiny_int.csv"
    wt = _tiny_csv(csv_path)
    _, pool = _split_mirror(csv_path, initial_train=30)
    batch = pool[:2]
    # 第 1 轮正常提交；第 2 轮 create 抛 429 式异常
    completions = _FailingCompletions(
        [_response([("submit_recommendations", {"sequences": list(batch), "reasons": ["a", "b"]})])],
        RuntimeError("Error code: 429 - bounded test transport is unavailable"))
    return csv_path, wt, batch, completions


def test_interrupted_run_keeps_completed_rounds_on_disk(tmp_path):
    csv_path, wt, batch, completions = _interrupted_setup(tmp_path)
    out_dir = tmp_path / "evolve_int"
    summary = run_evolution(str(csv_path), rounds=2, per_round=2, score_budget=4, initial_train=30,
                            reference_sequence=wt, seed=42, out_dir=out_dir,
                            client=_client(completions), model="mock")

    # 返回中断汇总（不再向上抛）：status / error 摘要 / 已完成轮数
    assert summary["status"] == "interrupted"
    assert summary["error"]["failed_round"] == 2
    assert summary["error"]["completed_rounds"] == 1
    assert summary["error"]["type"] == "RuntimeError"
    assert "429" in summary["error"]["message"]
    assert [r["round"] for r in summary["rounds"]] == [1]
    assert summary["rounds"][0]["status"] == "submitted"

    # 落盘的部分结果：per_round.csv 只含第 1 轮，summary.json 记 interrupted
    per_round = pd.read_csv(out_dir / "per_round.csv")
    assert per_round["round"].tolist() == [1, 1]
    assert per_round["variant"].tolist() == list(batch)
    on_disk = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "interrupted" and on_disk["error"]["failed_round"] == 2
    assert on_disk["cumulative_best"] == summary["cumulative_best"]
    # 第 1 轮日志完整收尾；第 2 轮留下 api_error 记录（可回放定位中断原因）
    entries1 = [json.loads(line) for line in
                (out_dir / "agent_round1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "round_end" for e in entries1)
    entries2 = [json.loads(line) for line in
                (out_dir / "agent_round2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "api_error" and "429" in e["result"]["error"] for e in entries2)


def test_tui_interrupted_run_points_last_run_dir_at_partial_results(tmp_path, monkeypatch, capsys):
    """DSH 场景：第 2 轮外部 429 后 TUI 曾显示"最近运行：无"、log/export 全不可用。"""
    monkeypatch.chdir(tmp_path)
    shell = ProtevoShell(env_path=tmp_path / ".env")
    csv_path, wt, _batch, completions = _interrupted_setup(tmp_path)
    shell.onecmd(f"open {csv_path} --ref {wt}")
    capsys.readouterr()
    shell._client_for_test = (_client(completions), "mock")
    shell.onecmd("run --rounds 2 --per-round 2 --budget 4")
    out = capsys.readouterr().out
    assert "运行中断" in out and "429" in out          # 中断信息不吞
    assert "部分结果" in out and "log / export / plot 照常可用" in out

    # 会话 last_run_dir 指向部分结果目录（修复前 last_out_dir 为 None）
    assert shell.last_out_dir is not None and shell.last_out_dir.is_dir()
    assert shell.last_summary["status"] == "interrupted"

    shell.onecmd("status")
    assert str(shell.last_out_dir) in capsys.readouterr().out  # 最近运行可见
    shell.onecmd("log")
    assert "第 1 轮：status=submitted" in capsys.readouterr().out  # 已完成轮可回放

    export_dir = tmp_path / "export_int"
    shell.onecmd(f"export --out {export_dir}")
    capsys.readouterr()
    for name in ("per_round.csv", "summary.json", "agent_round1.jsonl", "agent_round2.jsonl"):
        assert (export_dir / name).is_file(), name
    exported = json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))
    assert exported["status"] == "interrupted"

    shell.onecmd("plot trajectory")
    assert "轨迹:" in capsys.readouterr().out  # 图表对 interrupted 目录照常可用


# ============================================================
# 5. GB1 冻结路径（protein_config=None）零变化回归
# ============================================================

def test_legacy_path_ranked_scored_still_descending():
    # 旧路径排序键原样保留：恒按预测分降序（GB1 恒 higher，语义一致）
    known = pd.DataFrame([{"variant": "VDGV", "fitness": 1.0, "mutations": "synthetic"}])
    toolbox = ToolBox(known=known, eligible={"VDGA", "VDGC", "IDGV"},
                      predict_fn=lambda sequences: [1.0] * len(sequences),
                      config=ExperimentConfig(k=1, proposal_count=1, score_budget=1, knowledge_path=None))
    toolbox.score_cache = {"VDGA": 3.0, "VDGC": 1.0, "IDGV": 2.0}
    assert [row["variant"] for row in toolbox.ranked_scored()] == ["VDGA", "IDGV", "VDGC"]


def test_tui_predict_marker_follows_direction(tmp_path, monkeypatch, capsys):
    # TUI 展示标记：higher 标最大（▲，现行为）；lower 标最小（▼，修复后）
    monkeypatch.chdir(tmp_path)
    from protevo.protein_config import configure_protein
    csv_path = tmp_path / "tiny_marker.csv"
    wt = _tiny_csv(csv_path, n_rows=40)
    shell = ProtevoShell(env_path=tmp_path / ".env")

    df = pd.read_csv(csv_path)
    variants = [df["variant"].iloc[0], df["variant"].iloc[1]]

    shell.config = configure_protein(str(csv_path), reference_sequence=wt, higher_is_better=True)
    shell.dataset_label, shell.data_path, shell.n_rows = "marker", csv_path, len(df)
    shell.onecmd(f"predict {variants[0]} {variants[1]}")
    out_higher = capsys.readouterr().out
    scores_higher = shell.config.predict_fitness(variants)
    best_higher = variants[scores_higher.index(max(scores_higher))]
    marker_lines = [line for line in out_higher.splitlines() if "▲" in line or "▼" in line]
    assert len(marker_lines) == 1 and best_higher in marker_lines[0] and "▲" in marker_lines[0]

    shell.config = configure_protein(str(csv_path), reference_sequence=wt, higher_is_better=False)
    shell.onecmd(f"predict {variants[0]} {variants[1]}")
    out_lower = capsys.readouterr().out
    scores_lower = shell.config.predict_fitness(variants)
    best_lower = variants[scores_lower.index(min(scores_lower))]
    marker_lines = [line for line in out_lower.splitlines() if "▲" in line or "▼" in line]
    assert len(marker_lines) == 1 and best_lower in marker_lines[0] and "▼" in marker_lines[0]
