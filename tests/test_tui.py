"""TUI（protevo.tui）测试：命令处理器 .onecmd + capsys、首次启动引导（非 tty 不阻塞 +
.env 三键写入规则）、run/log/plot/export 全链路（mock 客户端）、无凭据降级、evolve on_round 回调。

无网络、无真实 LLM：mock 客户端复用 test_evolve 的 _Completions 模式；ask 的降级路径经
monkeypatch protevo.agent_v2.build_openai_client（configure_with_llm 在函数内延迟导入该名）。
"""
import json
import os
import re
import subprocess
import sys
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo import tui
from protevo.agent import MissingCredentials
from protevo.evolve import run_evolution
from protevo.tui import ProtevoShell, credentials_complete, run_setup_wizard, write_env_updates

PROJECT = Path(__file__).parent.parent
UBIQ_CSV = PROJECT / "data/external/ubiq_roscoe_2014/dms.csv"
AA = "ACDEFGHIKLMNPQRSTVWY"
IDS = count()


# ---------- mock 客户端脚手架（同 test_evolve / test_agent_v2_universal 模式） ----------

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


def _ubiq_pool(per_round=3, initial_train=None):
    """镜像 run_evolution(seed=42) 对 ubiq CSV 的划分：返回未测池前 per_round 条（mock 提交用）。"""
    df = pd.read_csv(UBIQ_CSV)
    initial_train = initial_train if initial_train is not None else min(2000, max(1, len(df) // 2))
    idx = np.random.RandomState(42).permutation(len(df))
    known = set(df["variant"].iloc[idx[:initial_train]])
    pool = [v for v in df["variant"] if v not in known]
    return list(pool[:per_round])


def _make_shell(tmp_path, monkeypatch):
    """chdir 到 tmp（run_evolution / 图表的相对输出目录落临时区），返回新 shell。"""
    monkeypatch.chdir(tmp_path)
    return ProtevoShell(env_path=tmp_path / ".env")


# ---------- .env 写入规则（三键管理 + 无关行不动 + 幂等） ----------

def test_write_env_updates_creates_and_preserves(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FOO=1\n# 注释行保留\nOPENAI_MODEL=old-model\n", encoding="utf-8")
    write_env_updates(env, {"OPENAI_BASE_URL": "https://api.deepseek.com",
                            "OPENAI_API_KEY": "sk-test-123",
                            "OPENAI_MODEL": "deepseek-chat"})
    text = env.read_text(encoding="utf-8")
    assert "FOO=1" in text and "# 注释行保留" in text          # 无关键/注释逐字保留
    assert "OPENAI_MODEL=deepseek-chat" in text                 # 已有键原位更新
    assert "OPENAI_BASE_URL=https://api.deepseek.com" in text   # 新键追加
    assert "OPENAI_API_KEY=sk-test-123" in text
    assert len(re.findall(r"^OPENAI_MODEL=", text, re.M)) == 1  # 无重复


def test_write_env_updates_idempotent_and_removal(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FOO=1\n", encoding="utf-8")
    updates = {"OPENAI_BASE_URL": "https://api.deepseek.com", "OPENAI_API_KEY": "k",
               "OPENAI_MODEL": "m1"}
    write_env_updates(env, updates)
    write_env_updates(env, updates)  # 二次 setup：更新不重复
    text = env.read_text(encoding="utf-8")
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        assert len(re.findall(rf"^{key}=", text, re.M)) == 1
    assert "FOO=1" in text
    write_env_updates(env, {"OPENAI_MODEL": "m2", "OPENAI_BASE_URL": None})  # 改官方端点→删 BASE_URL
    text = env.read_text(encoding="utf-8")
    assert "OPENAI_MODEL=m2" in text and "OPENAI_BASE_URL" not in text
    assert "OPENAI_API_KEY=k" in text and "FOO=1" in text


def test_write_env_updates_empty_file_created(tmp_path):
    env = tmp_path / ".env"  # 不存在 → 创建
    write_env_updates(env, {"OPENAI_MODEL": "gpt-4o-mini"})
    assert env.read_text(encoding="utf-8").strip() == "OPENAI_MODEL=gpt-4o-mini"


# ---------- 交互式向导（scripted stdin + mock getpass） ----------

def test_setup_wizard_writes_three_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=1\n", encoding="utf-8")
    answers = iter(["2", "", "y"])  # 端点选 DeepSeek；模型名回车取默认 deepseek-chat；确认写入
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(tui.getpass, "getpass", lambda prompt="": "sk-wizard-key")
    run_setup_wizard(env_path=env)
    text = env.read_text(encoding="utf-8")
    assert "OPENAI_BASE_URL=https://api.deepseek.com" in text
    assert "OPENAI_API_KEY=sk-wizard-key" in text
    assert "OPENAI_MODEL=deepseek-chat" in text  # DeepSeek 端点的建议默认值
    assert "FOO=1" in text


def test_setup_command_non_tty_does_not_block(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tui.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("setup")  # pytest 下 stdin 非 tty：打印引导文本，立即返回
    out = capsys.readouterr().out
    assert "非交互终端" in out and "setup" in out


# ---------- 非交互启动（子进程：echo help/quit | python -m protevo.tui） ----------

def _run_tui_subprocess(tmp_path, extra_args=()):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "OPENAI_MODEL",
                                                            "OPENAI_BASE_URL")}
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-m", "protevo.tui", *extra_args],
                          input="help\nquit\n", cwd=str(PROJECT), env=env,
                          text=True, capture_output=True, timeout=180)
    out_file = tmp_path / "tui_stdout.txt"  # 输出落临时文件再断言
    out_file.write_text(proc.stdout, encoding="utf-8")
    return proc, out_file.read_text(encoding="utf-8")


def test_startup_non_tty_does_not_block(tmp_path):
    proc, text = _run_tui_subprocess(tmp_path)
    assert proc.returncode == 0
    assert "再见" in text            # quit 生效 → 没有阻塞挂在 REPL
    assert "datasets" in text        # help 命令表正常输出
    assert "open" in text and "run" in text
    if not credentials_complete():   # 包根 .env 无凭据 → 打印一行凭据缺失说明
        assert "未检测到 LLM 凭据" in text
        assert "setup" in text


def test_startup_no_wizard_flag_skips_notice(tmp_path):
    proc, text = _run_tui_subprocess(tmp_path, extra_args=("--no-wizard",))
    assert proc.returncode == 0 and "再见" in text
    if not credentials_complete():
        assert "未检测到 LLM 凭据" not in text  # --no-wizard：跳过引导（连说明也不打）


# ---------- 基础命令：datasets / help / quit / 未知命令 ----------

def test_datasets_lists_four_builtins(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("datasets")
    out = capsys.readouterr().out
    for name in ("gb1", "ubiq", "gfp", "blat"):
        assert name in out
    assert "149361 行" in out and "1380 行" in out  # 行数来自 CSV/ meta.json


def test_help_and_quit(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("help")
    out = capsys.readouterr().out
    for command in ("datasets", "open", "run", "plot", "log", "export", "plan",
                    "doctor", "ask", "predict", "setup", "model", "status", "info"):
        assert command in out
    assert shell.onecmd("quit") is True
    assert "再见" in capsys.readouterr().out


def test_unknown_command_hint(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("frobnicate")
    assert "未知命令" in capsys.readouterr().out


def test_guidance_without_open(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    for command in ("run", "plot heatmap", "predict I3V", "info"):
        shell.onecmd(command)
    out = capsys.readouterr().out
    assert out.count("请先 open") == 4


# ---------- open 内置数据集：配置卡字段 ----------

def test_open_ubiq_config_card(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("open ubiq")
    out = capsys.readouterr().out
    assert "mutation_notation" in out          # 格式
    assert "75 个可变位点" in out               # 位点
    assert "显式提供" in out                    # 参考来源（meta.json wt_sequence）
    assert "higher is better" in out            # 方向（meta.json higher_is_better）
    assert "val ρ=0." in out                    # 选型与验证 Spearman
    assert "random_forest" in out and "ridge" in out  # 全部候选
    assert shell.config is not None and shell.dataset_label == "ubiq"
    assert re.match(r"^protevo \(ubiq\|(RF|Ridge) ρ=0\.\d+\)>\s*$", shell.prompt)


def test_open_then_predict(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("open ubiq")
    capsys.readouterr()
    shell.onecmd("predict I3V K6R")
    out = capsys.readouterr().out
    assert "I3V" in out and "K6R" in out
    assert re.search(r"I3V\s+-?\d", out) and re.search(r"K6R\s+-?\d", out)  # 原格式进、数值出


# ---------- run（mock 客户端 1 轮）+ log + plot + export 全链路 ----------

def _shell_after_open_ubiq(tmp_path, monkeypatch):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("open ubiq")
    return shell


def test_run_one_round_with_mock_client(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    capsys.readouterr()
    batch = _ubiq_pool(per_round=3)
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": batch, "reasons": ["r1", "r2", "r3"]})])])
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 3 --budget 6")
    out = capsys.readouterr().out
    assert "第 1/1 轮" in out and "预算余量" in out      # on_round 准实时一行
    assert "── 进化轨迹 ──" in out and "submitted" in out  # 轨迹小表
    assert shell.last_summary is not None and shell.last_summary["rounds"][0]["submitted"] == 3
    out_dir = Path(shell.last_summary["out_dir"])
    assert (out_dir / "summary.json").is_file() and (out_dir / "per_round.csv").is_file()
    assert (out_dir / "agent_round1.jsonl").is_file()
    frame = pd.read_csv(out_dir / "per_round.csv")
    assert frame["variant"].tolist() == batch           # mock 提交的 3 条全部落盘
    assert "输出目录" in out


def test_run_without_credentials_prints_setup_hint(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # evolve 在模块顶层 from .agent_v2 import build_openai_client → 补丁须打在 evolve 命名空间
    monkeypatch.setattr("protevo.evolve.build_openai_client",
                        lambda *a, **k: (_ for _ in ()).throw(
                            MissingCredentials("Set project OPENAI_API_KEY and OPENAI_MODEL")))
    shell.onecmd("run --rounds 1")
    out = capsys.readouterr().out
    assert "运行失败" in out and "setup" in out


def test_log_replay_after_run(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    batch = _ubiq_pool(per_round=2)
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": batch, "reasons": ["a", "b"]})])])
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    capsys.readouterr()
    shell.onecmd("log")
    out = capsys.readouterr().out
    assert "第 1 轮：status=submitted" in out and "submit_recommendations×1" in out
    shell.onecmd("log 1")
    out = capsys.readouterr().out
    assert "[round_start]" in out and "[tool_call] submit_recommendations" in out
    assert "提交 " in out and "a" in out               # submit 的变体 + 理由
    assert "[round_end] status=submitted" in out


def test_plot_trajectory_scatter_budget_png(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    batch = _ubiq_pool(per_round=2)
    script = [_response([("predict_fitness", {"sequences": list(batch)})]),
              _response([("submit_recommendations", {"sequences": list(batch),
                                                     "reasons": ["a", "b"]})])]
    completions = _Completions(script)
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    capsys.readouterr()
    out_dir = Path(shell.last_summary["out_dir"])
    shell.onecmd("plot trajectory")
    assert "轨迹:" in capsys.readouterr().out and (out_dir / "trajectory.png").is_file()
    shell.onecmd("plot scatter")
    assert "散点:" in capsys.readouterr().out and (out_dir / "scatter.png").is_file()
    shell.onecmd("plot budget")
    assert "预算:" in capsys.readouterr().out and (out_dir / "budget.png").is_file()


def test_plot_requires_run_hint(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    shell.onecmd("plot trajectory")
    assert "先 run" in capsys.readouterr().out


def test_plot_matplotlib_missing_degrades(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    batch = _ubiq_pool(per_round=2)
    completions = _Completions([
        _response([("submit_recommendations", {"sequences": list(batch), "reasons": ["a", "b"]})])])
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    capsys.readouterr()

    def _no_matplotlib():
        raise ImportError("No module named 'matplotlib'")
    monkeypatch.setattr(tui, "_matplotlib_pyplot", _no_matplotlib)
    shell.onecmd("plot trajectory")
    out = capsys.readouterr().out
    assert "轨迹:" in out and "matplotlib 未安装" in out
    assert not (Path(shell.last_summary["out_dir"]) / "trajectory.png").exists()


# ---------- plot heatmap（合成替换库）/ plot length（合成 indel 库） ----------

def _synthetic_substitution_csv(path, n_rows=120, seed=3):
    """60 aa WT 记谱格式合成替换库：3 个位点单/双突变，K/R 加分信号。"""
    rng = np.random.RandomState(seed)
    wt = "".join(rng.choice(list(AA), size=60, replace=True))
    rows, seen = [], set()
    while len(rows) < n_rows:
        picks = sorted(rng.choice([6, 12, 18], size=1 + int(rng.rand() < 0.5), replace=False))
        tokens = []
        signal = 0.0
        for p in picks:
            mut = AA[int(rng.randint(0, 20))]
            if mut == wt[p - 1]:
                continue
            tokens.append(f"{wt[p - 1]}{p}{mut}")
            signal += 1.0 if mut in "KR" else 0.0
        if not tokens:
            continue
        variant = ",".join(tokens)
        if variant in seen:
            continue
        seen.add(variant)
        rows.append({"variant": variant, "fitness": signal + rng.normal(0, 0.05)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return wt


def _synthetic_indel_csv(path, wt, n_rows=100, seed=5):
    """对给定 WT 合成 indel 记谱库：del / ins / dup 混合。"""
    rng = np.random.RandomState(seed)
    rows, seen = [], set()
    while len(rows) < n_rows:
        pos = int(rng.randint(2, len(wt) - 2))
        kind = rng.choice(["del", "ins", "dup"])
        if kind == "del":
            variant = f"{wt[pos - 1]}{pos}del"
        elif kind == "ins":
            # 记谱格式 V39_G40insAA：锚点残基对 + 各自位点号 + ins 序列
            variant = (f"{wt[pos - 1]}{pos}_{wt[pos]}{pos + 1}"
                       f"ins{''.join(rng.choice(list(AA), size=2))}")
        else:
            variant = f"{wt[pos - 1]}{pos}dup"
        if variant in seen:
            continue
        seen.add(variant)
        rows.append({"variant": variant, "fitness": rng.normal(0, 1)})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_plot_heatmap_on_substitution_library(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    csv_path = tmp_path / "subst.csv"
    wt = _synthetic_substitution_csv(csv_path)
    shell.onecmd(f"open {csv_path} --ref {wt}")
    capsys.readouterr()
    shell.onecmd("plot heatmap")
    out = capsys.readouterr().out
    assert "热图:" in out and "PNG" in out
    assert (tmp_path / "results/tui/heatmap.png").is_file()


def test_plot_length_on_indel_library(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    rng = np.random.RandomState(11)
    wt = "".join(rng.choice(list(AA), size=40, replace=True))
    csv_path = tmp_path / "indel.csv"
    _synthetic_indel_csv(csv_path, wt)
    shell.onecmd(f"open {csv_path} --ref {wt}")
    out = capsys.readouterr().out
    assert "indel_notation" in out                     # 走了 indel 专用路径
    shell.onecmd("plot length")
    out = capsys.readouterr().out
    assert "长度分布:" in out and f"WT {len(wt)}" in out
    assert (tmp_path / "results/tui/length.png").is_file()


def test_plot_cross_format_guidance(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    wt = _synthetic_substitution_csv(tmp_path / "subst.csv")
    shell.onecmd(f"open {tmp_path / 'subst.csv'} --ref {wt}")
    capsys.readouterr()
    shell.onecmd("plot length")                        # 替换库用 length → 指引 heatmap
    out = capsys.readouterr().out
    assert "plot heatmap" in out and "绘图失败" not in out

    indel_shell = _make_shell(tmp_path, monkeypatch)
    rng = np.random.RandomState(11)
    indel_wt = "".join(rng.choice(list(AA), size=40, replace=True))
    _synthetic_indel_csv(tmp_path / "indel.csv", indel_wt)
    indel_shell.onecmd(f"open {tmp_path / 'indel.csv'} --ref {indel_wt}")
    capsys.readouterr()
    indel_shell.onecmd("plot heatmap")                 # indel 库用 heatmap → 指引 length
    assert "plot length" in capsys.readouterr().out


def test_export_bundles_artifacts(tmp_path, monkeypatch, capsys):
    shell = _shell_after_open_ubiq(tmp_path, monkeypatch)
    batch = _ubiq_pool(per_round=2)
    script = [_response([("predict_fitness", {"sequences": list(batch)})]),
              _response([("submit_recommendations", {"sequences": list(batch),
                                                     "reasons": ["a", "b"]})])]
    completions = _Completions(script)
    shell._client_for_test = (SimpleNamespace(chat=SimpleNamespace(completions=completions)), "mock")
    shell.onecmd("run --rounds 1 --per-round 2 --budget 4")
    capsys.readouterr()
    out_dir = tmp_path / "export_out"
    shell.onecmd(f"export --out {out_dir}")
    out = capsys.readouterr().out
    assert "── 导出清单" in out
    for name in ("per_round.csv", "summary.json", "agent_round1.jsonl", "trajectory.png", "scatter.png",
                 "budget.png", "heatmap.png"):
        assert (out_dir / name).is_file(), name


# ---------- model / status / plan / doctor / ask ----------

def test_model_command_sets_session_model(tmp_path, monkeypatch, capsys):
    # do_model 直接写 os.environ：先用 monkeypatch 登记还原点，避免泄漏到后续测试
    if "OPENAI_MODEL" in os.environ:
        monkeypatch.setenv("OPENAI_MODEL", os.environ["OPENAI_MODEL"])
    else:
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("model deepseek-chat")
    assert "deepseek-chat" in capsys.readouterr().out
    shell.onecmd("model")
    out = capsys.readouterr().out
    assert "deepseek-chat" in out and "OPENAI_MODEL" not in out
    assert os.environ.get("OPENAI_MODEL") == "deepseek-chat"


def test_status_shows_session_state(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("status")
    out = capsys.readouterr().out
    assert "未 open" in out and "凭据" in out


def test_plan_without_credentials(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("open ubiq")
    capsys.readouterr()
    shell.onecmd("plan")
    out = capsys.readouterr().out
    assert "── 探测 + 规则计划 ──" in out
    assert "mutation_notation" in out            # 数据探测 + 格式判定
    assert "use_esm=" in out and "device=" in out


def test_ask_without_credentials_degrades(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr("protevo.agent_v2.build_openai_client",
                        lambda *a, **k: (_ for _ in ()).throw(
                            MissingCredentials("Set project OPENAI_API_KEY and OPENAI_MODEL")))
    shell.onecmd("ask 这台机器该怎么配")          # 无凭据：降级说明 + setup 引导，不崩
    out = capsys.readouterr().out
    assert "LLM 未参与" in out and "setup" in out


def test_doctor_runs_and_renders(tmp_path, monkeypatch, capsys):
    shell = _make_shell(tmp_path, monkeypatch)
    shell.onecmd("doctor")
    out = capsys.readouterr().out
    assert "protevo 环境体检报告" in out and "llm_credentials" in out


# ---------- evolve.py 的 on_round 回调（TUI 准实时的底层支撑） ----------

def _tiny_csv(path, n_rows=60, seed=7):
    rng = np.random.RandomState(seed)
    wt = "".join(rng.choice(list(AA), size=40, replace=True))
    rows, seen = [], set()
    while len(rows) < n_rows:
        p = int(rng.choice([5, 10, 15, 20]))  # 4 位点 × 19 残基 > n_rows，循环必然终止
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


def _run_one_round_with_mock(tmp_path, on_round):
    csv_path = tmp_path / "tiny.csv"
    wt = _tiny_csv(csv_path)
    df = pd.read_csv(csv_path)
    initial_train = min(2000, max(1, len(df) // 2))
    idx = np.random.RandomState(42).permutation(len(df))
    pool = [v for v in df["variant"] if v not in set(df["variant"].iloc[idx[:initial_train]])][:2]
    completions = _Completions([_response([("submit_recommendations",
                                            {"sequences": list(pool), "reasons": ["a", "b"]})])])
    return run_evolution(str(csv_path), rounds=1, per_round=2, score_budget=4,
                         initial_train=initial_train, reference_sequence=wt, seed=42,
                         out_dir=tmp_path / "out",
                         client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
                         model="mock", on_round=on_round)


def test_run_evolution_on_round_receives_summary(tmp_path):
    received = []
    summary = _run_one_round_with_mock(tmp_path, received.append)
    assert len(received) == 1
    assert received[0]["round"] == 1
    assert received[0]["submitted"] == 2
    assert received[0] == summary["rounds"][0]          # 与 summary 明细同构


def test_run_evolution_on_round_exception_swallowed(tmp_path):
    def _boom(_summary):
        raise RuntimeError("展示层故障不应影响进化")
    summary = _run_one_round_with_mock(tmp_path, _boom)  # 回调抛错被吞掉，主流程正常完成
    assert summary["rounds"][0]["status"] == "submitted"
