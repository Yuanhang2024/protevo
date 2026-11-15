# tests/test_setup_assistant.py — setup_assistant 单元测试
# 无 torch、无网络、无真实 pip/下载：环境全部打桩，LLM 用 SimpleNamespace mock 客户端，
# auto_execute 分支用 monkeypatch 桩掉 subprocess 与 ensure_esm_checkpoint。
# 合成数据与公式仅用于测试，不代表任何实验结果。
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo import setup_assistant as sa

PROJECT = Path(__file__).parent.parent
EXT = PROJECT / "data/external"


def _env(torch=True, cuda=True, vram=24.0, transformers=True, checkpoint=False, backend=None,
         core_deps_ok=True, in_venv=True, os_name="Windows-11-stub"):
    """构造 probe_environment 返回形状的环境桩。"""
    return {"torch": torch, "cuda": cuda, "backend": backend or ("cuda" if cuda else "unknown"),
            "device_name": "stub-gpu" if cuda else "unknown", "vram_gb": vram,
            "transformers": transformers, "checkpoint_ready": checkpoint, "checkpoint_path": None,
            "os": os_name, "in_venv": in_venv, "core_deps_ok": core_deps_ok}


def _full_csv(tmp_path, seed=1):
    """合成全长序列库：56 aa、两位点低频突变（众数=参考残基，非均匀库）。"""
    ref = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    rng = np.random.RandomState(seed)
    variants = []
    for _ in range(300):
        v = list(ref)
        if rng.rand() > 0.8:
            v[10] = "A" if ref[10] != "A" else "G"
        if rng.rand() > 0.8:
            v[25] = "L" if ref[25] != "L" else "M"
        variants.append("".join(v))
    csv = tmp_path / "full.csv"
    pd.DataFrame({"variant": variants, "fitness": rng.rand(300)}).to_csv(csv, index=False)
    return str(csv)


# ---------- probe_data：真实数据集格式判定 + uniform 护栏 ----------

def test_probe_real_datasets():
    # BLAT：indel 记谱（M1_S2insA）→ 含数字 + indel token → indel_notation
    blat = sa.probe_data(str(EXT / "blat_ecolx_gonzalez_2019/dms.csv"))
    assert blat["indel_suspect"] is True
    assert blat["digits_ratio"] >= 0.99
    assert sa.build_plan(blat, _env())["variant_format"] == "indel_notation"
    # GFP / Ubiq：替换记谱（K3R,...）→ 含数字无 indel token → mutation_notation
    for name in ("gfp_sarkisyan", "ubiq_roscoe_2014"):
        probe = sa.probe_data(str(EXT / name / "dms.csv"))
        assert probe["digits_ratio"] >= 0.99
        assert probe["indel_suspect"] is False
        assert sa.build_plan(probe, _env())["variant_format"] == "mutation_notation"


def test_probe_uniform_suspect(tmp_path):
    rng = np.random.RandomState(2)
    # 枚举型均匀库（GB1 风格）：众数/次众数 ≈ 1.0 → 不可靠
    variants = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 4)) for _ in range(3000)]
    csv = tmp_path / "uniform.csv"
    pd.DataFrame({"variant": variants, "fitness": rng.rand(3000)}).to_csv(csv, index=False)
    assert sa.probe_data(str(csv))["uniform_suspect"] is True
    # 富集型库：WT 残基 ~30%，众数显著占优 → 可靠
    wt, others = "VDGV", "ACDEFHIKLMNPQRSTWY"  # others 不含 G，与 WT 第 2 位不混淆
    enriched = ["".join(wt[p] if rng.rand() < 0.3 else rng.choice(list(others)) for p in range(4))
                for _ in range(1000)]
    csv2 = tmp_path / "enriched.csv"
    pd.DataFrame({"variant": enriched, "fitness": rng.rand(1000)}).to_csv(csv2, index=False)
    assert sa.probe_data(str(csv2))["uniform_suspect"] is False


# ---------- build_plan 四分支 ----------

def test_plan_gpu_indel_650m():
    # 分支 1：GPU + indel 数据 → ESM-2 650M（indel 可按 WT 重构全长序列）
    probe = sa.probe_data(str(EXT / "blat_ecolx_gonzalez_2019/dms.csv"))
    plan = sa.build_plan(probe, _env(vram=24.0))
    assert plan["use_esm"] is True
    assert plan["esm_model"] == "esm2_650m"
    assert plan["esm_model_id"] == "facebook/esm2_t33_650M_UR50D"
    assert plan["device"] == "auto"
    assert plan["install_torch"] is None  # torch 已装
    assert plan["reference_needed"] is True  # indel 库必须提供全长 WT


def test_plan_gpu_low_vram_8m(tmp_path):
    # 分支 2：GPU 低显存 → 降档 8M
    plan = sa.build_plan(sa.probe_data(_full_csv(tmp_path)), _env(vram=6.0))
    assert plan["esm_model"] == "esm2_8m"
    assert plan["esm_model_id"] == "facebook/esm2_t6_8M_UR50D"
    assert plan["use_esm"] is True
    assert "8M" in plan["esm_reason"]


def test_plan_no_torch_cpu(tmp_path):
    # 分支 3：无 torch → CPU 计划 + 三命令选一
    probe = sa.probe_data(_full_csv(tmp_path))
    plan = sa.build_plan(probe, _env(torch=False, cuda=False, vram=None, transformers=False))
    assert plan["device"] == "cpu"
    assert plan["use_esm"] is False and plan["esm_model"] == "none"
    assert plan["install_torch"] == sa.TORCH_COMMANDS["cpu"]  # backend unknown → 默认 wheel
    # 三命令显式存在；backend 为 cuda/rocm 时选择对应 wheel
    assert set(sa.TORCH_COMMANDS.values()) == {
        "pip install torch --index-url https://download.pytorch.org/whl/cu126",
        "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2",
        "pip install torch"}
    plan_cuda = sa.build_plan(probe, _env(torch=False, cuda=False, backend="cuda"))
    assert plan_cuda["install_torch"] == sa.TORCH_COMMANDS["cuda"]


def test_plan_mutation_string_skips_esm(tmp_path):
    # 分支 4：mutation_string 数据 → 无法重构全长序列，ESM 跳过
    rng = np.random.RandomState(3)
    wt, others = "VDGV", "ACDEFHIKLMNPQRSTWY"
    variants = ["".join(wt[p] if rng.rand() < 0.3 else rng.choice(list(others)) for p in range(4))
                for _ in range(400)]
    csv = tmp_path / "mstring.csv"
    pd.DataFrame({"variant": variants, "fitness": rng.rand(400)}).to_csv(csv, index=False)
    plan = sa.build_plan(sa.probe_data(str(csv)), _env())
    assert plan["variant_format"] == "mutation_string"
    assert plan["use_esm"] is False and plan["esm_model"] == "none"
    assert "全长" in plan["esm_reason"]


# ---------- configure_with_llm：降级 / 守卫 / 合法采纳 ----------

class ScriptedCompletions:
    """进程内 chat.completions 替身：弹出脚本化回复并记录每次请求。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def _content_response(payload):
    message = SimpleNamespace(role="assistant", content=payload, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _mock_client(script):
    return SimpleNamespace(chat=SimpleNamespace(completions=ScriptedCompletions(script)))


def test_configure_without_credentials(tmp_path):
    # ① 无凭据（environ 空，不触 os.environ）→ 降级返回规则计划
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={})
    assert result["llm_used"] is False
    assert "规则计划" in result["note"]
    assert result["chosen"] == sa._menu_defaults(result["plan"])
    assert result["disagreements"] == []
    assert result["warnings"] == []          # 无 LLM 路径也带空 warnings 字段


def test_configure_llm_guard_overwrites(tmp_path, monkeypatch):
    # ② mock 客户端：请求须含 probe 字段；无 cuda 却选 cuda → 守卫覆写并记录
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False))
    reply = {"chosen": {"device": "cuda", "esm_model": "none", "use_esm": False,
                        "install_torch": sa.TORCH_COMMANDS["cpu"], "checkpoint_action": "skip"},
             "rationale": "mock 理由", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="mock-model")

    assert result["llm_used"] is True
    # 请求 prompt 包含两个 probe 的原始字段
    request_text = json.dumps(client.chat.completions.requests[0]["messages"], ensure_ascii=False)
    for field in ("digits_ratio", "length_unique", "vram_gb", "checkpoint_ready", "device_name"):
        assert field in request_text
    # 恰好一次越界：device=cuda 与硬件事实矛盾 → 覆写为计划值 cpu
    assert len(result["disagreements"]) == 1
    d = result["disagreements"][0]
    assert d["field"] == "device" and d["llm"] == "cuda" and d["plan"] == "cpu"
    assert result["chosen"]["device"] == "cpu"
    # 其余合法字段原样采纳
    assert result["chosen"]["install_torch"] == sa.TORCH_COMMANDS["cpu"]
    assert result["rationale"] == "mock 理由"


def test_configure_llm_legal_reply_adopted(tmp_path, monkeypatch):
    # ③ 合法回复（与计划菜单默认一致）→ 原样采纳，无纠正
    env_stub = _env(vram=24.0)
    monkeypatch.setattr(sa, "probe_environment", lambda: env_stub)
    csv = _full_csv(tmp_path)
    expected = sa._menu_defaults(sa.build_plan(sa.probe_data(csv), env_stub))
    reply = {"chosen": expected, "rationale": "合法选择", "commands": ["echo hi"]}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=csv, client=client, model="mock-model")

    assert result["disagreements"] == []
    assert result["chosen"] == expected
    assert result["rationale"] == "合法选择"


def test_llm_prompt_contains_knowledge(tmp_path, monkeypatch):
    # 知识块注入 system prompt：首行 + 全文（原文未删改）+ "以守卫为准"句；
    # 且知识不成为绕过守卫的理由——无 cuda 却选 cuda 仍被覆写
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False))
    reply = {"chosen": {"device": "cuda", "esm_model": "none", "use_esm": False,
                        "install_torch": sa.TORCH_COMMANDS["cpu"], "checkpoint_action": "skip"},
             "rationale": "mock 理由", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="mock-model")

    system = client.chat.completions.requests[0]["messages"][0]["content"]
    assert "【模型选型校准知识（外部文献核实版）】" in system   # 知识块首行
    assert sa.MODEL_SELECTION_KNOWLEDGE in system               # 知识块全文注入
    assert "以守卫为准" in system
    assert result["disagreements"] and result["chosen"]["device"] == "cpu"  # 守卫行为不变


def test_configure_llm_low_vram_650m_warns_not_overwrites(tmp_path, monkeypatch):
    # 先验降级为警告：低显存（6GB）选 650M → 保留选择、记 warnings、不记 disagreements
    # （650M 推理约需 4GB 显存，10GB 阈值只是推荐先验；推荐层 build_plan 仍推荐 8M）
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(vram=6.0))
    reply = {"chosen": {"device": "auto", "esm_model": "esm2_650m", "use_esm": True,
                        "install_torch": None, "checkpoint_action": "skip"},
             "rationale": "低显存也要 650M", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="mock-model")

    assert result["chosen"]["esm_model"] == "esm2_650m"                       # 保留 LLM 选择
    assert result["chosen"]["use_esm"] is True
    assert result["warnings"] and any("OOM" in w or "4GB" in w for w in result["warnings"])
    assert all(d["field"] != "esm_model" for d in result["disagreements"])    # 未被改写
    assert result["plan"]["esm_model"] == "esm2_8m"        # 推荐层不动：<10GB 仍推荐 8M
    assert "650M" in sa.render_report(result)              # 报告展示警告节


def test_configure_llm_low_vram_factual_guards_still_overwrite(tmp_path, monkeypatch):
    # 事实类守卫未被误软化：同桩低显存但无 cuda 时选 device=cuda → 仍覆写为 cpu，且无 warnings
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(vram=6.0, cuda=False))
    reply = {"chosen": {"device": "cuda", "esm_model": "none", "use_esm": False,
                        "install_torch": None, "checkpoint_action": "skip"},
             "rationale": "mock 理由", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="mock-model")

    assert result["chosen"]["device"] == "cpu"
    assert [d["field"] for d in result["disagreements"]] == ["device"]    # 恰好一条事实覆写
    assert result["warnings"] == []                                       # 无先验分歧


def test_configure_llm_bad_json_falls_back(tmp_path, monkeypatch):
    # LLM 返回非 JSON → 整体回退规则计划（不崩溃）
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    client = _mock_client([_content_response("我觉得 CPU 挺好的")])
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="m")
    assert result["chosen"] == sa._menu_defaults(result["plan"])
    assert result["disagreements"] and "JSON" in result["disagreements"][0]["reason"]


# ---------- render_report 与命令块 ----------

def test_render_report(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False))
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={})
    report = sa.render_report(result)
    assert "configure_protein(" in report          # 最终调用串
    assert "pip install torch" in report           # 命令块
    assert "守卫纠正记录" in report and "dry-run" in report
    assert "算法说明" in report                     # LLM 不选模型的声明


# ---------- auto_execute：真实 pip/下载全部桩掉 ----------

def test_auto_execute_stubbed_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={}, auto_execute=True)
    assert calls == ["pip install torch"]           # 只发 cpu 安装命令（backend unknown）
    assert result["executed"][0]["returncode"] == 0


def test_auto_execute_checkpoint_download(tmp_path, monkeypatch):
    # 低显存 GPU + checkpoint 未就绪：规则计划默认 download → 调 ensure_esm_checkpoint(8M)
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(vram=6.0, transformers=True))
    received = {}

    def fake_ensure(path=None, model_id="facebook/esm2_t33_650M_UR50D", download=True):
        received["model_id"] = model_id
        return "D:/fake/esm2_t6_8M_UR50D"

    import protevo.embedding_service as embedding_service
    monkeypatch.setattr(embedding_service, "ensure_esm_checkpoint", fake_ensure, raising=False)
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={}, auto_execute=True)
    assert received["model_id"] == "facebook/esm2_t6_8M_UR50D"
    assert any("ensure_esm_checkpoint" in entry["command"] for entry in result["executed"])


# ---------- CLI 子进程（stdout 落临时文件再断言——Bash 吞 stdout） ----------

def test_cli_blat_exit_zero(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    env["PYTHONUTF8"] = "1"
    env["USERPROFILE"] = str(tmp_path)  # 隔离 HOME：确保无 setup_state.json，不走短路
    out = tmp_path / "cli_out.txt"
    with out.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [sys.executable, "-m", "protevo.setup_assistant",
             str(EXT / "blat_ecolx_gonzalez_2019/dms.csv")],
            cwd=str(PROJECT), env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=300)
    assert proc.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "configure_protein(" in text
    assert "indel_notation" in text
    assert "参考序列建议" in text    # indel 库必须提供全长 WT


# ---------- 已配置短路（幂等）：指纹匹配与廉价复验 ----------

def _state_dict(fingerprint, plan=None, essentials=None):
    return {"fingerprint": fingerprint, "configured_at": "2026-11-15T00:00:00+00:00",
            "essentials_passed": essentials or ["python", "core_deps"],
            "plan_summary": plan or {"device": "cpu", "esm_model": "none", "use_esm": False}}


def _write_state_file(path, state):
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def test_is_configured_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(backend="cuda", vram=24.0))
    fp = {"backend": "cuda", "cuda": True, "device_name": "stub-gpu", "vram_gb": 24.0}
    state = _state_dict(fp)
    _write_state_file(sa.STATE_PATH, state)

    ok, loaded = sa.is_configured()          # 指纹逐项一致 → 真
    assert ok is True and loaded == state

    state["fingerprint"]["vram_gb"] = 24.6   # vram 容差 ±1GB 内 → 真
    _write_state_file(sa.STATE_PATH, state)
    assert sa.is_configured()[0] is True

    state["fingerprint"]["vram_gb"] = 22.0   # 越容差 → 假
    _write_state_file(sa.STATE_PATH, state)
    assert sa.is_configured()[0] is False

    state["fingerprint"] = dict(fp, backend="rocm")  # backend 变化（换硬件）→ 假
    _write_state_file(sa.STATE_PATH, state)
    assert sa.is_configured()[0] is False

    sa.STATE_PATH.unlink()                   # 无状态文件 → 假
    assert sa.is_configured() == (False, {})


def test_is_configured_esm_reverify(tmp_path, monkeypatch):
    # 状态含 esm → 快速复验：torch 可导入 + checkpoint 仍就绪（全部廉价检查）
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(backend="cuda", vram=24.0))
    _write_state_file(sa.STATE_PATH, _state_dict(
        {"backend": "cuda", "cuda": True, "device_name": "stub-gpu", "vram_gb": 24.0},
        plan={"device": "auto", "esm_model": "esm2_650m", "use_esm": True}))

    monkeypatch.setattr(sa, "_torch_available", lambda: True)
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: None)  # checkpoint 被移走 → 假
    assert sa.is_configured()[0] is False

    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: Path("D:/fake/ckpt"))
    assert sa.is_configured()[0] is True

    monkeypatch.setattr(sa, "_torch_available", lambda: False)              # torch 不可导入 → 假
    assert sa.is_configured()[0] is False


def test_cli_short_circuit(tmp_path):
    # 伪造 setup_state.json（USERPROFILE 指向临时 HOME）→ 默认命令短路：exit 0、含"已配置"、无 LLM 痕迹
    home = tmp_path / "home"
    (home / ".cache/protevo").mkdir(parents=True)
    # 指纹须与无 hardware_probe 环境的兜底探测一致（backend unknown / vram None）
    _write_state_file(home / ".cache/protevo/setup_state.json",
                      _state_dict({"backend": "unknown", "cuda": False,
                                   "device_name": "unknown", "vram_gb": None}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    env["PYTHONUTF8"] = "1"
    env["USERPROFILE"] = str(home)
    out = tmp_path / "cli_short.txt"
    with out.open("w", encoding="utf-8") as handle:
        proc = subprocess.run([sys.executable, "-m", "protevo.setup_assistant"],
                              cwd=str(PROJECT), env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=300)
    assert proc.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "已配置" in text
    assert "--doctor" in text and "--force" in text     # 一行提示
    assert "MissingCredentials" not in text             # 未尝试 LLM 调用
    assert "规则计划" not in text and "配置选择" not in text  # 未走配置流程


# ---------- doctor：体检、冒烟与状态闭环 ----------

def test_doctor_platform_mismatch_warn(tmp_path, monkeypatch):
    # CPU 版 torch + 探测到 NVIDIA GPU → warn，fix 为 CUDA 重装命令（复用 TORCH_COMMANDS）
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    monkeypatch.setattr(sa, "_torch_info", lambda: ("2.5.1+cpu", "cpu"))
    report = sa.run_doctor(full=False)
    match = next(c for c in report["checks"] if c["name"] == "platform_match")
    assert match["status"] == "warn"
    assert match["fix"] == sa.TORCH_COMMANDS["cuda"]
    assert "NVIDIA" in match["detail"]


def test_doctor_checkpoint_missing_fail(tmp_path, monkeypatch):
    # 期望 ESM 但 checkpoint 缺失 → fail + ensure_esm_checkpoint 修复命令；FAIL 不写状态
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: None)
    report = sa.run_doctor(esm_expected="esm2_650m", full=False)
    check = next(c for c in report["checks"] if c["name"] == "checkpoint")
    assert check["status"] == "fail"
    assert "ensure_esm_checkpoint" in check["fix"]
    assert report["summary"] == "FAIL" and report["state_written"] is False
    assert not (tmp_path / "setup_state.json").exists()


def test_doctor_smokes_and_state_written(tmp_path, monkeypatch):
    # full 冒烟：嵌入冒烟走桩 EmbeddingService.encode；预测器冒烟跑真实合成 CSV；
    # essentials 无 fail → setup_state.json 落盘并与短路闭环
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(vram=24.0))
    monkeypatch.setattr(sa, "_torch_info", lambda: ("2.5.1+cu126", "cuda"))
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: tmp_path / "ckpt")

    class FakeEmbeddingService:  # 桩掉 EmbeddingService（含 encode），抓 doctor 的调用方式
        calls = []

        def __init__(self, model_path=None, device=None, **kwargs):
            self.calls.append(("init", model_path, device))

        def encode(self, sequences):
            self.calls.append(("encode", list(sequences)))
            return np.zeros((len(sequences), 1280), dtype=np.float32)

    import protevo.embedding_service as embedding_service
    monkeypatch.setattr(embedding_service, "EmbeddingService", FakeEmbeddingService)

    report = sa.run_doctor(full=True)

    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["embedding_smoke"]["status"] == "ok"
    assert by_name["predictor_smoke"]["status"] == "ok"     # 真实 configure_protein 冒烟
    assert by_name["platform_match"]["status"] == "ok"      # cuda 构建 = cuda 硬件
    assert report["summary"] in ("PASS", "WARN")            # transformers/llm 缺失记 warn 不阻塞
    assert FakeEmbeddingService.calls[-1] == ("encode", ["MKVWV"])
    assert ("init", str(tmp_path / "ckpt"), "cuda") in FakeEmbeddingService.calls

    # 通过 → 状态落盘；指纹与 essentials 实证一致；未实证 esm 不写入 plan
    state_path = tmp_path / "setup_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["fingerprint"]["backend"] == "cuda" and state["fingerprint"]["vram_gb"] == 24.0
    assert "predictor_smoke" in state["essentials_passed"]
    assert state["plan_summary"]["esm_model"] == "none"
    assert sa.is_configured()[0] is True                    # 闭环：落盘后短路为真


def test_doctor_autofix_repairs_checkpoint(tmp_path, monkeypatch):
    # --doctor --execute 路径核心：checkpoint fail → ensure_esm_checkpoint 修复 → 复检通过并写状态
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    available = {"path": None}
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: available["path"])
    monkeypatch.setattr(sa, "_weight_bytes", lambda ckpt: 2 * 1024 ** 3)  # 权重体积达标（桩）

    def fake_ensure(path=None, model_id="facebook/esm2_t33_650M_UR50D", download=True):
        available["path"] = tmp_path / "ckpt"
        return str(available["path"])

    import protevo.embedding_service as embedding_service
    monkeypatch.setattr(embedding_service, "ensure_esm_checkpoint", fake_ensure, raising=False)

    report = sa.run_doctor(esm_expected="esm2_650m", full=False)
    assert report["summary"] == "FAIL"

    fixed = sa._doctor_autofix(report, esm_expected="esm2_650m")
    by_name = {c["name"]: c for c in fixed["checks"]}
    assert by_name["checkpoint"]["status"] == "ok"
    assert fixed["state_written"] is True
    assert (tmp_path / "setup_state.json").exists()


def test_render_doctor(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False))
    report = sa.run_doctor(full=True)
    text = sa.render_doctor(report)
    assert "protevo 环境体检报告" in text
    assert "torch" in text and "pip install torch" in text   # warn 项附 fix 命令
    assert "总结" in text


# ---------- bootstrap 缺口：in_venv/core_deps_ok 探测、bootstrap 块与 install_core 菜单 ----------

def test_probe_environment_venv_and_core_keys():
    # probe_environment 增两键；本 venv（pytest 运行环境）内两者均为真
    env = sa.probe_environment()
    assert env["in_venv"] is True                    # sys.prefix != sys.base_prefix
    assert env["core_deps_ok"] is True               # pandas/sklearn/scipy 均可导入


def test_plan_bootstrap_block(tmp_path):
    # core_deps_ok=False → install_core=requirements，bootstrap 三步块按 OS 两式生成
    probe = sa.probe_data(_full_csv(tmp_path))
    plan = sa.build_plan(probe, _env(core_deps_ok=False, os_name="Windows-11-stub"))
    assert plan["install_core"] == "requirements"
    assert plan["bootstrap_commands"] == ["python -m venv .venv", r".venv\Scripts\activate",
                                          "pip install -r requirements.txt"]
    assert "requirements.txt 在包根" in plan["requirements_note"]

    plan_posix = sa.build_plan(probe, _env(core_deps_ok=False, os_name="Linux-5.15"))
    assert plan_posix["bootstrap_commands"] == ["python -m venv .venv", "source .venv/bin/activate",
                                                "pip install -r requirements.txt"]
    # 已装全 → 无 bootstrap
    assert sa.build_plan(probe, _env())["install_core"] == "skip"
    assert sa.build_plan(probe, _env())["bootstrap_commands"] == []


def test_guard_install_core_overwrite_when_deps_present(tmp_path, monkeypatch):
    # 核心依赖已全（桩）+ LLM 选 requirements → 覆写 skip 并记 disagreements（理由：核心依赖已就绪）
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    csv = _full_csv(tmp_path)
    expected = sa._menu_defaults(sa.build_plan(sa.probe_data(csv), _env()))
    reply = {"chosen": {**expected, "install_core": "requirements"},
             "rationale": "mock 理由", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=csv, client=client, model="mock-model")

    assert result["chosen"]["install_core"] == "skip"
    d = next(d for d in result["disagreements"] if d["field"] == "install_core")
    assert d["llm"] == "requirements" and "核心依赖已就绪" in d["reason"]
    assert result["commands"] == []                      # 无 bootstrap 命令进入


def test_guard_install_core_adopted_when_deps_missing(tmp_path, monkeypatch):
    # 核心依赖缺失（桩）+ LLM 选 requirements → 采纳，bootstrap 三步块进入 commands 首位
    env_stub = _env(core_deps_ok=False)
    monkeypatch.setattr(sa, "probe_environment", lambda: env_stub)
    csv = _full_csv(tmp_path)
    expected = sa._menu_defaults(sa.build_plan(sa.probe_data(csv), env_stub))
    assert expected["install_core"] == "requirements"
    reply = {"chosen": expected, "rationale": "mock 理由", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])

    result = sa.configure_with_llm(data_csv=csv, client=client, model="mock-model")

    assert result["disagreements"] == []
    assert result["chosen"]["install_core"] == "requirements"
    assert result["commands"][:3] == result["plan"]["bootstrap_commands"]
    assert result["commands"][0] == "python -m venv .venv"
    # 降级（无凭据）路径同样带 bootstrap：chosen 默认即 requirements
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(core_deps_ok=False))
    degraded = sa.configure_with_llm(data_csv=csv, environ={})
    assert degraded["chosen"]["install_core"] == "requirements"
    assert degraded["commands"][:3] == degraded["plan"]["bootstrap_commands"]


def test_doctor_core_deps_fix_requirements(tmp_path, monkeypatch):
    # 打桩 _module_available 缺 sklearn → core_deps fail，fix 为 requirements.txt（包根），
    # 逐包安装降为 detail 里的备选
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "_module_available", lambda name: name != "sklearn")
    report = sa.run_doctor(full=False)
    check = next(c for c in report["checks"] if c["name"] == "core_deps")
    assert check["status"] == "fail"
    assert "requirements.txt" in check["fix"] and "包根" in check["fix"]
    assert "备选" in check["detail"] and "scikit-learn" in check["detail"]


def test_doctor_venv_check(tmp_path, monkeypatch):
    # 第 10 项 venv：不在虚拟环境（sys.prefix 打桩）→ warn + 两式创建命令；在 → ok
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())

    monkeypatch.setattr(sa.sys, "base_prefix", "/usr/local")
    monkeypatch.setattr(sa.sys, "prefix", "/usr/local")     # prefix == base_prefix → 不在 venv
    report = sa.run_doctor(full=False)
    check = next(c for c in report["checks"] if c["name"] == "venv")
    assert check["status"] == "warn"
    assert ".venv" in check["fix"] and "Scripts" in check["fix"] and "source" in check["fix"]  # 两式

    monkeypatch.setattr(sa.sys, "prefix", "/some/venv")     # prefix != base_prefix → 在 venv
    report = sa.run_doctor(full=False)
    check = next(c for c in report["checks"] if c["name"] == "venv")
    assert check["status"] == "ok"


def test_doctor_autofix_core_deps(tmp_path, monkeypatch):
    # --doctor --execute：core_deps fail → 自动 pip install -r requirements.txt（子进程桩）
    # → 复检该项 ok 并写状态
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    available = {"sklearn": False}
    monkeypatch.setattr(sa, "_module_available",
                        lambda name: name != "sklearn" or available["sklearn"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs.get("cwd")))
        available["sklearn"] = True                     # pip 安装后复检可导入
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sa.subprocess, "run", fake_run)

    report = sa.run_doctor(full=False)
    assert report["summary"] == "FAIL"

    fixed = sa._doctor_autofix(report)
    assert calls and calls[0][0] == "pip install -r requirements.txt"
    assert calls[0][1] == str(PROJECT)                  # 在包根执行（requirements.txt 所在）
    by_name = {c["name"]: c for c in fixed["checks"]}
    assert by_name["core_deps"]["status"] == "ok"       # 复检通过
    assert fixed["state_written"] is True
    assert (tmp_path / "setup_state.json").exists()
