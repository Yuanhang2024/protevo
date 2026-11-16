# tests/test_dsh_note012_fixes.py — DSH 组合压力测试（protevo-combination-stress-001，
# 证据：组合压力测试报告.md + gpu-cache/facts-8m-append.json + environment-cases/observations.json）
# 发现的 6 项问题修复的回归测试：
#   1【严重】8M 编码维度不匹配：embed_dim 取自模型 config.hidden_size（无 torch 档 mock 320 维；
#             GPU 档有本机 8M checkpoint 时真实编码 1 条 → (1, 320)）
#   2【严重】doctor 的 8M checkpoint 阈值：实测 31,384,292 bytes（< 30MiB）判 ok；
#             零字节权重仍判 fail（保留半下载/损坏检测）
#   3【严重】doctor 误读 CUDA 后端：backend="CUDA 12.8" 前缀匹配 → platform_match ok 不建议 CPU
#   4【功能缺口】通用自动续跑：诚实处理——docstring 与 README 声明不支持（断言文案存在）
#   5【健壮性】chosen 列表 / fingerprint 列表 / .cache 被文件占用 / 空 CSV → 可操作结果不崩溃
#   6【一致性】安装命令绑定 sys.executable；torch 已可用时不生成安装命令
import contextlib
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from protevo import setup_assistant as sa
from protevo.embedding_service import EmbeddingService

PROJECT = Path(__file__).parent.parent

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
# GPU 档 8M 真实 checkpoint：本机 D 盘不存在时回退压力测试工作区的镜像副本
MODEL_8M_CANDIDATES = (
    Path("D:/ChimeraModel_Lite/esm2_t6_8M_UR50D"),
    Path(r"C:\Users\lijin\Desktop\ZJU张强老师申请\笔试试题\test-workspace"
         r"\protevo-combination-stress-001\gpu-cache\mirror-8m"),
)
MODEL_8M = next((p for p in MODEL_8M_CANDIDATES if (p / "config.json").is_file()), None)
requires_8m = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS and MODEL_8M is not None),
    reason="GPU 档需要 torch+transformers 与本机 8M checkpoint（openfold 环境）",
)


def _env(torch=True, cuda=True, vram=24.0, transformers=True, backend=None,
          core_deps_ok=True, os_name="Windows-11-stub"):
    """构造 probe_environment 返回形状的环境桩（backend 可传完整后端串如 'CUDA 12.8'）。"""
    return {"torch": torch, "cuda": cuda, "backend": backend or ("cuda" if cuda else "unknown"),
            "device_name": "stub-gpu" if cuda else "unknown", "vram_gb": vram,
            "transformers": transformers, "checkpoint_ready": False, "checkpoint_path": None,
            "os": os_name, "in_venv": True, "core_deps_ok": core_deps_ok}


def _full_csv(tmp_path, seed=1):
    """合成全长序列库：56 aa、两位点低频突变（与 test_setup_assistant 同口径）。"""
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


class _ScriptedCompletions:
    """进程内 chat.completions 替身：弹出脚本化回复。"""

    def __init__(self, script):
        self.script = list(script)

    def create(self, **kwargs):
        return self.script.pop(0)


def _content_response(payload):
    message = SimpleNamespace(role="assistant", content=payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _mock_client(script):
    return SimpleNamespace(chat=SimpleNamespace(completions=_ScriptedCompletions(script)))


# ---------- 修复 1：8M 编码维度 ----------

class _FakeEsmModel:
    """桩模型：仅提供 config.hidden_size / num_hidden_layers 与 eval().to() 链。"""

    def __init__(self, hidden_size, num_hidden_layers):
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)

    def eval(self):
        return self

    def to(self, device):
        return self


def _install_fake_torch_stack(monkeypatch, hidden_size=320, num_hidden_layers=6):
    """向 sys.modules 注入假 torch/transformers（含 __spec__，find_spec 探测不炸）。

    EmbeddingService.__init__ 惰性导入两者；假 torch 无 CUDA → device 用 'cpu'，
    前向由测试打桩（_forward），tokenizer/model 不真正参与计算。
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.version = SimpleNamespace(cuda="12.8", hip=None)
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        get_device_properties=lambda i: SimpleNamespace(name="stub", total_memory=0))

    @contextlib.contextmanager
    def _no_grad():
        yield

    fake_torch.no_grad = _no_grad
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = SimpleNamespace(
        from_pretrained=lambda path: _FakeEsmModel(hidden_size, num_hidden_layers))
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path: object())
    fake_transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", None)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_embed_dim_follows_hidden_size_no_torch(tmp_path, monkeypatch):
    """无 torch 档：mock hidden_size=320 → encode 输出 (n, 320)，磁盘缓存按实际维度落盘/加载。"""
    _install_fake_torch_stack(monkeypatch, hidden_size=320, num_hidden_layers=6)
    cache = tmp_path / "esm_cache_8m"
    svc = EmbeddingService("D:/fake/esm2_t6_8M_UR50D", device="cpu", cache_dir=str(cache))
    assert svc.embed_dim == 320

    def fake_forward(seqs):
        out = {}
        for s in seqs:
            vec = np.full(320, 0.25, dtype=np.float32)
            svc._remember(s, vec)  # 与真实 _forward 同口径登记缓存（置脏 → 落盘）
            out[s] = vec
        return out

    monkeypatch.setattr(svc, "_forward", fake_forward)
    out = svc.encode(["ACDEFG", "ACDEFG"])  # 含重复序列：缓存去重路径
    assert out.shape == (2, 320) and out.dtype == np.float32

    # 磁盘缓存按实际维度（320）存储，档案 dim 与之一致
    vectors = np.load(cache / "embeddings.npz")["embeddings"]
    assert vectors.shape[1] == 320
    assert json.loads((cache / "model.json").read_text(encoding="utf-8"))["dim"] == 320

    # 新构造服务从盘加载（同样 320 维），编码逐位一致且不再触发前向
    reloaded = EmbeddingService("D:/fake/esm2_t6_8M_UR50D", device="cpu", cache_dir=str(cache))
    monkeypatch.setattr(reloaded, "_forward",
                        lambda seqs: pytest.fail("磁盘缓存命中不应触发前向"))
    assert np.array_equal(reloaded.encode(["ACDEFG", "ACDEFG"]), out)


def test_embed_dim_650m_anchor_unchanged(tmp_path, monkeypatch):
    """650M 文档锚点不变：mock hidden_size=1280 → (n, 1280)（回归钉子，行为零变化）。"""
    _install_fake_torch_stack(monkeypatch, hidden_size=1280, num_hidden_layers=33)
    svc = EmbeddingService("D:/fake/esm2_t33_650M_UR50D", device="cpu")
    assert svc.embed_dim == 1280
    monkeypatch.setattr(svc, "_forward",
                        lambda seqs: {s: np.full(1280, 0.5, dtype=np.float32) for s in seqs})
    assert svc.encode(["ACDE"]).shape == (1, 1280)


@requires_8m
def test_encode_real_8m_one_sequence_shape_320(tmp_path):
    """GPU 档：真实 8M checkpoint 编码 1 条序列 → (1, 320) 无 ValueError。"""
    svc = EmbeddingService(str(MODEL_8M), device="cuda" if HAS_TORCH and _cuda_ok() else "cpu",
                           cache_dir=str(tmp_path / "cache8m"))
    assert svc.embed_dim == 320
    emb = svc.encode(["MKVWV"])
    assert emb.shape == (1, 320) and emb.dtype == np.float32 and np.isfinite(emb).all()


def _cuda_ok():
    import torch
    return torch.cuda.is_available()


# ---------- 修复 2：doctor 的 8M checkpoint 阈值 ----------

def test_doctor_8m_real_weight_size_ok(tmp_path, monkeypatch):
    """镜像 8M 权重实测 31,384,292 bytes（差 30MiB 阈值 72,988）不得再被误拒。"""
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: tmp_path / "ckpt8m")
    monkeypatch.setattr(sa, "_weight_bytes", lambda ckpt: 31_384_292)
    report = sa.run_doctor(esm_expected="esm2_8m", full=False)
    check = next(c for c in report["checks"] if c["name"] == "checkpoint")
    assert check["status"] == "ok"


def test_doctor_zero_byte_weights_still_fail(tmp_path, monkeypatch):
    """阈值放宽不丢半下载/损坏检测：零字节权重仍 fail（observations.json 的 checkpoint-ready 场景）。"""
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    monkeypatch.setattr(sa, "_find_esm_checkpoint", lambda *a, **k: tmp_path / "ckpt650m")
    monkeypatch.setattr(sa, "_weight_bytes", lambda ckpt: 0)
    report = sa.run_doctor(esm_expected="esm2_650m", full=False)
    check = next(c for c in report["checks"] if c["name"] == "checkpoint")
    assert check["status"] == "fail" and report["summary"] == "FAIL"


# ---------- 修复 3：platform_match 前缀匹配完整后端串 ----------

def test_doctor_platform_match_cuda_12_8_ok(tmp_path, monkeypatch):
    """backend='CUDA 12.8'（hardware_probe 的完整后端串）→ ok，不再误判无 GPU / 建议 CPU torch。"""
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(backend="CUDA 12.8"))
    monkeypatch.setattr(sa, "_torch_info", lambda: ("2.11.0", "CUDA 12.8", True))
    report = sa.run_doctor(full=False)
    match = next(c for c in report["checks"] if c["name"] == "platform_match")
    assert match["status"] == "ok"
    assert "无 GPU" not in match["detail"]
    assert not (match.get("fix") or "").startswith(sa.TORCH_COMMANDS["cpu"])  # 不建议 CPU torch
    # torch 项展示同步正确：CUDA 构建不再被误标为 CPU 版
    torch_check = next(c for c in report["checks"] if c["name"] == "torch")
    assert "CUDA" in torch_check["detail"]


def test_doctor_platform_match_rocm_full_string_ok(tmp_path, monkeypatch):
    """ROCm 同理：backend='ROCm 6.2' + rocm 构建 → ok。"""
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(backend="ROCm 6.2"))
    monkeypatch.setattr(sa, "_torch_info", lambda: ("2.4.0", "rocm"))
    report = sa.run_doctor(full=False)
    match = next(c for c in report["checks"] if c["name"] == "platform_match")
    assert match["status"] == "ok"


# ---------- 修复 4：自动续跑的诚实声明 ----------

def test_evolve_docstring_and_readme_declare_no_resume():
    """不实现续跑（新功能非 bug 修复），但 docstring 与 README 必须明确声明。"""
    from protevo import evolve
    assert "不支持自动续跑" in (evolve.run_evolution.__doc__ or "")
    assert "status=interrupted" in (evolve.run_evolution.__doc__ or "")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert "不支持自动续跑" in readme and "interrupted" in readme
    readme_en = (PROJECT / "README_EN.md").read_text(encoding="utf-8")
    assert "not supported" in readme_en and "interrupted" in readme_en


# ---------- 修复 5：异常结构容错 ----------

def test_guard_choices_non_dict_falls_back_to_rule_plan(tmp_path, monkeypatch):
    """chosen 为列表（observations.json 的 reply-shape 场景）→ 回退规则计划，不抛 AttributeError。"""
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    reply = {"chosen": [1, 2], "rationale": "坏结构回复", "commands": []}
    client = _mock_client([_content_response(json.dumps(reply))])
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), client=client, model="mock-model")
    assert result["chosen"] == sa._menu_defaults(result["plan"])
    assert any(d.get("field") == "chosen" for d in result["disagreements"])
    assert result["rationale"] == "坏结构回复"  # 整体回退但保留原文供人工核对


def test_is_configured_fingerprint_non_dict_unconfigured(tmp_path, monkeypatch):
    """状态文件 fingerprint 为列表（cache-shape 场景）→ 视为不匹配（未配置），不崩溃。"""
    monkeypatch.setattr(sa, "STATE_PATH", tmp_path / "setup_state.json")
    (tmp_path / "setup_state.json").write_text(
        json.dumps({"fingerprint": [1], "configured_at": "2026-11-16T00:00:00+00:00",
                    "plan_summary": {"device": "cpu", "esm_model": "none", "use_esm": False}}),
        encoding="utf-8")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    assert sa.is_configured() == (False, {})


def test_write_state_cache_path_occupied_by_regular_file(tmp_path, monkeypatch):
    """.cache 路径被普通文件占用（state-write 场景）→ 可操作的 RuntimeError 而非 FileExistsError。"""
    blocker = tmp_path / "state_dir"       # 将作为 STATE_PATH.parent 的路径先占为普通文件
    blocker.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(sa, "STATE_PATH", blocker / "setup_state.json")
    monkeypatch.setattr(sa, "probe_environment", lambda: _env())
    with pytest.raises(RuntimeError, match="普通文件占用"):
        sa._write_state()


def test_probe_data_empty_csv_clear_valueerror(tmp_path):
    """空 CSV（仅表头）probe_data：ValueError 含'无有效数据'（原报 cannot convert float NaN to integer）。"""
    csv = tmp_path / "empty.csv"
    pd.DataFrame({"variant": pd.Series([], dtype=object),
                  "fitness": pd.Series([], dtype=float)}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="无有效数据"):
        sa.probe_data(str(csv))


def test_configure_protein_empty_csv_clear_valueerror(tmp_path):
    """空 CSV（仅表头）configure_protein：ValueError 含'无有效数据'（原报 KeyError pop from an empty set）。"""
    csv = tmp_path / "empty.csv"
    pd.DataFrame({"variant": pd.Series([], dtype=object),
                  "fitness": pd.Series([], dtype=float)}).to_csv(csv, index=False)
    from protevo.protein_config import configure_protein
    with pytest.raises(ValueError, match="无有效数据"):
        configure_protein(str(csv))


def test_configure_protein_nan_fitness_rejected(tmp_path):
    """fitness 含 NaN：显式 ValueError 含'无有效数据'（原静默训练出 val ρ=-999 垃圾结果）。"""
    csv = tmp_path / "nan.csv"
    pd.DataFrame({"variant": ["ACDE", "ACDK", "ACDQ"], "fitness": [1.0, float("nan"), 0.5]}
                 ).to_csv(csv, index=False)
    from protevo.protein_config import configure_protein
    with pytest.raises(ValueError, match="无有效数据"):
        configure_protein(str(csv))


# ---------- 修复 6：安装命令与说明一致 + 绑定解释器 ----------

def test_generated_install_commands_bind_interpreter(tmp_path, monkeypatch):
    """无 torch + 核心依赖缺失：bootstrap 与 torch 安装命令均绑定 sys.executable（非裸 pip）。"""
    monkeypatch.setattr(sa, "probe_environment",
                        lambda: _env(torch=False, cuda=False, vram=None, transformers=False,
                                     core_deps_ok=False))
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={})
    assert result["commands"]
    for command in result["commands"]:
        if "pip" in command:
            assert command.startswith(f'"{sys.executable}" -m pip'), command
    assert any("pip install -r requirements.txt" in c and sys.executable in c
               for c in result["commands"])                      # bootstrap 安装步绑定
    assert any("pip install torch" in c and sys.executable in c
               for c in result["commands"])                      # torch 安装命令绑定


def test_no_torch_install_command_when_torch_available(tmp_path, monkeypatch):
    """torch 已可用（chosen.install_torch=None）：commands 不得含任何 torch 安装命令——
    与说明文字'无需安装 torch'保持一致（真实 Luna 建议的自相矛盾案例）。"""
    monkeypatch.setattr(sa, "probe_environment", lambda: _env(torch=True, cuda=True))
    result = sa.configure_with_llm(data_csv=_full_csv(tmp_path), environ={})
    assert result["chosen"]["install_torch"] is None
    assert not any("install torch" in c for c in result["commands"])
