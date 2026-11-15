"""按需 ESM-2 嵌入服务测试：两档。

第一档（无 torch，本项目 .venv 即可跑）：
  - 模块级 import 不加载 torch（惰性导入契约）
  - 无 torch 环境构造 EmbeddingService → 含 "torch" 字样的明确 ImportError
  - 长度护栏：> 1022 残基抛 ValueError，恰好 1022 合法
第二档（GPU 冒烟，无 torch/transformers 自动跳过；openfold 环境全跑）：
  - 3 条短序列（20-56 aa）：shape (3, 1280)、float32、无 NaN
  - 同序列二次 encode 走内存缓存且逐位一致
  - 与 embed_gb1.py 核心口径（BOS 偏移 +1、末层 mean-pool）余弦相似度 > 0.999
  - cache_dir 落盘（npz + json）并在新构造时加载
  - esm_layer 层选择（中层 vs 默认末层余弦 < 0.999 证明生效；越界 ValueError）与 batch_size 小批量

运行：
  .venv/Scripts/python.exe -m pytest tests/test_embedding_service.py -q
  D:/Anaconda3/envs/openfold/python.exe -m pytest tests/test_embedding_service.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from protevo.embedding_service import (
    EMBED_DIM,
    MAX_SEQUENCE_LENGTH,
    EmbeddingService,
    validate_sequence_lengths,
)

MODEL_PATH = "D:/ChimeraModel_Lite/esm2_t33_650M_UR50D"
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
requires_model = pytest.mark.skipif(
    not (HAS_TORCH and HAS_TRANSFORMERS),
    reason="GPU 冒烟档需要 torch+transformers（D:/Anaconda3/envs/openfold）",
)

# GB1 WT（56 aa，取自 embed_gb1.py）+ 两条短序列；长度互不相同以覆盖批量 padding 路径
SEQUENCES = [
    "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "ACDEFGHIKLMNPQRSTVWY",
    "GKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
]


# ---------- 第一档：无 torch 可跑 ----------

def test_module_import_is_lazy():
    """模块级导入本文件不得把 torch 带进 sys.modules（无 torch 环境不崩溃）。"""
    before = "torch" in sys.modules
    import protevo.embedding_service
    assert hasattr(protevo.embedding_service, "EmbeddingService")
    assert ("torch" in sys.modules) == before  # import 前后 torch 状态不变


@pytest.mark.skipif(HAS_TORCH and HAS_TRANSFORMERS, reason="torch+transformers 齐备时构造不会走到导入失败")
def test_constructor_raises_importerror_without_torch():
    """无 torch 环境：构造抛含 'torch' 字样的明确 ImportError。"""
    with pytest.raises(ImportError, match="torch"):
        EmbeddingService("D:/nonexistent-model")


def test_length_guard():
    """长度护栏：恰好 1022 残基合法；> 1022 抛 ValueError（消息含上限说明）。"""
    validate_sequence_lengths(["ACDE", "A" * MAX_SEQUENCE_LENGTH])  # 边界值合法
    with pytest.raises(ValueError, match="1022"):
        validate_sequence_lengths(["ACDE", "A" * (MAX_SEQUENCE_LENGTH + 1)])
    with pytest.raises(ValueError, match="1022"):
        validate_sequence_lengths(["W" * 2000])


# ---------- 第二档：GPU 冒烟 ----------

@pytest.fixture(scope="module")
def service(tmp_path_factory):
    import torch
    # CUDA 不可用时降级 cpu 验证功能（RTX 5090 正常走 cuda）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = str(tmp_path_factory.mktemp("esm_cache"))
    return EmbeddingService(MODEL_PATH, device=device, cache_dir=cache_dir)


@requires_model
def test_encode_shape_dtype_finite(service):
    emb = service.encode(SEQUENCES)
    assert emb.shape == (3, EMBED_DIM)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()  # 无 NaN/Inf


@requires_model
def test_encode_cache_hit_consistent(service):
    first = service.encode(SEQUENCES)
    assert all(s in service._cache for s in SEQUENCES)  # 已进内存缓存
    second = service.encode(SEQUENCES)                  # 全部命中缓存，不触发推理
    assert np.array_equal(first, second)                # 逐位一致


@requires_model
def test_consistency_with_embed_gb1(service):
    """与 embed_gb1.py 核心口径一致性：复制其提取核心几行，余弦相似度 > 0.999。"""
    seq = SEQUENCES[0]
    ours = service.encode([seq])[0]
    # embed_gb1.py Esm2Embedder.extract 核心逻辑：tokenizer 加 BOS/EOS →
    # output_hidden_states=True → 末层 hidden（t33 即 hidden_states[33]）→ 残基 i 取行 i+1
    encoded = service.tokenizer([seq], return_tensors="pt").to(service.device)
    with service.torch.no_grad():
        outputs = service.model(**encoded, output_hidden_states=True)
    reference = outputs.hidden_states[-1][0, 1:len(seq) + 1, :].float().cpu().numpy().mean(axis=0)
    cosine = float(np.dot(ours, reference) / (np.linalg.norm(ours) * np.linalg.norm(reference)))
    print(f"cosine={cosine:.6f} max_abs_diff={np.abs(ours - reference).max():.3e}")
    assert cosine > 0.999


@requires_model
def test_disk_cache_roundtrip(service):
    """cache_dir 落盘（npz + json 索引）且新构造的服务从盘加载、编码结果一致。"""
    cache = Path(service.cache_dir)
    assert (cache / "embeddings.npz").exists()
    assert (cache / "index.json").exists()

    reloaded = EmbeddingService(MODEL_PATH, device=service.device, cache_dir=service.cache_dir)
    assert all(s in reloaded._disk for s in SEQUENCES)  # 构造时已从盘加载，无需重新推理
    assert np.array_equal(reloaded.encode(SEQUENCES), service.encode(SEQUENCES))


# ---------- 硬件自适应 + checkpoint 获取（新增能力） ----------

from protevo.embedding_service import (  # noqa: E402  追加段自带导入，保持对现有内容的零改动
    ensure_esm_checkpoint,
    hardware_probe,
    resolve_device,
)

# 设计说明：resolve_device 采用"无 torch 时抛含 'torch' 的 ImportError"方案
# （另一种设计"保证不被调用"不可测，故舍去），与 EmbeddingService 构造失败契约一致。


@pytest.mark.skipif(HAS_TORCH, reason="仅在无 torch 环境验证降级返回（本项目 .venv）")
def test_hardware_probe_without_torch():
    """无 torch：hardware_probe 不抛错，返回全量降级键值。"""
    assert hardware_probe() == {
        "torch": False, "cuda": False, "backend": None,
        "device_name": None, "vram_gb": None, "transformers": False,
    }


@pytest.mark.skipif(HAS_TORCH, reason="仅在无 torch 环境验证降级报错（本项目 .venv）")
def test_resolve_device_without_torch_raises_importerror():
    """无 torch：resolve_device 抛含 'torch' 的 ImportError（设计为可直接调用）。"""
    with pytest.raises(ImportError, match="torch"):
        resolve_device("auto")


def test_configure_protein_esm_device_default_is_auto():
    """回归钉子：configure_protein 的 esm_device 默认值必须为 'auto'（透传给 EmbeddingService 解析）。"""
    import inspect

    from protevo import protein_config
    assert inspect.signature(protein_config.configure_protein).parameters["esm_device"].default == "auto"


@requires_model
def test_resolve_device_auto_prefers_cuda():
    """auto 解析：CUDA 可用（openfold + RTX 5090）→ 'cuda'；显式 'cpu' 原样返回。"""
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cpu") == "cpu"


@requires_model
def test_hardware_probe_with_cuda():
    """有 torch + CUDA：backend 形如 'CUDA x'，设备名/显存被探测，transformers 就位。"""
    probe = hardware_probe()
    assert probe["torch"] is True
    assert probe["cuda"] is True
    assert probe["backend"].startswith("CUDA")
    assert probe["device_name"]  # RTX 5090 应有名有姓
    assert probe["vram_gb"] and probe["vram_gb"] > 8
    assert probe["transformers"] is True


@requires_model
def test_service_device_auto_backend_cuda():
    """device='auto'：实例 device 解析为 'cuda'，backend 以 'CUDA' 开头。"""
    svc = EmbeddingService(MODEL_PATH, device="auto")
    assert svc.device == "cuda"
    assert svc.backend.startswith("CUDA")


@requires_model
def test_service_explicit_cpu_works():
    """显式 device='cpu'：可构造、可编码（慢但可用）。"""
    svc = EmbeddingService(MODEL_PATH, device="cpu")
    assert svc.device == "cpu"
    # 注：svc.backend 报告的是硬件探针的 torch 构建后端（本机为 "CUDA x"），与实例
    # 选用的 device 无关，故此处不对其取值做断言
    emb = svc.encode(["ACDEFGHIKLMNPQRSTVWY"])
    assert emb.shape == (1, EMBED_DIM)
    assert np.isfinite(emb).all()


@requires_model
def test_ensure_esm_checkpoint_reuses_local_path():
    """本机已有 checkpoint（含 config.json）：原样返回同一字符串，不触发下载。"""
    assert ensure_esm_checkpoint(MODEL_PATH) == MODEL_PATH


@requires_model
def test_ensure_esm_checkpoint_download_failure_message(tmp_path, monkeypatch):
    """下载失败分支：RuntimeError 消息含手动命令两式（直连 huggingface-cli + hf-mirror 镜像前缀）。"""
    import huggingface_hub

    def _boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    target = tmp_path / "esm2_t33_650M"
    with pytest.raises(RuntimeError) as excinfo:
        ensure_esm_checkpoint(str(target))
    msg = str(excinfo.value)
    assert "huggingface-cli download facebook/esm2_t33_650M_UR50D" in msg
    assert f"--local-dir {target}" in msg
    assert "HF_ENDPOINT=https://hf-mirror.com" in msg

def test_disk_cache_rejects_model_swap(tmp_path):
    # 缓存目录绑定单一 ESM 模型：meta 记录 650M，请求 8M → 在 torch 导入前即被守卫拒绝（两档环境都跑）
    import json as _json
    cache = tmp_path / "esm_cache"
    cache.mkdir()
    (cache / "model.json").write_text(
        _json.dumps({"model_path": "D:/models/esm2_t33_650M_UR50D", "dim": 1280}), encoding="utf-8")
    with pytest.raises(ValueError, match="其他 ESM 模型"):
        EmbeddingService(model_path="D:/models/esm2_t6_8M_UR50D", cache_dir=str(cache))


# ---------- 嵌入层选择与批量大小（esm_layer / batch_size，GPU 档） ----------
# 越界校验需加载模型取 num_hidden_layers（无 torch 环境构造在 torch 导入处即抛 ImportError，
# 到不了层校验），故归 GPU 档；默认层口径零变化由上方 test_consistency_with_embed_gb1 回归。

@requires_model
@pytest.mark.parametrize("bad_layer", [34, 100])
def test_esm_layer_out_of_range_raises(bad_layer):
    """越界层号 → ValueError，消息含可用范围说明（t33：整数层号 0..33，0=embedding 层输出）。"""
    with pytest.raises(ValueError, match=r"越界.*范围 0\.\.33"):
        EmbeddingService(MODEL_PATH, esm_layer=bad_layer)


@requires_model
def test_esm_layer_middle_differs_from_last(service):
    """esm_layer=6（中层）：可编码、shape/dtype/有限性正常，且与默认末层向量不同——
    余弦相似 < 0.999，证明层参数真实生效而非被静默忽略。"""
    mid = EmbeddingService(MODEL_PATH, esm_layer=6)
    seq = SEQUENCES[1]
    emb_mid = mid.encode([seq])[0]
    emb_last = service.encode([seq])[0]
    assert emb_mid.shape == (EMBED_DIM,) and emb_mid.dtype == np.float32
    assert np.isfinite(emb_mid).all()
    cosine = float(np.dot(emb_mid, emb_last) / (np.linalg.norm(emb_mid) * np.linalg.norm(emb_last)))
    print(f"layer6-vs-last cosine={cosine:.6f}")
    assert cosine < 0.999


@requires_model
def test_small_batch_size_encodes_across_batches():
    """batch_size=4：6 条序列分两批编码，shape/dtype/有限性正常（批量参数真实生效）。"""
    svc = EmbeddingService(MODEL_PATH, batch_size=4)
    seqs = SEQUENCES + [s[:10] + "G" + s[11:] for s in SEQUENCES]  # 3 WT + 3 单点变体
    emb = svc.encode(seqs)
    assert emb.shape == (6, EMBED_DIM)
    assert emb.dtype == np.float32 and np.isfinite(emb).all()
def test_disk_cache_rejects_layer_swap(tmp_path):
    # 同模型换嵌入层：缓存档案 esm_layer=6，请求默认 last → 守卫拒绝（torch 导入前）
    import json as _json
    cache = tmp_path / "esm_cache"
    cache.mkdir()
    (cache / "model.json").write_text(
        _json.dumps({"model_path": "D:/models/650M", "esm_layer": 6, "dim": 1280}), encoding="utf-8")
    with pytest.raises(ValueError, match="另一嵌入层"):
        EmbeddingService(model_path="D:/models/650M", cache_dir=str(cache))
