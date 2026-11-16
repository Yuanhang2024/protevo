# setup_assistant.py — LLM 辅助环境配置助手
# 设计原则：探测与建议规则确定性；LLM 是配置器/顾问——在受约束菜单内提出配置、解释、
# 答疑、生成命令；执行前需 --execute 确认（默认 dry-run）；科学选型（RF/Ridge/MLP/ESM
# 谁好）仍由 configure_protein 的验证 Spearman 协议决定，LLM 不替数据选模型。
#
# 三层结构：
#   1. 确定性探测：probe_data（轻量抽样，不训练）/ probe_environment（hardware_probe 契约
#      + ESM checkpoint 探测 + OS）；embedding_service 的并行契约函数缺失时兜底为 unknown。
#   2. 确定性建议：build_plan——格式判定 / reference 建议 / torch 三命令 / ESM 按 VRAM 分档，
#      规则全部显式可测。
#   3. LLM 层：configure_with_llm——LLM 只能在受约束菜单内选 chosen，守卫逐字段校验，
#      越出菜单或与硬件事实矛盾即覆写为确定性计划值并记录 disagreements；
#      先验类分歧（低显存选 650M——10GB 阈值是保守先验，650M 推理约需 4GB）仅记
#      warnings 不覆写。
#   4. 幂等与体检：is_configured 已配置短路（硬件指纹 + 廉价复验，不发 LLM 调用）；
#      run_doctor 逐项体检（依赖/torch/平台匹配/checkpoint/冒烟），通过即写入状态文件闭环。
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from protevo.protein_config import _MODE_DOMINANCE_THRESHOLD, configure_protein
from protevo.variant_notation import has_indel

# 与 protein_config._detect_format_and_positions 一致的判定阈值
DIGITS_RATIO_THRESHOLD = 0.5   # 采样变体含数字占比 ≥ 0.5 → 记谱类（氨基酸序列永不含数字）
FULL_LENGTH_THRESHOLD = 30     # 等长无数字且长度 > 30 → 全长序列；≤ 30 走突变串判定
VRAM_650M_GB = 10.0            # VRAM ≥ 10GB 才建议 ESM-2 650M，否则降档 8M
ESM_MAX_LENGTH = 1022          # ESM-2 位置编码上限（与 embedding_service 护栏一致）

# torch 安装命令三选一（按 backend）；ROCm 版 PyTorch 的设备名仍是 "cuda"
TORCH_COMMANDS = {
    "cuda": "pip install torch --index-url https://download.pytorch.org/whl/cu126",
    "rocm": "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2",
    "cpu": "pip install torch",
}
ESM_MODEL_IDS = {
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_8m": "facebook/esm2_t6_8M_UR50D",
}
# ESM-2 checkpoint 的本地探测位置（只探测不下载）；键与菜单 esm_model 一致
CHECKPOINT_CANDIDATES = {
    "esm2_650m": (Path.home() / ".cache/protevo/esm2_t33_650M",
                  Path("D:/ChimeraModel_Lite/esm2_t33_650M_UR50D")),
    "esm2_8m": (Path.home() / ".cache/protevo/esm2_t6_8M_UR50D",
                Path("D:/ChimeraModel_Lite/esm2_t6_8M_UR50D")),
}
# checkpoint 权重文件最小体积（抓半下载/损坏的零字节/极小文件；留 ~5% 余量：镜像下载的
# 实际字节数可略小于名义值——8M 实测 31,384,292 bytes 曾被 30MiB=31,457,280 阈值误拒，差 72,988）
CHECKPOINT_MIN_WEIGHTS = {"esm2_650m": int(1024 ** 3 * 0.95), "esm2_8m": 30_000_000}
# 已配置状态文件与指纹容差
STATE_PATH = Path.home() / ".cache/protevo/setup_state.json"
VRAM_TOLERANCE_GB = 1.0

# LLM 可选字段菜单（守卫校验的合法域）
MENU = {
    "device": ("auto", "cpu", "cuda"),
    "esm_model": ("none", "esm2_650m", "esm2_8m"),
    "use_esm": ("bool",),
    "install_torch": ("null-or-command",),  # None 或 TORCH_COMMANDS 三者之一
    "install_core": ("skip", "requirements"),  # requirements = bootstrap（venv+激活+pip -r）
    "checkpoint_action": ("skip", "download", "manual"),
}


# ---------------------------------------------------------------------------
# 第一层：确定性探测
# ---------------------------------------------------------------------------

def probe_data(csv_path, reference_sequence=None, sample_rows=20000):
    """轻量抽样探测变体 CSV（只读不训练）。

    返回字段：n_rows / columns / variant_col / sampled_rows / digits_ratio /
    length_unique / max_length / indel_suspect / uniform_suspect / reference_given / csv_path。
    uniform_suspect 仅对无数字等长数据计算（记谱格式的 WT 来自记谱字母，无需众数推断），
    判定与 protein_config._infer_reference_mode 同口径：任一位点 众数/次众数 < 1.5。
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        # 空数据显式拒绝：0 行时 lengths.max() 为 NaN，此前在 int() 处报不可读的
        # "cannot convert float NaN to integer"（DSH 组合压力测试发现）
        raise ValueError(f"CSV 无有效数据：{csv_path} 含 0 行数据（只有表头或完全为空）")
    variant_col = next((c for c in ("variant", "Variants", "mutations", "seq", "sequence")
                        if c in df.columns), None)
    if variant_col is None:
        object_cols = df.select_dtypes(include=["object"]).columns
        if len(object_cols) == 0:
            raise ValueError(f"CSV 中找不到变体列: {list(df.columns)}")
        variant_col = object_cols[0]

    # 变体列先剔除缺失值再转字符串：空行变体（读入为 NaN）在部分 pandas 版本下
    # astype(str) 仍保留 float NaN，进入逐字符扫描会抛 TypeError
    sample = df[variant_col].dropna().astype(str).head(sample_rows)
    if sample.empty:
        raise ValueError(f"CSV 无有效数据：变体列 '{variant_col}' 全为空/缺失值（{csv_path}）")
    digits_ratio = float(pd.Series([any(c.isdigit() for c in v) for v in sample]).mean())
    # indel token 抽样窗口 2000（protein_config 用前 1000，同量级）
    indel_suspect = bool(any(has_indel(v) for v in sample.head(2000)))
    lengths = sample.str.len()
    length_unique = int(lengths.nunique())
    max_length = int(lengths.max())

    uniform_suspect = False
    if digits_ratio == 0.0 and length_unique == 1:
        scan = sample.head(5000)  # 众数优势比对 5000 行采样即稳定
        for p in range(max_length):
            counts = scan.str[p].value_counts()
            if len(counts) > 1 and counts.iloc[0] < counts.iloc[1] * _MODE_DOMINANCE_THRESHOLD:
                uniform_suspect = True
                break

    return {
        "csv_path": str(csv_path).replace("\\", "/"),
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "variant_col": str(variant_col),
        "sampled_rows": int(len(sample)),
        "digits_ratio": round(digits_ratio, 4),
        "length_unique": length_unique,
        "max_length": max_length,
        "indel_suspect": indel_suspect,
        "uniform_suspect": uniform_suspect,
        "reference_given": reference_sequence is not None,
    }


def probe_environment():
    """探测运行环境：hardware_probe 契约 + ESM checkpoint 就绪性 + OS。

    embedding_service.hardware_probe 由并行开发维护：缺失或抛错一律兜底 unknown，
    不影响本模块导入与规则计划。
    """
    hw = {}
    try:
        from protevo.embedding_service import hardware_probe
        probed = hardware_probe()
        if isinstance(probed, dict):
            hw = probed
    except Exception:
        hw = {}

    try:
        vram = float(hw.get("vram_gb"))
    except (TypeError, ValueError):
        vram = None

    checkpoint = _find_esm_checkpoint()
    return {
        "torch": hw.get("torch") is True,
        "cuda": hw.get("cuda") is True,
        "backend": hw.get("backend") or "unknown",
        "device_name": hw.get("device_name") or "unknown",
        "vram_gb": vram,
        "transformers": hw.get("transformers") is True,
        "checkpoint_ready": checkpoint is not None,
        "checkpoint_path": (str(checkpoint).replace("\\", "/") if checkpoint else None),
        "os": platform.platform(),
        "in_venv": sys.prefix != sys.base_prefix,
        "core_deps_ok": all(_module_available(m) for m in ("pandas", "sklearn", "scipy")),
    }


def _find_esm_checkpoint(esm_key="esm2_650m"):
    """在默认缓存与本机目录中探测 ESM-2 checkpoint（含 config.json 即视为目录有效）。"""
    for candidate in CHECKPOINT_CANDIDATES.get(esm_key, ()):
        if (candidate / "config.json").exists():
            return candidate
    return None


def _weight_bytes(checkpoint_dir):
    """checkpoint 目录内最大权重文件体积（*.safetensors / *.bin）；无权重返回 None。"""
    weights = list(checkpoint_dir.glob("*.safetensors")) + list(checkpoint_dir.glob("*.bin"))
    return max((w.stat().st_size for w in weights), default=None)


def _module_available(name):
    """模块可导入性探测（find_spec，不执行模块体）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _torch_available():
    return _module_available("torch")


def _torch_info():
    """返回 (version, build) 或 None；build ∈ cuda/rocm/cpu（按 torch.version 判定）。"""
    try:
        import torch
        if getattr(torch.version, "hip", None):
            build = "rocm"
        elif torch.version.cuda:
            build = "cuda"
        else:
            build = "cpu"
        return torch.__version__, build
    except Exception:
        return None


def _gpu_backend_key(backend):
    """后端串规范化：'CUDA 12.8'/'cuda' → 'cuda'；'ROCm 6.2'/'rocm' → 'rocm'；其余 → None。

    hardware_probe 返回完整后端串（如 "CUDA 12.8"），而命令/菜单键是短键（cuda/rocm）——
    此前 platform_match 对 backend 做精确匹配（hw in ("cuda", "rocm")），把 "CUDA 12.8"
    误判为无 GPU 并建议 CPU torch（DSH 组合压力测试发现）。统一用本函数做前缀匹配。
    """
    text = str(backend or "").upper()
    if text.startswith("CUDA"):
        return "cuda"
    if text.startswith("ROCM"):
        return "rocm"
    return None


def _pip_bound(command):
    """裸 pip 命令绑定当前解释器：'pip install X' → '"<解释器>" -m pip install X'。

    以独立 venv 解释器启动时 PATH 中常无可定位的裸 pip（DSH 组合压力测试发现），
    生成与执行的安装命令统一绑定 sys.executable；非 pip 开头的命令原样返回。
    """
    command = str(command)
    return f'"{sys.executable}" -m {command}' if command.startswith("pip ") else command


def _llm_credentials():
    """.env 三变量约定（与 agent_v2.build_openai_client 一致）中前两个是否存在。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass
    return bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("OPENAI_MODEL"))


def _ensure_esm_available():
    """embedding_service.ensure_esm_checkpoint 契约是否可用（只探测导入，不调用）。"""
    try:
        from protevo.embedding_service import ensure_esm_checkpoint  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 第二层：确定性建议
# ---------------------------------------------------------------------------

def build_plan(data_probe, env_probe):
    """显式规则生成确定性计划。data_probe=None（未提供数据）时格式判 unknown。

    规则（全部可测）：
      格式: digits_ratio≥0.5 且 indel_suspect → indel_notation；digits_ratio≥0.5 →
            mutation_notation；等长且长>30 → full；否则 mutation_string。
      reference: uniform_suspect 且未给 → 建议显式提供（GB1 众数 WLVS≠VDGV 案例）；
            indel 格式未给 → 必须提供（configure_protein 硬约束）。
      torch: 已装 → 无命令；未装按 backend 三选一（cuda/rocm/cpu wheel）。
      蛋白模型: torch+cuda → ESM-2，VRAM≥10GB → 650M，否则 8M；torch 无 cuda → 8M（CPU 慢）；
            无 torch → none + 安装命令；数据为 mutation_string/mutation_notation 或
            full 且长>1022 → use_esm=False（无法重构全长序列 / 超位置编码上限）。
      device: 有 GPU → auto；否则 cpu。
    """
    torch_ok = env_probe.get("torch") is True
    cuda_ok = env_probe.get("cuda") is True
    backend = env_probe.get("backend") or "unknown"
    vram = env_probe.get("vram_gb")

    # --- 格式判定 ---
    if data_probe is None:
        variant_format, format_reason = "unknown", "未提供数据文件，跳过格式判定"
    elif data_probe["digits_ratio"] >= DIGITS_RATIO_THRESHOLD and data_probe["indel_suspect"]:
        variant_format = "indel_notation"
        format_reason = (f"含数字比例 {data_probe['digits_ratio']:.2f} 且采样检出 indel 记谱"
                         f"（del/ins/dup）")
    elif data_probe["digits_ratio"] >= DIGITS_RATIO_THRESHOLD:
        variant_format = "mutation_notation"
        format_reason = f"含数字比例 {data_probe['digits_ratio']:.2f}（氨基酸序列永不含数字）"
    elif data_probe["length_unique"] == 1 and data_probe["max_length"] > FULL_LENGTH_THRESHOLD:
        variant_format = "full"
        format_reason = f"等长（{data_probe['length_unique']} 种）且长度 {data_probe['max_length']} > {FULL_LENGTH_THRESHOLD}"
    else:
        variant_format = "mutation_string"
        if data_probe["length_unique"] > 1:
            format_reason = (f"无数字但长度不一致（{data_probe['length_unique']} 种）——"
                             f"configure_protein 将显式报错，indel 数据请改用记谱格式")
        else:
            format_reason = f"无数字等长短串（{data_probe['max_length']} aa ≤ {FULL_LENGTH_THRESHOLD}）"

    # --- reference 建议 ---
    reference_needed, reference_note = False, None
    if data_probe is not None and data_probe["uniform_suspect"] and not data_probe["reference_given"]:
        reference_needed = True
        reference_note = ("位点残基分布接近均匀（枚举型库），众数推断的参考序列不可靠——"
                          "GB1 全枚举库众数为 WLVS 而真实 WT 是 VDGV。请显式传入 reference_sequence。")
    if variant_format == "indel_notation" and (data_probe or {}).get("reference_given") is not True:
        reference_needed = True
        reference_note = "indel 库必须提供全长 reference_sequence（1-indexed 坐标核对与序列重构依赖它）。"

    # --- torch 安装命令 ---
    if torch_ok:
        install_torch, torch_reason = None, "torch 已安装"
    else:
        backend_key = _gpu_backend_key(backend) or backend  # 'CUDA 12.8' → 'cuda'；unknown/cpu 原样
        install_torch = TORCH_COMMANDS.get(backend_key, TORCH_COMMANDS["cpu"])
        extra = "（ROCm 版 PyTorch 的设备名仍为 cuda）" if backend_key == "rocm" else ""
        torch_reason = f"torch 未安装，按 backend={backend} 选择安装命令{extra}"

    # --- bootstrap（核心依赖缺失时 commands 首位插入三步块，按 OS 两式；安装步绑定解释器）---
    core_deps_ok = env_probe.get("core_deps_ok") is True
    if core_deps_ok:
        install_core, bootstrap_commands = "skip", []
    else:
        windows = "Windows" in str(env_probe.get("os") or "")
        bootstrap_commands = (["python -m venv .venv", r".venv\Scripts\activate",
                               _pip_bound("pip install -r requirements.txt")] if windows else
                              ["python -m venv .venv", "source .venv/bin/activate",
                               _pip_bound("pip install -r requirements.txt")])
        install_core = "requirements"

    # --- 蛋白模型（环境侧资格 → VRAM 分档 → 数据侧否决）---
    esm_env_ok = torch_ok
    if not torch_ok:
        esm_model, esm_env_reason = "none", "torch 不可用，无法运行 ESM-2"
    elif cuda_ok and vram is not None and vram >= VRAM_650M_GB:
        esm_model, esm_env_reason = "esm2_650m", f"CUDA 可用且 VRAM {vram:g}GB ≥ {VRAM_650M_GB:g}GB"
    elif cuda_ok:
        esm_model = "esm2_8m"
        esm_env_reason = (f"CUDA 可用但 VRAM {'未知' if vram is None else f'{vram:g}GB'} < {VRAM_650M_GB:g}GB，"
                          f"建议 8M（精度略低但可跑）")
    else:
        esm_model, esm_env_reason = "esm2_8m", "torch 可用但无 CUDA，ESM-2 8M 可在 CPU 运行（较慢）"

    esm_data_veto = None
    if variant_format in ("mutation_string", "mutation_notation"):
        esm_data_veto = "突变串/记谱格式无法重构全长序列，跳过 ESM 候选"
    elif variant_format == "full" and data_probe and data_probe["max_length"] > ESM_MAX_LENGTH:
        esm_data_veto = f"序列长度 {data_probe['max_length']} > {ESM_MAX_LENGTH}（ESM-2 位置编码上限）"

    use_esm = esm_env_ok and esm_data_veto is None
    if esm_data_veto:
        esm_model, esm_reason = "none", f"{esm_env_reason}；且{esm_data_veto}"
    else:
        esm_reason = esm_env_reason

    # --- checkpoint 状态（事实层）与菜单默认 ---
    if env_probe.get("checkpoint_ready"):
        checkpoint_status, checkpoint_action_default = "ready", "skip"
    elif esm_model != "none" and _ensure_esm_available():
        checkpoint_status, checkpoint_action_default = "downloadable", "download"
    else:
        checkpoint_status, checkpoint_action_default = "unavailable", "skip"

    # --- transformers（ESM 运行的另一依赖）---
    install_transformers = (None if env_probe.get("transformers") or esm_model == "none"
                            else _pip_bound("pip install transformers"))

    plan = {
        "variant_format": variant_format,
        "format_reason": format_reason,
        "reference_needed": reference_needed,
        "reference_note": reference_note,
        "device": "auto" if cuda_ok else "cpu",
        "use_esm": use_esm,
        "esm_model": esm_model,
        "esm_model_id": ESM_MODEL_IDS.get(esm_model),
        "esm_reason": esm_reason,
        "install_torch": install_torch,
        "install_transformers": install_transformers,
        "core_deps_ok": core_deps_ok,
        "install_core": install_core,
        "bootstrap_commands": bootstrap_commands,
        "requirements_note": (None if core_deps_ok
                              else "bootstrap 三步（建 venv → 激活 → 安装）；requirements.txt 在包根"),
        "checkpoint_path": env_probe.get("checkpoint_path"),
        "checkpoint_status": checkpoint_status,
        "checkpoint_action_default": checkpoint_action_default,
        "torch_reason": torch_reason,
        "algorithms_note": ("sklearn 系预测器（RandomForest / Ridge / MLP）零配置内置；选型由 "
                            "configure_protein 的种子化验证 Spearman 协议自动完成——"
                            "本助手与 LLM 均不替数据选模型。"),
    }
    plan["configure_call"] = _configure_call(data_probe, plan)
    return plan


def _configure_call(data_probe, plan, chosen=None):
    """生成预期 configure_protein(...) 完整调用串（含建议参数）。"""
    chosen = chosen or _menu_defaults(plan)
    csv_path = (data_probe or {}).get("csv_path") or "your_data.csv"
    args = [f"{csv_path!r}"]
    if plan["reference_needed"]:
        args.append("reference_sequence='WT序列'")  # 需用户补齐，占位提示
    if chosen["use_esm"] and chosen["esm_model"] != "none":
        model_ref = plan.get("checkpoint_path") or ESM_MODEL_IDS.get(chosen["esm_model"])
        device = "cuda" if chosen["device"] in ("auto", "cuda") else "cpu"
        args += [f"use_esm=True", f"esm_model_path={model_ref!r}", f"esm_device={device!r}"]
    return f"configure_protein({', '.join(args)})"


def _menu_defaults(plan):
    """确定性计划的菜单默认值（降级路径与守卫覆写的基准）。"""
    return {
        "device": plan["device"],
        "esm_model": plan["esm_model"],
        "use_esm": plan["use_esm"],
        "install_torch": plan["install_torch"],
        "install_core": plan["install_core"],
        "checkpoint_action": plan["checkpoint_action_default"],
    }


# ---------------------------------------------------------------------------
# 第三层：LLM 受约束配置器
# ---------------------------------------------------------------------------

# 模型选型校准知识（外部文献核实版）——原文使用，置信级别与限定词不得删改。
# 仅注入 LLM 系统提示词，供解释与菜单内选择建议；菜单与守卫逻辑零改动，
# 知识不得成为绕过守卫的理由。条目判定与来源链接见 docs/knowledge_calibration.md。
MODEL_SELECTION_KNOWLEDGE = """【模型选型校准知识（外部文献核实版）】
1 [文献强支持] 位点特异组合库、样本≤5万、插值任务：one-hot+线性/树模型可与冻结PLM嵌入相当甚至更强（FLIP/PLMFit 在GB1/AAV证实）；但全蛋白多位点、高N的DMS设定下PLM嵌入监督明显更强，勿外推。
2 [文献部分支持] 冻结PLM嵌入+浅层头是标准低成本做法、可作默认起点；系统对比显示任务微调（全参/LoRA）几乎总能再提升，"微调非必需"应读作"非必需但常有增益"。
3 [仅内证] "浅层MLP头优于线性头"是本项目观察；公开基准同测两类头但无一致赢家，应作为待调超参而非默认假设。
4 [文献强支持] 监督小数据下中小PLM（150M–650M）可与大模型相当甚至反超；零样本打分则不然：ESM-2 8M→650M规模收益大（Spearman 0.23 vs 0.42），两设定不可混用。
5 [文献强支持] 嵌入提取应显式消融：全序列mean-pooling与突变区域池化（mut-mean）在FLIP中并列使用、各有适用，无普适赢家。
6 [文献强支持] sklearn随机森林/树模型仅CPU（Array API GPU名单不含树模型）；GPU树需RAPIDS cuML或XGBoost/LightGBM另建依赖栈。
7 [文献强支持] ESM-2位置嵌入表1024 token（含<cls>/<eos>），氨基酸上限1022；esm-extract默认1022截断，超长需自行分块且超出预训练范围。
8 [文献强支持] PLM嵌入与自回归打分天然兼容indel变长序列，固定位置one-hot/BLOSUM62/MSA不能；但ESM masked-marginal零样本打分同样不适用于indel，且indel整体仍难于替换（最佳零样本Spearman~0.47）。"""

_SYSTEM_PROMPT_TEMPLATE = """你是 protevo 的环境配置助手。根据下方探测事实，在受约束菜单内给出最终配置并解释。

## 数据探测（只读事实）
{data_probe}

## 环境探测（只读事实）
{env_probe}

## 确定性规则计划（推荐基线）
{plan}

{knowledge}

知识块用于解释与选择建议，与硬件/数据事实矛盾时以守卫为准。

## 受约束菜单（chosen 各字段只能取以下值）
- device: auto | cpu | cuda（无 GPU 时仅 cpu 合法）
- esm_model: none | esm2_650m | esm2_8m（无 torch 时仅 none 合法；VRAM<10GB 时选 650M 会保留选择并记录警告——650M 推理约需 4GB 显存，若 OOM 请改用 8M 或 CPU）
- use_esm: true | false（突变串/记谱格式数据只允许 false）
- install_torch: null 或以下三命令之一:
  - pip install torch --index-url https://download.pytorch.org/whl/cu126
  - pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  - pip install torch
- install_core: skip | requirements（核心依赖 pandas/sklearn/scipy 缺失时选 requirements——bootstrap 三步: python -m venv .venv → 激活 → pip install -r requirements.txt，requirements.txt 在包根；已装全时 requirements 会被守卫覆写为 skip）
- checkpoint_action: skip | download | manual

其余字段只读。科学选型（RF/Ridge/MLP/ESM 谁好）由 configure_protein 的验证 Spearman 协议决定，你不替数据选模型。

只返回一个 JSON 对象，不要任何其他文字或代码块标记：
{{"chosen": {{"device": ..., "esm_model": ..., "use_esm": ..., "install_torch": ..., "install_core": ..., "checkpoint_action": ...}}, "rationale": "...", "commands": ["..."]}}
rationale 用中文，解释每个选择；若用户提了问题，先回答问题再给配置理由。"""


def configure_with_llm(data_csv=None, question=None, client=None, model=None,
                       environ=None, auto_execute=False, reference_sequence=None):
    """LLM 辅助配置主入口。

    流程：双 probe → 确定性计划 → LLM 在受约束菜单内选 chosen → 守卫逐字段校验
    （越界/与硬件事实矛盾 → 覆写为计划值并记入 disagreements；先验类分歧如低显存选
    650M → 保留选择仅记入 warnings）→ 默认 dry-run 只返回，auto_execute=True 才执行
    命令。无 .env 凭据时直接降级返回规则计划。
    """
    data_probe = probe_data(data_csv, reference_sequence=reference_sequence) if data_csv else None
    env_probe = probe_environment()
    plan = build_plan(data_probe, env_probe)
    chosen_defaults = _menu_defaults(plan)

    result = {
        "data_probe": data_probe, "env_probe": env_probe, "plan": plan,
        "llm_used": False, "llm_model": None, "llm_reply": None,
        "chosen": dict(chosen_defaults), "rationale": "LLM 未参与，以下为确定性规则计划。",
        "disagreements": [], "warnings": [], "commands": [], "question": question, "answer": None,
        "note": None, "executed": None,
    }

    # --- LLM 凭据（复用 agent_v2 的 .env 三变量约定）；缺失则降级规则计划 ---
    if client is None:
        try:
            from protevo.agent_v2 import build_openai_client
            client, model = build_openai_client(SimpleNamespace(llm_timeout=120), environ)
        except Exception as exc:
            result["note"] = f"LLM 未配置（{type(exc).__name__}: {exc}），以下为规则计划。"
            _finalize(result, plan, auto_execute)
            return result

    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        data_probe=json.dumps(data_probe, ensure_ascii=False, default=str),
        env_probe=json.dumps(env_probe, ensure_ascii=False, default=str),
        plan=json.dumps({k: v for k, v in plan.items() if k != "configure_call"},
                        ensure_ascii=False, default=str),
        knowledge=MODEL_SELECTION_KNOWLEDGE)
    user_message = question or "请根据探测事实给出配置。"
    response = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": prompt},
                               {"role": "user", "content": user_message}], temperature=0)
    reply = response.choices[0].message.content or ""

    parsed, disagreements, warnings = _parse_reply(reply), [], []
    if parsed is None:
        disagreements.append({"field": "(整体)", "llm": reply[:200], "plan": "规则计划",
                              "reason": "LLM 返回不是合法 JSON，整体回退确定性计划"})
        chosen = dict(chosen_defaults)
        rationale = "（LLM 返回无法解析，已回退规则计划）"
    else:
        chosen, disagreements, warnings = _guard_choices(parsed.get("chosen") or {}, plan, env_probe)
        rationale = str(parsed.get("rationale") or "")

    result.update({"llm_used": True, "llm_model": model, "llm_reply": reply,
                   "chosen": chosen, "rationale": rationale, "disagreements": disagreements,
                   "warnings": warnings,
                   "answer": rationale if question else None,
                   "llm_commands": [str(c) for c in (parsed or {}).get("commands") or []]})
    _finalize(result, plan, auto_execute)
    return result


def _parse_reply(reply):
    """解析 LLM 回复中的单个 JSON 对象；容错剥离 markdown 代码块围栏。"""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        return None


def _guard_choices(raw, plan, env_probe):
    """守卫校验：逐字段检查 LLM 的 chosen，分两级处理。

      事实类矛盾（越出菜单 / 无 torch 选 ESM / 无 GPU 选 cuda / 数据格式否决 /
      自相矛盾等——表示或资源事实）→ 覆写为计划值并记入 disagreements；
      先验类分歧（VRAM<10GB 含未知时选 650M——10GB 阈值是保守先验，650M 推理实际
      约需 4GB 显存）→ 保留 LLM 选择，仅记入 warnings，OOM 风险由文案提示兜底。

    返回 (chosen, disagreements, warnings)。矛盾类型 → 覆写行为见每条规则注释。
    """
    if not isinstance(raw, dict):
        # 容错：chosen 非 JSON 对象（如列表/字符串，DSH 组合压力测试的坏结构回复）→
        # 整体回退规则计划并记 disagreements，不抛 AttributeError
        return (dict(_menu_defaults(plan)),
                [{"field": "chosen", "llm": raw, "plan": "规则计划",
                  "reason": "chosen 不是 JSON 对象（需要 dict），整体回退确定性计划"}],
                [])
    defaults = _menu_defaults(plan)
    cuda_ok = env_probe.get("cuda") is True
    torch_ok = env_probe.get("torch") is True
    vram = env_probe.get("vram_gb")
    checkpoint_ready = env_probe.get("checkpoint_ready") is True
    data_veto = plan["esm_model"] == "none" and not plan["use_esm"]
    disagreements = []
    warnings = []

    def check(field, value, ok, reason):
        """ok=False 时覆写为计划值并记录；True 时采纳。"""
        if ok:
            return value
        disagreements.append({"field": field, "llm": value, "plan": defaults[field], "reason": reason})
        return defaults[field]

    chosen = {}
    for field in defaults:
        value = raw.get(field, None)
        present = field in raw
        if field == "install_core" and not present:
            # 菜单新增字段（bootstrap 期）：旧版回复无此键 → 静默补计划默认值，
            # 不记 disagreements（原有字段的"字段缺失"记录语义不变）
            chosen[field] = defaults[field]
            continue
        if field == "device":
            ok = value in MENU["device"] and (cuda_ok or value == "cpu")
            chosen[field] = check(field, value, ok and present,
                                  "越出菜单或环境无 GPU（仅 cpu 合法）" if present else "字段缺失")
        elif field == "esm_model":
            ok = value in MENU["esm_model"]
            if ok and value != "none":
                if not torch_ok:
                    ok = False
                elif data_veto:
                    ok = False
            chosen[field] = check(field, value, ok and present,
                                  "越出菜单或与 torch/数据格式事实矛盾" if present else "字段缺失")
            # 先验只推荐不拦截：VRAM<10GB（含未知/无 GPU）时选 650M——保留选择，
            # 记警告不覆写（650M 推理约需 4GB 显存，OOM 时可退 8M 或 CPU）
            if (present and value == "esm2_650m" and torch_ok and not data_veto
                    and not (cuda_ok and vram is not None and vram >= VRAM_650M_GB)):
                vram_text = f"{vram:g}GB" if vram is not None else "未知"
                warnings.append(
                    f"esm_model=esm2_650m：650M 推理约需 4GB 显存，当前探测 {vram_text}"
                    f"（低于 {VRAM_650M_GB:g}GB 推荐阈值，先验仅警告不拦截），"
                    f"若 OOM 请改用 esm2_8m 或 CPU")
        elif field == "use_esm":
            ok = isinstance(value, bool)
            if ok and value and ((not torch_ok) or data_veto or chosen.get("esm_model") == "none"):
                ok = False
            chosen[field] = check(field, value, ok and present,
                                  "越出菜单或与 torch/数据格式/esm_model 事实矛盾" if present else "字段缺失")
        elif field == "install_torch":
            ok = value is None or value in TORCH_COMMANDS.values()
            if ok and value is not None and (torch_ok or value != defaults[field]):
                ok = False  # torch 已装 / 与 backend 事实不符
            chosen[field] = check(field, value, ok and present,
                                  "越出菜单或 torch 已装/与 backend 事实不符" if present else "字段缺失")
        elif field == "install_core":
            # 核心依赖已全 → requirements 被覆写为 skip；缺失时 skip/requirements 两值均合法
            if value not in MENU["install_core"]:
                ok, reason = False, "越出菜单"
            elif value == "requirements" and plan.get("core_deps_ok"):
                ok, reason = False, "核心依赖已就绪"
            else:
                ok, reason = True, ""
            chosen[field] = check(field, value, ok and present, reason)
        else:  # checkpoint_action
            ok = value in MENU["checkpoint_action"]
            if ok and value == "download" and (checkpoint_ready or plan["checkpoint_status"] == "unavailable"):
                ok = False
            chosen[field] = check(field, value, ok and present,
                                  "越出菜单或 checkpoint 已就绪/不可下载" if present else "字段缺失")
    return chosen, disagreements, warnings


def _finalize(result, plan, auto_execute):
    """由校验后的 chosen 派生可执行命令与最终调用串；auto_execute=True 时执行。"""
    chosen = result["chosen"]
    commands = []
    if chosen.get("install_core") == "requirements":
        commands.extend(plan["bootstrap_commands"])  # bootstrap 三步块固定在 commands 首位
    if chosen["install_torch"] and plan["install_torch"] is not None:
        # 计划侧判定 torch 已就绪（install_torch=None）时不生成安装命令，保证 commands
        # 与"无需安装 torch"的探测事实一致（DSH 组合压力测试发现的说明-命令矛盾）
        commands.append(_pip_bound(chosen["install_torch"]))
    if plan["install_transformers"] and chosen["use_esm"] and chosen["esm_model"] != "none":
        commands.append(plan["install_transformers"])
    result["commands"] = commands
    result["configure_call"] = _configure_call(result["data_probe"], plan, chosen)
    if auto_execute:
        result["executed"] = _execute(commands, chosen, plan)


def _execute(commands, chosen, plan):
    """执行命令（每步前打印，subprocess 捕获输出）+ 可选 checkpoint 下载。"""
    executed = []
    for command in commands:
        if "activate" in command:
            # 激活只对交互式 shell 生效、不跨 subprocess——跳过并注明，避免假失败
            print(f"[execute] 跳过（激活需在交互式 shell 中执行）: {command}")
            executed.append({"command": command, "returncode": None,
                             "note": "激活步骤需在交互式 shell 中手动执行"})
            continue
        print(f"[execute] {command}")
        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        executed.append({"command": command, "returncode": proc.returncode,
                         "stdout_tail": (proc.stdout or "")[-2000:], "stderr_tail": (proc.stderr or "")[-2000:]})
    if chosen["checkpoint_action"] == "download" and plan["esm_model_id"]:
        model_id = plan["esm_model_id"]
        print(f"[execute] 下载 ESM checkpoint: {model_id}")
        try:
            from protevo.embedding_service import ensure_esm_checkpoint
            path = ensure_esm_checkpoint(model_id=model_id)
            executed.append({"command": f"ensure_esm_checkpoint(model_id={model_id!r})",
                             "path": str(path)})
        except Exception as exc:
            executed.append({"command": f"ensure_esm_checkpoint(model_id={model_id!r})",
                             "error": f"{type(exc).__name__}: {exc}"})
    return executed


# ---------------------------------------------------------------------------
# 报告与 CLI
# ---------------------------------------------------------------------------

def render_report(result):
    """渲染中文报告：探测表 / LLM 选择与理由 / 守卫纠正 / 命令块 / 最终调用串。"""
    lines = ["=" * 62, "protevo 环境配置助手报告", "=" * 62]
    lines += ["", "— 数据探测 —"]
    if result["data_probe"]:
        for key, value in result["data_probe"].items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  未提供数据文件")
    lines += ["", "— 格式判定 —",
              f"  variant_format: {result['plan']['variant_format']}",
              f"  依据: {result['plan']['format_reason']}"]
    lines += ["", "— 环境探测 —"]
    for key, value in result["env_probe"].items():
        lines.append(f"  {key}: {value}")

    lines += ["", "— 配置选择" + ("（LLM）—" if result["llm_used"] else "（规则计划）—")]
    for key in ("device", "esm_model", "use_esm", "install_torch", "install_core", "checkpoint_action"):
        lines.append(f"  {key}: {result['chosen'][key]}")
    lines.append(f"  理由: {result['rationale']}")
    if result.get("note"):
        lines.append(f"  说明: {result['note']}")

    lines += ["", "— 守卫纠正记录 —"]
    if result["disagreements"]:
        for d in result["disagreements"]:
            lines.append(f"  字段 {d['field']}: LLM 选 {d['llm']!r} → 覆写为 {d['plan']!r}（{d['reason']}）")
    else:
        lines.append("  无（LLM 选择全部合法或未参与）")

    if result.get("warnings"):  # 先验分歧：保留 LLM 选择仅提示（无则不显示该节）
        lines += ["", "— 守卫警告（先验分歧，保留 LLM 选择）—"]
        for warning in result["warnings"]:
            lines.append(f"  {warning}")

    if result.get("question"):
        lines += ["", "— 问答 —", f"  问: {result['question']}", f"  答: {result['answer'] or result['rationale']}"]

    lines += ["", "— 命令" + ("（已执行）" if result.get("executed") else "（dry-run，加 --execute 才执行）—")]
    if result["commands"]:
        lines += ["  ```", *[f"  {c}" for c in result["commands"]], "  ```"]
        if result["plan"].get("requirements_note") and result["chosen"].get("install_core") == "requirements":
            lines.append(f"  {result['plan']['requirements_note']}")
    else:
        lines.append("  无需安装命令")

    lines += ["", "— 最终调用 —", "  from protevo.protein_config import configure_protein",
              f"  config = {result['configure_call']}"]
    if result["plan"]["algorithms_note"]:
        lines += ["", "— 算法说明 —", f"  {result['plan']['algorithms_note']}"]
    if result["plan"]["reference_note"]:
        lines += ["", "— 参考序列建议 —", f"  {result['plan']['reference_note']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 第四层：已配置短路（幂等）与环境体检（doctor）
# ---------------------------------------------------------------------------

def _hardware_fingerprint(env_probe=None):
    """硬件指纹：backend / cuda / device_name 逐项相等，vram 容差 ±1GB。"""
    env_probe = env_probe or probe_environment()
    return {"backend": env_probe.get("backend"), "cuda": env_probe.get("cuda"),
            "device_name": env_probe.get("device_name"), "vram_gb": env_probe.get("vram_gb")}


def _fingerprint_matches(stored, current):
    # 容错：指纹字段被写成非 dict（如列表，DSH 组合压力测试的坏结构状态文件）→
    # 视为不匹配（未配置），不抛 AttributeError
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return False
    for key in ("backend", "cuda", "device_name"):
        if stored.get(key) != current.get(key):
            return False
    vram_stored, vram_now = stored.get("vram_gb"), current.get("vram_gb")
    if vram_stored is None or vram_now is None:
        return vram_stored is None and vram_now is None
    return abs(vram_stored - vram_now) <= VRAM_TOLERANCE_GB


def _read_state():
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(env_probe=None, plan_summary=None, essentials=None):
    """写入/刷新配置状态。调用时机：doctor 通过（essentials 无 fail）；或 --execute 成功后。"""
    state = {"fingerprint": _hardware_fingerprint(env_probe or probe_environment()),
             "configured_at": datetime.now(timezone.utc).isoformat(),
             "essentials_passed": list(essentials or []),
             "plan_summary": dict(plan_summary or {})}
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except (FileExistsError, NotADirectoryError) as exc:
        # .cache 路径被普通文件占用（DSH 组合压力测试的场景）：给可操作消息而非裸异常
        raise RuntimeError(
            f"无法写入配置状态：{STATE_PATH.parent} 已被普通文件占用（此处需要目录）。"
            f"请删除或移动该文件后重新运行：{STATE_PATH.parent}"
        ) from exc
    return state


def is_configured():
    """已配置短路判定，全部廉价检查（不做嵌入前向）。

    状态文件存在 且 指纹与当前探测一致（backend/cuda/device_name 逐项相等、vram ±1GB）
    且 快速复验通过（状态含 esm 时要求 torch 可导入 + checkpoint 仍就绪）。
    指纹不匹配（换硬件）→ 视为未配置，走正常流程。
    """
    state = _read_state()
    if not state:
        return False, {}
    if not _fingerprint_matches(state.get("fingerprint") or {}, _hardware_fingerprint()):
        return False, {}
    plan = state.get("plan_summary") or {}
    esm_in_state = (plan.get("esm_model") not in (None, "", "none")) or plan.get("use_esm") is True
    if esm_in_state:
        if not _torch_available():
            return False, {}
        if _find_esm_checkpoint(plan.get("esm_model") or "esm2_650m") is None:
            return False, {}
    return True, state


def _render_configured(state):
    """已配置短路时的简短报告（配置时间 + 指纹要点 + 一行提示）。"""
    fp = state.get("fingerprint") or {}
    plan = state.get("plan_summary") or {}
    return "\n".join([
        "=" * 62, "protevo 环境已配置（短路跳过）", "=" * 62,
        f"  配置时间: {state.get('configured_at', '未知')}",
        f"  指纹: backend={fp.get('backend')} | cuda={fp.get('cuda')} | "
        f"device_name={fp.get('device_name')} | vram_gb={fp.get('vram_gb')}",
        f"  摘要: device={plan.get('device')} | esm_model={plan.get('esm_model')} | use_esm={plan.get('use_esm')}",
        "", "  完整体检用 --doctor，重新配置用 --force"])


def _expected_esm(esm_expected=None):
    """doctor 的 checkpoint 检查目标：显式参数优先，其次状态文件里的 esm。"""
    if esm_expected in CHECKPOINT_CANDIDATES:
        return esm_expected
    plan = _read_state().get("plan_summary") or {}
    if plan.get("esm_model") in CHECKPOINT_CANDIDATES:
        return plan["esm_model"]
    return None


def _write_smoke_csv(path):
    """预测器冒烟用合成 20 行小 CSV（确定性，无随机成分）。"""
    variants = [a + b + "GV" for a in "ACDEF" for b in "ACDE"]
    pd.DataFrame({"variant": variants, "fitness": [float(i % 7) for i in range(len(variants))]}
                 ).to_csv(path, index=False)


def _doctor_plan_summary(env_probe, expected_esm, checks):
    """状态文件的 plan_summary：只记录体检实证过的配置（esm 仅在 checkpoint 检查通过时写入）。"""
    esm_verified = expected_esm is not None and any(
        c["name"] == "checkpoint" and c["status"] == "ok" for c in checks)
    return {"variant_format": "unknown",
            "device": "auto" if env_probe.get("cuda") else "cpu",
            "use_esm": esm_verified,
            "esm_model": expected_esm if esm_verified else "none",
            "install_torch": (None if _torch_info() is not None
                              else TORCH_COMMANDS.get(_gpu_backend_key(env_probe.get("backend"))
                                                      or env_probe.get("backend"), TORCH_COMMANDS["cpu"])),
            "install_core": "skip" if env_probe.get("core_deps_ok") is True else "requirements",
            "checkpoint_action": "skip"}


def run_doctor(esm_expected=None, full=True):
    """环境体检：逐项检查并给出 fix；essentials 无 fail（warn 允许）时写入状态文件。

    检查项：python ≥3.10 / 核心依赖 / torch 版本与构建 / 平台-硬件匹配 /
    transformers+hub / checkpoint（esm_expected 或状态含 esm 时）/ 嵌入冒烟（full）/
    预测器冒烟（full，真实合成 CSV 走 configure_protein）/ LLM 凭据（可选项，只 warn）/
    虚拟环境（第 10 项，仅 warn 不计 essentials fail）。
    """
    env_probe = probe_environment()
    expected_esm = _expected_esm(esm_expected)
    checks = []

    def add(name, status, detail, fix=None):
        checks.append({"name": name, "status": status, "detail": detail, "fix": fix})

    # 1. Python ≥ 3.10
    vi = sys.version_info
    if (vi.major, vi.minor) >= (3, 10):
        add("python", "ok", f"Python {vi.major}.{vi.minor}.{vi.micro}")
    else:
        add("python", "fail", f"Python {vi.major}.{vi.minor}.{vi.micro} < 3.10", "升级到 Python ≥ 3.10")

    # 2. 核心依赖（sklearn 层）
    missing = [m for m in ("pandas", "sklearn", "scipy") if not _module_available(m)]
    if missing:
        add("core_deps", "fail",
            f"缺失: {', '.join(missing)}（备选逐包安装: {_pip_bound('pip install pandas scikit-learn scipy')}）",
            f"{_pip_bound('pip install -r requirements.txt')}（requirements.txt 在包根；--doctor --execute 可自动修复）")
    else:
        add("core_deps", "ok", "pandas / sklearn / scipy 均可导入")

    # 3. torch：版本与构建（CUDA / ROCm / CPU）
    info = _torch_info()
    if info is None:
        add("torch", "warn", "torch 未安装——核心 sklearn 管线不受影响，ESM 候选不可用",
            _pip_bound(TORCH_COMMANDS.get(env_probe.get("backend"), TORCH_COMMANDS["cpu"])))
    else:
        build = _gpu_backend_key(info[1]) or str(info[1])  # 容错完整后端串（如 'CUDA 12.8'）
        label = {"cuda": "CUDA", "rocm": "ROCm"}.get(build, "CPU")
        add("torch", "ok", f"torch {info[0]}（{label} 版）")

    # 4. 平台-硬件匹配：torch 构建与 hardware_probe 的 GPU 事实比对
    if info is None:
        add("platform_match", "ok", "跳过：torch 未安装（见 torch 项）")
    else:
        build = _gpu_backend_key(info[1]) or str(info[1])
        hw_gpu = _gpu_backend_key(env_probe.get("backend"))  # 'CUDA 12.8'/'cuda' → 'cuda'
        if build == hw_gpu or (build == "cpu" and hw_gpu is None):
            add("platform_match", "ok", f"torch {build} 版与硬件匹配（{env_probe.get('device_name')}）")
        else:
            target = hw_gpu or "cpu"
            gpu_desc = {"cuda": "NVIDIA", "rocm": "AMD"}.get(hw_gpu, "无")
            add("platform_match", "warn",
                f"检测到{gpu_desc} GPU（backend={env_probe.get('backend')}）但 torch 为 {build} 版——ESM 无法用该 GPU 加速",
                _pip_bound(TORCH_COMMANDS[target]))

    # 5. transformers / huggingface_hub（ESM 依赖）
    missing = [m for m in ("transformers", "huggingface_hub") if not _module_available(m)]
    if missing:
        add("transformers", "warn", f"缺失: {', '.join(missing)}（只影响 ESM 嵌入）",
            _pip_bound("pip install transformers huggingface_hub"))
    else:
        add("transformers", "ok", "transformers / huggingface_hub 均可导入")

    # 6. checkpoint：目标目录 config.json + 权重体积（650M ≈1GB / 8M ≈30MB，各留 5% 余量）
    if expected_esm is None:
        add("checkpoint", "ok", "未要求 ESM（未显式指定且状态不含 esm），跳过")
    else:
        fix = (f"python -m protevo.setup_assistant --doctor --execute（自动调 "
               f"ensure_esm_checkpoint(model_id='{ESM_MODEL_IDS[expected_esm]}') 下载）")
        ckpt = _find_esm_checkpoint(expected_esm)
        if ckpt is None:
            add("checkpoint", "fail",
                f"{expected_esm} checkpoint 未找到（搜索: "
                f"{', '.join(str(c) for c in CHECKPOINT_CANDIDATES[expected_esm])}）", fix)
        else:
            weight = _weight_bytes(ckpt)
            threshold = CHECKPOINT_MIN_WEIGHTS[expected_esm]
            if weight is None or weight < threshold:
                add("checkpoint", "fail",
                    f"{ckpt} 权重文件缺失或过小（{weight} bytes < {threshold}）", fix)
            else:
                add("checkpoint", "ok", f"{ckpt}（权重 {weight / 1024 ** 3:.2f}GB）")

    if full:
        # 7. 嵌入冒烟：EmbeddingService 编码 1 条短序列（抓安装损坏/驱动异常）
        if info is None:
            add("embedding_smoke", "warn", "无 torch，跳过嵌入冒烟")
        else:
            smoke_ckpt = _find_esm_checkpoint(expected_esm or "esm2_650m")
            if smoke_ckpt is None:
                add("embedding_smoke", "warn", "checkpoint 未就绪，跳过嵌入冒烟（先修复 checkpoint 项）")
            else:
                try:
                    from protevo.embedding_service import EmbeddingService
                    device = "cuda" if env_probe.get("cuda") else "cpu"
                    service = EmbeddingService(model_path=str(smoke_ckpt), device=device)
                    service.encode(["MKVWV"])
                    add("embedding_smoke", "ok", f"1 条短序列编码成功（{smoke_ckpt.name}, {device}）")
                except Exception as exc:
                    add("embedding_smoke", "fail", f"{type(exc).__name__}: {exc}",
                        "检查 GPU 驱动 / 按 torch 与 transformers 项的 fix 重装")

        # 8. 预测器冒烟：合成 20 行 CSV 真实走 configure_protein（抓 sklearn 层问题）
        try:
            with tempfile.TemporaryDirectory() as tmp:
                smoke_csv = Path(tmp) / "smoke.csv"
                _write_smoke_csv(smoke_csv)
                with contextlib.redirect_stdout(io.StringIO()):
                    configure_protein(str(smoke_csv), auto_select_predictor=False)
            add("predictor_smoke", "ok", "合成 20 行 CSV 经 configure_protein 训练成功（sklearn 层正常）")
        except Exception as exc:
            add("predictor_smoke", "fail", f"{type(exc).__name__}: {exc}",
                _pip_bound("pip install pandas scikit-learn scipy"))

    # 9. LLM 凭据（可选项：缺失只影响 Agent/顾问层，不记 fail）
    if _llm_credentials():
        add("llm_credentials", "ok", ".env 含 OPENAI_API_KEY / OPENAI_MODEL")
    else:
        add("llm_credentials", "warn",
            ".env 缺 OPENAI_API_KEY / OPENAI_MODEL——只影响 Agent/顾问层，规则计划与训练不受影响",
            "在项目根 .env 写入 OPENAI_API_KEY 与 OPENAI_MODEL")

    # 10. 虚拟环境（仅 warn 不计 essentials fail）：避免污染全局站点包
    if sys.prefix != sys.base_prefix:
        add("venv", "ok", f"运行于虚拟环境: {sys.prefix}")
    else:
        add("venv", "warn", "不在虚拟环境中运行——建议在虚拟环境中安装运行，避免污染全局站点包",
            "python -m venv .venv，然后激活（Windows: .venv\\Scripts\\activate；POSIX: source .venv/bin/activate）")

    statuses = [c["status"] for c in checks]
    summary = "FAIL" if "fail" in statuses else ("WARN" if "warn" in statuses else "PASS")
    report = {"checks": checks, "summary": summary, "state_written": False}
    # essentials 无 fail（warn 允许）→ 写入/刷新状态，与短路闭环；llm_credentials 为可选项不计入
    if summary != "FAIL":
        essentials = [c["name"] for c in checks
                      if c["name"] != "llm_credentials" and c["status"] != "fail"]
        _write_state(env_probe=env_probe, plan_summary=_doctor_plan_summary(env_probe, expected_esm, checks),
                     essentials=essentials)
        report["state_written"] = True
    return report


def _doctor_autofix(report, esm_expected=None):
    """--doctor --execute：修复可修复项后复检一次。

    可修复项：核心依赖缺失 → pip install -r requirements.txt（在包根执行，捕获输出）；
    checkpoint 缺失 → ensure_esm_checkpoint 下载。
    """
    repaired = False
    if any(c["name"] == "core_deps" and c["status"] == "fail" for c in report["checks"]):
        command = _pip_bound("pip install -r requirements.txt")
        print(f"[fix] {command}（requirements.txt 在包根）")
        proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[1]))
        if proc.returncode != 0:
            print(f"[fix] pip 返回非零: {(proc.stderr or proc.stdout or '')[-300:]}")
        repaired = True  # 成败交由复检裁决
    expected = _expected_esm(esm_expected)
    if (expected is not None and _find_esm_checkpoint(expected) is None
            and any(c["name"] == "checkpoint" and c["status"] == "fail" for c in report["checks"])):
        print(f"[fix] 下载 ESM checkpoint: {ESM_MODEL_IDS[expected]}")
        try:
            from protevo.embedding_service import ensure_esm_checkpoint
            ensure_esm_checkpoint(model_id=ESM_MODEL_IDS[expected])
            repaired = True
        except Exception as exc:
            print(f"[fix] 修复失败: {type(exc).__name__}: {exc}")
    return run_doctor(esm_expected, full=True) if repaired else report


def render_doctor(report):
    """体检中文报告：逐项 ✓/⚠/✗ + fix 命令 + 总结与状态写入说明。"""
    mark = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    lines = ["=" * 62, "protevo 环境体检报告（doctor）", "=" * 62]
    width = max((len(c["name"]) for c in report["checks"]), default=0)
    for check in report["checks"]:
        lines.append(f"  [{mark[check['status']]}] {check['name'].ljust(width)}  {check['detail']}")
        if check.get("fix"):
            lines.append(f"      fix: {check['fix']}")
    summary = report["summary"]
    note = "（存在 fail 项，未写入配置状态）" if summary == "FAIL" else "（essentials 无 fail，warn 允许）"
    lines += ["", f"总结: {summary} {note}"]
    if report.get("state_written"):
        lines.append(f"已写入配置状态: {STATE_PATH}（下次默认运行将短路；指纹随硬件变化自动失效）")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m protevo.setup_assistant",
        description="LLM 辅助环境配置助手：探测数据与环境，给出受约束的配置建议（默认 dry-run；"
                    "已配置时短路跳过，完整体检用 --doctor）")
    parser.add_argument("data_csv", nargs="?", default=None, help="变体数据 CSV 路径（可选）")
    parser.add_argument("--reference", default=None, help="显式参考序列（如 GB1: VDGV / 全长 WT）")
    parser.add_argument("--ask", default=None, help="向 LLM 提问（无凭据时仅打印规则计划）")
    parser.add_argument("--execute", action="store_true", help="实际执行安装/下载命令（默认 dry-run）")
    parser.add_argument("--doctor", action="store_true", help="环境体检（依赖/torch/平台匹配/checkpoint/冒烟）")
    parser.add_argument("--force", action="store_true", help="忽略已配置短路，强制重新走配置流程")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # Windows 重定向输出编码兜底
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.doctor:
        report = run_doctor(full=True)
        if args.execute:
            report = _doctor_autofix(report)
        print(render_doctor(report))
        return 0 if report["summary"] != "FAIL" else 1

    # 已配置短路：未给 --force/--doctor/--ask/--execute → 打印简短报告退出，不发 LLM 调用
    if not (args.force or args.ask or args.execute):
        configured, state = is_configured()
        if configured:
            print(_render_configured(state))
            return 0

    result = configure_with_llm(data_csv=args.data_csv, question=args.ask,
                                auto_execute=args.execute, reference_sequence=args.reference)
    # --execute 成功后写入状态（全部步骤 returncode 0 且无 error；无步骤也视为无事可做的成功）
    if args.execute and result["executed"] is not None and all(
            entry.get("returncode", 0) == 0 and "error" not in entry for entry in result["executed"]):
        chosen, plan = result["chosen"], result["plan"]
        _write_state(plan_summary={"variant_format": plan["variant_format"],
                                   "device": chosen["device"], "esm_model": chosen["esm_model"],
                                   "use_esm": chosen["use_esm"], "install_torch": chosen["install_torch"],
                                   "install_core": chosen["install_core"],
                                   "checkpoint_action": chosen["checkpoint_action"]},
                     essentials=[entry["command"] for entry in result["executed"]])
        print(f"已写入配置状态: {STATE_PATH}（下次默认运行将短路；指纹随硬件变化自动失效）")
    print(render_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
