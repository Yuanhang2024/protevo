"""通用化修复的回归测试：三格式支持 + 顺序无关推断 + 均匀库护栏 + 种子化切分。

对应 protein_config.py 的四处修复：
  缺口1 突变记谱格式（V39D,G40E）解析与配置
  缺口2 位点检测/WT 推断的顺序依赖（旧版采样前 100/200 条）
  缺口3 训练/验证顺序切分偏差（旧版无 shuffle，排序数据会偏置模型选择）
  缺口4 indel / 非标准残基的显式边界
合成数据与公式仅用于测试，不代表任何实验结果。
"""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from protevo.protein_config import (
    AA_ALPHABET,
    ProteinConfig,
    configure_protein,
    parse_mutation_notation,
)

PROJECT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


# ---------- 记谱解析 ----------

def test_parse_notation_basic():
    assert parse_mutation_notation("V39D,G40E") == [("V", 39, "D"), ("G", 40, "E")]
    assert parse_mutation_notation("V39D G40E") == [("V", 39, "D"), ("G", 40, "E")]
    assert parse_mutation_notation("K3R:V55A") == [("K", 3, "R"), ("V", 55, "A")]  # ProteinGym 原生冒号分隔
    assert parse_mutation_notation(":") == []
    assert parse_mutation_notation("WT") == []
    with pytest.raises(ValueError, match="V39"):
        parse_mutation_notation("V39??")
    with pytest.raises(ValueError):
        parse_mutation_notation("39D")  # 缺野生型字母


def _write_csv(path, variants, fitness, columns=("variant", "fitness")):
    pd.DataFrame({columns[0]: variants, columns[1]: fitness}).to_csv(path, index=False)
    return path


# ---------- 缺口1：mutation_notation 格式 ----------

def test_notation_config(tmp_path):
    # 合成记谱库：位点 39/40 可变，A 类残基 fitness 高
    rng = np.random.RandomState(0)
    variants, fitness = [], []
    for v0 in "ACDE":
        for v1 in "ACDE":
            variants.append(f"V39{v0},G40{v1}")
            fitness.append(3.0 * (v0 == "D") + 2.0 * (v1 == "E") + rng.rand() * 0.1)
    variants.append(":")           # 野生型行
    fitness.append(1.0)
    variants.append("V39*")        # 终止密码子 → 应被丢弃
    fitness.append(0.0)
    csv = _write_csv(tmp_path / "notation.csv", variants, fitness)

    cfg = configure_protein(str(csv))

    assert cfg.variant_format == "mutation_notation"
    assert cfg.mutable_positions == [39, 40]          # 1-indexed 蛋白编号
    assert cfg.wt_by_position == {39: "V", 40: "G"}   # WT 从记谱字母推断
    assert cfg.reference_sequence == "VG"

    # 编码正确性："V39D" → 39 位 D、40 位取 WT 的 G
    x = cfg.encode_variant("V39D")
    assert x[AA_ALPHABET.index("D")] == 1
    assert x[20 + AA_ALPHABET.index("G")] == 1
    assert x.sum() == 2
    # Agent 运行时新生成的候选（含复合突变）按同一实现编码
    x2 = cfg.encode_variant("V39D,G40E")
    assert x2[AA_ALPHABET.index("D")] == 1 and x2[20 + AA_ALPHABET.index("E")] == 1
    # 野生型 ":" → 全部取 WT
    assert cfg.encode_variant(":").tolist() == cfg.encode_variant("V39V,G40G").tolist()

    # 终止密码子变体被丢弃且不崩溃（16 记谱 + 1 WT，丢 1 条 → 16 条入库）
    assert cfg.predictor_info["validation_spearman"] > 0.8


def test_notation_explicit_reference(tmp_path):
    # 提供全长参考序列时，WT 按蛋白编号切片
    ref = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 3] for i in range(50))  # 50 aa
    variants = [f"{ref[38]}39D", f"{ref[38]}39D,{ref[39]}40E"]
    csv = _write_csv(tmp_path / "notation_ref.csv", variants, [1.0, 2.0])
    cfg = configure_protein(str(csv), reference_sequence=ref)
    assert cfg.wt_by_position == {39: ref[38], 40: ref[39]}
    assert cfg.reference_source == "provided"


# ---------- 缺口2：顺序无关的位点检测与 WT 推断 ----------

def _enriched_string_data(seed=0, n=400):
    # 富集型库：WT 残基（V/D/G/V）出现 ~30%，其余分摊 → 众数可可靠推断
    rng = np.random.RandomState(seed)
    wt, others = "VDGV", "ACDEFHIKLMNPQRSTWY"
    variants, fitness = [], []
    for _ in range(n):
        v = "".join(wt[p] if rng.rand() < 0.3 else others[rng.randint(18)] for p in range(4))
        variants.append(v)
        fitness.append(sum(c == wt[i] for i, c in enumerate(v)) + rng.rand())
    return variants, fitness


def test_order_invariance(tmp_path):
    variants, fitness = _enriched_string_data()
    df = pd.DataFrame({"variant": variants, "fitness": fitness})
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    df.to_csv(csv_a, index=False)
    shuffled.to_csv(csv_b, index=False)

    cfg_a = configure_protein(str(csv_a), auto_select_predictor=False)
    cfg_b = configure_protein(str(csv_b), auto_select_predictor=False)

    # 旧版 bug：位点检测（前 100 条采样）与 WT 推断（前 200 条采样）随行序改变
    assert cfg_a.mutable_positions == cfg_b.mutable_positions == [0, 1, 2, 3]
    assert cfg_a.reference_sequence == cfg_b.reference_sequence == "VDGV"
    assert cfg_a.reference_source == cfg_b.reference_source == "inferred"


def test_full_format_positions(tmp_path):
    # 全长序列：差异位置 = 可变位点，恒定位置不进编码
    ref = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    rng = np.random.RandomState(1)
    variants = []
    for _ in range(300):
        v = list(ref)
        if rng.rand() > 0.8:              # 位点 10、25 各 20% 突变 → 众数=参考残基
            v[10] = "A" if ref[10] != "A" else "G"
        if rng.rand() > 0.8:
            v[25] = "L" if ref[25] != "L" else "M"
        variants.append("".join(v))
    fitness = [3.0 * (v[10] != ref[10]) + np.random.RandomState(i).rand() for i, v in enumerate(variants)]
    csv = _write_csv(tmp_path / "full.csv", variants, fitness)

    cfg = configure_protein(str(csv))
    assert cfg.variant_format == "full"
    assert cfg.mutable_positions == [10, 25]
    assert cfg.reference_sequence == ref
    assert cfg.reference_source == "inferred"
    x = cfg.encode_variant(variants[0])
    assert x[AA_ALPHABET.index(variants[0][10])] == 1
    assert x[20 + AA_ALPHABET.index(variants[0][25])] == 1


# ---------- 均匀库护栏（枚举型库众数不可靠）----------

def test_uniform_library_flag(tmp_path):
    # 枚举型均匀库：众数是缺失噪声，必须标记不可靠而非静默给出错误 WT
    rng = np.random.RandomState(2)
    variants = ["".join(AA_ALPHABET[rng.randint(20)] for _ in range(4)) for _ in range(3000)]
    fitness = list(rng.rand(3000))
    csv = _write_csv(tmp_path / "uniform.csv", variants, fitness)
    cfg = configure_protein(str(csv), auto_select_predictor=False)
    assert cfg.reference_source == "inferred_uniform_unreliable"


# ---------- 缺口3：种子化随机切分 ----------

def test_split_not_sequential(tmp_path):
    # 按 fitness 降序导出的数据：旧版顺序切分会让验证集全是低适应度同组样本
    # （Spearman 退化），新版本种子化随机切分可正常评估
    rng = np.random.RandomState(3)
    rows = []
    for pos0 in "AC":
        for _ in range(80):
            v = pos0 + "".join(rng.choice(list("DEFG")) for _ in range(3))
            fit = (10.0 if pos0 == "A" else 1.0) + rng.rand()
            rows.append((v, fit))
    rows.sort(key=lambda r: -r[1])  # 模拟按 fitness 排序导出
    csv = _write_csv(tmp_path / "sorted.csv", [r[0] for r in rows], [r[1] for r in rows])

    cfg = configure_protein(str(csv))
    sp = cfg.predictor_info["validation_spearman"]
    # 旧版顺序切分下验证集全为同组样本，Spearman 退化为 ~0 或 nan
    assert np.isfinite(sp) and sp > 0.6
    # 同一数据两次配置结果可复现
    cfg2 = configure_protein(str(csv))
    assert cfg.predictor_info["validation_spearman"] == cfg2.predictor_info["validation_spearman"]


# ---------- 缺口4：显式边界 ----------

def test_indel_rejected(tmp_path):
    csv = _write_csv(tmp_path / "indel.csv", ["VDGV", "VDG", "AAAA"], [1.0, 0.5, 0.2])
    with pytest.raises(ValueError, match="indel"):
        configure_protein(str(csv), auto_select_predictor=False)


def test_nonstandard_residue_rejected(tmp_path):
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "ok.csv", variants, fitness)
    cfg = configure_protein(str(csv), auto_select_predictor=False)
    with pytest.raises(ValueError, match="非标准残基"):
        cfg.encode_variant("UDGV")  # U（硒代半胱氨酸）不在 20 标准氨基酸内


# ---------- 真实 GB1 数据回归 ----------

def test_gb1_real_regression(tmp_path):
    # 20k 子样本：格式/位点/护栏行为与显式参考路径均应正确
    df = pd.read_csv(f"{PROJECT}/data/processed/all_variants.csv").iloc[::7].reset_index(drop=True)
    csv = tmp_path / "gb1_sub.csv"
    df.to_csv(csv, index=False)

    cfg = configure_protein(str(csv))
    assert cfg.variant_format == "mutation_string"
    assert cfg.mutable_positions == [0, 1, 2, 3]
    assert cfg.reference_source == "inferred_uniform_unreliable"  # 枚举库护栏
    assert cfg.predictor_info["validation_spearman"] > 0.15

    # 显式传入 WT：报告与 Demo 的推荐用法
    cfg2 = configure_protein(str(csv), reference_sequence="VDGV")
    assert cfg2.reference_sequence == "VDGV"
    assert cfg2.reference_source == "provided"
    # 编码与 v1 协议一致：80 维 one-hot，每变体恰有 4 个 1
    x = cfg2.encode_variant("VDGV")
    assert x.shape == (80,) and x.sum() == 4
    preds = cfg2.predict_fitness(["VDGV", "VDGA"])
    assert len(preds) == 2 and all(isinstance(p, float) for p in preds)


# ---------- fitness 方向归一化（低=好）----------

def test_lower_is_better(tmp_path):
    # IC50/结合能类数据：数值越低越好。合成 fitness = -(信号)，
    # configure 后内部按 -fitness 训练，predict_fitness 翻回原始标度——
    # 排序 top（预测值最小）应是原始值最小（信号最强）的变体
    rng = np.random.RandomState(11)
    variants, fitness = [], []
    for v0 in "ACDE":
        for v1 in "ACDE":
            for v2 in "ACDE":
                variants.append(v0 + v1 + v2)
                fitness.append(-(3.0 * (v0 == "D") + 2.0 * (v1 == "D") + (v2 == "D") + rng.rand() * 0.1))
    csv = _write_csv(tmp_path / "low_better.csv", variants, fitness)

    cfg = configure_protein(str(csv), higher_is_better=False)

    assert cfg.fitness_higher_is_better is False
    assert cfg.predictor_info["fitness_higher_is_better"] is False
    # 下游统一"高=好"口径：内部 y=-fitness 下验证 Spearman 显著为正
    assert cfg.predictor_info["validation_spearman"] > 0.8
    # 返回原始标度：DDD（原始 fitness 最小、最优）预测最低，AAA 最高
    preds = cfg.predict_fitness(["DDD", "ADD", "AAA"])
    assert np.argmin(preds) == 0
    assert preds[0] < preds[1] < preds[2]


# ---------- ESM-2 候选优雅降级（.venv 无 torch）----------

def test_esm_skip_mutation_string_format(tmp_path, capsys):
    # 突变串格式无法重构全长序列 → 打印说明并跳过 ESM，RF/Ridge 选型照常完成
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "esm_str.csv", variants, fitness)

    cfg = configure_protein(str(csv), use_esm=True, esm_model_path="D:/no/such/model")

    assert "跳过 ESM 候选" in capsys.readouterr().out
    assert cfg.predictor_info["selected"] in ("random_forest", "ridge")
    assert "esm2_ridge" not in cfg.predictor_info["all_candidates"]


def test_esm_skip_without_torch(tmp_path, capsys):
    # full 格式 + use_esm=True：无 torch 环境下 EmbeddingService 导入/构建失败
    # → 优雅跳过且不崩溃，RF/Ridge 选型与预测照常
    ref = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    rng = np.random.RandomState(5)
    variants = []
    for _ in range(300):
        v = list(ref)
        if rng.rand() > 0.8:
            v[10] = "A" if ref[10] != "A" else "G"
        if rng.rand() > 0.8:
            v[25] = "L" if ref[25] != "L" else "M"
        variants.append("".join(v))
    fitness = [3.0 * (v[10] != ref[10]) + 2.0 * (v[25] != ref[25]) + np.random.RandomState(i).rand()
               for i, v in enumerate(variants)]
    csv = _write_csv(tmp_path / "esm_full.csv", variants, fitness)

    cfg = configure_protein(str(csv), use_esm=True, esm_model_path="D:/no/such/model")

    out = capsys.readouterr().out
    assert "跳过 ESM 候选" in out
    assert cfg.predictor_info["selected"] in ("random_forest", "ridge")
    assert "esm2_ridge" not in cfg.predictor_info["all_candidates"]
    assert cfg.predictor_info["validation_spearman"] > 0.6
    # 带 use_esm 的调用路径不崩溃：配置结果仍可正常预测
    preds = cfg.predict_fitness([variants[0], variants[1]])
    assert len(preds) == 2 and all(isinstance(p, float) for p in preds)


def test_esm_skip_long_sequence(tmp_path, capsys):
    # >1022 aa 的 full 格式：ESM 不可用，打印警告并跳过，其余配置正常
    ref = "ACDEFGHIKLMNPQRSTVWY" * 52  # 1040 aa > 1022
    rng = np.random.RandomState(6)
    variants = []
    for _ in range(200):
        v = list(ref)
        if rng.rand() > 0.7:
            v[10] = "P" if ref[10] != "P" else "G"
        variants.append("".join(v))
    fitness = [2.0 * (v[10] != ref[10]) + rng.rand() for v in variants]
    csv = _write_csv(tmp_path / "esm_long.csv", variants, fitness)

    cfg = configure_protein(str(csv), use_esm=True, esm_model_path="D:/no/such/model")

    out = capsys.readouterr().out
    assert "1022" in out and "跳过 ESM 候选" in out
    assert cfg.variant_format == "full"
    assert len(cfg.reference_sequence) == 1040
    assert "esm2_ridge" not in cfg.predictor_info["all_candidates"]


# ---------- ESM 候选池（GPU 档：openfold + 真实 checkpoint，无 torch 自动跳过）----------

_ESM_MODEL_PATH = "D:/ChimeraModel_Lite/esm2_t33_650M_UR50D"
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
requires_model = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_TRANSFORMERS),
    reason="GPU 档需要 torch+transformers（D:/Anaconda3/envs/openfold）",
)


@requires_model
def test_esm_pool_contains_ridge_and_mlp(tmp_path):
    # 小合成 full 数据 + 真实 checkpoint：嵌入池同时给出 esm2_ridge 与 esm2_mlp
    # （两候选共享同一嵌入矩阵、嵌入只算一次），两者 Spearman 均为有限值
    ref = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    rng = np.random.RandomState(5)
    variants = []
    for _ in range(150):
        v = list(ref)
        if rng.rand() > 0.8:
            v[10] = "A" if ref[10] != "A" else "G"
        if rng.rand() > 0.8:
            v[25] = "L" if ref[25] != "L" else "M"
        variants.append("".join(v))
    fitness = [3.0 * (v[10] != ref[10]) + 2.0 * (v[25] != ref[25]) + np.random.RandomState(i).rand()
               for i, v in enumerate(variants)]
    csv = _write_csv(tmp_path / "esm_pool.csv", variants, fitness)

    cfg = configure_protein(str(csv), use_esm=True, esm_model_path=_ESM_MODEL_PATH)

    pool = cfg.predictor_info["all_candidates"]
    assert "esm2_ridge" in pool and "esm2_mlp" in pool
    assert np.isfinite(pool["esm2_ridge"]) and np.isfinite(pool["esm2_mlp"])
    # 规格（243 组消融同规格）记入 predictor_info
    assert "hidden_layer_sizes=(256,64)" in cfg.predictor_info["esm2_mlp_spec"]
    assert cfg.predictor_info["selected"] in ("random_forest", "ridge", "esm2_ridge", "esm2_mlp")
    # 选型后预测照常（takes_sequences 适配器对 Ridge/MLP 一致）
    preds = cfg.predict_fitness([variants[0], variants[1]])
    assert len(preds) == 2 and all(isinstance(p, float) for p in preds)


# ---------- indel_notation：插入/缺失/重复库（变长序列 → 序列级预测器）----------

def _indel_library(seed=0, n=240):
    """合成 indel 库：WT 60 aa（标准字母），变体混合 sub/del/ins/dup。

    fitness = 2*重构长度 + 小噪声——composition_features 的长度维可直接学会，
    20 维组分占比是无信号维度（检验 Ridge 正则下不捣乱）。
    """
    wt = "".join(AA_ALPHABET[(i * 7 + 3) % 20] for i in range(60))
    rng = np.random.RandomState(seed)
    variants, fitness = [":"], [120.0 + rng.rand() * 0.5]  # WT 行：长度 60
    for i in range(n):
        kind = i % 4
        if kind == 0:    # 单点替换：长度不变
            pos = 11 + rng.randint(39)
            v, length = f"{wt[pos - 1]}{pos}D", 60
        elif kind == 1:  # 单残基缺失：59 aa
            pos = 11 + rng.randint(39)
            v, length = f"{wt[pos - 1]}{pos}del", 59
        elif kind == 2:  # 插入 1-5 aa：61-65 aa
            after = 11 + rng.randint(38)
            ins = "".join(AA_ALPHABET[rng.randint(20)] for _ in range(1 + rng.randint(5)))
            v = f"{wt[after - 1]}{after}_{wt[after]}{after + 1}ins{ins}"
            length = 60 + len(ins)
        else:            # 重复 1-3 aa：61-63 aa
            start = 11 + rng.randint(37)
            end = start + rng.randint(3)
            v = f"{wt[start - 1]}{start}_{wt[end - 1]}{end}dup"
            length = 60 + (end - start + 1)
        variants.append(v)
        fitness.append(2.0 * length + rng.rand() * 0.5)
    return wt, variants, fitness


def test_indel_config_composition(tmp_path):
    # 无 torch 环境下选型为 composition_ridge；长度信号可学，predict 对更长插入打更高分
    wt, variants, fitness = _indel_library()
    csv = _write_csv(tmp_path / "indel.csv", variants, fitness)

    cfg = configure_protein(str(csv), reference_sequence=wt)

    assert cfg.variant_format == "indel_notation"
    assert cfg.predictor_info["selected"] == "composition_ridge"
    assert cfg.predictor_info["encoding"] == "composition+length"
    assert cfg.predictor_info["validation_spearman"] > 0.8
    assert cfg.reference_sequence == wt and cfg.wt_full_sequence == wt
    assert cfg.reference_source == "provided"
    # mutable_positions = 观察到的锚定位点并集（1-indexed，仅信息展示）
    assert cfg.mutable_positions == sorted(cfg.mutable_positions)
    assert all(1 <= p <= 60 for p in cfg.mutable_positions)
    # 记谱串直接进预测器：更长插入 → 更高预测分（fitness 由长度主导）
    short = f"{wt[19]}20_{wt[20]}21insA"            # +1 aa
    long_ins = f"{wt[19]}20_{wt[20]}21insAAAAAAAA"  # +8 aa
    preds = cfg.predict_fitness([short, long_ins])
    assert preds[1] > preds[0]
    # 野生型行同样可预测（重构回 WT 本身）
    assert len(cfg.predict_fitness([":"])) == 1


def test_indel_requires_reference(tmp_path):
    # 缺全长 WT：1-indexed 坐标系与重构无基准 → 大声失败
    wt, variants, fitness = _indel_library(n=40)
    csv = _write_csv(tmp_path / "indel_noref.csv", variants, fitness)
    with pytest.raises(ValueError, match="全长"):
        configure_protein(str(csv))


def test_indel_invalid_row_fails_loud(tmp_path):
    # 混入非法记谱（X3del：X 与参考 3 位残基不符）→ ValueError 列出违规示例
    wt, variants, fitness = _indel_library(n=40)
    variants[7] = "X3del"
    csv = _write_csv(tmp_path / "indel_bad.csv", variants, fitness)
    with pytest.raises(ValueError) as excinfo:
        configure_protein(str(csv), reference_sequence=wt)
    msg = str(excinfo.value)
    assert "X3del" in msg and "非法" in msg  # 违规示例出现在报错里，不静默丢弃


def test_indel_encode_variant_raises(tmp_path):
    # 变长序列无固定位点 one-hot：encode_variant 在 indel 配置下显式抛错
    wt, variants, fitness = _indel_library(n=40)
    csv = _write_csv(tmp_path / "indel_enc.csv", variants, fitness)
    cfg = configure_protein(str(csv), reference_sequence=wt)
    with pytest.raises(ValueError, match="one-hot"):
        cfg.encode_variant(variants[1])


def test_indel_use_esm_graceful_without_torch(tmp_path, capsys):
    # use_esm=True 但 .venv 无 torch：打印说明并跳过，composition_ridge 照常选型与预测
    wt, variants, fitness = _indel_library()
    csv = _write_csv(tmp_path / "indel_esm.csv", variants, fitness)

    cfg = configure_protein(str(csv), reference_sequence=wt,
                            use_esm=True, esm_model_path="D:/no/such/model")

    out = capsys.readouterr().out
    assert "跳过 ESM 候选" in out
    assert cfg.predictor_info["selected"] == "composition_ridge"
    assert "esm2_ridge" not in cfg.predictor_info["all_candidates"]
    preds = cfg.predict_fitness([variants[0], variants[1]])
    assert len(preds) == 2 and all(isinstance(p, float) for p in preds)


# ---------- 候选池放开：predictor_pool / extra_candidates（先验只作推荐不作拦截）----------

def _mlp_friendly_library(seed=0, n=2000, sigma=0.1):
    """MLP 友好信号合成库：fitness 由两位点残基潜在性质值的双线性交互生成。

    fitness = 4*(x0-0.5)*(x1-0.5) + 噪声，x 为残基潜在性质值（按字母序线性赋值）。
    纯交互信号：Ridge（one-hot 线性）完全看不见（ρ≈0），RF 只能按 one-hot 格子记忆
    （每格样本有限），MLP 可学出残基嵌入逼近该交互 → full 池下 onehot_mlp 胜出。
    """
    rng = np.random.RandomState(seed)
    latent = {aa: AA_ALPHABET.index(aa) / 19.0 for aa in AA_ALPHABET}
    variants, fitness = [], []
    for _ in range(n):
        v = [AA_ALPHABET[rng.randint(20)] for _ in range(4)]
        variants.append("".join(v))
        fitness.append(4.0 * (latent[v[0]] - 0.5) * (latent[v[1]] - 0.5) + rng.randn() * sigma)
    return variants, fitness


def test_full_pool_adds_onehot_mlp(tmp_path):
    # full 模式：onehot_mlp 入池且数值有限；池模式/纳入清单/先验说明记入 predictor_info
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "pool_full.csv", variants, fitness)
    cfg = configure_protein(str(csv), predictor_pool="full")

    pool = cfg.predictor_info["all_candidates"]
    assert {"random_forest", "ridge", "onehot_mlp"} <= set(pool)
    assert np.isfinite(pool["onehot_mlp"])
    assert cfg.predictor_info["pool"]["mode"] == "full"
    assert cfg.predictor_info["pool"]["included"] == list(pool)
    assert "先验" in cfg.predictor_info["pool_notes"]


def test_full_pool_mlp_friendly_signal_wins(tmp_path):
    # 反证可赢：双线性交互信号下 full 模式选中 onehot_mlp——证明先验放开不是摆设
    # （推荐集默认不进 one-hot MLP，是推荐不是拦截；换池即可翻盘）
    variants, fitness = _mlp_friendly_library()
    csv = _write_csv(tmp_path / "mlp_win.csv", variants, fitness)

    cfg = configure_protein(str(csv), predictor_pool="full")

    assert cfg.predictor_info["selected"] == "onehot_mlp"
    assert cfg.predictor_info["validation_spearman"] > 0.8


def test_extra_candidates_append_and_record(tmp_path):
    # extra_candidates=["onehot_mlp"] 在 recommended 模式下单独追加生效；
    # 表示层面不可行（替换库请求 composition_*）与未知名字忽略并记入排除清单
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "pool_extra.csv", variants, fitness)

    cfg = configure_protein(str(csv), extra_candidates=["onehot_mlp", "composition_mlp", "bogus"])

    pool = cfg.predictor_info["all_candidates"]
    assert "onehot_mlp" in pool and np.isfinite(pool["onehot_mlp"])
    assert cfg.predictor_info["pool"]["mode"] == "recommended"
    excluded = cfg.predictor_info["pool"]["excluded"]
    assert "composition_mlp" in excluded and excluded["composition_mlp"]
    assert "bogus" in excluded and "未知" in excluded["bogus"]
    assert cfg.predictor_info["selected"] in ("random_forest", "ridge", "onehot_mlp")


def test_mutation_format_esm_excluded_recorded(tmp_path, capsys):
    # 突变串格式 + use_esm：ESM 表示层面不可行（无全长背景序列）——打印跳过且
    # 两候选记入排除清单（表示事实，非先验，两种池模式下同样排除）
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "esm_excluded.csv", variants, fitness)

    cfg = configure_protein(str(csv), use_esm=True, esm_model_path="D:/no/such/model")

    assert "跳过 ESM 候选" in capsys.readouterr().out
    excluded = cfg.predictor_info["pool"]["excluded"]
    assert set(excluded) >= {"esm2_ridge", "esm2_mlp"}
    assert "全长" in excluded["esm2_ridge"]


def test_indel_full_pool_composition_mlp(tmp_path):
    # indel full 池：composition×Ridge 与 composition×MLP 并存（无 torch 路径）
    wt, variants, fitness = _indel_library()
    csv = _write_csv(tmp_path / "indel_full_pool.csv", variants, fitness)

    cfg = configure_protein(str(csv), reference_sequence=wt, predictor_pool="full")

    pool = cfg.predictor_info["all_candidates"]
    assert {"composition_ridge", "composition_mlp"} <= set(pool)
    assert np.isfinite(pool["composition_ridge"]) and np.isfinite(pool["composition_mlp"])
    assert cfg.predictor_info["pool"]["mode"] == "full"


def test_invalid_predictor_pool_rejected(tmp_path):
    # 非法池模式显式报错（不静默回退）
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "pool_bad.csv", variants, fitness)
    with pytest.raises(ValueError, match="predictor_pool"):
        configure_protein(str(csv), predictor_pool="everything")


# ---------- 写死规则参数化收尾：variant_format / dominance_threshold / val_fraction / model_kwargs ----------
# （先验只作默认，不作拦截：四处写死规则全部开放为参数，默认值与原行为逐字节一致）

def _short_full_length_library(seed=17, n=240, seq_len=12):
    """12 aa 全长短肽库：全部位点可变——≤30 aa 且全位点多样，自动检测会误判为突变串。

    fitness = 位点 0 残基性质值 + 0.5×位点 5 性质值 + 小噪声，one-hot 线性模型可学。
    """
    rng = np.random.RandomState(seed)
    ref = "".join(AA_ALPHABET[rng.randint(20)] for _ in range(seq_len))
    latent = {aa: AA_ALPHABET.index(aa) / 19.0 for aa in AA_ALPHABET}
    variants, fitness = [], []
    for _ in range(n):
        v = list(ref)
        for p in range(seq_len):
            if rng.rand() < 0.5:
                v[p] = AA_ALPHABET[rng.randint(20)]
        variants.append("".join(v))
        fitness.append(latent[v[0]] + 0.5 * latent[v[5]] + rng.rand() * 0.1)
    return ref, variants, fitness


def test_variant_format_override_full(tmp_path):
    # 检测误判场景：12 aa 小肽全部位点可变 → 自动判为 mutation_string；
    # 显式 variant_format="full" 跳过检测走全长路径，format_source 记 "override"
    ref, variants, fitness = _short_full_length_library()
    csv = _write_csv(tmp_path / "short_full.csv", variants, fitness)

    cfg_auto = configure_protein(str(csv), auto_select_predictor=False)
    assert cfg_auto.variant_format == "mutation_string"            # 误判前提成立
    assert cfg_auto.predictor_info["format_source"] == "auto"

    cfg = configure_protein(str(csv), variant_format="full")
    assert cfg.variant_format == "full"
    assert cfg.predictor_info["format_source"] == "override"
    assert cfg.mutable_positions == list(range(12))                # 全长路径：差异位点扫描
    # 编码走全长分支：按 0-indexed 下标从完整序列取残基，12 位点 × 20 = 240 维
    x = cfg.encode_variant(variants[0])
    assert x.shape == (12 * 20,) and x.sum() == 12
    assert cfg.predictor_info["validation_spearman"] > 0.5         # 信号在新路径下可学


def test_variant_format_override_invalid(tmp_path):
    # 非法格式值：显式报错并列出四合法值（不静默回退）
    ref, variants, fitness = _short_full_length_library(n=40)
    csv = _write_csv(tmp_path / "fmt_bad.csv", variants, fitness)
    with pytest.raises(ValueError,
                       match="full.*mutation_string.*mutation_notation.*indel_notation"):
        configure_protein(str(csv), variant_format="sequence")


def _weakly_enriched_data(seed=21, n=400):
    """弱富集库：每位点 WT ~50%、指定次残基 ~25%、其余噪声均摊——
    众数/次众数 ≈ 1.9~2.3，落在 [1.2, 5.0)：阈值 1.2 判 inferred、5.0 判 unreliable。"""
    rng = np.random.RandomState(seed)
    wt, second = "VDGV", "AQWE"
    variants, fitness = [], []
    for _ in range(n):
        v = []
        for p in range(4):
            r = rng.rand()
            if r < 0.5:
                v.append(wt[p])
            elif r < 0.75:
                v.append(second[p])
            else:
                others = [aa for aa in AA_ALPHABET if aa not in (wt[p], second[p])]
                v.append(others[rng.randint(18)])
        variants.append("".join(v))
        fitness.append(sum(c == wt[p] for p, c in enumerate(v)) + rng.rand())
    return variants, fitness


def test_dominance_threshold_param(tmp_path):
    # 同一弱富集库（众数/次众数 ≈ 2.0）：阈值 1.2 → inferred，5.0 → unreliable（阈值生效）
    variants, fitness = _weakly_enriched_data()
    csv = _write_csv(tmp_path / "weak_enriched.csv", variants, fitness)

    cfg_low = configure_protein(str(csv), auto_select_predictor=False, dominance_threshold=1.2)
    assert cfg_low.reference_source == "inferred"

    cfg_high = configure_protein(str(csv), auto_select_predictor=False, dominance_threshold=5.0)
    assert cfg_high.reference_source == "inferred_uniform_unreliable"


def test_val_fraction_param(tmp_path):
    # val_fraction=0.5：训练占比 80%→50%（验证集 20%→50%），划分点移动 → 验证
    # Spearman 在新划分下可算且与默认不同；0.3 越界 → ValueError
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "valfrac.csv", variants, fitness)

    cfg_default = configure_protein(str(csv))
    cfg_half = configure_protein(str(csv), val_fraction=0.5)

    sp = cfg_half.predictor_info["validation_spearman"]
    assert np.isfinite(sp) and sp > 0.5                            # 新划分下 Spearman 可算
    assert sp != cfg_default.predictor_info["validation_spearman"]  # 划分点确已移动

    with pytest.raises(ValueError, match="val_fraction"):
        configure_protein(str(csv), val_fraction=0.3)


def _heavy_noise_library(seed=33, n=400, sigma=1.0):
    """强噪声库：弱信号（0.3×WT 命中数）+ 大噪声——Ridge alpha=10 与 alpha=100
    的验证 Spearman 可分（正则强度不同给出不同排序）。"""
    rng = np.random.RandomState(seed)
    wt, others = "VDGV", "ACDEFHIKLMNPQRSTWY"
    variants, fitness = [], []
    for _ in range(n):
        v = "".join(wt[p] if rng.rand() < 0.3 else others[rng.randint(18)] for p in range(4))
        variants.append(v)
        fitness.append(0.3 * sum(c == wt[i] for i, c in enumerate(v)) + rng.randn() * sigma)
    return variants, fitness


def test_model_kwargs_ridge_alpha(tmp_path):
    # model_kwargs={"ridge": {"alpha": 100}}：覆盖记入 model_kwargs_applied，
    # 且 ridge 行为确已改变（强噪声数据下验证 Spearman 与默认 alpha=10 不同）
    variants, fitness = _heavy_noise_library()
    csv = _write_csv(tmp_path / "mkw.csv", variants, fitness)

    cfg_default = configure_protein(str(csv))
    assert cfg_default.predictor_info["model_kwargs_applied"] == {}   # 默认无覆盖

    cfg = configure_protein(str(csv), model_kwargs={"ridge": {"alpha": 100}})
    assert cfg.predictor_info["model_kwargs_applied"] == {"ridge": {"alpha": 100}}
    assert cfg.predictor_info["all_candidates"]["ridge"] != \
        cfg_default.predictor_info["all_candidates"]["ridge"]


def test_model_kwargs_unknown_name(tmp_path):
    # 未知候选名：ValueError 列出本格式可覆盖名
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "mkw_bad.csv", variants, fitness)
    with pytest.raises(ValueError, match="random_forest.*ridge.*onehot_mlp"):
        configure_protein(str(csv), model_kwargs={"bogus": {"alpha": 1}})


def test_model_kwargs_sklearn_passthrough(tmp_path):
    # 已知名下的未知超参键不在此处拦截：透传给 sklearn，由其自行报错
    variants, fitness = _enriched_string_data()
    csv = _write_csv(tmp_path / "mkw_key.csv", variants, fitness)
    with pytest.raises((ValueError, TypeError)):
        configure_protein(str(csv), model_kwargs={"ridge": {"not_a_param": 1}})
