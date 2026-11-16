# protein_config.py — 通用蛋白进化 Agent 的自动配置层
# 用户只需提供一个 CSV，系统自动完成：检测变体格式 → 检测突变位点 → 推断参考序列 → 训练预测器
#
# 支持四种变体格式：
#   full               全长序列（每行一条完整氨基酸序列，如 GFP 库）
#   mutation_string    突变串（仅可突变位点的残基串，如 GB1 的 "VDGV"）
#   mutation_notation  突变记谱（如 "V39D,G40E" / "V39D G40E" / ":"，ProteinGym/MaveDB 惯例）
#   indel_notation     含插入/缺失/重复的记谱（"V39del" / "V39_G40insAA" / "V39dup"，protevo.variant_notation 文法），
#                      必须提供全长 reference_sequence；变长序列 one-hot 失效 → 序列级预测器（composition / ESM-2）
#
# 明确边界（显式报错，不静默处理）：
#   - 全长序列库变体长度不一致时抛错（indel 变体请改用 indel_notation 记谱 + 全长 WT 走专用路径）
#   - 20 标准氨基酸之外的残基（U/X/B/Z/O/* 等）：编码时抛错；记谱中的终止密码子 (*) 变体在配置时丢弃并计数
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from protevo.variant_notation import (
    composition_features,
    has_indel,
    parse_variant,
    reconstruct,
    validate_variant,
)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_MUTATION_TOKEN = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
# 位点众数对次众数的优势比低于此值 → 参考序列推断不可靠
# （枚举型均匀库如 GB1 的 20^4 全枚举：众数/次众数 ≈ 1.0，众数只是缺失模式的噪声；
#   富集型库野生型显著占优：优势比通常 > 3。二残基 50/50 平分同样不可靠）
_MODE_DOMINANCE_THRESHOLD = 1.5
# variant_format 覆盖参数的合法值（None = 自动检测，现行为）
_VARIANT_FORMATS = ("full", "mutation_string", "mutation_notation", "indel_notation")


def parse_mutation_notation(variant: str) -> list:
    """解析突变记谱："V39D,G40E" → [("V", 39, "D"), ("G", 40, "E")]。

    野生型记法（":"、"WT"、空串）→ []。
    三元组为 (野生型残基, 1-indexed 蛋白编号, 突变残基)。
    分隔符支持逗号/分号/空白/冒号——冒号是 ProteinGym 原生多点突变分隔符（"K3R:V55A"），
    整串恰为 ":" 时先命中上方的野生型分支，二者无歧义。
    """
    variant = str(variant).strip()
    if variant in (":", "", "WT", "wt"):
        return []
    triples = []
    for token in re.split(r"[:,;\s]+", variant):
        if not token:
            continue
        m = _MUTATION_TOKEN.match(token)
        if not m:
            raise ValueError(f"无法解析突变记谱 token: {token!r}（期望形如 V39D）")
        triples.append((m.group(1), int(m.group(2)), m.group(3)))
    return triples


@dataclass
class ProteinConfig:
    """蛋白无关的配置对象——Agent 的 5 个工具全部通过它访问蛋白信息。

    mutable_positions 语义因格式而异：
      full / mutation_string   0-indexed 序列下标
      mutation_notation        1-indexed 蛋白编号（数据库惯例）
      indel_notation           1-indexed 锚定位点并集（仅信息展示——变长序列无固定编码位）
    可变位点集合在配置时固定；配置后新出现在恒定位点的突变不会被编码捕获
    （one-hot 固定维度设计的固有边界，开放位点需求应走 ESM 嵌入路径）。
    """

    reference_sequence: str          # 参考序列（突变串/记谱格式下为各可变位点的 WT 残基串）
    mutable_positions: list          # 可突变位点
    variant_format: str = "full"     # "full" | "mutation_string" | "mutation_notation"
    predictor: object = None         # 训练好的适应度预测器
    predictor_info: dict = field(default_factory=dict)
    aa_alphabet: str = AA_ALPHABET
    reference_source: str = "provided"  # provided | inferred | inferred_uniform_unreliable
    fitness_higher_is_better: bool = True  # False（IC50/结合能等低=好）时按 -fitness 训练，predict_fitness 返回原始标度
    wt_by_position: dict = field(default_factory=dict, repr=False, compare=False)  # 记谱格式: {蛋白编号: WT 残基}
    wt_full_sequence: str = ""    # indel_notation 专属：全长 WT 序列（1-indexed 坐标系与记谱重构的基准）

    def encode_variant(self, variant: str) -> np.ndarray:
        """将变体编码为 one-hot 特征向量（自动适配可编码格式）。"""
        if self.variant_format == "indel_notation":
            # 变长序列使固定位点 one-hot 失效——indel 库只能走序列级预测器
            raise ValueError("indel 库无固定位置 one-hot 编码，请用序列级预测器")
        residues = self._extract_residues(variant)
        m = np.zeros(len(self.mutable_positions) * 20)
        for i, aa in enumerate(residues):
            if aa not in self.aa_alphabet:
                raise ValueError(
                    f"非标准残基 {aa!r}（位点 {self.mutable_positions[i]}）——仅支持 20 标准氨基酸"
                )
            m[i * 20 + self.aa_alphabet.index(aa)] = 1
        return m

    def _extract_residues(self, variant: str) -> list:
        if self.variant_format == "full":
            return [variant[p] for p in self.mutable_positions]
        if self.variant_format == "mutation_string":
            return list(variant)
        # mutation_notation：未突变的可变位点取野生型残基（与 Agent 新生成候选的记谱格式一致）
        muts = {p: mt for _, p, mt in parse_mutation_notation(variant)}
        return [muts.get(p, self.wt_by_position[p]) for p in self.mutable_positions]

    def predict_fitness(self, variants: list) -> list:
        """对变体列表预测 fitness（低=好方向时返回原始标度）。"""
        if self.variant_format == "indel_notation":
            # indel 预测器直接消费记谱串（内部 记谱→重构→特征→预测），绕过 one-hot 编码
            preds = np.asarray(self.predictor.predict(list(variants)), dtype=float)
            if not self.fitness_higher_is_better:
                preds = -preds  # 训练用 y=-fitness，翻回原始标度返回
            return preds.tolist()
        if getattr(self.predictor, "takes_sequences", False):
            # ESM 预测器直接消费全长序列，绕过 one-hot 编码
            preds = np.asarray(self.predictor.predict(list(variants)), dtype=float)
        else:
            X = np.array([self.encode_variant(v) for v in variants])
            preds = np.asarray(self.predictor.predict(X), dtype=float)
        if not self.fitness_higher_is_better:
            preds = -preds  # 训练用 y=-fitness，翻回原始标度返回
        return preds.tolist()


class _ESM2EmbeddingPredictor:
    """ESM-2 嵌入 + sklearn 回归器（Ridge / MLP）的推理适配器：predict() 直接接受全长序列列表。"""

    takes_sequences = True  # predict_fitness 据此跳过 one-hot 编码

    def __init__(self, service, model):
        self._service = service
        self._model = model

    def predict(self, sequences):
        return self._model.predict(self._service.encode(list(sequences)))


# model_kwargs 可覆盖的候选名（按路径；先验只作默认不作拦截——超参同样可整体改写）
# 替换格式（full / mutation_string / mutation_notation）的池矩阵可构造者
_SUBSTITUTION_MODEL_NAMES = ("random_forest", "ridge", "onehot_mlp", "esm2_ridge", "esm2_mlp")
# indel 格式的池矩阵可构造者
_INDEL_MODEL_NAMES = ("composition_ridge", "composition_mlp", "esm2_ridge", "esm2_mlp")


def _check_model_kwargs(model_kwargs, allowed):
    """校验 model_kwargs 的候选名均为本路径可构造者，未知名显式报错（列出可覆盖名）。"""
    if not model_kwargs:
        return
    unknown = sorted(set(model_kwargs) - set(allowed))
    if unknown:
        raise ValueError(f"model_kwargs 含未知候选名 {unknown}；本格式可覆盖: {'/'.join(allowed)}")


def _make_model(model_cls, name, base_kwargs, model_kwargs=None):
    """统一模型构造入口：base_kwargs 为该候选的默认超参，model_kwargs 按候选名覆盖合并。

    如 model_kwargs={"ridge": {"alpha": 1.0}} → ridge 构造时 alpha 改 1.0，其余默认超参
    不变；已知名下的未知超参键不在此处拦截，直接透传给 sklearn 由其自行报错。
    model_kwargs 为 None（默认）时不覆盖——构造结果与直接 model_cls(**base_kwargs) 一致。
    """
    kwargs = dict(base_kwargs)
    if model_kwargs and name in model_kwargs:
        kwargs.update(model_kwargs[name])
    return model_cls(**kwargs)


def _applied_model_kwargs(model_kwargs, constructed_names):
    """筛出实际作用到已构造候选的覆盖（未入池/未构建成功的候选不算生效），供 predictor_info 记录。"""
    return {n: model_kwargs[n] for n in (model_kwargs or {}) if n in constructed_names}


def _mlp_candidate(model_kwargs=None, name="mlp"):
    """MLP 候选的统一规格（与 243 组预测器消融同规格）：esm2_mlp / onehot_mlp / composition_mlp 共用。

    name 为调用侧的候选名（onehot_mlp / composition_mlp / esm2_mlp），据此应用 model_kwargs
    的对应超参覆盖（三种 MLP 共用本规格，按各自名字分别覆盖）；默认无覆盖时与写死规格一致。
    """
    return _make_model(
        MLPRegressor, name,
        dict(hidden_layer_sizes=(256, 64), alpha=1e-4, early_stopping=True,
             random_state=42, max_iter=500),
        model_kwargs,
    )


# predictor_info 里的 esm2_mlp 规格说明（供审计/复现，无需读源码）
_ESM2_MLP_SPEC = ("MLPRegressor(hidden_layer_sizes=(256,64), alpha=1e-4, early_stopping=True, "
                  "random_state=42, max_iter=500)——与 243 组消融同规格，与 esm2_ridge 共享同一嵌入矩阵")


def _esm2_mlp_spec_text(model_kwargs=None):
    """esm2_mlp 的实际构造规格说明：合并 model_kwargs['esm2_mlp'] 覆盖后的真实超参。

    无覆盖时逐字返回默认规格常量（回归不变）；有覆盖时按合并结果渲染，
    保证 spec 与 model_kwargs_applied 描述的是同一组构造参数。"""
    override = (model_kwargs or {}).get("esm2_mlp")
    if not override:
        return _ESM2_MLP_SPEC
    merged = dict(hidden_layer_sizes=(256, 64), alpha=1e-4, early_stopping=True,
                  random_state=42, max_iter=500)
    merged.update(override)

    def _render(value):
        if isinstance(value, (tuple, list)):  # 层结构按 sklearn repr 风格渲染：(256,64) / (8,)
            inner = ",".join(repr(v) for v in value)
            return f"({inner},)" if len(value) == 1 else f"({inner})"
        return repr(value)

    body = ", ".join(f"{key}={_render(value)}" for key, value in merged.items())
    return (f"MLPRegressor({body})——超参经 model_kwargs['esm2_mlp'] 覆盖，"
            "与 esm2_ridge 共享同一嵌入矩阵")

# 候选池先验说明：先验只作推荐不作拦截，写入 predictor_info["pool_notes"]
_POOL_NOTES_RECOMMENDED = (
    "recommended 推荐集 = 内证与文献校准的支持面：one-hot 系 {RF, Ridge}——243 组消融中 "
    "one-hot 特征上 MLP 三连败（2K 0.344 / 10K 0.309 / 50K 0.556，vs RF 0.543/0.589/0.655），"
    "故 MLP 不默认进 one-hot/composition 池；ESM 系 {Ridge, MLP}（消融总榜第二组合）。"
    "先验仅作推荐，可用 predictor_pool='full' 或 extra_candidates 放开，不构成拦截。"
)
_POOL_NOTES_FULL = (
    "full 池 = 该格式下全部表示层面可行组合；先验不参与拦截，仅作推荐说明"
    "（构成理由见 recommended 注）。表示层面不可行的组合仍排除并记于 pool['excluded']。"
)


def _expand_candidate_pool(candidates, predictor_pool, extra_candidates, mlp_name, make_mlp_entry, blocked):
    """按 predictor_pool / extra_candidates 扩充候选池，返回排除清单 {名: 原因}。

    mlp_name / make_mlp_entry：该格式表示层上 MLP 候选的名字与构造器（替换库为裸
    sklearn 模型，indel 为 (模型, 序列级特征函数)）；blocked：本格式表示层面不可行
    （或需 use_esm 独立开启）的候选名 → 原因。先验只作推荐不作拦截——full 即入池，
    推荐理由见 _POOL_NOTES_*；extra_candidates 中已在池中的名字幂等跳过。
    """
    excluded = {}
    if predictor_pool == "full":
        candidates[mlp_name] = make_mlp_entry()
    for name in extra_candidates or []:
        if name in candidates:
            continue  # 已在池中（幂等）
        if name == mlp_name:
            candidates[mlp_name] = make_mlp_entry()
        elif name in blocked:
            excluded[name] = blocked[name]
        else:
            excluded[name] = f"未知候选名（该格式可用：{'/'.join([*candidates, mlp_name])}）"
    return excluded


class _NotationSeqPredictor:
    """indel 库的序列级预测器适配器：predict() 直接接受记谱串列表。

    内部管线 记谱 → reconstruct(wt) 全长重构 → featurize 序列特征 → model.predict。
    featurize 为序列级特征函数（list[str] → ndarray）：composition_features（CPU 基线）
    或 ESM 嵌入（EmbeddingService.encode），两者对上层完全同构。
    """

    def __init__(self, wt: str, featurize, model):
        self._wt = wt
        self._featurize = featurize
        self._model = model

    def predict(self, notations: list) -> np.ndarray:
        seqs = [reconstruct(str(v), self._wt) for v in notations]
        return self._model.predict(self._featurize(seqs))


def _build_esm_candidate(train_seqs, val_seqs, y_tr, y_val, model_path, device, cache_dir=None,
                         model_kwargs=None):
    """惰性构建 ESM-2 嵌入候选（esm2_ridge + esm2_mlp）。返回 (candidates, err_msg)。

    嵌入只算一次：两候选共享同一嵌入矩阵（GPU 推理是唯一大头），MLP 只多一次
    CPU 侧训练。candidates 形如 {"esm2_ridge": (predictor, spearman),
    "esm2_mlp": (predictor, spearman)}。model_kwargs 按候选名（esm2_ridge / esm2_mlp）
    覆盖超参，透传自 configure_protein。
    EmbeddingService 在函数体内导入（.venv 无 torch 时本模块仍可正常 import）；
    导入失败 / torch 缺失 / CUDA 不可用等任何异常都不抛出——ESM 是可选增强，
    返回 err_msg 由调用方打印并跳过（两候选一并跳过）。
    """
    try:
        from protevo.embedding_service import EmbeddingService
    except ImportError as e:
        return None, f"EmbeddingService 导入失败（{e}），跳过 ESM 候选"
    try:
        service = EmbeddingService(model_path=model_path, device=device, cache_dir=cache_dir)
        X_tr = service.encode(list(train_seqs))
        X_val = service.encode(list(val_seqs))
        candidates = {}
        ridge = _make_model(Ridge, "esm2_ridge", {"alpha": 10}, model_kwargs).fit(X_tr, y_tr)
        candidates["esm2_ridge"] = (
            _ESM2EmbeddingPredictor(service, ridge),
            spearmanr(y_val, ridge.predict(X_val)).statistic,
        )
        mlp = _mlp_candidate(model_kwargs, "esm2_mlp").fit(X_tr, y_tr)
        candidates["esm2_mlp"] = (
            _ESM2EmbeddingPredictor(service, mlp),
            spearmanr(y_val, mlp.predict(X_val)).statistic,
        )
        return candidates, None
    except Exception as e:
        return None, f"ESM 候选构建失败（{type(e).__name__}: {e}），跳过 ESM 候选"


def _build_esm_seq_candidate(train_seqs, val_seqs, y_tr, y_val, model_path, device, cache_dir=None,
                             model_kwargs=None):
    """惰性构建 ESM-2 序列级候选（esm2_ridge + esm2_mlp；仿 _build_esm_candidate，但不做预测器包装）。

    返回 (candidates, err_msg)：candidates 形如 {"esm2_ridge": (model, featurize, spearman),
    "esm2_mlp": (model, featurize, spearman)}，model 是作用于嵌入的裸 sklearn 回归器，
    featurize = EmbeddingService.encode（list[str] → 嵌入矩阵），二者交由
    _NotationSeqPredictor 与 composition 特征统一组合。嵌入只算一次、两候选共享。
    model_kwargs 按候选名（esm2_ridge / esm2_mlp）覆盖超参，透传自 configure_protein。
    任何失败都不抛出——返回 err_msg 由调用方打印并跳过（两候选一并跳过；
    ESM 是可选增强）。
    """
    try:
        from protevo.embedding_service import EmbeddingService
    except ImportError as e:
        return None, f"EmbeddingService 导入失败（{e}），跳过 ESM 候选"
    try:
        service = EmbeddingService(model_path=model_path, device=device, cache_dir=cache_dir)
        X_tr = service.encode(list(train_seqs))
        X_val = service.encode(list(val_seqs))
        candidates = {}
        ridge = _make_model(Ridge, "esm2_ridge", {"alpha": 10}, model_kwargs).fit(X_tr, y_tr)
        candidates["esm2_ridge"] = (
            ridge, service.encode, spearmanr(y_val, ridge.predict(X_val)).statistic,
        )
        mlp = _mlp_candidate(model_kwargs, "esm2_mlp").fit(X_tr, y_tr)
        candidates["esm2_mlp"] = (
            mlp, service.encode, spearmanr(y_val, mlp.predict(X_val)).statistic,
        )
        return candidates, None
    except Exception as e:
        return None, f"ESM 候选构建失败（{type(e).__name__}: {e}），跳过 ESM 候选"


def _scan_residues(variants, seq_len):
    """全量扫描每个位置的残基集合。

    修复旧版仅采样前 100 条的顺序依赖 bug：若 CSV 按 fitness 排序，
    采样会漏检后期才出现突变的位置。
    """
    s = pd.Series(variants)
    return [set(s.str[p]) for p in range(seq_len)]


def _detect_format_and_positions(variants, variant_format=None):
    """确定性检测变体格式 + 全量扫描可变位点。返回 (format, mutable_positions, seq_len)。

    variant_format 给定合法值之一时跳过自动检测直接采用（检测误判时的人工覆盖：
    如 12 aa 小肽全部位点可变会被自动判为突变串，实为全长短肽库时显式指定 "full"）。
    full / mutation_string 仍做等长校验与全量位点扫描（mutation_string 的可变位点
    固定为全部下标，full 仍为差异位置）；两种记谱格式只返回格式名，位点由各自分支解析。

    自动检测的判定规则：
      1. 变体含数字 → 记谱类（氨基酸序列永不含数字）；采样中任一变体含 indel token
         （del/ins/dup）→ indel_notation，否则为纯替换记谱，位置由记谱解析得到
      2. 变体长度不一致 → 抛错（indel 变体应改用 indel_notation 记谱 + 全长 WT）
      3. 短串（≤30 aa）且每个位置都观察到多种残基 → 突变串
      4. 其余 → 全长序列，可变位点 = 全量扫描的差异位置
    """
    if variant_format in ("mutation_notation", "indel_notation"):
        return variant_format, None, None
    if variant_format is None:
        for v in variants[:1000]:
            if any(c.isdigit() for c in str(v)):
                if any(has_indel(str(x)) for x in variants[:1000]):
                    return "indel_notation", None, None
                parse_mutation_notation(v)  # 提前校验可解析性，给出明确报错
                return "mutation_notation", None, None
    lengths = {len(str(v)) for v in variants}
    if len(lengths) > 1:
        raise ValueError(
            f"变体长度不一致 {sorted(lengths)[:5]}——全长序列库仅支持等长替换；"
            f"indel 变体请改用 indel_notation 记谱（如 V39del）并提供全长 reference_sequence"
        )
    seq_len = lengths.pop()
    residues_by_pos = _scan_residues(variants, seq_len)
    varying = [p for p, residues in enumerate(residues_by_pos) if len(residues) > 1]
    if variant_format == "mutation_string":
        return "mutation_string", list(range(seq_len)), seq_len
    if variant_format == "full":
        return "full", varying, seq_len
    if seq_len <= 30 and len(varying) == seq_len:
        return "mutation_string", list(range(seq_len)), seq_len
    return "full", varying, seq_len


def _infer_reference_mode(variants, seq_len, dominance_threshold=_MODE_DOMINANCE_THRESHOLD):
    """全量众数推断参考序列。返回 (reference, source)。

    dominance_threshold：可靠性阈值——任一位点 众数计数/次众数计数 低于该值 →
    众数被视为均匀库噪声而非野生型，标记 inferred_uniform_unreliable（默认 1.5，
    语义见 _MODE_DOMINANCE_THRESHOLD 注释；富集程度不同的库可经 configure_protein 调整）。

    source 三种取值：
      provided                     用户显式传入
      inferred                     富集型库，众数显著占优（优势比 ≥ 阈值），可信
      inferred_uniform_unreliable  众数不占优（枚举库/平分位点），是噪声而非野生型
    """
    s = pd.Series(variants)
    ref, unreliable = [], False
    for p in range(seq_len):
        counts = s.str[p].value_counts()
        ref.append(counts.index[0])
        if len(counts) > 1 and counts.iloc[0] < counts.iloc[1] * dominance_threshold:
            unreliable = True
    return "".join(ref), ("inferred_uniform_unreliable" if unreliable else "inferred")


def _configure_indel_library(
    variants, fitness, reference_sequence, mutable_positions,
    auto_select_predictor, random_state, higher_is_better,
    use_esm, esm_model_path, esm_device, esm_cache_dir,
    predictor_pool, extra_candidates, val_fraction, model_kwargs, format_source,
) -> ProteinConfig:
    """indel（插入/缺失/重复）库的专用配置路径。

    变长序列使固定位点 one-hot 失效：以全长 WT 为坐标基准重构每条记谱的全长序列，
    训练序列级预测器（CPU 基线 composition+length；可选 ESM-2 嵌入）。
    选型协议与替换库路径一致：种子化随机切分 + 验证 Spearman + higher_is_better 归一化；
    val_fraction / model_kwargs / format_source 透传自 configure_protein（划分比例与
    超参覆盖同样适用于本路径）。
    """
    # 全长 WT 必填：1-indexed 坐标核对（validate_variant）与序列重构（reconstruct）都依赖它
    if reference_sequence is None:
        raise ValueError("indel 库必须提供全长 reference_sequence——1-indexed WT 坐标系与序列重构依赖它")
    wt = str(reference_sequence)

    # model_kwargs 未知候选名显式报错（本路径可构造者与池矩阵一致）
    _check_model_kwargs(model_kwargs, _INDEL_MODEL_NAMES)

    # 全量校验（大声失败，不静默丢弃）：列出至多 3 条违规示例
    invalid = [(v, err) for v in variants if (err := validate_variant(str(v), wt)) is not None]
    if invalid:
        examples = "；".join(f"{v!r}（{err}）" for v, err in invalid[:3])
        raise ValueError(f"indel 库含 {len(invalid)} 条对参考序列非法的记谱，违规示例: {examples}")

    # 记谱 → 变长全长序列（后续特征与预测都以序列为输入）
    seqs = [reconstruct(str(v), wt) for v in variants]

    # 观察到的锚定位点并集（1-indexed 蛋白编号，仅作信息展示——变长序列无固定编码位）：
    # sub 触及 pos；del 触及 start..end；ins 锚定于 after 与 after+1 之间；dup 触及 start..end+1
    positions = set()
    for v in variants:
        for m in parse_variant(str(v)):
            if m["type"] == "sub":
                positions.add(m["pos"])
            elif m["type"] == "del":
                positions.update(range(m["start"], m["end"] + 1))
            elif m["type"] == "ins":
                positions.update((m["after"], m["after"] + 1))
            else:  # dup
                positions.update(range(m["start"], m["end"] + 2))
    mutable = sorted(positions)
    if mutable_positions is not None:
        mutable = mutable_positions

    config = ProteinConfig(
        reference_sequence=wt,
        mutable_positions=mutable,
        variant_format="indel_notation",
        reference_source="provided",
        wt_full_sequence=wt,
        fitness_higher_is_better=higher_is_better,
    )

    # fitness 方向归一化（与替换库路径同协议：内部按"高=好"训练，predict_fitness 返回原始标度）
    if not higher_is_better:
        fitness = -fitness

    # --- 种子化随机划分（与替换库路径同一协议；比例经 val_fraction 参数化）---
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(seqs))
    split = int(len(seqs) * val_fraction)
    tr, va = idx[:split], idx[split:]
    seqs_tr, seqs_va = [seqs[i] for i in tr], [seqs[i] for i in va]
    y_tr, y_va = fitness[tr], fitness[va]

    # --- 训练序列级预测器 ---
    if auto_select_predictor:
        # 候选池：name → (回归模型, 序列级特征函数 list[str] → ndarray)
        # 推荐集（recommended，默认）= CPU 基线 composition×Ridge；full = 放开 composition×MLP
        # （先验只作推荐不作拦截，见 _POOL_NOTES_*）。one-hot 系在本格式表示层面不可行
        # （变长序列无固定位置坐标），经 extra_candidates 请求时记原因排除
        candidates = {
            "composition_ridge": (_make_model(Ridge, "composition_ridge", {"alpha": 10}, model_kwargs),
                                  composition_features),
        }  # CPU 基线：20 组分占比 + 长度
        pool_excluded = _expand_candidate_pool(
            candidates, predictor_pool, extra_candidates,
            mlp_name="composition_mlp",
            make_mlp_entry=lambda: (_mlp_candidate(model_kwargs, "composition_mlp"), composition_features),
            blocked={
                "random_forest": "表示层面不可行：变长序列无固定位置 one-hot 坐标",
                "ridge": "表示层面不可行：变长序列无固定位置 one-hot 坐标",
                "onehot_mlp": "表示层面不可行：变长序列无固定位置 one-hot 坐标",
                "esm2_ridge": "ESM 候选由 use_esm=True 开启（需嵌入环境）",
                "esm2_mlp": "ESM 候选由 use_esm=True 开启（需嵌入环境）",
            },
        )
        scores = {}
        for name, (model, featurize) in candidates.items():
            model.fit(featurize(seqs_tr), y_tr)
            scores[name] = spearmanr(y_va, model.predict(featurize(seqs_va))).statistic
        best_name = max(scores, key=scores.get)

        # ESM-2 嵌入候选（可选增强：任何不可用都打印说明并优雅跳过；两候选一并处理）
        if use_esm:
            if max(len(s) for s in seqs) > 1022:
                print("⚠️  重构序列长度超过 1022（ESM-2 上限），跳过 ESM 候选")
            else:
                esm_cands, err = _build_esm_seq_candidate(
                    seqs_tr, seqs_va, y_tr, y_va, esm_model_path, esm_device, esm_cache_dir,
                    model_kwargs=model_kwargs,
                )
                if err:
                    print(f"⚠️  {err}")
                else:
                    for esm_name, (esm_model, featurize_e, sp_e) in esm_cands.items():
                        candidates[esm_name] = (esm_model, featurize_e)
                        scores[esm_name] = sp_e
                        if sp_e > scores[best_name]:
                            best_name = esm_name

        model, featurize = candidates[best_name]
        config.predictor = _NotationSeqPredictor(wt, featurize, model)
        config.predictor_info = {
            "selected": best_name,
            "validation_spearman": round(scores[best_name], 4),
            "random_state": random_state,
            "fitness_higher_is_better": higher_is_better,
            "encoding": "composition+length" if best_name == "composition_ridge" else "esm2",
            "all_candidates": {n: round(s, 4) for n, s in scores.items()},
            "format_source": format_source,
            "model_kwargs_applied": _applied_model_kwargs(model_kwargs, candidates),
        }
        if "esm2_mlp" in scores:
            # spec 反映实际构造参数（合并 model_kwargs['esm2_mlp'] 覆盖后的真实超参）
            config.predictor_info["esm2_mlp_spec"] = _esm2_mlp_spec_text(model_kwargs)
        config.predictor_info["pool"] = {
            "mode": predictor_pool,
            "included": list(config.predictor_info["all_candidates"].keys()),
            "excluded": pool_excluded,
        }
        config.predictor_info["pool_notes"] = (
            _POOL_NOTES_RECOMMENDED if predictor_pool == "recommended" else _POOL_NOTES_FULL
        )
    else:
        model = _make_model(Ridge, "composition_ridge", {"alpha": 10}, model_kwargs).fit(
            composition_features(seqs), fitness
        )
        config.predictor = _NotationSeqPredictor(wt, composition_features, model)
        config.predictor_info = {
            "selected": "composition_ridge",
            "fitness_higher_is_better": higher_is_better,
            "encoding": "composition+length",
            "format_source": format_source,
            "model_kwargs_applied": _applied_model_kwargs(model_kwargs, ["composition_ridge"]),
        }

    print(f"✅ Protein configured:")
    print(f"   Format: {config.variant_format}")
    print(f"   Mutable positions ({len(mutable)}): {mutable[:12]}{'...' if len(mutable) > 12 else ''}"
          f"（1-indexed 锚点并集，仅信息展示）")
    print(f"   Reference (provided): {wt[:20]}{'...' if len(wt) > 20 else ''} ({len(wt)}aa)")
    print(f"   Fitness direction: "
          f"{'higher is better' if higher_is_better else 'lower is better（内部按 -fitness 训练，predict_fitness 返回原始标度）'}")
    print(f"   Predictor: {config.predictor_info.get('selected', 'N/A')} "
          f"(val ρ={config.predictor_info.get('validation_spearman', 'N/A')}, "
          f"encoding={config.predictor_info.get('encoding', 'N/A')})")
    return config


def configure_protein(
    data_path: str,
    variant_col: str = None,
    fitness_col: str = None,
    reference_sequence: str = None,
    mutable_positions: list = None,
    auto_select_predictor: bool = True,
    random_state: int = 42,
    higher_is_better: bool = True,
    use_esm: bool = False,
    esm_model_path: str = r"D:/ChimeraModel_Lite/esm2_t33_650M_UR50D",
    esm_device: str = "auto",
    esm_cache_dir: str = None,
    predictor_pool: str = "recommended",
    extra_candidates: list = None,
    variant_format: str = None,
    dominance_threshold: float = _MODE_DOMINANCE_THRESHOLD,
    val_fraction: float = 0.8,
    model_kwargs: dict = None,
) -> ProteinConfig:
    """
    从数据文件自动配置蛋白进化 Agent。

    最小用法：
        config = configure_protein("my_data.csv")

    带选项（枚举型均匀库必须显式传 reference_sequence，否则参考序列不可靠）：
        config = configure_protein(
            "gb1_data.csv",
            variant_col="Variants",
            fitness_col="Fitness",
            reference_sequence="VDGV",   # 突变串/记谱格式传 WT 残基串或全长序列均可
        )

    mutable_positions 覆盖语义：full/mutation_string 传 0-indexed 下标，
    mutation_notation 传 1-indexed 蛋白编号。

    自动完成：
        1. 检测变体列和 fitness 列（如未指定）
        2. 检测变体格式（full / mutation_string / mutation_notation / indel_notation）
        3. 检测突变位点（全量扫描，无顺序依赖）
        4. 推断参考序列（全量众数 + 均匀库可靠性标记）
        5. 种子化随机划分，训练并选择最优预测器

    indel_notation（插入/缺失/重复记谱，如 "V39del" / "V39_G40insAA" / "V39dup"）：
        必须传全长 reference_sequence（1-indexed 坐标核对与序列重构的基准）。
        记谱按 WT 重构为变长全长序列后训练序列级预测器（composition_ridge /
        可选 esm2_ridge、esm2_mlp），encode_variant 不可用（无固定位置 one-hot）。

    higher_is_better=False：IC50/结合能等"越低越好"的数据。内部按 -fitness 训练，
    模型选择 / 验证 Spearman / Agent 工具统一按"高=好"工作，predict_fitness 返回原始标度。

    use_esm=True：候选池额外加入 ESM-2 嵌入候选 esm2_ridge（Ridge）与 esm2_mlp（MLP，
    与 243 组预测器消融同规格；两候选共享同一嵌入矩阵，嵌入只算一次）（需
    protevo.embedding_service + torch，full 格式与 indel_notation 重构序列可用——
    突变串/记谱格式无法重构全长序列）。不可用时打印说明并一并跳过，不抛错。

    esm_cache_dir：ESM 嵌入磁盘缓存目录（透传 EmbeddingService(cache_dir=...)，
    npz 向量 + json 索引，跨进程复用）；None 关闭。run_cross_protein 复测用它提速。

    predictor_pool：候选池模式（先验只作推荐不作拦截）。
      "recommended"（默认）= 内证与文献校准的推荐集：替换库 one-hot×{RF, Ridge}、
      indel composition×Ridge、use_esm 时 ESM×{Ridge, MLP}（推荐理由记于
      predictor_info["pool_notes"]）；
      "full" = 放开该格式下全部表示层面可行组合（替换库另加 onehot_mlp、
      indel 另加 composition_mlp）。表示层面不可行的组合（indel 上的 one-hot 系、
      突变串/记谱上的 ESM/composition）在两种模式下都排除，原因记于
      predictor_info["pool"]["excluded"]——这是表示事实，不是先验。

    extra_candidates：按名向候选池细粒度追加（如 ["onehot_mlp"]，与 predictor_pool
    可叠加）；已在池中幂等跳过，表示层面不可行或未知的名字忽略并记入 excluded。

    variant_format：覆盖变体格式的自动检测（None = 自动检测，现行为）。检测误判时可
    显式指定四合法值之一——"full" | "mutation_string" | "mutation_notation" |
    "indel_notation"——跳过检测直接采用（典型场景：12 aa 小肽全部位点可变会被自动
    判为突变串，实为全长短肽库时指定 "full"）。覆盖后仍走该格式的全部位点/参考/
    编码分支；predictor_info["format_source"] 记录 "auto" | "override"；非法值 ValueError。

    dominance_threshold：参考序列众数推断的可靠性阈值（默认 1.5，见
    _MODE_DOMINANCE_THRESHOLD）。任一可变位点的 众数计数/次众数计数 低于该值 →
    众数被视为均匀库噪声，参考序列标记 inferred_uniform_unreliable。富集程度较低
    的库可调低（如 1.2），要求更严格占优可调高（如 5.0）。

    val_fraction：训练/验证划分比例——前 val_fraction 比例样本训练、其余验证
    （默认 0.8，即原先写死的 80/20）。需 0.5 ≤ val_fraction < 1.0，否则 ValueError；
    random_state 语义不变（同一 seed 下划分可复现）。

    model_kwargs：按候选名覆盖预测器超参（先验只作默认不作拦截——超参同样可改写）：
        configure_protein(csv, model_kwargs={
            "ridge": {"alpha": 1.0},
            "random_forest": {"n_estimators": 200},
            "esm2_mlp": {"hidden_layer_sizes": (128,)},
        })
    可覆盖名 = 该格式池矩阵实际可构造者：替换库 random_forest / ridge / onehot_mlp /
    esm2_ridge / esm2_mlp（三种 MLP 候选共用同一规格，按各自名字覆盖）；indel 库
    composition_ridge / composition_mlp / esm2_ridge / esm2_mlp。未知候选名
    ValueError 列出可覆盖名；已知名下的未知超参键不拦截，直接透传给 sklearn 由其
    自行报错；实际生效的覆盖（候选确实入池并构建）记于
    predictor_info["model_kwargs_applied"]，默认空字典。
    """
    df = pd.read_csv(data_path)
    if df.empty:
        # 空数据显式拒绝：0 行此前会在深层报不可读的 KeyError 'pop from an empty set'
        raise ValueError(f"CSV 无有效数据：{data_path} 含 0 行数据（只有表头或完全为空）")

    # 池模式非法值显式报错（不静默回退）
    if predictor_pool not in ("recommended", "full"):
        raise ValueError(f"predictor_pool 需为 'recommended' 或 'full'，收到 {predictor_pool!r}")
    # 覆盖参数的非法值同样显式报错（先验只作默认，不作拦截——但给错值要大声失败）
    if variant_format is not None and variant_format not in _VARIANT_FORMATS:
        raise ValueError(
            f"variant_format 需为 {'/'.join(_VARIANT_FORMATS)} 之一（或 None 自动检测），"
            f"收到 {variant_format!r}"
        )
    if not 0.5 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction 需在 [0.5, 1.0) 内，收到 {val_fraction!r}")

    # --- 1. 自动检测列名 ---
    if variant_col is None:
        for candidate in ["variant", "Variants", "mutations", "seq", "sequence"]:
            if candidate in df.columns:
                variant_col = candidate
                break
        else:
            variant_col = df.select_dtypes(include=["object"]).columns[0]

    if fitness_col is None:
        for candidate in ["fitness", "Fitness", "score", "activity", "value"]:
            if candidate in df.columns:
                fitness_col = candidate
                break
        else:
            fitness_col = df.select_dtypes(include=[np.number]).columns[0]

    variants = df[variant_col].astype(str).values
    fitness = df[fitness_col].values
    # 空/NaN fitness 显式拒绝（DSH 组合压力测试发现）：NaN 进入训练会静默产出垃圾
    # 验证分数（val ρ=-999），此前报错形态为不可读的深层 IndexError/类型错误
    if len(fitness) == 0:
        raise ValueError(f"CSV 无有效数据：{data_path} 的 {fitness_col} 列为空（0 行）")
    nan_mask = pd.isna(fitness)
    if nan_mask.any():
        raise ValueError(
            f"CSV 无有效数据：fitness 列 '{fitness_col}' 含 {int(nan_mask.sum())} 个缺失/NaN 值"
            f"（共 {len(fitness)} 行，{data_path}）——请清洗缺失行后重试")

    # --- 2. 格式检测与位点扫描（variant_format 显式指定时跳过检测直接采用）---
    format_source = "override" if variant_format is not None else "auto"
    variant_format, mutable, seq_len = _detect_format_and_positions(variants, variant_format)
    if variant_format == "indel_notation":
        # indel（插入/缺失/重复）库：变长序列 one-hot 失效，走序列级预测器专用路径
        return _configure_indel_library(
            variants, fitness, reference_sequence, mutable_positions,
            auto_select_predictor, random_state, higher_is_better,
            use_esm, esm_model_path, esm_device, esm_cache_dir,
            predictor_pool, extra_candidates, val_fraction, model_kwargs, format_source,
        )
    wt_by_position, ref_source = {}, "provided"

    if variant_format == "mutation_notation":
        # 丢弃含终止密码子 (*) 的变体（在 20 氨基酸编码中无意义）
        mask = ~pd.Series(variants).str.contains(r"\*", regex=True).values
        n_dropped = int((~mask).sum())
        if n_dropped:
            print(f"⚠️  丢弃 {n_dropped} 条含终止密码子 (*) 的变体")
            variants, fitness = variants[mask], fitness[mask]
        parsed = [parse_mutation_notation(v) for v in variants]
        positions = sorted({p for triples in parsed for _, p, _ in triples})
        if reference_sequence is not None:
            # 按 1-indexed 蛋白编号从全长参考序列切片
            wt_by_position = {p: reference_sequence[p - 1] for p in positions}
        else:
            # WT 残基来自记谱自身的野生型字母（全数据取众数，容错个别笔误）
            votes = {}
            for triples in parsed:
                for wt, p, _ in triples:
                    votes.setdefault(p, Counter())[wt] += 1
            wt_by_position = {p: votes[p].most_common(1)[0][0] for p in positions}
            ambiguous = [p for p in positions if len(votes[p]) > 1]
            if ambiguous:
                print(f"⚠️  以下位点的野生型记谱字母不一致（取众数）: {ambiguous[:10]}")
        mutable = positions
        reference_sequence = "".join(wt_by_position[p] for p in positions)
    else:
        if reference_sequence is None:
            reference_sequence, ref_source = _infer_reference_mode(variants, seq_len, dominance_threshold)
            if ref_source == "inferred_uniform_unreliable":
                print("⚠️  位点残基分布接近均匀（枚举型库），众数推断的参考序列不可靠。")
                print("    建议显式传入 reference_sequence（如 GB1: 'VDGV'）。")

    if mutable_positions is not None:
        mutable = mutable_positions

    # --- 3. 编码（与推理时共用同一实现，杜绝训练/服务编码漂移）---
    config = ProteinConfig(
        reference_sequence=reference_sequence,
        mutable_positions=mutable,
        variant_format=variant_format,
        reference_source=ref_source,
        wt_by_position=wt_by_position,
        fitness_higher_is_better=higher_is_better,
    )
    X = np.array([config.encode_variant(v) for v in variants])

    # --- fitness 方向归一化 ---
    # 低=好（IC50/结合能）时内部取 -fitness 训练，使模型选择、验证 Spearman、Agent
    # 工具统一按"高=好"工作；predict_fitness 返回时翻回原始标度
    if not higher_is_better:
        fitness = -fitness

    # --- 4. 种子化随机划分（比例经 val_fraction 参数化，默认 0.8 即 80/20）---
    # 修复旧版顺序 80/20 切分：DMS 数据常按 fitness 排序导出，顺序切分会让验证集
    # 只覆盖低适应度区，模型选择系统性有偏
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    split = int(len(X) * val_fraction)
    tr, va = idx[:split], idx[split:]
    X_tr, X_val, y_tr, y_val = X[tr], X[va], fitness[tr], fitness[va]

    # --- 5. 训练预测器 ---
    # model_kwargs 未知候选名显式报错（替换格式可构造者与池矩阵一致；两种选型模式共用）
    _check_model_kwargs(model_kwargs, _SUBSTITUTION_MODEL_NAMES)
    if auto_select_predictor:
        # 简化的模型选择（正式实验可用 models.py 的完整 7-config 协议）
        # 推荐集（recommended，默认）= 内证与文献校准的支持面：one-hot 系 {RF, Ridge}——
        # 243 组消融中 one-hot 特征上 MLP 三连败（2K 0.344 / 10K 0.309 / 50K 0.556，
        # vs RF 0.543 / 0.589 / 0.655），故 MLP 不默认进 one-hot 池；ESM 系 {Ridge, MLP}
        # （消融总榜第二）。先验只作推荐不作拦截：predictor_pool='full' 放开 onehot_mlp，
        # extra_candidates 细粒度追加（推荐理由同时记于 predictor_info["pool_notes"]）
        candidates = {
            "random_forest": _make_model(
                RandomForestRegressor, "random_forest",
                dict(n_estimators=100, min_samples_leaf=1, max_features="sqrt",
                     random_state=42, n_jobs=4),
                model_kwargs,
            ),
            "ridge": _make_model(Ridge, "ridge", {"alpha": 10}, model_kwargs),
        }
        pool_excluded = _expand_candidate_pool(
            candidates, predictor_pool, extra_candidates,
            mlp_name="onehot_mlp",
            make_mlp_entry=lambda: _mlp_candidate(model_kwargs, "onehot_mlp"),
            blocked={
                "composition_ridge": "composition 候选仅定义于 indel 路径（替换格式的表示层为 one-hot / ESM）",
                "composition_mlp": "composition 候选仅定义于 indel 路径（替换格式的表示层为 one-hot / ESM）",
                "esm2_ridge": "ESM 候选由 use_esm=True 开启（需嵌入环境）",
                "esm2_mlp": "ESM 候选由 use_esm=True 开启（需嵌入环境）",
            },
        )
        best_name, best_sp, best_model = None, -999, None
        for name, model in candidates.items():
            model.fit(X_tr, y_tr)
            sp = spearmanr(y_val, model.predict(X_val)).statistic
            if sp > best_sp:
                best_name, best_sp, best_model = name, sp, model

        # --- ESM-2 嵌入候选（可选增强，任何不可用都优雅降级为跳过；两候选一并处理）---
        esm_scores = {}
        if use_esm:
            if variant_format != "full":
                print("ℹ️  突变串/记谱格式无法重构全长序列，跳过 ESM 候选")
                # 表示层面的硬排除（非先验）：无全长背景序列，ESM 嵌入不可计算——记入排除清单
                pool_excluded["esm2_ridge"] = pool_excluded["esm2_mlp"] = (
                    "表示层面不可行：突变串/记谱格式无法重构全长序列"
                )
            elif seq_len > 1022:
                print(f"⚠️  序列长度 {seq_len} > 1022（ESM-2 上限），ESM 不可用，跳过 ESM 候选")
            else:
                esm_cands, err = _build_esm_candidate(
                    variants[tr], variants[va], y_tr, y_val, esm_model_path, esm_device, esm_cache_dir,
                    model_kwargs=model_kwargs,
                )
                if err:
                    print(f"⚠️  {err}")
                else:
                    for esm_name, (esm_model, sp) in esm_cands.items():
                        esm_scores[esm_name] = sp
                        if sp > best_sp:
                            best_name, best_sp, best_model = esm_name, sp, esm_model

        config.predictor = best_model
        config.predictor_info = {
            "selected": best_name,
            "validation_spearman": round(best_sp, 4),
            "random_state": random_state,
            "fitness_higher_is_better": higher_is_better,
            "all_candidates": {n: round(spearmanr(y_val, m.predict(X_val)).statistic, 4)
                               for n, m in candidates.items()},
            "format_source": format_source,
            "model_kwargs_applied": _applied_model_kwargs(model_kwargs, [*candidates, *esm_scores]),
        }
        config.predictor_info["all_candidates"].update(
            {n: round(s, 4) for n, s in esm_scores.items()}
        )
        if "esm2_mlp" in esm_scores:
            # spec 反映实际构造参数（合并 model_kwargs['esm2_mlp'] 覆盖后的真实超参）
            config.predictor_info["esm2_mlp_spec"] = _esm2_mlp_spec_text(model_kwargs)
        config.predictor_info["pool"] = {
            "mode": predictor_pool,
            "included": list(config.predictor_info["all_candidates"].keys()),
            "excluded": pool_excluded,
        }
        config.predictor_info["pool_notes"] = (
            _POOL_NOTES_RECOMMENDED if predictor_pool == "recommended" else _POOL_NOTES_FULL
        )
    else:
        config.predictor = _make_model(
            RandomForestRegressor, "random_forest",
            dict(n_estimators=100, min_samples_leaf=1, max_features="sqrt",
                 random_state=42, n_jobs=4),
            model_kwargs,
        ).fit(X, fitness)
        config.predictor_info = {
            "selected": "random_forest_default",
            "fitness_higher_is_better": higher_is_better,
            "format_source": format_source,
            "model_kwargs_applied": _applied_model_kwargs(model_kwargs, ["random_forest"]),
        }

    print(f"✅ Protein configured:")
    print(f"   Format: {variant_format}")
    print(f"   Mutable positions ({len(mutable)}): {mutable[:12]}{'...' if len(mutable) > 12 else ''}")
    print(f"   Reference ({config.reference_source}): "
          f"{reference_sequence[:20]}{'...' if len(reference_sequence) > 20 else ''} "
          f"({len(reference_sequence)}aa)")
    print(f"   Fitness direction: "
          f"{'higher is better' if config.fitness_higher_is_better else 'lower is better（内部按 -fitness 训练，predict_fitness 返回原始标度）'}")
    print(f"   Predictor: {config.predictor_info.get('selected', 'N/A')} "
          f"(val ρ={config.predictor_info.get('validation_spearman', 'N/A')})")

    return config
