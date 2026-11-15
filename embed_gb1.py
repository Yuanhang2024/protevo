# ESM-2 残基级嵌入提取：GB1 四位点变体（39/40/41/54）→ 多层 pooled + 末层 concat
# 后端：transformers（AutoModel/AutoTokenizer，本地 HF 格式 checkpoint），非 fair-esm。
#
# 输入：含 variant 列（4 字符突变串，依次对应 39/40/41/54 位）的 CSV，保持文件行序。
# 输出（写入 --output 目录）：
#   esm2_pooled_L{layer}.npy   4 位点嵌入取平均     → (N, 1280) × 每层
#   esm2_concat_L{last}.npy    末层 4 位点拼接      → (N, 5120)
#   variants.txt               变体顺序（与 .npy 行号一一对应）
#   embed_log.json             处理日志（checkpoint hash、耗时、环境）
#
# 用法：
#   全量（子任务 D）：
#     python embed_gb1.py --input data/processed/all_variants.csv --output data/esm2_embeddings/ --layers 20 25 30 33
#   小批量验证（子任务 A，前 100 个变体）：
#     python embed_gb1.py --input data/processed/seed-11/train.csv --output data/esm2_embeddings/validation_100 --limit 100 --validate --report validation_report.txt
#
# 索引约定（关键）：EsmTokenizer 会在序列前加 <cls>(BOS)、后加 <eos>，
# 0-indexed 残基 i 的模型输出索引为 i+1。本项目位点 0-indexed (38,39,40,53) → 模型索引 (39,40,41,54)。
# 层约定：transformers 的 hidden_states[k] 即第 k 层输出（hidden_states[0] 为 embedding 层），
# 与 fair-esm 的 repr_layers=k 一致；hidden_states 共 num_hidden_layers+1 = 34 项。
import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ESM2_PATH = r"D:\ChimeraModel_Lite\esm2_t33_650M_UR50D"
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS_1 = (39, 40, 41, 54)                    # 1-indexed 突变位点
POSITIONS_0 = (38, 39, 40, 53)                    # 0-indexed 残基下标
MODEL_IDX = tuple(p + 1 for p in POSITIONS_0)     # BOS 偏移后 → (39, 40, 41, 54)
EMBED_DIM = 1280                                  # ESM-2 650M 隐藏维
DEFAULT_LAYERS = (20, 25, 30, 33)


# variant 四个字母依次替换 39/40/41/54 位（1-indexed），其余位置保留参考氨基酸。
def variant_to_sequence(variant, reference=WT):
    if not isinstance(variant, str) or re.fullmatch(f"[{AMINO_ACIDS}]{{4}}", variant) is None:
        raise ValueError(f"Invalid four-site variant: {variant!r}")
    if "".join(reference[p - 1] for p in POSITIONS_1) != "VDGV":
        raise ValueError("Reference positions 39/40/41/54 must map to VDGV")
    sequence = list(reference)
    for position, residue in zip(POSITIONS_1, variant):
        sequence[position - 1] = residue
    return "".join(sequence)


# 读入 CSV 并校验变体格式；--limit 取文件前 N 行（保持行序，行号即 .npy 行号）。
def load_variants(input_path, variant_column="variant", limit=None):
    table = pd.read_csv(input_path)
    if variant_column not in table:
        raise ValueError(f"Column {variant_column!r} not found; available: {table.columns.tolist()}")
    variants = [str(v).strip().upper() for v in table[variant_column].tolist()]
    if limit is not None:
        variants = variants[:limit]
    for v in variants:
        if re.fullmatch(f"[{AMINO_ACIDS}]{{4}}", v) is None:
            raise ValueError(f"Invalid variant: {v!r}")
    if len(set(variants)) != len(variants):
        raise ValueError("Input contains duplicate variants")
    return variants


# sha256 分块读取，供处理日志记录 checkpoint 指纹。
def file_sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Esm2Embedder:
    """加载本地 ESM-2，一次前向传播同时取多层隐藏状态（仅推理，不训练）。"""

    def __init__(self, model_path=ESM2_PATH, layers=DEFAULT_LAYERS, batch_size=32, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.layers = tuple(sorted(set(int(l) for l in layers)))
        self.last_layer = self.layers[-1]
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval().to(device)
        n_layers = self.model.config.num_hidden_layers
        if self.last_layer > n_layers:
            raise ValueError(f"Requested layer {self.last_layer} exceeds model depth {n_layers}")

    # 返回 {f"pooled_L{l}": (N,1280), f"concat_L{last}": (N,5120)}
    def extract(self, variants):
        model_idx = list(MODEL_IDX)
        n = len(variants)
        pooled = {layer: np.empty((n, EMBED_DIM), dtype=np.float32) for layer in self.layers}
        concat = np.empty((n, EMBED_DIM * len(model_idx)), dtype=np.float32)
        total_batches = (n + self.batch_size - 1) // self.batch_size
        started = time.perf_counter()
        for start in range(0, n, self.batch_size):
            batch_variants = variants[start:start + self.batch_size]
            sequences = [variant_to_sequence(v) for v in batch_variants]
            encoded = self.tokenizer(sequences, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**encoded, output_hidden_states=True)
            # hidden_states[k]: (B, seq_len+2, 1280)；位点残基级嵌入按模型索引取行
            for layer in self.layers:
                hidden = outputs.hidden_states[layer][:, model_idx, :].float().cpu().numpy()
                pooled[layer][start:start + len(batch_variants)] = hidden.mean(axis=1)
            last = outputs.hidden_states[self.last_layer][:, model_idx, :].float().cpu().numpy()
            concat[start:start + len(batch_variants)] = last.reshape(len(batch_variants), -1)
            done = start // self.batch_size + 1
            if done % 20 == 0 or done == total_batches:
                print(f"  batch {done}/{total_batches} ({(time.perf_counter() - started):.1f}s)")
        result = {f"pooled_L{layer}": pooled[layer] for layer in self.layers}
        result[f"concat_L{self.last_layer}"] = concat
        result["_seconds"] = time.perf_counter() - started
        return result


# 行归一化后点积 = 余弦相似度矩阵。
def cosine_matrix(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def save_outputs(output_dir, variants, arrays, args, checkpoint_sha):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, array in arrays.items():
        if name == "_seconds":
            continue
        path = output_dir / f"esm2_{name}.npy"
        np.save(path, array)
        saved[f"esm2_{name}.npy"] = list(array.shape)
    (output_dir / "variants.txt").write_text("\n".join(variants) + "\n", encoding="utf-8")
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "variant_column": args.variant_column,
        "n_variants": len(variants),
        "limit": args.limit,
        "layers": list(arrays_key_layers(arrays)),
        "batch_size": args.batch_size,
        "device": device_name(),
        "torch_version": torch_version(),
        "transformers_version": transformers_version(),
        "checkpoint": ESM2_PATH,
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_seconds": round(float(arrays["_seconds"]), 2),
        "output_files": saved,
    }
    (output_dir / "embed_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return saved


# ---- 环境信息辅助（延迟导入，避免脚本未用 GPU 时也加载 torch）----
def device_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def torch_version():
    import torch
    return torch.__version__


def transformers_version():
    import transformers
    return transformers.__version__


def arrays_key_layers(arrays):
    return tuple(int(name.split("_L")[1]) for name in arrays if name.startswith("pooled_L"))


# ---- 验证套件（--validate）：维度 / NaN / 区分度 / 范数 / 索引偏移 ----
def validate_and_report(output_dir, variants, arrays, embedder, report_path, input_path):
    lines = []
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    pooled_names = sorted((n for n in arrays if n.startswith("pooled_L")), key=lambda n: int(n.split("_L")[1]))
    concat_name = next(n for n in arrays if n.startswith("concat_L"))

    lines.append("=" * 72)
    lines.append("ESM-2 嵌入验证报告（embed_gb1.py, transformers 后端）")
    lines.append(f"生成时间   : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"输入       : {input_path}（前 {len(variants)} 个变体，保持文件行序）")
    lines.append(f"设备       : {device_name()}（{embedder.device}）")
    lines.append(f"版本       : torch {torch_version()}, transformers {transformers_version()}")
    lines.append(f"checkpoint : {ESM2_PATH}")
    lines.append(f"位点(1-idx): {POSITIONS_1} → 模型索引 {MODEL_IDX}（BOS 偏移 +1）")
    lines.append("=" * 72)

    # 1. 维度检查
    lines.append("")
    lines.append("1. 嵌入维度")
    for name in pooled_names:
        shape = arrays[name].shape
        check(f"{name} 形状 {shape}", shape == (len(variants), EMBED_DIM))
    cshape = arrays[concat_name].shape
    check(f"{concat_name} 形状 {cshape}", cshape == (len(variants), EMBED_DIM * 4))

    # 2. NaN / Inf
    lines.append("")
    lines.append("2. 数值健全性（NaN/Inf）")
    for name in pooled_names + [concat_name]:
        finite = bool(np.isfinite(arrays[name]).all())
        check(f"{name} 全部有限（无 NaN/Inf）", finite)

    # 3. 变体区分度：两两余弦相似度（pooled 末层）
    lines.append("")
    lines.append(f"3. 变体区分度（{pooled_names[-1]}，N={len(variants)} 两两余弦相似度）")
    sim = cosine_matrix(arrays[pooled_names[-1]])
    iu = np.triu_indices(len(variants), k=1)
    pair_sims = sim[iu]
    lines.append(f"   pair 数={len(pair_sims)}  min={pair_sims.min():.6f}  mean={pair_sims.mean():.6f}  max={pair_sims.max():.6f}")
    check("任意两变体余弦相似度 < 1.0（嵌入互不相同）", bool(pair_sims.max() < 1.0),
          f"max={pair_sims.max():.6f}")
    dup_rows = len(arrays[pooled_names[-1]]) - len(np.unique(arrays[pooled_names[-1]], axis=0))
    check("无重复嵌入行", dup_rows == 0, f"重复行数={dup_rows}")
    k = min(5, len(variants))
    lines.append(f"   前 {k} 个变体 {variants[:k]} 的 {k}x{k} 余弦矩阵：")
    for row in sim[:k, :k]:
        lines.append("     " + "  ".join(f"{x:7.4f}" for x in row))

    # 4. 前 5 个变体嵌入 L2 范数
    lines.append("")
    lines.append("4. 前 5 个变体嵌入 L2 范数")
    header = "variant  " + "".join(f"{n:>14}" for n in pooled_names + [concat_name])
    lines.append("   " + header)
    for i, v in enumerate(variants[:k]):
        norms = [np.linalg.norm(arrays[n][i]) for n in pooled_names + [concat_name]]
        lines.append(f"   {v:8s}" + "".join(f"{x:14.4f}" for x in norms))

    # 5. 索引偏移验证
    lines.append("")
    lines.append("5. 索引偏移验证（BOS +1）")
    lines.append("   5a. token 级核对：tokenized[p+1] 应精确解码为残基 p 的字符")
    offset_ok = True
    rows = ["WT(VDGV)"] + variants[:3]
    seqs = [WT] + [variant_to_sequence(v) for v in variants[:3]]
    for label, seq in zip(rows, seqs):
        ids = embedder.tokenizer(seq, return_tensors="pt")["input_ids"][0].tolist()
        tokens = embedder.tokenizer.convert_ids_to_tokens(ids)
        mapping = []
        for p in POSITIONS_0:
            match = tokens[p + 1] == seq[p]
            offset_ok &= match
            mapping.append(f"{p}->{p+1}:{tokens[p+1]}({'=' if match else '!='}{seq[p]})")
        lines.append(f"   {label:10s} " + "  ".join(mapping))
    check("位点 (38,39,40,53) 的模型索引 (39,40,41,54) token 精确对应", offset_ok)
    lines.append("   5b. 经验核对：WT(VDGV) 与 ADGV（仅 39 位 V→A）各提取位点的余弦相似度")
    diag = embedder.extract(["VDGV", "ADGV"])
    wt_concat, mut_concat = diag[f"concat_L{embedder.last_layer}"]
    pos_sims = []
    for j, p in enumerate(POSITIONS_1):
        a = wt_concat[j * EMBED_DIM:(j + 1) * EMBED_DIM]
        b = mut_concat[j * EMBED_DIM:(j + 1) * EMBED_DIM]
        pos_sims.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        lines.append(f"     位点 {p}: cos={pos_sims[-1]:.6f}")
    check("39 位（突变位点）相似度最低，偏移方向正确", int(np.argmin(pos_sims)) == 0,
          f"argmin={POSITIONS_1[int(np.argmin(pos_sims))]}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"结论：{sum(checks)}/{len(checks)} 项通过" + ("，全部通过" if all(checks) else "，存在失败项"))
    lines.append("=" * 72)
    report = "\n".join(lines) + "\n"
    Path(report_path).write_text(report, encoding="utf-8")
    print(report)
    return all(checks)


def parse_args():
    parser = argparse.ArgumentParser(description="GB1 四位点变体的 ESM-2 多层残基级嵌入提取")
    parser.add_argument("--input", required=True, help="含 variant 列的 CSV")
    parser.add_argument("--variant-column", default="variant", help="变体列名（默认 variant）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS),
                        help="隐藏层号（默认 20 25 30 33；concat 取最大层）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个变体")
    parser.add_argument("--device", default=None, help="cuda/cpu（默认自动）")
    parser.add_argument("--validate", action="store_true", help="计算后运行验证套件并写报告")
    parser.add_argument("--report", default=None, help="验证报告路径（默认 <output>/validation_report.txt）")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = load_variants(args.input, args.variant_column, args.limit)
    print(f"载入 {len(variants)} 个变体（{args.input}），层 {sorted(args.layers)}")
    checkpoint_sha = file_sha256(Path(ESM2_PATH) / "model.safetensors")
    print(f"checkpoint sha256: {checkpoint_sha[:16]}…")
    embedder = Esm2Embedder(ESM2_PATH, layers=args.layers, batch_size=args.batch_size, device=args.device)
    print(f"模型已加载于 {embedder.device}（{device_name()}）")
    arrays = embedder.extract(variants)
    print(f"推理耗时 {arrays['_seconds']:.1f}s")
    saved = save_outputs(args.output, variants, arrays, args, checkpoint_sha)
    for name, shape in saved.items():
        print(f"已保存 {name} {shape}")
    if args.validate:
        report_path = args.report or str(Path(args.output) / "validation_report.txt")
        ok = validate_and_report(args.output, variants, arrays, embedder, report_path, args.input)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
