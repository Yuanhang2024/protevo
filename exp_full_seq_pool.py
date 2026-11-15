# 实验一：全序列均值池化 —— ESM-2 650M 第 33 层，取全部 56 个残基位置的嵌入均值。
# 与 embed_gb1.py 的 pooled（仅 4 个突变位点）不同，本脚本覆盖全序列上下文。
#
# 索引约定（同 embed_gb1.py）：EsmTokenizer 前加 <cls>(BOS)、后加 <eos>，
# 0-indexed 残基 i 的模型输出索引为 i+1；均值取模型索引 1..56（剔除 BOS/EOS）。
# 所有变体均为 56 aa 等长替换序列，tokenize 后长度恒为 58，无 padding 歧义。
#
# 输入：data/processed/all_variants.csv（149361 行，行序即输出 .npy 行号）
# 输出：data/esm2_embeddings/esm2_fullseq_L33.npy (149361, 1280)
# 用法：D:\Anaconda3\envs\openfold\python.exe -I exp_full_seq_pool.py [--batch-size 64] [--limit N]
import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
ESM2_PATH = r"D:\ChimeraModel_Lite\esm2_t33_650M_UR50D"
PROGRESS_LOG = os.path.join(ROOT, "results", "predictor_v2", "exp_progress.log")
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS_1 = (39, 40, 41, 54)   # 1-indexed 突变位点
POSITIONS_0 = (38, 39, 40, 53)   # 0-indexed 残基下标
LAYER = 33                       # hidden_states[33] = 末层（hidden_states[0] 为 embedding 层）
SEQ_LEN = 56
EMBED_DIM = 1280


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# variant 四个字母依次替换 39/40/41/54 位（1-indexed），其余位置保留参考氨基酸。
def variant_to_sequence(variant, reference=WT):
    if not isinstance(variant, str) or re.fullmatch(f"[{AMINO_ACIDS}]{{4}}", variant) is None:
        raise ValueError(f"Invalid four-site variant: {variant!r}")
    sequence = list(reference)
    for position, residue in zip(POSITIONS_1, variant):
        sequence[position - 1] = residue
    return "".join(sequence)


def load_variants(input_path, limit=None):
    table = pd.read_csv(input_path)
    variants = [str(v).strip().upper() for v in table["variant"].tolist()]
    if limit is not None:
        variants = variants[:limit]
    for v in variants:
        if re.fullmatch(f"[{AMINO_ACIDS}]{{4}}", v) is None:
            raise ValueError(f"Invalid variant: {v!r}")
    if len(set(variants)) != len(variants):
        raise ValueError("Input contains duplicate variants")
    return variants


def main():
    parser = argparse.ArgumentParser(description="GB1 全序列均值池化（ESM-2 650M L33）")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个变体（冒烟测试用）")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    variants = load_variants(os.path.join(ROOT, "data", "processed", "all_variants.csv"), args.limit)
    n = len(variants)
    log(f"实验一开始：全序列均值池化，{n} 个变体，batch_size={args.batch_size}，层 L{LAYER}")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（GPU 推理）")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(ESM2_PATH)
    model = AutoModel.from_pretrained(ESM2_PATH).eval().to(device)

    # BOS 偏移核对：残基 p 的 token 应精确位于模型索引 p+1
    seq0 = variant_to_sequence(variants[0])
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(seq0)["input_ids"])
    assert len(tokens) == SEQ_LEN + 2, f"token 长度异常: {len(tokens)}"
    assert all(tokens[p + 1] == seq0[p] for p in POSITIONS_0), "BOS 偏移校验失败"
    log(f"  BOS 偏移校验通过：位点(0-idx) {POSITIONS_0} → token 索引 {tuple(p + 1 for p in POSITIONS_0)}")

    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    total_batches = (n + args.batch_size - 1) // args.batch_size
    started = time.perf_counter()
    for start in range(0, n, args.batch_size):
        batch = variants[start:start + args.batch_size]
        sequences = [variant_to_sequence(v) for v in batch]
        encoded = tokenizer(sequences, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[LAYER]
        # hidden: (B, 58, 1280)；均值取模型索引 1..56（56 个残基，剔除 BOS/EOS）
        out[start:start + len(batch)] = hidden[:, 1:SEQ_LEN + 1, :].float().mean(dim=1).cpu().numpy()
        done = start // args.batch_size + 1
        if done % 100 == 0 or done == total_batches:
            elapsed = time.perf_counter() - started
            log(f"  实验一批次 {done}/{total_batches}（{elapsed:.0f}s，预计共 {elapsed / done * total_batches:.0f}s）")

    if not np.isfinite(out).all():
        raise ValueError("嵌入含 NaN/Inf")
    if out.std(axis=0).min() <= 0:
        raise ValueError("存在零方差维度")
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    out_path = os.path.join(ROOT, "data", "esm2_embeddings", f"esm2_fullseq_L33{suffix}.npy")
    np.save(out_path, out)
    log(f"实验一完成：{os.path.basename(out_path)} shape={out.shape} "
        f"耗时={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
