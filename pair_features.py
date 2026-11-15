# 追加实验：逐位点嵌入 + 逐对点积特征 —— ESM-2 650M 第 33 层，4 个突变位点各自的残基级嵌入（不池化）。
#
# 输出（行序与 data/processed/all_variants.csv 一一对应）：
#   esm2_per_residue_L33.npy  (149361, 4, 1280)  依次为位点 39/40/41/54（1-indexed）
#   esm2_pairwise_dot.npy     (149361, 6)        C(4,2)=6 对点积，顺序：
#     e39·e40, e39·e41, e39·e54, e40·e41, e40·e54, e41·e54
# 自检：per_residue 展平应与已有 esm2_concat_L33.npy 一致（同一模型/层/位点，抽样 2000 行 allclose）。
# 用法：D:\Anaconda3\envs\openfold\python.exe -I pair_features.py [--batch-size 64] [--limit N]
import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
EMB_DIR = os.path.join(ROOT, "data", "esm2_embeddings")
ESM2_PATH = r"D:\ChimeraModel_Lite\esm2_t33_650M_UR50D"
PROGRESS_LOG = os.path.join(ROOT, "results", "predictor_v2", "exp_progress.log")
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS_1 = (39, 40, 41, 54)                 # 1-indexed 突变位点
POSITIONS_0 = (38, 39, 40, 53)                 # 0-indexed 残基下标
MODEL_IDX = tuple(p + 1 for p in POSITIONS_0)  # (39, 40, 41, 54)，BOS 偏移 +1
LAYER = 33
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 对应 e39·e40 … e41·e54
EMBED_DIM = 1280


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    parser = argparse.ArgumentParser(description="GB1 逐位点 L33 嵌入 + 6 对点积特征")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个变体（冒烟测试用）")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    variants = load_variants(os.path.join(ROOT, "data", "processed", "all_variants.csv"), args.limit)
    n = len(variants)
    log(f"逐位点实验开始：{n} 个变体，batch_size={args.batch_size}，层 L{LAYER}，位点(1-idx) {POSITIONS_1}")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（GPU 推理）")
    tokenizer = AutoTokenizer.from_pretrained(ESM2_PATH)
    model = AutoModel.from_pretrained(ESM2_PATH).eval().to("cuda")

    seq0 = variant_to_sequence(variants[0])
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(seq0)["input_ids"])
    assert all(tokens[p + 1] == seq0[p] for p in POSITIONS_0), "BOS 偏移校验失败"
    log("  BOS 偏移校验通过")

    per_res = np.empty((n, len(MODEL_IDX), EMBED_DIM), dtype=np.float32)
    pairwise = np.empty((n, len(PAIRS)), dtype=np.float32)
    model_idx = list(MODEL_IDX)
    total_batches = (n + args.batch_size - 1) // args.batch_size
    started = time.perf_counter()
    for start in range(0, n, args.batch_size):
        batch = variants[start:start + args.batch_size]
        sequences = [variant_to_sequence(v) for v in batch]
        encoded = tokenizer(sequences, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[LAYER]
        # hidden: (B, 58, 1280)；按模型索引取 4 个位点 → (B, 4, 1280)
        res = hidden[:, model_idx, :].float().cpu().numpy()
        per_res[start:start + len(batch)] = res
        for k, (i, j) in enumerate(PAIRS):
            pairwise[start:start + len(batch), k] = (res[:, i, :] * res[:, j, :]).sum(axis=1)
        done = start // args.batch_size + 1
        if done % 100 == 0 or done == total_batches:
            elapsed = time.perf_counter() - started
            log(f"  逐位点批次 {done}/{total_batches}（{elapsed:.0f}s，预计共 {elapsed / done * total_batches:.0f}s）")

    if not np.isfinite(per_res).all() or not np.isfinite(pairwise).all():
        raise ValueError("输出含 NaN/Inf")
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    per_path = os.path.join(EMB_DIR, f"esm2_per_residue_L33{suffix}.npy")
    dot_path = os.path.join(EMB_DIR, f"esm2_pairwise_dot{suffix}.npy")
    np.save(per_path, per_res)
    np.save(dot_path, pairwise)

    # 自检：与已有 concat_L33 抽样比对（同模型/层/位点，应逐元素一致）
    if args.limit is None:
        concat = np.load(os.path.join(EMB_DIR, "esm2_concat_L33.npy"), mmap_mode="r")
        rng = np.random.RandomState(0)
        sample = rng.choice(n, size=2000, replace=False)
        flat = per_res.reshape(n, -1)[sample]
        ref = np.asarray(concat[sample])
        close = np.allclose(flat, ref, atol=1e-4, rtol=1e-4)
        log(f"  自检 per_residue 展平 vs esm2_concat_L33（抽样 2000 行 allclose）: {close}")
        if not close:
            raise ValueError("per_residue 与已有 concat_L33 不一致，请检查层/位点/偏移设置")

    log(f"逐位点实验完成：esm2_per_residue_L33{suffix}.npy {per_res.shape} + "
        f"esm2_pairwise_dot{suffix}.npy {pairwise.shape}，耗时={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
