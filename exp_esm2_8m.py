# 实验三：ESM-2 8M（facebook/esm2_t6_8M_UR50D，hidden=320，6 层）下载并提取嵌入。
#
# 两种池化均取末层 L6（hidden_states[6]，hidden_states[0] 为 embedding 层）：
#   mutpos ：4 个突变位点（0-idx 38/39/40/53 → 模型索引 39/40/41/54）均值 → esm2_8m_mutpos.npy  (N, 320)
#   fullseq：全部 56 个残基（模型索引 1..56，剔除 BOS/EOS）均值       → esm2_8m_fullseq.npy (N, 320)
#
# 网络：优先直连 HuggingFace；失败自动改走镜像 HF_ENDPOINT=https://hf-mirror.com 并指定
#       cache_dir（data/hf_cache_esm2_8m，独立目录）重试。
# 用法：D:\Anaconda3\envs\openfold\python.exe -I exp_esm2_8m.py [--batch-size 256] [--limit N]
import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
EMB_DIR = os.path.join(ROOT, "data", "esm2_embeddings")
CACHE_DIR = os.path.join(ROOT, "data", "hf_cache_esm2_8m")
PROGRESS_LOG = os.path.join(ROOT, "results", "predictor_v2", "exp_progress.log")
MODEL_ID = "facebook/esm2_t6_8M_UR50D"
WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS_1 = (39, 40, 41, 54)
POSITIONS_0 = (38, 39, 40, 53)
MODEL_IDX = tuple(p + 1 for p in POSITIONS_0)   # (39, 40, 41, 54)
N_LAYERS = 6
SEQ_LEN = 56
EMBED_DIM = 320


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


def load_model():
    """直连下载；失败改走 hf-mirror 镜像 + 独立 cache_dir 重试。"""
    from transformers import AutoModel, AutoTokenizer
    try:
        log(f"  下载/加载 {MODEL_ID}（直连 HuggingFace）")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID)
        source = "huggingface.co"
    except Exception as exc:
        log(f"  直连失败（{type(exc).__name__}: {str(exc)[:120]}），改用镜像 hf-mirror.com + cache_dir 重试")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        source = "hf-mirror.com"
    return tokenizer, model, source


def main():
    parser = argparse.ArgumentParser(description="GB1 ESM-2 8M 嵌入提取（mutpos + fullseq）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个变体（冒烟测试用）")
    args = parser.parse_args()

    import torch

    variants = load_variants(os.path.join(ROOT, "data", "processed", "all_variants.csv"), args.limit)
    n = len(variants)
    log(f"实验三开始：ESM-2 8M，{n} 个变体，batch_size={args.batch_size}，层 L{N_LAYERS}（末层）")
    tokenizer, model, source = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    log(f"  模型已加载（来源 {source}，hidden_size={model.config.hidden_size}，"
        f"num_layers={model.config.num_hidden_layers}，device={device}）")
    if model.config.hidden_size != EMBED_DIM or model.config.num_hidden_layers != N_LAYERS:
        raise ValueError("模型结构与预期不符（应为 8M/6 层/320 维）")

    seq0 = variant_to_sequence(variants[0])
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(seq0)["input_ids"])
    assert len(tokens) == SEQ_LEN + 2, f"token 长度异常: {len(tokens)}"
    assert all(tokens[p + 1] == seq0[p] for p in POSITIONS_0), "BOS 偏移校验失败"

    mutpos = np.empty((n, EMBED_DIM), dtype=np.float32)
    fullseq = np.empty((n, EMBED_DIM), dtype=np.float32)
    model_idx = list(MODEL_IDX)
    total_batches = (n + args.batch_size - 1) // args.batch_size
    started = time.perf_counter()
    for start in range(0, n, args.batch_size):
        batch = variants[start:start + args.batch_size]
        sequences = [variant_to_sequence(v) for v in batch]
        encoded = tokenizer(sequences, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[N_LAYERS]
        # hidden: (B, 58, 320)
        mutpos[start:start + len(batch)] = hidden[:, model_idx, :].float().mean(dim=1).cpu().numpy()
        fullseq[start:start + len(batch)] = hidden[:, 1:SEQ_LEN + 1, :].float().mean(dim=1).cpu().numpy()
        done = start // args.batch_size + 1
        if done % 100 == 0 or done == total_batches:
            elapsed = time.perf_counter() - started
            log(f"  实验三批次 {done}/{total_batches}（{elapsed:.0f}s）")

    for name, arr in (("mutpos", mutpos), ("fullseq", fullseq)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 嵌入含 NaN/Inf")
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    mut_path = os.path.join(EMB_DIR, f"esm2_8m_mutpos{suffix}.npy")
    full_path = os.path.join(EMB_DIR, f"esm2_8m_fullseq{suffix}.npy")
    np.save(mut_path, mutpos)
    np.save(full_path, fullseq)
    log(f"实验三完成：esm2_8m_mutpos{suffix}.npy {mutpos.shape} + "
        f"esm2_8m_fullseq{suffix}.npy {fullseq.shape}，耗时={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
