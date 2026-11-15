# 实验二：多层拼接 —— 将已有 4 层 pooled 嵌入沿特征维拼接（无需重新推理，秒级完成）。
#
# 输入：data/esm2_embeddings/esm2_pooled_L{20,25,30,33}.npy
#       （embed_gb1.py 全量产出，行序与 data/processed/all_variants.csv 一一对应）
# 输出：data/esm2_embeddings/esm2_multilayer_pooled.npy  (149361, 5120) = [L20 | L25 | L30 | L33]
import os
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
EMB_DIR = os.path.join(ROOT, "data", "esm2_embeddings")
PROGRESS_LOG = os.path.join(ROOT, "results", "predictor_v2", "exp_progress.log")
LAYERS = (20, 25, 30, 33)
EMBED_DIM = 1280


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("实验二开始：多层拼接 L20|L25|L30|L33")
    n_expected = len(pd.read_csv(os.path.join(ROOT, "data", "processed", "all_variants.csv")))
    arrays = []
    for layer in LAYERS:
        path = os.path.join(EMB_DIR, f"esm2_pooled_L{layer}.npy")
        arr = np.asarray(np.load(path, mmap_mode="r"))
        log(f"  加载 esm2_pooled_L{layer}.npy shape={arr.shape}")
        if arr.shape != (n_expected, EMBED_DIM):
            raise ValueError(f"esm2_pooled_L{layer}.npy 形状 {arr.shape} != {(n_expected, EMBED_DIM)}")
        arrays.append(arr)
    merged = np.concatenate(arrays, axis=1)
    if not np.isfinite(merged).all():
        raise ValueError("拼接结果含 NaN/Inf")
    out_path = os.path.join(EMB_DIR, "esm2_multilayer_pooled.npy")
    np.save(out_path, merged)
    log(f"实验二完成：esm2_multilayer_pooled.npy shape={merged.shape} dtype={merged.dtype}")


if __name__ == "__main__":
    main()
