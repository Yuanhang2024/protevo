# 统一评估：9 特征 × 3 模型 × 3 训练量 × 3 种子 = 243 组完整网格。
#
# 特征（行序均与 data/processed/all_variants.csv 一一对应）：
#   onehot_80d        4 位点 × 20 AA one-hot（80 维，变体串即时编码，基线）
#   L33_mutpos        esm2_pooled_L33.npy        (1280)  650M 4 突变位点均值池化
#   L33_fullseq       esm2_fullseq_L33.npy       (1280)  650M 全 56 残基均值池化（实验一）
#   multilayer_5120   esm2_multilayer_pooled.npy (5120)  L20|L25|L30|L33 拼接（实验二）
#   8m_mutpos_320     esm2_8m_mutpos.npy         (320)   8M 4 突变位点均值池化（实验三）
#   8m_fullseq_320    esm2_8m_fullseq.npy        (320)   8M 全序列均值池化（实验三）
#   per_residue_flat  esm2_per_residue_L33.npy   (5120)  4 位点逐残基嵌入展平（追加实验）
#   pairdot_concat    per_residue_flat + 6 点积  (5126)  追加实验
#   pairdot_only      esm2_pairwise_dot.npy      (6)     仅 6 对点积（交互信号单独检验）
#
# 模型：RF(n_estimators=100, min_samples_leaf=1, max_features='sqrt')；
#       Ridge(alpha=10)+StandardScaler；MLP(256,64)+StandardScaler（adam, alpha=1e-4,
#       batch 256, early_stopping, max_iter 200，正则化沿用 run_predictor_v2.py 的约定——
#       sklearn 无 dropout，以 weight decay 替代）。
#       注：RF 首版用 sklearn 默认 max_features=1.0（全部特征参与每次分裂），实测 5120 维 × 50k
#       单拟合约 2.3h、全网格约 30h，不可行；经确认统一改为 'sqrt'（高维 RF 回归标准做法，
#       对 80 维 one-hot 与 5120 维嵌入一视同仁）。首版部分结果归档于
#       results/predictor_v2/full_ablation_runs_rf_mf1.0_partial.csv（RF 格子与 sqrt 版不可比）。
# 数据划分：验证集固定 2000 条（RandomState(0) 置换后取前 2000，全部 243 组共用）；
#           训练子集从其余 147,361 条中按种子 RandomState(seed) 无放回抽样。
# 训练量：2000 / 10000 / 50000；种子：11 / 22 / 33。
# 输出：results/predictor_v2/full_ablation_runs.csv（逐组流式追加）
#       results/predictor_v2/full_ablation.csv（243 行，按 val Spearman 降序）
#       results/predictor_v2/full_ablation_summary.csv（按 feature×model×train_size 聚合）
# 用法：D:\Anaconda3\envs\openfold\python.exe -I evaluate_all_variants.py
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
EMB_DIR = os.path.join(ROOT, "data", "esm2_embeddings")
OUT_DIR = os.path.join(ROOT, "results", "predictor_v2")
PROGRESS_LOG = os.path.join(OUT_DIR, "exp_progress.log")
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
TRAIN_SIZES = [2000, 10000, 50000]
SEEDS = [11, 22, 33]
VAL_SIZE = 2000
# 便宜的排前面：结果流式落盘，重计算（RF×5120 维）靠后
FEATURE_ORDER = ["onehot_80d", "pairdot_only", "8m_mutpos_320", "8m_fullseq_320",
                 "L33_mutpos", "L33_fullseq", "per_residue_flat", "pairdot_concat",
                 "multilayer_5120"]
MODEL_ORDER = ["ridge", "mlp", "rf"]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def one_hot_encode(variants):
    """4-position x 20-AA one-hot (80 dims)，与 run_predictor_v2.py 基线一致。"""
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    X = np.zeros((len(variants), 4 * 20), dtype=np.float32)
    for r, v in enumerate(variants):
        for p, ch in enumerate(v):
            X[r, p * 20 + aa_to_idx[ch]] = 1.0
    return X


def make_model(name, seed):
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10))])
    if name == "mlp":
        return Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,          # weight decay：dropout 替代（sklearn 无 dropout）
            batch_size=256,
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=200,
            random_state=seed,
        ))])
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=1,
                                     max_features="sqrt", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="9 特征统一评估网格")
    parser.add_argument("--resume", action="store_true",
                        help="跳过 full_ablation_runs.csv 中已有的 (feature,model,size,seed) 组合，续跑缺口")
    parser.add_argument("--force-model", default=None,
                        help="配合 --resume：强制重算该模型的全部组合（如换 RF 超参后 rf）")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    table = pd.read_csv(os.path.join(ROOT, "data", "processed", "all_variants.csv"))
    assert table["variant"].is_unique
    variants = table["variant"].to_numpy()
    fitness = table["fitness"].to_numpy(dtype=np.float64)
    n = len(table)

    # 特征获取器：给定行索引返回设计矩阵（大文件走 mmap，按需 gather）
    per_res = np.load(os.path.join(EMB_DIR, "esm2_per_residue_L33.npy"), mmap_mode="r")
    pair_dot = np.load(os.path.join(EMB_DIR, "esm2_pairwise_dot.npy"), mmap_mode="r")
    per_res_flat = per_res.reshape(n, -1)  # mmap 上的连续视图，gather 时才物化
    emb_files = {
        "L33_mutpos": "esm2_pooled_L33.npy",
        "L33_fullseq": "esm2_fullseq_L33.npy",
        "multilayer_5120": "esm2_multilayer_pooled.npy",
        "8m_mutpos_320": "esm2_8m_mutpos.npy",
        "8m_fullseq_320": "esm2_8m_fullseq.npy",
    }
    mmaps = {k: np.load(os.path.join(EMB_DIR, f), mmap_mode="r") for k, f in emb_files.items()}
    for k, arr in list(mmaps.items()) + [("per_residue_flat", per_res_flat),
                                         ("pairdot_only", pair_dot)]:
        if arr.shape[0] != n:
            raise ValueError(f"{k} 行数 {arr.shape[0]} != {n}")

    def get_X(feature, idx):
        if feature == "onehot_80d":
            return one_hot_encode(variants[idx])
        if feature == "per_residue_flat":
            return np.asarray(per_res_flat[idx])
        if feature == "pairdot_only":
            return np.asarray(pair_dot[idx])
        if feature == "pairdot_concat":
            return np.concatenate([per_res_flat[idx], pair_dot[idx]], axis=1)
        return np.asarray(mmaps[feature][idx])

    # 固定验证集：RandomState(0) 置换取前 2000；训练子集按种子从其余抽样
    perm = np.random.RandomState(0).permutation(n)
    val_idx = perm[:VAL_SIZE]
    pool = perm[VAL_SIZE:]
    yval = fitness[val_idx]

    total = len(FEATURE_ORDER) * len(TRAIN_SIZES) * len(MODEL_ORDER) * len(SEEDS)
    log(f"统一评估开始：{len(FEATURE_ORDER)} 特征 × {len(MODEL_ORDER)} 模型 × "
        f"{len(TRAIN_SIZES)} 训练量 × {len(SEEDS)} 种子 = {total} 组，验证集固定 {VAL_SIZE} 条")

    runs_path = os.path.join(OUT_DIR, "full_ablation_runs.csv")
    header = "feature,model,train_size,seed,val_spearman,val_mse,fit_seconds,rf_max_features\n"
    done_set = set()
    if args.resume and os.path.exists(runs_path):
        old = pd.read_csv(runs_path)
        for _, r in old.iterrows():
            if args.force_model is None or r["model"] != args.force_model:
                done_set.add((r["feature"], r["model"], int(r["train_size"]), int(r["seed"])))
        log(f"续跑模式：runs.csv 已有 {len(old)} 行，跳过 {len(done_set)} 组"
            + (f"（{args.force_model} 强制重算）" if args.force_model else ""))
        f_mode = "a"
    else:
        f_mode = "w"

    done = 0
    with open(runs_path, f_mode, encoding="utf-8") as f:
        if f_mode == "w":
            f.write(header)

    total_todo = total - len(done_set)
    for feature in FEATURE_ORDER:
        Xval = get_X(feature, val_idx)
        for size in TRAIN_SIZES:
            # 同一 (feature, size) 下三个种子共用训练索引抽样逻辑：RandomState(seed)
            train_sets = {seed: np.random.RandomState(seed).choice(pool, size=size, replace=False)
                          for seed in SEEDS}
            for model_name in MODEL_ORDER:
                for seed in SEEDS:
                    if (feature, model_name, size, seed) in done_set:
                        continue
                    train_idx = train_sets[seed]
                    Xtr = get_X(feature, train_idx)
                    ytr = fitness[train_idx]
                    t_fit = time.perf_counter()
                    model = make_model(model_name, seed)
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xval)
                    fit_s = time.perf_counter() - t_fit
                    rho = float(spearmanr(yval, pred).statistic)
                    mse = float(np.mean((yval - pred) ** 2))
                    done += 1
                    mf_tag = "sqrt" if model_name == "rf" else ""
                    with open(runs_path, "a", encoding="utf-8") as f:
                        f.write(f"{feature},{model_name},{size},{seed},{rho:.6f},{mse:.6f},{fit_s:.1f},{mf_tag}\n")
                    log(f"  [{done:3d}/{total_todo}] {feature:16s} {model_name:5s} n={size:5d} "
                        f"seed={seed:2d}: val rho={rho:.4f} mse={mse:.4f} ({fit_s:.0f}s, "
                        f"累计 {time.time() - t0:.0f}s)")
                    del model, Xtr

    df = pd.read_csv(runs_path)
    full = df.sort_values("val_spearman", ascending=False).reset_index(drop=True)
    full_path = os.path.join(OUT_DIR, "full_ablation.csv")
    full.to_csv(full_path, index=False)

    agg = (df.groupby(["feature", "model", "train_size"])
             .agg(val_spearman_mean=("val_spearman", "mean"),
                  val_spearman_std=("val_spearman", "std"),
                  val_mse_mean=("val_mse", "mean"),
                  fit_seconds_mean=("fit_seconds", "mean"))
             .reset_index()
             .sort_values("val_spearman_mean", ascending=False)
             .reset_index(drop=True))
    summary_path = os.path.join(OUT_DIR, "full_ablation_summary.csv")
    agg.to_csv(summary_path, index=False)

    log("统一评估完成。Top 10（按 mean val Spearman，种子聚合）：")
    for _, r in agg.head(10).iterrows():
        log(f"  {r.feature:16s} {r.model:5s} n={int(r.train_size):5d}: "
            f"{r.val_spearman_mean:.4f} ± {r.val_spearman_std:.4f}")
    log(f"统一评估总耗时 {time.time() - t0:.0f}s；输出：full_ablation.csv（{len(full)} 行）、"
        f"full_ablation_summary.csv（{len(agg)} 行）、full_ablation_runs.csv")


if __name__ == "__main__":
    main()
