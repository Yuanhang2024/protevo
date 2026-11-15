"""Predictor v2: ESM-2 embedding based fitness prediction vs RF + one-hot baseline.

Layers: ESM-2 pooled L20/L25/L30/L33 (1280-d) + concat L33 (5120-d).
Models: (a) Ridge(alpha=1.0) raw; (b) StandardScaler + PCA(256) + Ridge(alpha=1.0);
        (c) shallow MLP 1280->64->1 ReLU (sklearn MLPRegressor, weight decay +
            early stopping as dropout substitute -- sklearn has no dropout).
Seeds: 11, 22, 33 (same splits as prior experiments).
Metrics: validation (and hidden test) Spearman, MSE, Top-10 hit rate.
Outputs: results/predictor_v2/model_comparison.csv, best_model.joblib, summary.json.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)
EMB_DIR = os.path.join(ROOT, "data", "esm2_embeddings")
PROC_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "results", "predictor_v2")
SEEDS = [11, 22, 33]
TOP_K = 10
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

LAYER_FILES = {
    "L20": "esm2_pooled_L20.npy",
    "L25": "esm2_pooled_L25.npy",
    "L30": "esm2_pooled_L30.npy",
    "L33": "esm2_pooled_L33.npy",
    "concat_L33": "esm2_concat_L33.npy",
}


def topk_hit_rate(y_true, y_pred, k=TOP_K):
    """Fraction of the true top-k variants recovered in the predicted top-k."""
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def evaluate(y_true, y_pred):
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "top10_hit": float(topk_hit_rate(y_true, y_pred)),
    }


def make_model(name, seed):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "pca_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=256, random_state=0)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=1e-4,  # weight decay: dropout substitute (sklearn has no dropout)
                batch_size=128,
                early_stopping=True,
                n_iter_no_change=20,
                max_iter=500,
                random_state=seed,
            )),
        ])
    raise ValueError(name)


def one_hot_encode(variants):
    """4-position x 20-AA one-hot (80 dims), consistent with the RF baseline."""
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    X = np.zeros((len(variants), 4 * 20), dtype=np.float32)
    for r, v in enumerate(variants):
        for p, ch in enumerate(v):
            X[r, p * 20 + aa_to_idx[ch]] = 1.0
    return X


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # 1. variant -> row index map (embedding rows follow all_variants.csv order)
    all_v = pd.read_csv(os.path.join(PROC_DIR, "all_variants.csv"))
    assert all_v["variant"].is_unique, "variants must be unique"
    var_to_row = {v: i for i, v in enumerate(all_v["variant"])}
    print(f"all_variants: {len(all_v)} rows, unique variants OK")

    # 2. embeddings via mmap; gather rows per split on demand
    emb_mmaps = {k: np.load(os.path.join(EMB_DIR, f), mmap_mode="r")
                 for k, f in LAYER_FILES.items()}

    def gather(layer, variants):
        rows = np.fromiter((var_to_row[v] for v in variants), dtype=np.int64,
                           count=len(variants))
        return np.asarray(emb_mmaps[layer][rows])

    # 3-4. loop seeds x layers x models
    records = []
    model_names = ["ridge", "pca_ridge", "mlp"]
    splits = {}
    for seed in SEEDS:
        sp = os.path.join(PROC_DIR, f"seed-{seed}")
        splits[seed] = {
            "train": pd.read_csv(os.path.join(sp, "train.csv")),
            "val": pd.read_csv(os.path.join(sp, "validation.csv")),
            "test": pd.read_csv(os.path.join(sp, "hidden", "test.csv")),
        }
    # sanity: no split overlap
    for seed in SEEDS:
        s = splits[seed]
        assert not (set(s["train"].variant) & set(s["val"].variant))
        assert not (set(s["train"].variant) & set(s["test"].variant))

    for layer in LAYER_FILES:
        for model_name in model_names:
            for seed in SEEDS:
                s = splits[seed]
                Xtr = gather(layer, s["train"].variant)
                Xva = gather(layer, s["val"].variant)
                Xte = gather(layer, s["test"].variant)
                ytr = s["train"].fitness.to_numpy()
                yva = s["val"].fitness.to_numpy()
                yte = s["test"].fitness.to_numpy()

                model = make_model(model_name, seed)
                model.fit(Xtr, ytr)
                m_val = evaluate(yva, model.predict(Xva))
                m_test = evaluate(yte, model.predict(Xte))
                records.append({"layer": layer, "model": model_name, "seed": seed,
                                **{f"val_{k}": v for k, v in m_val.items()},
                                **{f"test_{k}": v for k, v in m_test.items()}})
                print(f"[{time.time()-t0:7.1f}s] {layer:10s} {model_name:9s} "
                      f"seed {seed}: val rho={m_val['spearman']:.4f} "
                      f"mse={m_val['mse']:.4f} top10={m_val['top10_hit']:.1f} | "
                      f"test rho={m_test['spearman']:.4f}")

    # 7. RF + one-hot baseline on identical splits
    for seed in SEEDS:
        s = splits[seed]
        Xtr = one_hot_encode(s["train"].variant)
        Xva = one_hot_encode(s["val"].variant)
        Xte = one_hot_encode(s["test"].variant)
        ytr = s["train"].fitness.to_numpy()
        yva = s["val"].fitness.to_numpy()
        yte = s["test"].fitness.to_numpy()
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(Xtr, ytr)
        m_val = evaluate(yva, rf.predict(Xva))
        m_test = evaluate(yte, rf.predict(Xte))
        records.append({"layer": "onehot", "model": "random_forest", "seed": seed,
                        **{f"val_{k}": v for k, v in m_val.items()},
                        **{f"test_{k}": v for k, v in m_test.items()}})
        print(f"[{time.time()-t0:7.1f}s] onehot     rf        seed {seed}: "
              f"val rho={m_val['spearman']:.4f} mse={m_val['mse']:.4f} "
              f"top10={m_val['top10_hit']:.1f} | test rho={m_test['spearman']:.4f}")

    # 5. aggregate over seeds
    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c.startswith(("val_", "test_"))]
    agg = (df.groupby(["layer", "model"])[metric_cols]
             .mean().reset_index())
    agg["std_val_spearman"] = (df.groupby(["layer", "model"])["val_spearman"]
                               .std().reset_index(drop=True))
    agg = agg.sort_values("val_spearman", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    agg.to_csv(csv_path, index=False)

    # 6. best (layer, model) by mean val Spearman
    best = agg.iloc[0]
    print("\n" + "=" * 96)
    print("Mean over seeds [11, 22, 33]  (sorted by val Spearman)")
    print("=" * 96)
    hdr = (f"{'layer':<10s} {'model':<14s} {'val_rho':>8s} {'val_MSE':>9s} "
           f"{'val_top10':>9s} | {'test_rho':>8s} {'test_MSE':>9s} {'test_top10':>10s}")
    print(hdr)
    print("-" * 96)
    for _, r in agg.iterrows():
        print(f"{r.layer:<10s} {r.model:<14s} {r.val_spearman:8.4f} "
              f"{r.val_mse:9.4f} {r.val_top10_hit:9.2f} | "
              f"{r.test_spearman:8.4f} {r.test_mse:9.4f} {r.test_top10_hit:10.2f}")
    print("=" * 96)
    print(f"BEST: layer={best.layer}, model={best.model}, "
          f"val Spearman={best.val_spearman:.4f} (+-{best.std_val_spearman:.4f}), "
          f"val top10={best.val_top10_hit:.2f}")

    rf_row = agg[(agg.layer == "onehot") & (agg.model == "random_forest")].iloc[0]
    print(f"RF baseline: val Spearman={rf_row.val_spearman:.4f}, "
          f"val top10={rf_row.val_top10_hit:.2f}")
    esm_best = agg[agg.layer != "onehot"].iloc[0]
    print(f"Best ESM-2 combo: layer={esm_best.layer}, model={esm_best.model}, "
          f"val Spearman={esm_best.val_spearman:.4f}")
    verdict = ("ESM-2 WINS" if esm_best.val_spearman > rf_row.val_spearman
               else "RF one-hot WINS (negative result for ESM-2 on this task)")
    print(f"Comparison verdict: {verdict}")

    # 9. save models (refit on seed-11 train, the canonical seed)
    # best_model.joblib = overall winner; esm2_best_model.joblib = best ESM-2 combo
    def fit_bundle(row):
        seed = 11
        s = splits[seed]
        if row.layer == "onehot":
            Xtr = one_hot_encode(s["train"].variant)
            est = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        else:
            Xtr = gather(row.layer, s["train"].variant)
            est = make_model(row.model, seed)
        est.fit(Xtr, s["train"].fitness.to_numpy())
        return {
            "estimator": est,
            "layer": row.layer,
            "layer_file": LAYER_FILES.get(row.layer, None),
            "model_name": row.model,
            "fit_seed": seed,
            "feature_note": ("4x20 one-hot" if row.layer == "onehot" else
                             "embedding rows follow data/processed/all_variants.csv order"),
            "val_spearman_mean_3seeds": float(row.val_spearman),
            "val_top10_mean_3seeds": float(row.val_top10_hit),
        }

    overall_best = best if best.layer != "onehot" else rf_row
    joblib.dump(fit_bundle(overall_best), os.path.join(OUT_DIR, "best_model.joblib"))
    joblib.dump(fit_bundle(esm_best), os.path.join(OUT_DIR, "esm2_best_model.joblib"))

    summary = {
        "seeds": SEEDS,
        "embeddings": LAYER_FILES,
        "dropout_note": ("sklearn MLPRegressor has no dropout; used alpha=1e-4 "
                         "weight decay + early_stopping as the regularizer"),
        "best_layer": best.layer,
        "best_model": best.model,
        "best_val_spearman": float(best.val_spearman),
        "best_val_spearman_std": float(best.std_val_spearman),
        "best_esm2_layer": esm_best.layer,
        "best_esm2_model": esm_best.model,
        "best_esm2_val_spearman": float(esm_best.val_spearman),
        "rf_onehot_val_spearman": float(rf_row.val_spearman),
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed_records": records,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'best_model.joblib')} (overall best: "
          f"{overall_best.layer}/{overall_best.model})")
    print(f"Saved: {os.path.join(OUT_DIR, 'esm2_best_model.joblib')} "
          f"(best ESM-2: {esm_best.layer}/{esm_best.model})")
    print(f"Saved: {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
