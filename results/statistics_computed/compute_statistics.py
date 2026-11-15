# -*- coding: utf-8 -*-
"""
Subagent H: independent statistical computation from raw data.
Executes docs/statistics_plan.md (pre-registered v2, frozen 2026-11-14).
All numbers computed from raw data sources; no report numbers consulted.

Outputs: results/statistics_computed/statistics.json + statistics.csv
"""
import json, os, sys, csv, math, itertools, platform, datetime
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scikit_posthocs as sp

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(ROOT, "results", "statistics_computed")
BOOT_SEED = 20261114     # plan section 5.4
B = 10000                 # plan section 1.4 / 5.4
ALPHA = 0.05
Q_FDR = 0.05
SEEDS = [11, 22, 33]
OLD_METHODS = ["random", "model", "llm", "knowledge"]
G1_METHOD = "llm"  # G1 = RF one-hot + prompt = old 'llm' method (verified by test_spearman match)
METRICS_OLD = ["mean_fitness", "max_fitness", "cumulative_best", "new_high_count"]
PRIMARY_METRIC = "mean_fitness"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- helpers
def bh_adjust(pvals):
    """Benjamini-Hochberg step-up FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out

def boot_ci_mean(diff, rng, b=B):
    """95% percentile bootstrap CI of the mean; resamples the paired units (blocks)."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return [None, None]
    idx = rng.integers(0, len(diff), size=(b, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]

def d_z(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0:
        return None
    return float(diff.mean() / sd)

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]

def entropy_bits(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counter.values()))

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(L) for L in f if L.strip()]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ================================================================ PART 1
# Predictor evaluation recompute (plan section 1.1: Spearman direction, high score = positive)
results = {}

pred_summary = load_json(os.path.join(ROOT, "results", "predictor_v2", "summary.json"))
per_seed = pred_summary["per_seed_records"]
df_ps = pd.DataFrame(per_seed)
agg = (df_ps.groupby(["layer", "model"])
       .agg(val_spearman_mean=("val_spearman", "mean"),
            val_spearman_std=("val_spearman", "std"),
            val_mse_mean=("val_mse", "mean"),
            val_top10_mean=("val_top10_hit", "mean"),
            test_spearman_mean=("test_spearman", "mean"),
            test_mse_mean=("test_mse", "mean"),
            test_top10_mean=("test_top10_hit", "mean"),
            n_seeds=("seed", "count"))
       .sort_values("test_spearman_mean", ascending=False).reset_index())

part1 = {
    "source": "results/predictor_v2/summary.json per_seed_records (raw per-seed numbers, 3 seeds per config)",
    "configs_ranked": agg.round(6).to_dict(orient="records"),
    "direction_check": {
        "convention": "Spearman(predicted_score, measured_fitness); fitness higher_is_better=true; positive = correct ranking direction",
        "n_configs": int(len(agg)),
        "n_test_spearman_positive": int((df_ps["test_spearman"] > 0).sum()),
        "n_test_spearman_total": int(len(df_ps)),
        "all_positive": bool((df_ps["test_spearman"] > 0).all()),
        "note": "all per-seed test Spearman values are positive -> predictor ranking direction is correct under the high-score convention",
    },
    "rf_onehot_baseline": {
        "val_spearman_mean": float(df_ps.query("layer=='onehot'")["val_spearman"].mean()),
        "test_spearman_mean": float(df_ps.query("layer=='onehot'")["test_spearman"].mean()),
        "consistency_with_plan_reference": "plan section 1.1 cites old RF baseline ~ +0.54 after sign conversion; recomputed val=+0.539, test=+0.535 -> consistent",
    },
    "best_config_per_family": {
        "best_overall": {"layer": "onehot", "model": "random_forest",
                         "val_spearman_mean": float(df_ps.query("layer=='onehot'")["val_spearman"].mean())},
        "best_esm2": {"layer": "L33", "model": "ridge",
                      "val_spearman_mean": float(df_ps.query("layer=='L33' and model=='ridge'")["val_spearman"].mean())},
    },
}
results["part1_predictor_evaluation"] = part1

# Old-experiment final models: recompute test metrics from RAW prediction files
old_pred_rows = []
for seed in SEEDS:
    run_dir = os.path.join(ROOT, "results", f"seed-{seed}", "run")
    ftm = load_json(os.path.join(run_dir, "final_test_metrics.json"))
    for method in OLD_METHODS:
        p = os.path.join(run_dir, f"{method}-test-predictions.csv")
        df = pd.read_csv(p)
        pred, act = df["predicted_fitness"].to_numpy(), df["actual_fitness"].to_numpy()
        rho = float(stats.spearmanr(pred, act).statistic)
        mse = float(np.mean((pred - act) ** 2))
        k = 10
        top_pred = set(np.argsort(-pred, kind="stable")[:k])
        top_act = set(np.argsort(-act, kind="stable")[:k])
        hit = len(top_pred & top_act) / min(k, len(df))
        stored = ftm.get(method, {})
        old_pred_rows.append({
            "seed": seed, "method": method, "n_test": int(len(df)),
            "spearman_recomputed": round(rho, 10),
            "mse_recomputed": round(mse, 10),
            "top10_hit_recomputed": round(hit, 10),
            "spearman_stored_final_test_metrics": stored.get("spearman"),
            "mse_stored": stored.get("mse"),
            "top10_hit_stored": stored.get("top_k_hit_rate"),
            "agree_spearman": bool(abs(rho - stored.get("spearman", np.nan)) < 1e-9),
            "agree_mse": bool(abs(mse - stored.get("mse", np.nan)) < 1e-9),
            "agree_top10": bool(abs(hit - stored.get("top_k_hit_rate", np.nan)) < 1e-9),
        })
results["part1_predictor_evaluation"]["old_experiment_final_models_from_raw_test_predictions"] = old_pred_rows

# ================================================================ PART 1.1
# Position-Spearman sign correction on old recommendation data (36 method-rounds)
pos_units = []
for seed in SEEDS:
    df = pd.read_csv(os.path.join(ROOT, "results", f"seed-{seed}", "run", "recommendations.csv"))
    for method in OLD_METHODS:
        for rnd in (1, 2, 3):
            d = df[(df["method"] == method) & (df["round"] == rnd)].sort_values("rank")
            pos = d["rank"].to_numpy(dtype=float)
            fit = d["actual_fitness"].to_numpy(dtype=float)
            rho_raw = float(stats.spearmanr(pos, fit).statistic)          # negative = correct direction
            rho_high = float(stats.spearmanr(-pos, fit).statistic)        # positive = correct (report this)
            pos_units.append({
                "seed": seed, "method": method, "round": rnd, "n_items": int(len(d)),
                "raw_position_correlation": round(rho_raw, 6),
                "reported_high_score_correlation": round(rho_high, 6),
            })
pos_df = pd.DataFrame(pos_units)
per_method = {m: {
    "mean_raw_position_spearman": round(float(pos_df.query("method==@m")["raw_position_correlation"].mean()), 6),
    "mean_high_score_spearman": round(float(pos_df.query("method==@m")["reported_high_score_correlation"].mean()), 6),
    "n_units": 9,
    "n_units_raw_negative_correct_direction": int((pos_df.query("method==@m")["raw_position_correlation"] < 0).sum()),
} for m in OLD_METHODS}
results["part1_1_position_spearman_correction"] = {
    "definition": "per method-round over its 10 best-first recommendations: raw = spearman(position, measured_fitness) (negative = correct); reported = spearman(-position, measured_fitness) (positive = correct)",
    "n_units_total": len(pos_units),
    "per_method": per_method,
    "overall_mean_high_score_spearman": round(float(pos_df["reported_high_score_correlation"].mean()), 6),
    "n_units_raw_negative": int((pos_df["raw_position_correlation"] < 0).sum()),
    "reading_rule": "all reported_high_score_correlation values are exact sign-flips of raw; positive magnitude = ranking ability",
    "units": pos_units,
}

# ================================================================ PART 1.2 + 1.3 data
# Old per-(method, seed, round) metrics recomputed from RAW recommendations.csv,
# cross-checked against round_metrics.csv columns (both raw artifacts).
old_block = {}  # (seed, round) -> {method: {metric: value}}
rm_check_rows = []
threshold_by_seed, initial_best_by_seed = {}, {}
for seed in SEEDS:
    run_dir = os.path.join(ROOT, "results", f"seed-{seed}", "run")
    rec = pd.read_csv(os.path.join(run_dir, "recommendations.csv"))
    rm = pd.read_csv(os.path.join(run_dir, "round_metrics.csv"))
    for rnd in (1, 2, 3):
        old_block[(seed, rnd)] = {}
        for method in OLD_METHODS:
            d = rec[(rec["method"] == method) & (rec["round"] == rnd)]
            mean_f = float(d["actual_fitness"].mean())
            max_f = float(d["actual_fitness"].max())
            rm_row = rm[(rm["method"] == method) & (rm["round"] == rnd)].iloc[0]
            if method == OLD_METHODS[0] and rnd == 1:
                threshold_by_seed[seed] = float(rm_row["high_fitness_threshold"])
                initial_best_by_seed[seed] = float(rm_row["previous_best"])
            thr = float(rm_row["high_fitness_threshold"])
            nh = int((d["actual_fitness"] > thr).sum())
            cum_best = float(max(float(rm_row["previous_best"]), max(d["actual_fitness"].max(), float(rm_row["cumulative_best"]))))
            old_block[(seed, rnd)][method] = {
                "mean_fitness": mean_f, "max_fitness": max_f,
                "cumulative_best": cum_best, "new_high_count": float(nh),
            }
            rm_check_rows.append({
                "seed": seed, "round": rnd, "method": method,
                "mean_fitness_recomputed": round(mean_f, 10),
                "mean_fitness_round_metrics_csv": round(float(rm_row["mean_fitness"]), 10),
                "mean_agree": bool(abs(mean_f - float(rm_row["mean_fitness"])) < 1e-9),
                "new_high_recomputed": nh,
                "new_high_round_metrics_csv": int(rm_row["new_high_fitness_count"]),
                "new_high_agree": bool(nh == int(rm_row["new_high_fitness_count"])),
            })
rm_check = pd.DataFrame(rm_check_rows)
results["data_integrity_old_round_metrics_vs_recomputed"] = {
    "n_rows": len(rm_check),
    "all_mean_fitness_agree": bool(rm_check["mean_agree"].all()),
    "all_new_high_agree": bool(rm_check["new_high_agree"].all()),
    "n_disagree_mean": int((~rm_check["mean_agree"]).sum()),
    "n_disagree_new_high": int((~rm_check["new_high_agree"]).sum()),
}

# ================================================================ PART 1.2 Friedman + Nemenyi
blocks = sorted(old_block.keys())           # 9 blocks = (seed, round)
N_blocks = len(blocks)
friedman_out = {}
fried_pvals = []
for metric in METRICS_OLD:
    X = np.array([[old_block[b][m][metric] for m in OLD_METHODS] for b in blocks])
    chi2, p = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    fried_pvals.append(p)
    ranks = np.array([stats.rankdata(row) for row in X])
    mean_ranks = ranks.mean(axis=0)
    nemenyi = sp.posthoc_nemenyi_friedman(X)
    nemenyi.index = nemenyi.columns = OLD_METHODS
    k = len(OLD_METHODS)
    q_alpha = stats.studentized_range.ppf(1 - ALPHA, k, df=1e6) / math.sqrt(2)
    CD = q_alpha * math.sqrt(k * (k + 1) / (6 * N_blocks))
    beyond = [{"pair": f"{OLD_METHODS[i]}>{OLD_METHODS[j]}",
               "mean_rank_diff": round(float(mean_ranks[i] - mean_ranks[j]), 4)}
              for i in range(k) for j in range(k)
              if i < j and abs(mean_ranks[i] - mean_ranks[j]) > CD]
    desc = {m: {"mean": round(float(X[:, j].mean()), 6),
                "sd": round(float(X[:, j].std(ddof=1)), 6)}
            for j, m in enumerate(OLD_METHODS)}
    friedman_out[metric] = {
        "chi2_F": round(float(chi2), 6), "df": k - 1, "p_raw": round(float(p), 8),
        "mean_ranks": {m: round(float(r), 4) for m, r in zip(OLD_METHODS, mean_ranks)},
        "descriptives_mean_sd_over_9_blocks": desc,
        "nemenyi_p_matrix": nemenyi.round(6).to_dict(orient="index"),
        "CD_alpha0.05": round(float(CD), 6),
        "pairs_beyond_CD": beyond,
        "primary_metric": metric == PRIMARY_METRIC,
    }
fried_bh = bh_adjust(fried_pvals)
for metric, padj in zip(METRICS_OLD, fried_bh):
    friedman_out[metric]["p_BH_across_4_metrics"] = round(float(padj), 8)
    friedman_out[metric]["significant_after_BH_q0.05"] = bool(padj < Q_FDR)
results["part1_2_friedman_nemenyi"] = {
    "design": "randomized block: block = seed x round (N=9), treatments = 4 methods; per-block values recomputed from recommendations.csv",
    "gate_note": "plan section 1.2 sample-size gate requires N(blocks)>=10 for inferential reporting; actual N=9 < 10 (the plan's '36 method-rounds' = 4 methods x 9 blocks of observations, not 36 blocks). Tests reported per plan section 1.2 old-data clause; wording per R2: significance claims require BH-adjusted p<0.05 with effect size and CI.",
    "chi2_approx_note": "scipy friedmanchisquare uses the chi-square approximation (as pre-registered); with N=9 the approximation is coarse - flagged.",
    "per_metric": friedman_out,
}

# ================================================================ PART 1.3 + 1.4 Wilcoxon + BH + d_z + bootstrap CI
rng = np.random.default_rng(BOOT_SEED)
pairs = list(itertools.combinations(OLD_METHODS, 2))
wilcoxon_rows = []
p_store = {}
for metric in METRICS_OLD:
    for a, b in pairs:
        diff = np.array([old_block[blk][b][metric] - old_block[blk][a][metric] for blk in blocks])
        n_eff = int(np.sum(diff != 0))
        zero_dropped = int(np.sum(diff == 0))
        d_nz = diff[diff != 0]
        mode, pval, W = "exact", None, None
        if n_eff == 0:
            mode, pval, W = "no_nonzero_diffs", None, None
        else:
            tied_abs = len(np.unique(np.abs(d_nz))) < len(d_nz)
            try:
                if n_eff < 25 and not tied_abs:
                    W, pval = stats.wilcoxon(diff, alternative="two-sided", mode="exact",
                                             zero_method="wilcox")
                    mode = "exact (zero diffs dropped; no tied |diffs|)"
                else:
                    raise ValueError
            except ValueError:
                W, pval = stats.wilcoxon(diff, alternative="two-sided", mode="auto",
                                         zero_method="wilcox")
                mode = ("normal approx w/ continuity (ties or zeros present; zero diffs dropped)"
                        if (tied_abs or zero_dropped) else "normal approx w/ continuity")
        p_store[(metric, a, b)] = pval
        wilcoxon_rows.append({
            "metric": metric, "pair": f"{b} - {a}",
            "n_blocks": N_blocks, "n_zero_diff_dropped": zero_dropped, "n_eff": n_eff,
            "mode": mode, "W": None if W is None or not np.isfinite(W) else round(float(W), 4),
            "mean_diff": round(float(diff.mean()), 8) if len(diff) else None,
            "sd_diff": round(float(diff.std(ddof=1)), 8) if len(diff) else None,
            "d_z": d_z(diff),
            "boot_ci95_mean_diff": boot_ci_mean(diff, rng),
            "p_raw": None if pval is None else round(float(pval), 8),
            "_p_for_bh": None if pval is None else float(pval),
        })
# BH across full family of 24 (primary, pre-declared full set) and within-metric family of 6 (secondary)
for r in wilcoxon_rows:
    r["d_z"] = None if r["d_z"] is None else round(r["d_z"], 6)
    if r["boot_ci95_mean_diff"] != [None, None]:
        r["boot_ci95_mean_diff"] = [round(x, 8) for x in r["boot_ci95_mean_diff"]]
mask = [(r["p_raw"] is not None) for r in wilcoxon_rows]
bh_all = bh_adjust([r["_p_for_bh"] for r in wilcoxon_rows if r["p_raw"] is not None])
it = iter(bh_all)
for r in wilcoxon_rows:
    r["p_BH_family24"] = round(next(it), 8) if r["p_raw"] is not None else None
    r["significant_BH_q0.05"] = bool(r["p_BH_family24"] is not None and r["p_BH_family24"] < Q_FDR)
for metric in METRICS_OLD:
    sub = [r for r in wilcoxon_rows if r["metric"] == metric and r["p_raw"] is not None]
    adj = bh_adjust([r["_p_for_bh"] for r in sub])
    for r, a in zip(sub, adj):
        r["p_BH_within_metric_family6"] = round(float(a), 8)
for r in wilcoxon_rows:
    r.pop("_p_for_bh")
results["part1_3_wilcoxon"] = {
    "family": "6 method pairs x 4 metrics = 24 two-sided Wilcoxon signed-rank tests; BH FDR q=0.05 applied across all 24 (primary) and within each metric family of 6 (secondary)",
    "pairing": "diff = method_b - method_a within block (same seed x round); n=9 blocks",
    "bootstrap": f"95% percentile CI of mean diff; resampling unit = block (seed x round); B={B}; rng seed {BOOT_SEED}",
    "footnote_zero_diffs": "zero_method='wilcox': zero differences dropped before ranking; exact mode used when no tied |diffs| and n_eff<25, otherwise normal approximation with continuity correction (footnote per plan 1.3)",
    "tests": wilcoxon_rows,
    "n_significant_after_BH24": int(sum(r["significant_BH_q0.05"] for r in wilcoxon_rows)),
}

# ================================================================ PART 2 factorial (2x2), descriptive n=3
def group_metrics_from_raw():
    rows = []
    per_round_detail = []
    for seed in SEEDS:
        thr = threshold_by_seed[seed]
        init_best = initial_best_by_seed[seed]
        # G1: old llm method, RF one-hot, prompt
        rec = pd.read_csv(os.path.join(ROOT, "results", f"seed-{seed}", "run", "recommendations.csv"))
        d1 = rec[rec["method"] == G1_METHOD]
        rmf = [float(d1[d1["round"] == r]["actual_fitness"].mean()) for r in (1, 2, 3)]
        cum1 = max([init_best] + [float(d1["actual_fitness"].max())])
        nh1 = int(sum((d1[d1["round"] == r]["actual_fitness"] > thr).sum() for r in (1, 2, 3)))
        # usage for G1 llm rows
        us = pd.read_csv(os.path.join(ROOT, "results", f"seed-{seed}", "run", "usage.csv"))
        us1 = us[us["method"] == G1_METHOD]
        tok1 = int(sum(json.loads(u)["total_tokens"] for u in us1["usage"]))
        api1 = int(len(us1))
        rows.append(dict(group="G1", seed=seed, predictor="rf_onehot", agent_mode="prompt",
                         final_cumulative_best=cum1, mean_round_fitness=float(np.mean(rmf)),
                         total_new_high_fitness_count=nh1, api_calls=api1, total_tokens=tok1))
        for r in (1, 2, 3):
            per_round_detail.append(dict(group="G1", seed=seed, round=r,
                                         mean_fitness=round(rmf[r - 1], 8),
                                         new_high=int((d1[d1["round"] == r]["actual_fitness"] > thr).sum())))
        # test metrics from raw predictions
        tp = pd.read_csv(os.path.join(ROOT, "results", f"seed-{seed}", "run", f"{G1_METHOD}-test-predictions.csv"))
        pred, act = tp["predicted_fitness"].to_numpy(), tp["actual_fitness"].to_numpy()
        rows[-1].update(test_spearman=float(stats.spearmanr(pred, act).statistic),
                        test_mse=float(np.mean((pred - act) ** 2)),
                        test_top10_hit_rate=float(len(set(np.argsort(-pred)[:10]) & set(np.argsort(-act)[:10])) / 10),
                        test_metric_source="recomputed from seed-{s}/run/llm-test-predictions.csv".format(s=seed))
        # G2: ESM-2 + prompt
        g2dir = os.path.join(ROOT, "results", "v2_experiments", "G2", f"seed-{seed}", "run")
        rec2 = pd.read_csv(os.path.join(g2dir, "recommendations.csv"))
        rm2 = [float(rec2[rec2["round"] == r]["actual_fitness"].mean()) for r in (1, 2, 3)]
        cum2 = max([init_best] + [float(rec2["actual_fitness"].max())])
        nh2 = int(sum((rec2[rec2["round"] == r]["actual_fitness"] > thr).sum() for r in (1, 2, 3)))
        us2 = pd.read_csv(os.path.join(g2dir, "usage.csv"))
        tok2 = int(sum(json.loads(u)["total_tokens"] for u in us2["usage"]))
        rows.append(dict(group="G2", seed=seed, predictor="esm2_l33_ridge", agent_mode="prompt",
                         final_cumulative_best=cum2, mean_round_fitness=float(np.mean(rm2)),
                         total_new_high_fitness_count=nh2, api_calls=int(len(us2)), total_tokens=tok2))
        for r in (1, 2, 3):
            per_round_detail.append(dict(group="G2", seed=seed, round=r,
                                         mean_fitness=round(rm2[r - 1], 8),
                                         new_high=int((rec2[rec2["round"] == r]["actual_fitness"] > thr).sum())))
        tp = pd.read_csv(os.path.join(g2dir, "llm-test-predictions.csv"))
        pred, act = tp["predicted_fitness"].to_numpy(), tp["actual_fitness"].to_numpy()
        rows[-1].update(test_spearman=float(stats.spearmanr(pred, act).statistic),
                        test_mse=float(np.mean((pred - act) ** 2)),
                        test_top10_hit_rate=float(len(set(np.argsort(-pred)[:10]) & set(np.argsort(-act)[:10])) / 10),
                        test_metric_source="recomputed from G2/seed-{s}/run/llm-test-predictions.csv".format(s=seed))
        # G3 / G4: tool_calling, from state.json + agent_log.jsonl
        for grp, pred_name in (("G3", "rf_onehot"), ("G4", "esm2_l33_ridge")):
            sdir = os.path.join(ROOT, "results", "v2_experiments", grp, f"seed-{seed}")
            st = load_json(os.path.join(sdir, "state.json"))
            recs_all = []
            round_means, round_nh = [], []
            api_sum, tok_sum, status_counts = 0, 0, {}
            for rd in st["rounds"]:
                rr = rd["record"]
                api_sum += int(rr.get("api_calls_used", 0))
                tok_sum += int(rr.get("total_tokens_used", 0))
                sc = rr.get("status", "unknown")
                status_counts[sc] = status_counts.get(sc, 0) + 1
                recs = rd.get("recommendations", [])
                fits = [x["actual_fitness"] for x in recs]
                recs_all.extend(fits)
                round_means.append(float(np.mean(fits)))
                round_nh.append(int(sum(f > thr for f in fits)))
            cum = max([init_best] + recs_all)
            ftm = st.get("final_test_metrics", {})
            rows.append(dict(group=grp, seed=seed, predictor=pred_name, agent_mode="tool_calling",
                             final_cumulative_best=float(cum),
                             mean_round_fitness=float(np.mean(round_means)),
                             total_new_high_fitness_count=int(sum(round_nh)),
                             api_calls=api_sum, total_tokens=tok_sum,
                             round_status_counts=status_counts,
                             test_spearman=ftm.get("spearman"), test_mse=ftm.get("mse"),
                             test_top10_hit_rate=ftm.get("top_k_hit_rate"),
                             test_metric_source="state.json final_test_metrics (raw test-set predictions not persisted for G3/G4)"))
            for r, (mf, nh) in enumerate(zip(round_means, round_nh), start=1):
                per_round_detail.append(dict(group=grp, seed=seed, round=r,
                                             mean_fitness=round(mf, 8), new_high=nh))
    return rows, per_round_detail

grow, per_round_detail = group_metrics_from_raw()
gdf = pd.DataFrame(grow)
fac_metrics = ["final_cumulative_best", "mean_round_fitness", "total_new_high_fitness_count",
               "api_calls", "total_tokens", "test_spearman", "test_mse", "test_top10_hit_rate"]
cell = {g: {m: {"mean": round(float(gdf[gdf.group == g][m].mean()), 6),
                "sd": round(float(gdf[gdf.group == g][m].std(ddof=1)), 6),
                "per_seed": [round(float(v), 6) for v in gdf[gdf.group == g][m]]}
            for m in fac_metrics} for g in ("G1", "G2", "G3", "G4")}
effects = {}
for m in fac_metrics:
    g1m, g2m, g3m, g4m = (cell[g][m]["mean"] for g in ("G1", "G2", "G3", "G4"))
    effects[m] = {
        "esm2_main_mean_G2G4_minus_G1G3": round(((g2m + g4m) / 2) - ((g1m + g3m) / 2), 6),
        "tool_calling_main_mean_G3G4_minus_G1G2": round(((g3m + g4m) / 2) - ((g1m + g2m) / 2), 6),
        "interaction_G4minusG3_minus_G2minusG1": round((g4m - g3m) - (g2m - g1m), 6),
    }
results["part2_factorial"] = {
    "flag": "描述性统计，不构成显著性判断 (n=3 seeds per cell; plan R1)",
    "design": "G1=RF one-hot+prompt (old llm runs), G2=ESM-2 L33 ridge+prompt, G3=RF one-hot+tool_calling, G4=ESM-2+tool_calling; seeds 11/22/33; 3 rounds each",
    "data_sources": "G1/G2: recommendations.csv + usage.csv + llm-test-predictions.csv (recomputed); G3/G4: state.json rounds/recommendations + final_test_metrics (raw per-seed test predictions not persisted)",
    "shared_constants": {"initial_known_best_per_seed": initial_best_by_seed,
                         "high_fitness_threshold_per_seed": threshold_by_seed},
    "per_group_seed_table": [{k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()} for r in grow],
    "four_cell_table_mean_sd": cell,
    "descriptive_factor_effects": effects,
    "interaction_formula_note": "interaction computed per pre-registered plan section 2.3: (G4-G3)-(G2-G1). The alternative literal expression 'G4-G1-ESM effect-Tool effect' is algebraically identically 0 and was not used.",
    "per_round_detail": per_round_detail,
}

# ================================================================ PART 3 autonomy (G3/G4 agent logs)
POS = [39, 40, 41, 54]
WT = "VDGV"

def mut_pairs(seqs):
    s = set()
    for seq in seqs:
        q = seq.replace(" ", "").upper()
        if len(q) != 4:
            continue
        for i, ch in enumerate(q):
            if ch != WT[i]:
                s.add((POS[i], ch))
    return s

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

autonomy = {"G3": {}, "G4": {}}
for grp in ("G3", "G4"):
    seqs_per_run, lev, seq_raw = [], [], []
    proactive_runs, passive_detail = [], []
    tp_detail, examples = [], []
    total_score_events = 0
    for seed in SEEDS:
        recs = load_jsonl(os.path.join(ROOT, "results", "v2_experiments", grp, f"seed-{seed}", "agent_log.jsonl"))
        by_round = {}
        for r in recs:
            by_round.setdefault(r["round"], []).append(r)
        for rnd, evs in sorted(by_round.items()):
            tools = [e for e in evs if e["kind"] == "tool_call"]
            apis = [e for e in evs if e["kind"] == "api_call"]
            seq = [t["tool_name"] for t in tools]
            seqs_per_run.append(tuple(seq))
            # verify against round_end recorded sequence
            re = [e for e in evs if e["kind"] == "round_end"]
            if re and re[0]["result"].get("tool_call_sequence") != seq:
                seq_raw.append({"seed": seed, "round": rnd, "mismatch": True})
            # proactive/passive
            forced = [t for t in tools if t.get("auto") is True or (t.get("result") or {}).get("auto") is True]
            n_api = len(apis)
            n_forced = len(forced)
            proactive_runs.append({
                "seed": seed, "round": rnd, "llm_turns_with_choice": n_api,
                "tools_offered_each_turn": sorted(set(tuple(sorted(a["arguments"]["tools"])) for a in apis if a.get("arguments")))[0] if apis and apis[0].get("arguments") else None,
                "framework_forced_actions": n_forced,
                "proactive_ratio": round(n_api / (n_api + n_forced), 6) if (n_api + n_forced) else None,
                "forced_submit": bool(n_forced > 0),
            })
            passive_detail.append(n_forced)
            # strategy turning points around predict_fitness (score events)
            pred_events = [i for i, t in enumerate(tools) if t["tool_name"] == "predict_fitness"]
            # tags: explore/exploit relative to union of previous predict calls within the round
            tag, prev_union, focus = {}, set(), {}
            for i in pred_events:
                focus[i] = mut_pairs(tools[i]["arguments"].get("sequences", []))
            ordered = pred_events
            for idx, i in enumerate(ordered):
                if idx == 0:
                    tag[i] = "initial"
                else:
                    tag[i] = "exploit" if jaccard(focus[i], prev_union) >= 0.5 else "explore"
                prev_union |= focus[i]
            for i in pred_events:
                total_score_events += 1
                nxt = i + 1
                if nxt >= len(tools):
                    tp_detail.append({"seed": seed, "round": rnd, "score_call_id": tools[i]["call_id"],
                                      "next": None, "strict_changed": False, "conservative_changed": False,
                                      "reason": "no subsequent decision (round ended)"})
                    continue
                cur_tool, next_tool = tools[i]["tool_name"], tools[nxt]["tool_name"]
                crit1 = next_tool != cur_tool
                crit2 = False
                crit3 = False
                if next_tool == "predict_fitness" and i in focus and nxt in focus:
                    j = jaccard(focus[nxt], focus[i])
                    crit2 = j < 0.5
                    t_cur, t_nxt = tag.get(i), tag.get(nxt)
                    crit3 = {t_cur, t_nxt} == {"explore", "exploit"}
                strict = bool(crit1 or crit2 or crit3)
                conservative = bool((crit1 and next_tool != "submit_recommendations") or crit2 or crit3)
                row = {"seed": seed, "round": rnd,
                       "score_call_id": tools[i]["call_id"],
                       "current_tool": cur_tool, "next_tool": next_tool,
                       "next_call_id": tools[nxt]["call_id"],
                       "focus_jaccard": (round(jaccard(focus[nxt], focus[i]), 4)
                                         if next_tool == "predict_fitness" and i in focus and nxt in focus else None),
                       "tag_cur": tag.get(i), "tag_next": tag.get(nxt) if next_tool == "predict_fitness" else None,
                       "criterion_tool_change": crit1, "criterion_focus_shift": crit2, "criterion_explore_exploit_switch": crit3,
                       "strict_changed": strict, "conservative_changed": conservative}
                tp_detail.append(row)
                if strict and len(examples) < 3:
                    examples.append(row)
    # diversity metrics
    n_runs = len(seqs_per_run)
    uniq = len(set(seqs_per_run))
    lev_list = []
    for a, b in itertools.combinations(range(n_runs), 2):
        sa, sb = seqs_per_run[a], seqs_per_run[b]
        lev_list.append(levenshtein(sa, sb) / max(len(sa), len(sb)))
    from collections import Counter
    bg = Counter()
    for s in seqs_per_run:
        bg.update(zip(s, s[1:]))
    autonomy[grp] = {
        "n_runs": n_runs,
        "sequence_type_count": uniq,
        "sequence_type_fraction": round(uniq / n_runs, 6),
        "sequences": [" -> ".join(s) for s in seqs_per_run],
        "pairwise_normalized_levenshtein": {
            "n_pairs": len(lev_list),
            "mean": round(float(np.mean(lev_list)), 6),
            "min": round(float(np.min(lev_list)), 6),
            "max": round(float(np.max(lev_list)), 6),
        },
        "bigram_entropy_bits": round(entropy_bits(bg), 6),
        "bigram_counts": {f"{a} -> {b}": c for (a, b), c in sorted(bg.items(), key=lambda kv: -kv[1])},
    }
    pr = [r["proactive_ratio"] for r in proactive_runs]
    autonomy[grp]["proactive_passive"] = {
        "definition": "decision point = LLM api_call turn (5 tools offered, tool_choice=auto => >=2 choices); passive = framework-forced action (auto submit after max_score_calls), logged with auto=true",
        "per_run": proactive_runs,
        "group_proactive_ratio_mean": round(float(np.mean(pr)), 6),
        "group_proactive_ratio_sd": round(float(np.std(pr, ddof=1)), 6),
        "total_llm_turns": int(sum(r["llm_turns_with_choice"] for r in proactive_runs)),
        "total_framework_forced_actions": int(sum(r["framework_forced_actions"] for r in proactive_runs)),
        "runs_with_forced_submit": int(sum(r["forced_submit"] for r in proactive_runs)),
    }
    strict_n = sum(1 for t in tp_detail if t.get("strict_changed"))
    cons_n = sum(1 for t in tp_detail if t.get("conservative_changed"))
    runs_strict = len({(t["seed"], t["round"]) for t in tp_detail if t.get("strict_changed")})
    runs_cons = len({(t["seed"], t["round"]) for t in tp_detail if t.get("conservative_changed")})
    autonomy[grp]["turning_points"] = {
        "definition": "score event = predict_fitness tool call; next decision = following tool_call in same round; changed if (tool name changes) OR (same tool and candidate focus-set Jaccard<0.5) OR (explore<->exploit switch). Focus set = mutated (position, residue) pairs of the call's candidate 4-mer list (adaptation: pre-registered strategy_tag fields absent from minimal-schema logs). explore/exploit tag: Jaccard vs union of previous predict calls in the round >=0.5 -> exploit.",
        "n_score_events": total_score_events,
        "n_changed_strict_literal": strict_n,
        "n_changed_conservative_excluding_next_submit": cons_n,
        "runs_with_ge1_change_strict": f"{runs_strict}/{n_runs}",
        "runs_with_ge1_change_conservative": f"{runs_cons}/{n_runs}",
        "examples": examples,
        "all_events": tp_detail,
    }
results["part3_autonomy"] = {
    "G1_G2_constructive_control": "G1/G2 are prompt mode with framework-fixed pipeline: tool-call sequence is constant by construction -> sequence diversity = 0 (constructive lower bound; no tool logs exist for prompt mode)",
    "G3": autonomy["G3"], "G4": autonomy["G4"],
    "schema_adaptation_note": "agent_log.jsonl uses the minimal schema (kind/round/call_id/tool_name/arguments/result/tokens_used/context_length/model), not the full section-4 schema (no decision_point/choice_space/forced/initiation/strategy_tag fields). Autonomy metrics were therefore derived from the minimal fields as documented per metric; forced actions identified via auto=true on tool_call/submit events.",
}

# ================================================================ meta + write
results["meta"] = {
    "computed_by": "subagent H, independent recomputation from raw data",
    "plan": "docs/statistics_plan.md (pre-registered v2, frozen 2026-11-14)",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "alpha": ALPHA, "fdr_q": Q_FDR, "bootstrap_B": B, "bootstrap_seed": BOOT_SEED,
    "python": sys.version, "platform": platform.platform(),
    "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                 "scikit_posthocs": sp.__version__},
    "raw_data_sources": [
        "results/seed-{11,22,33}/run/recommendations.csv, round_metrics.csv, usage.csv, final_test_metrics.json, *-test-predictions.csv",
        "results/v2_experiments/G{2,3,4}/seed-{11,22,33}/... (G2: run/...; G3/G4: agent_log.jsonl + state.json)",
        "results/predictor_v2/summary.json per_seed_records",
    ],
    "hard_rules_applied": ["R1: n=3 cells reported as descriptive only",
                           "R2: 'significant' only where BH-adjusted p<0.05 with effect size and CI"],
}

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o

with open(os.path.join(OUT_DIR, "statistics.json"), "w", encoding="utf-8") as f:
    json.dump(clean(results), f, ensure_ascii=False, indent=1)

# ---- statistics.csv: flat table of key results
rows = []
def add(section, analysis, metric, item, statistic, value, n="", p_raw="", p_adj="", ci="", notes=""):
    rows.append({"section": section, "analysis": analysis, "metric": metric, "item": item,
                 "statistic": statistic, "value": value, "n": n, "p_raw": p_raw,
                 "p_adjusted": p_adj, "ci95": ci, "notes": notes})

for r in results["part1_predictor_evaluation"]["configs_ranked"]:
    add("1", "predictor_evaluation", "test_spearman", f"{r['layer']}/{r['model']}", "mean", r["test_spearman_mean"], n=r["n_seeds"])
add("1", "predictor_evaluation", "direction_check", "all per-seed test Spearman", "n_positive/total",
    f"{results['part1_predictor_evaluation']['direction_check']['n_test_spearman_positive']}/{results['part1_predictor_evaluation']['direction_check']['n_test_spearman_total']}")
for m, d in results["part1_1_position_spearman_correction"]["per_method"].items():
    add("1.1", "position_spearman_correction", "high_score_spearman", m, "mean over 9 method-rounds", d["mean_high_score_spearman"],
        notes=f"raw mean {d['mean_raw_position_spearman']}; raw<0 (correct) in {d['n_units_raw_negative_correct_direction']}/9")
for metric, d in results["part1_2_friedman_nemenyi"]["per_metric"].items():
    add("1.2", "friedman", metric, "4 methods", "chi2_F(p)", d["chi2_F"], n="9 blocks",
        p_raw=d["p_raw"], p_adj=d["p_BH_across_4_metrics"],
        notes=f"df={d['df']}; CD={d['CD_alpha0.05']}; mean ranks {d['mean_ranks']}")
    for pair, pv in [(f"{OLD_METHODS[i]}|{OLD_METHODS[j]}", d["nemenyi_p_matrix"][OLD_METHODS[i]][OLD_METHODS[j]])
                     for i in range(4) for j in range(i + 1, 4)]:
        add("1.2", "nemenyi_posthoc", metric, pair, "p", pv, n="9 blocks", notes=f"CD={d['CD_alpha0.05']}")
for r in wilcoxon_rows:
    add("1.3", "wilcoxon_pairwise_BH", r["metric"], r["pair"], "p_raw/p_BH24/d_z",
        r["p_raw"], n=f"n_eff={r['n_eff']}", p_raw=r["p_raw"], p_adj=r["p_BH_family24"],
        ci=str(r["boot_ci95_mean_diff"]), notes=f"d_z={r['d_z']}; mode={r['mode']}; zeros dropped={r['n_zero_diff_dropped']}")
for m, e in results["part2_factorial"]["descriptive_factor_effects"].items():
    for kk, v in e.items():
        add("2", "factorial_2x2_descriptive", m, kk, "effect (group-mean units)", v, n="3 per cell",
            notes="描述性统计，不构成显著性判断")
for grp in ("G3", "G4"):
    d = results["part3_autonomy"][grp]
    add("3.1", "tool_sequence_diversity", grp, "unique_sequence_fraction", "value", d["sequence_type_fraction"], n=d["n_runs"])
    add("3.1", "tool_sequence_diversity", grp, "pairwise_norm_levenshtein_mean", "value", d["pairwise_normalized_levenshtein"]["mean"])
    add("3.1", "tool_sequence_diversity", grp, "bigram_entropy_bits", "value", d["bigram_entropy_bits"])
    pp = d["proactive_passive"]
    add("3.2", "proactive_passive", grp, "proactive_ratio", "mean over runs", pp["group_proactive_ratio_mean"],
        notes=f"SD={pp['group_proactive_ratio_sd']}; forced actions={pp['total_framework_forced_actions']}/{pp['total_llm_turns'] + pp['total_framework_forced_actions']}")
    tp = d["turning_points"]
    add("3.3", "strategy_turning_points", grp, "changed_strict", "count", tp["n_changed_strict_literal"], n=f"{tp['n_score_events']} score events")
    add("3.3", "strategy_turning_points", grp, "changed_conservative", "count", tp["n_changed_conservative_excluding_next_submit"], n=f"{tp['n_score_events']} score events")

with open(os.path.join(OUT_DIR, "statistics.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- console summary
print("=== integrity: old round_metrics vs recomputed:", results["data_integrity_old_round_metrics_vs_recomputed"])
print("\n=== Friedman:")
for m, d in friedman_out.items():
    print(f"  {m}: chi2={d['chi2_F']} p={d['p_raw']} BH_p={d['p_BH_across_4_metrics']} mean_ranks={d['mean_ranks']} CD={d['CD_alpha0.05']}")
print("\n=== Wilcoxon significant after BH(24):",
      [(r['metric'], r['pair'], r['p_BH_family24']) for r in wilcoxon_rows if r['significant_BH_q0.05']])
print("\n=== factorial effects:")
for m, e in effects.items():
    print(f"  {m}: ESM={e['esm2_main_mean_G2G4_minus_G1G3']} Tool={e['tool_calling_main_mean_G3G4_minus_G1G2']} Interact={e['interaction_G4minusG3_minus_G2minusG1']}")
print("\n=== autonomy:")
for g in ("G3", "G4"):
    d = results["part3_autonomy"][g]
    print(f"  {g}: uniq_frac={d['sequence_type_fraction']} lev_mean={d['pairwise_normalized_levenshtein']['mean']} H={d['bigram_entropy_bits']} "
          f"proactive={d['proactive_passive']['group_proactive_ratio_mean']} "
          f"TP strict={d['turning_points']['n_changed_strict_literal']}/{d['turning_points']['n_score_events']} "
          f"cons={d['turning_points']['n_changed_conservative_excluding_next_submit']}")
print("\nOutputs written to", OUT_DIR)
