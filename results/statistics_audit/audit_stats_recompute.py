# Independent recomputation of statistics.json parts (subagent I) - NOT using subagent H outputs
import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

recs = pd.concat([pd.read_csv(f"results/seed-{s}/run/recommendations.csv") for s in (11, 22, 33)], ignore_index=True)

# ---------- part1_1: position spearman per method-round ----------
units = []
for (s, m, r), g in recs.groupby(["seed", "method", "round"]):
    g = g.sort_values("rank")
    rho = spearmanr(g["rank"].values, g["actual_fitness"].values).statistic
    units.append({"seed": s, "method": m, "round": r, "n": len(g), "raw_rho": rho})
u = pd.DataFrame(units)
print("n units:", len(u))
for m in ["random", "model", "llm", "knowledge"]:
    sub = u[u.method == m]
    print(f"{m:9s} n={len(sub)} mean_raw={sub['raw_rho'].mean():+.6f} mean_high_score={-sub['raw_rho'].mean():+.6f} n_raw_negative={(sub['raw_rho']<0).sum()}")
overall = -u["raw_rho"].mean()
print("overall mean high-score:", round(overall, 6), " n raw negative:", int((u['raw_rho'] < 0).sum()))

# ---------- part1_2 Friedman ----------
rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
blocks = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
print("\nblocks shape:", blocks.shape)
for metric in ["mean_fitness", "max_fitness", "new_high_fitness_count"]:
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values=metric)
    chi2, p = friedmanchisquare(piv["random"], piv["model"], piv["llm"], piv["knowledge"])
    ranks = piv.rank(axis=1, ascending=False).mean()
    print(f"{metric}: chi2={chi2:.6f} p={p:.8f} mean_ranks={dict(ranks.round(4))}")
# cumulative_best: last round cumulative per method
cb = rm[rm["round"] == 3].set_index(["seed", "method"])["cumulative_best"].unstack()
# block value = final cumulative best (same for all rounds within seed? use final)
cbi = cb.reindex(blocks.index)
chi2, p = friedmanchisquare(cbi["random"], cbi["model"], cbi["llm"], cbi["knowledge"])
print(f"cumulative_best(final per seed x 3 rounds dup): chi2={chi2:.6f} p={p:.8f}")
# proper: use per-seed final cumulative (n=3)? plan used 9 blocks with cumulative_best column as-is per round
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="cumulative_best")
chi2, p = friedmanchisquare(piv["random"], piv["model"], piv["llm"], piv["knowledge"])
ranks = piv.rank(axis=1, ascending=False).mean()
print(f"cumulative_best(per-round): chi2={chi2:.6f} p={p:.8f} mean_ranks={dict(ranks.round(4))}")

# Nemenyi CD: k=4, N=9
from scipy.stats import studentized_range
q = studentized_range.ppf(0.95, 4, np.inf)
CD = q * np.sqrt(4 * 5 / (6 * 9))
print(f"Nemenyi q_0.05(k=4)= {q:.6f}, CD = {CD:.6f}")

# descriptives over 9 blocks
for metric in ["mean_fitness", "max_fitness", "new_high_fitness_count", "cumulative_best"]:
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values=metric)
    line = {m: (round(piv[m].mean(), 6), round(piv[m].std(ddof=1), 6)) for m in piv.columns}
    print(metric, line)

# ---------- part1_3 Wilcoxon (recompute a subset) ----------
def wilco(m_a, m_b, metric):
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values=metric)
    d = (piv[m_b] - piv[m_a]).values
    nz = d[d != 0]
    mode_exact = len(np.unique(np.abs(nz))) == len(nz) and len(nz) < 25
    res = wilcoxon(piv[m_b], piv[m_a], zero_method="wilcox",
                   method="exact" if mode_exact else "approx", correction=(not mode_exact))
    dz = np.mean(d) / np.std(d, ddof=1)
    return dict(W=res.statistic, p=res.pvalue, mean_diff=np.mean(d), sd=np.std(d, ddof=1), dz=dz,
                n_zero=len(d) - len(nz), n_eff=len(nz), exact=mode_exact)

checks = [
    ("mean_fitness", "random", "model"),
    ("mean_fitness", "random", "llm"),
    ("mean_fitness", "random", "knowledge"),
    ("mean_fitness", "model", "llm"),
    ("mean_fitness", "model", "knowledge"),
    ("mean_fitness", "llm", "knowledge"),
    ("max_fitness", "random", "llm"),
    ("new_high_count", "random", "model"),
    ("new_high_count", "random", "llm"),
    ("new_high_count", "random", "knowledge"),
    ("cumulative_best", "random", "model"),
]
metric_map = {"new_high_count": "new_high_fitness_count"}
print("\n== my Wilcoxon recomputation ==")
my_tests = []
for met, a, b in checks:
    col = metric_map.get(met, met)
    out = wilco(a, b, col)
    out.update(metric=met, pair=f"{b} - {a}")
    my_tests.append(out)
    print(f"{met:16s} {b}-{a:8s} W={out['W']:7.2f} p={out['p']:.7f} diff={out['mean_diff']:+.5f} dz={out['dz']:+.4f} n0={out['n_zero']} exact={out['exact']}")

# ---------- BH check ----------
# verify family24 BH from published p_raw list in statistics.json (read file but only use it to cross-check my p values)
sj = json.load(open(r"results/statistics_computed/statistics.json", encoding="utf-8"))
pub = sj["part1_3_wilcoxon"]["tests"]
print("\n== published vs my p (subset) ==")
for t in pub:
    key = (t["metric"], t["pair"])
    for mt in my_tests:
        pub_pair = mt["pair"]
        if t["metric"] == mt["metric"] and t["pair"] == pub_pair:
            print(f"{t['metric']:16s} {t['pair']:18s} pub_p={t['p_raw']:.7f} my_p={mt['p']:.7f} "
                  f"pub_W={t['W']} my_W={mt['W']} pub_dz={t['d_z']:+.4f} my_dz={mt['dz']:+.4f} "
                  f"pub_ci={t['boot_ci95_mean_diff']} match={abs(t['p_raw']-mt['p'])<1e-6}")

# BH family24 from my recomputed subset check: verify published BH on one known value
def bh(pvals):
    p = np.asarray(pvals); n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1)
    return out

pub_p = [t["p_raw"] for t in pub]
pub_bh = [t["p_BH_family24"] for t in pub]
my_bh = bh(pub_p)
print("\nBH family24 max abs diff (published vs my BH of published p_raw):", np.abs(my_bh - np.array(pub_bh)).max())
print("n significant BH24 (mine):", int((my_bh < 0.05).sum()), " published:", sj["part1_3_wilcoxon"]["n_significant_after_BH24"])

# ---------- bootstrap CI recompute (mean_fitness knowledge - random) ----------
rng = np.random.default_rng(20261114)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
d = (piv["knowledge"] - piv["random"]).values
bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
ci = np.percentile(bs, [2.5, 97.5])
print(f"\nmy bootstrap CI knowledge-random mean_fitness: [{ci[0]:.5f}, {ci[1]:.5f}] "
      f"(published [0.45574537, 2.31532356])")
