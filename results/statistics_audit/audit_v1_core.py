# Independent recomputation of v1 core tables from raw CSVs (subagent I)
import pandas as pd, numpy as np, json, glob, os

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

recs = pd.concat([pd.read_csv(f"results/seed-{s}/run/recommendations.csv") for s in (11, 22, 33)], ignore_index=True)
rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)

print("== basic counts ==")
print("total recommendations:", len(recs))
print("per method:", recs.groupby("method").size().to_dict())
print("unique variants (all seeds/methods):", recs["variant"].nunique())
print("method-rounds:", recs.groupby(["method"]).size().shape[0], "x3 seeds ->", len(rm), "rows in round_metrics")

print("\n== per-round mean fitness (mean over 3 seeds, sample SD ddof=1) ==")
for m in ["random", "model", "llm", "knowledge"]:
    sub = recs[recs.method == m]
    row = []
    for r in (1, 2, 3):
        seed_means = sub[sub["round"] == r].groupby("seed")["actual_fitness"].mean()
        row.append(f"R{r}: {seed_means.mean():.3f} ± {seed_means.std(ddof=1):.3f}")
    print(f"{m:10s}", " | ".join(row))

print("\n== 新增最高 fitness per seed (max over that seed's 30 recs of the method) ==")
for m in ["random", "model", "llm", "knowledge"]:
    per_seed = recs[recs.method == m].groupby("seed")["actual_fitness"].max()
    print(f"{m:10s}", {k: round(v, 4) for k, v in per_seed.items()},
          f"mean={per_seed.mean():.3f} sd={per_seed.std(ddof=1):.3f}")

print("\n== initial best per seed (from round_metrics cumulative / state?) ==")
# initial best: check selection.json per seed
for s in (11, 22, 33):
    sel = json.load(open(f"results/seed-{s}/selection/selection.json"))
    keys = list(sel.keys())
    # search for initial best info
    txt = json.dumps(sel)
    print(s, "selection keys:", keys)

print("\n== 超过初始最佳 (final cumulative best > initial best) ==")
for s in (11, 22, 33):
    st = json.load(open(f"results/seed-{s}/run/state.json"))
    # look for initial known best
    def find_key(obj, key, path=""):
        hits = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if key in str(k).lower():
                    hits.append((path + "/" + str(k), v if not isinstance(v, (dict, list)) else type(v).__name__))
                hits += find_key(v, key, path + "/" + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:3]):
                hits += find_key(v, key, path + f"[{i}]")
        return hits
    print(s, find_key(st, "initial")[:6])

print("\n== overall mean fitness per method (all 90 recs) ==")
for m in ["random", "model", "llm", "knowledge"]:
    sub = recs[recs.method == m]
    per_seed = sub.groupby("seed")["actual_fitness"].mean()
    print(f"{m:10s} all-rec mean={sub['actual_fitness'].mean():.3f}  seed-mean-of-means={per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f}")

print("\n== 高于WT count (WT fitness=1) ==")
for m in ["random", "model", "llm", "knowledge"]:
    sub = recs[recs.method == m]
    print(f"{m:10s}", int((sub["actual_fitness"] > 1).sum()), "/", len(sub), " (>1);  >=1:", int((sub["actual_fitness"] >= 1).sum()))

print("\n== usage v1 ==")
tot_req, tot_tok = 0, 0
for s in (11, 22, 33):
    u = pd.read_csv(f"results/seed-{s}/run/usage.csv")
    print(s, "usage cols:", u.columns.tolist(), "rows:", len(u))
    tot_req += len(u)
    if "total_tokens" in u.columns:
        tot_tok += pd.to_numeric(u["total_tokens"], errors="coerce").fillna(0).sum()
print("total requests:", tot_req, "total tokens:", tot_tok)
