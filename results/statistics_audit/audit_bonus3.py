import pandas as pd, numpy as np, json, os, glob

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== bonus-learning: mean_plus_std vs greedy mean fitness across seeds 44/55/66 ==")
for arm in ("mean_plus_std", "greedy"):
    vals, nrec, nrounds = [], 0, 0
    for s in (44, 55, 66):
        base = rf"results/bonus-learning/seed-{s}/arms/global/{arm}"
        recs = glob.glob(base + "/round-*/recommendations.csv")
        for p in recs:
            df = pd.read_csv(p)
            nrec += len(df)
            nrounds += 1
            vals.append(df["actual_fitness"].mean())
    vals = np.array(vals)
    # README claims "mean_plus_std的推荐均值为3.4787，greedy为3.4194" - likely pooled mean over all recs
    print(arm, "rounds:", nrounds, "recs:", nrec, "mean of round means:", round(vals.mean(), 4))

# pooled over all recs
for arm in ("mean_plus_std", "greedy"):
    allf = []
    for s in (44, 55, 66):
        for p in glob.glob(rf"results/bonus-learning/seed-{s}/arms/global/{arm}/round-*/recommendations.csv"):
            allf.append(pd.read_csv(p)["actual_fitness"].values)
    f = np.concatenate(allf)
    print(arm, "pooled mean over recs:", round(f.mean(), 4), "n =", len(f))

print("\n== total rounds/recs bonus (108 rounds, 1080 recs, seeds 44/55/66) ==")
tot_r, tot_n = 0, 0
for s in (44, 55, 66):
    for p in glob.glob(rf"results/bonus-learning/seed-{s}/arms/*/round-*/recommendations.csv"):
        tot_r += 1
        tot_n += len(pd.read_csv(p))
print("all arms rounds:", tot_r, "recs:", tot_n)

print("\n== bonus-graph: 6 conditions x 10 recs, 11 API requests, 142187 tokens ==")
g = glob.glob(r"results/bonus-graph/**/*.csv", recursive=True) + glob.glob(r"results/bonus-graph/**/*.json", recursive=True)
print([os.path.relpath(x, "results/bonus-graph") for x in g][:20])
