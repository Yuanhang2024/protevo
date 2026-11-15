import pandas as pd, numpy as np, json, os, glob

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

pv = json.load(open(r"results/predictor_v2/summary.json", encoding="utf-8"))
df = pd.DataFrame(pv["per_seed_records"])
print("predictor_v2: n rows:", len(df), "configs:", df.groupby(['layer','model']).ngroups)
print("test_spearman all positive:", bool((df['test_spearman'] > 0).all()), " count:", int((df['test_spearman']>0).sum()), "/", len(df))
g = df.groupby(["layer", "model"])["val_spearman"].agg(["mean", "std", "count"])
print(g.sort_values("mean", ascending=False).head(5).round(6).to_string())

print("\n== bonus numbers ==")
# mean_plus_std 3.4787, greedy 3.4194
for p in glob.glob(r"results/bonus-learning/**/*.csv", recursive=True)[:50]:
    pass
# find selection strategy results
cands = glob.glob(r"results/bonus-learning/*", recursive=True)
print([os.path.basename(x) for x in cands][:30])

# try to find acquisition comparison table
for p in glob.glob(r"results/bonus-learning/**/*acquisition*.csv", recursive=True) + \
         glob.glob(r"results/bonus-learning/**/*mean_plus_std*", recursive=True) + \
         glob.glob(r"results/bonus-learning/**/*summary*", recursive=True):
    print("\n--", p)
    if p.endswith(".csv"):
        print(pd.read_csv(p).to_string()[:1200])
    else:
        print(open(p, encoding="utf-8").read()[:600])
