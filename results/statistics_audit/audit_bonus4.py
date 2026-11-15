import pandas as pd, numpy as np, json, os, glob

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== all bonus-learning recommendations (any depth) ==")
tot_r, tot_n = 0, 0
by = {}
for p in glob.glob(r"results/bonus-learning/**/recommendations.csv", recursive=True):
    n = len(pd.read_csv(p))
    tot_r += 1; tot_n += n
    key = p.split(os.sep)[2] + "/" + os.sep.join(p.split(os.sep)[3:-2])
    by[key] = by.get(key, 0) + n
print("total rounds:", tot_r, "total recs:", tot_n)
print("breakdown:", {k: v for k, v in sorted(by.items())})

print("\n== bonus-graph: recs and usage ==")
tot = 0
req, tok = 0, 0
for p in glob.glob(r"results/bonus-graph/**/recommendations.csv", recursive=True):
    n = len(pd.read_csv(p)); tot += n
for p in glob.glob(r"results/bonus-graph/**/usage.csv", recursive=True):
    u = pd.read_csv(p)
    req += len(u)
    for _, r in u.iterrows():
        try:
            j = json.loads(r["usage"]) if isinstance(r["usage"], str) else r["usage"]
            t = j.get("total_tokens") or (j.get("prompt_tokens", 0) + j.get("completion_tokens", 0))
            tok += t or 0
        except Exception:
            pass
print("graph recs:", tot, "api requests:", req, "tokens:", tok)

# condition_results.csv
cr = pd.read_csv(r"results/bonus-graph/condition_results.csv")
print(cr.to_string()[:1500])
