import pandas as pd, numpy as np, json, os
from scipy.stats import wilcoxon

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)

print("== tie structure in new_high_count diffs vs random ==")
for m in ["model", "llm", "knowledge"]:
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values="new_high_fitness_count")
    d = (piv[m] - piv["random"]).values
    absd = np.abs(d[d != 0])
    uniq = len(np.unique(absd))
    print(f"{m}-random: diffs={d.tolist()} n_eff={len(absd)} unique|diffs|={uniq} ties={len(absd)-uniq > 0}")
    # exact p (as if no ties)
    ex = wilcoxon(piv[m], piv["random"], zero_method="wilcox", method="exact")
    ap = wilcoxon(piv[m], piv["random"], zero_method="wilcox", method="approx", correction=True)
    print(f"   exact p={ex.pvalue:.7f}  approx p={ap.pvalue:.7f}")

print("\n== v1 events.jsonl candidate entries (llm+knowledge) ==")
tot = 0
for s in (11, 22, 33):
    for line in open(f"results/seed-{s}/run/events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        k = e.get("kind", "")
        if k in ("llm_candidates", "candidates", "proposal"):
            tot += 1
    # print kinds once
kinds = {}
for s in (11, 22, 33):
    for line in open(f"results/seed-{s}/run/events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
print("v1 event kinds:", kinds)
# candidate proposal entries: look at structure of an llm proposal event
for line in open(r"results/seed-11/run/events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("kind") in ("llm_proposal", "proposal", "llm_candidates"):
        s_ = json.dumps(e)[:800]
        print(s_)
        break
