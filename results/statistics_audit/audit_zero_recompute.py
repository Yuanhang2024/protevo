# Independent Spearman recomputation for zero-context batches (subagent I)
import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr, kendalltau

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

table = pd.read_excel(r"data/raw/elife-16965-supp1-v4.xlsx", sheet_name="Supplementary Dataset 1")
variants = table["Variants"].astype(str).str.upper().str.strip()
fitness = pd.to_numeric(table["Fitness"], errors="coerce")
keep = variants.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}") & fitness.notna() & np.isfinite(fitness) & (fitness >= 0)
frame = pd.DataFrame({"variant": variants[keep].values, "fitness": fitness[keep].astype(float).values})
frame = frame.groupby("variant", as_index=False)["fitness"].first()
print("S1 unique variants:", len(frame))
fit = frame.set_index("variant")["fitness"].to_dict()

for b in (1, 2, 3):
    d = json.load(open(rf"results/critique-followup-20261113/zero_context/rank_batch{b}.json", encoding="utf-8"))
    ranked = d["ranked_variants"]
    pos = np.arange(len(ranked))
    f = np.array([fit[v] for v in ranked])
    rho_raw = spearmanr(pos, f).statistic
    tau = kendalltau(pos, f).statistic
    # top-10 intersection: model top10 (first 10) vs actual top10 in batch
    model_top10 = set(ranked[:10])
    actual_top10 = set(np.array(ranked)[np.argsort(-f)[:10]])
    inter = len(model_top10 & actual_top10)
    print(f"batch {b}: n={len(ranked)} raw spearman(position,fitness)={rho_raw:+.4f} "
          f"high-score-direction={-rho_raw:+.4f} kendall={tau:+.4f} top10_overlap={inter}")

# global top10/top100 check for free recall
order = frame.sort_values(["fitness", "variant"], ascending=[False, True])
g10 = set(order["variant"].head(10)); g100 = set(order["variant"].head(100))
fr = json.load(open(r"results/critique-followup-20261113/zero_context/free_recall.json", encoding="utf-8"))["top_variants"]
print("free recall n:", len(fr), "in global top10:", len([v for v in fr if v in g10]), "in top100:", len([v for v in fr if v in g100]))
print("global top10 list:", order["variant"].head(10).tolist())
