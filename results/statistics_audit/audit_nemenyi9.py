import numpy as np, pandas as pd, os, re
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp
from scipy.stats import studentized_range

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
cols = [piv[m].values for m in ("random", "model", "llm", "knowledge")]

# Approach A: pooled groups
resA = sp.posthoc_nemenyi(cols, dist="tukey")
resA.index = resA.columns = ["random", "model", "llm", "knowledge"]
print("A pooled-groups tukey:")
print(resA.round(6).to_string())

resA2 = sp.posthoc_nemenyi(cols, dist="chi")
resA2.index = resA2.columns = ["random", "model", "llm", "knowledge"]
print("\nA pooled-groups chi:")
print(resA2.round(6).to_string())

# Approach B: manual with Friedman block ranks, tukey p formula variants
rankdiff = {"random-model": 1.3333 - 2.5, "random-llm": 1.3333 - 3.2778, "random-knowledge": 1.3333 - 2.8889,
            "model-llm": 2.5 - 3.2778, "model-knowledge": 2.5 - 2.8889, "llm-knowledge": 3.2778 - 2.8889}
se = np.sqrt(4 * 5 / (6 * 9))
print("\nB manual variants (q = |d|/se):")
for k, d in rankdiff.items():
    q = abs(d) / se
    p_v1 = 2 * (1 - studentized_range.cdf(q, 4, np.inf))
    p_v2 = 2 * (1 - studentized_range.cdf(q / np.sqrt(2), 4, np.inf))
    p_v3 = 1 - studentized_range.cdf(q, 4, np.inf)
    print(f"{k:16s} d={d:+.4f} q={q:.4f}  p={p_v1:.6f}  p(/sqrt2)={p_v2:.6f}  1-sided={p_v3:.6f}")

print("\npublished: random-model 0.220784, random-llm 0.007635, random-knowledge 0.05175, "
      "model-llm 0.576973, model-knowledge 0.919384, llm-knowledge 0.919384")
