import numpy as np, pandas as pd, os, json
from scipy.stats import studentized_range
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

try:
    import scikit_posthocs as sp
    print("scikit_posthocs version:", sp.__version__)
except ImportError as e:
    print("not installed:", e)
    sp = None

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
mat = piv[["random", "model", "llm", "knowledge"]].values

if sp:
    res = sp.posthoc_nemenyi(mat)
    res.index = res.columns = ["random", "model", "llm", "knowledge"]
    print("\nmy posthoc_nemenyi (mean_fitness):")
    print(res.round(6).to_string())
    print("\npublished matrix:")
    pub = {"random": {"random": 1.0, "model": 0.220784, "llm": 0.007635, "knowledge": 0.05175},
           "model": {"random": 0.220784, "model": 1.0, "llm": 0.576973, "knowledge": 0.919384},
           "llm": {"random": 0.007635, "model": 0.576973, "llm": 1.0, "knowledge": 0.919384},
           "knowledge": {"random": 0.05175, "model": 0.919384, "llm": 0.919384, "knowledge": 1.0}}
    print(pd.DataFrame(pub).round(6).to_string())
    # compare
    diff = (res.round(6) - pd.DataFrame(pub)).abs().max().max()
    print("max abs diff:", diff)

# also check scikit_posthocs CD via its own critical difference function usage
# and the tukey/hsd style: sp.posthoc_nemenyi uses |z| = diff / sqrt(k(k+1)/(6N)) then
# p = studentized_range.sf equivalent with /sqrt(2)
q = 1.9444 / 0.608581
p1 = 2 * (1 - studentized_range.cdf(q / np.sqrt(2), 4, np.inf))
print("\nmanual with /sqrt(2):", p1)
