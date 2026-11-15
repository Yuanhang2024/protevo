import numpy as np, pandas as pd, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
wide = piv.reset_index(drop=True)[["random", "model", "llm", "knowledge"]].astype(float)

# block format: pass transposed? scikit_posthocs block input = DataFrame n_samples x n_groups
res = sp.posthoc_nemenyi(wide.T)  # try transposed (4 x 9): rows=groups?
print("wide.T result shape:", res.shape)
print(res.round(6).to_string())
print()
res2 = sp.posthoc_nemenyi(wide)
print("wide result shape:", res2.shape)
print(res2.round(6).to_string())

pub = pd.DataFrame({"random": {"random": 1.0, "model": 0.220784, "llm": 0.007635, "knowledge": 0.05175},
                    "model": {"random": 0.220784, "model": 1.0, "llm": 0.576973, "knowledge": 0.919384},
                    "llm": {"random": 0.007635, "model": 0.576973, "llm": 1.0, "knowledge": 0.919384},
                    "knowledge": {"random": 0.05175, "model": 0.919384, "llm": 0.919384, "knowledge": 1.0}})
for name, r in (("wide.T", res), ("wide", res2)):
    try:
        print(name, "max abs diff vs published:", (r - pub).abs().max().max())
    except Exception as e:
        print(name, "compare err:", e)
