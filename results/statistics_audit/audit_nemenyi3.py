import numpy as np, pandas as pd, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)

for metric in ("mean_fitness", "max_fitness", "new_high_fitness_count"):
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values=metric)
    wide = piv.reset_index(drop=True)[["random", "model", "llm", "knowledge"]]
    res = sp.posthoc_nemenyi(wide.to_numpy())
    res.index = res.columns = list(wide.columns)
    print(f"\n== {metric} my posthoc_nemenyi ==")
    print(res.round(6).to_string())

pub = {"random": {"random": 1.0, "model": 0.220784, "llm": 0.007635, "knowledge": 0.05175},
       "model": {"random": 0.220784, "model": 1.0, "llm": 0.576973, "knowledge": 0.919384},
       "llm": {"random": 0.007635, "model": 0.576973, "llm": 1.0, "knowledge": 0.919384},
       "knowledge": {"random": 0.05175, "model": 0.919384, "llm": 0.919384, "knowledge": 1.0}}
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
wide = piv.reset_index(drop=True)[["random", "model", "llm", "knowledge"]]
mine = sp.posthoc_nemenyi(wide.to_numpy())
mine.index = mine.columns = list(wide.columns)
print("\nmax abs diff (mine vs published, mean_fitness):", (mine - pd.DataFrame(pub)).abs().max().max())
