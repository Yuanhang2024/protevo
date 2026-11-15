import numpy as np, pandas as pd, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
wide = piv.reset_index(drop=True)[["random", "model", "llm", "knowledge"]].astype(float)

# long format
long_df = wide.reset_index().melt(id_vars="index", var_name="group", value_name="val")
res = sp.posthoc_nemenyi(long_df, val_col="val", group_col="group")
res = res.reindex(index=["random", "model", "llm", "knowledge"], columns=["random", "model", "llm", "knowledge"])
print("long-format posthoc_nemenyi (mean_fitness):")
print(res.round(6).to_string())

pub = pd.DataFrame({"random": {"random": 1.0, "model": 0.220784, "llm": 0.007635, "knowledge": 0.05175},
                    "model": {"random": 0.220784, "model": 1.0, "llm": 0.576973, "knowledge": 0.919384},
                    "llm": {"random": 0.007635, "model": 0.576973, "llm": 1.0, "knowledge": 0.919384},
                    "knowledge": {"random": 0.05175, "model": 0.919384, "llm": 0.919384, "knowledge": 1.0}})
try:
    print("max abs diff vs published:", (res - pub).abs().max().max())
except Exception as e:
    print("compare err:", e)
