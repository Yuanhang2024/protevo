import numpy as np, pandas as pd, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
pub = {"mean_fitness": {"random": {"random": 1.0, "model": 0.220784, "llm": 0.007635, "knowledge": 0.05175},
                        "model": {"random": 0.220784, "model": 1.0, "llm": 0.576973, "knowledge": 0.919384},
                        "llm": {"random": 0.007635, "model": 0.576973, "llm": 1.0, "knowledge": 0.919384},
                        "knowledge": {"random": 0.05175, "model": 0.919384, "llm": 0.919384, "knowledge": 1.0}}}

for metric in ("mean_fitness", "max_fitness", "new_high_fitness_count"):
    piv = rm.pivot_table(index=["seed", "round"], columns="method", values=metric)
    long_df = piv.reset_index().melt(id_vars=["seed", "round"], var_name="group", value_name="val")
    long_df["block"] = long_df["seed"].astype(str) + "-" + long_df["round"].astype(str)
    res = sp.posthoc_nemenyi(long_df, val_col="val", group_col="group", block_col="block")
    order = ["random", "model", "llm", "knowledge"]
    res = res.reindex(index=order, columns=order)
    print(f"== {metric} (block Nemenyi) ==")
    print(res.round(6).to_string())
    if metric in pub:
        p = pd.DataFrame(pub[metric])
        print("max abs diff vs published:", round((res - p).abs().max().max(), 8))
    print()
