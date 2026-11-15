import numpy as np, pandas as pd, os, inspect
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp
from scipy.stats import studentized_range

rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
piv = rm.pivot_table(index=["seed", "round"], columns="method", values="mean_fitness")
cols = [piv[m].values for m in ("random", "model", "llm", "knowledge")]

# internal helpers
src = inspect.getsource(sp._posthocs)
i = src.find("def __nemenyi")
print(src[i:i+900])
