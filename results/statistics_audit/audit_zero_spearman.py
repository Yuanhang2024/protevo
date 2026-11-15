# Independent recomputation of zero-context Spearman direction (subagent I)
# Uses raw S1 data or processed full fitness. First find the full landscape fitness table.
import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

# The samples() in zero_context.py used global fitness over variant-ID-sorted measured S1.
# analyze_stats.load_global_fitness - check its source
src = open(r"results/critique-followup-20261113/analyze_stats.py", encoding="utf-8").read()
idx = src.find("def load_global_fitness")
print(src[idx:idx+800])
