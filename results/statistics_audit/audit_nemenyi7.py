import numpy as np, pandas as pd, os, inspect
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
import scikit_posthocs as sp

print(inspect.signature(sp.posthoc_nemenyi))
print(inspect.getsource(sp.posthoc_nemenyi)[:2200])
