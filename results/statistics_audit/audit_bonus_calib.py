import pandas as pd, numpy as np, json, os, glob, re

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== api_check.json (49 tokens claim) ==")
print(open(r"results/api_check.json", encoding="utf-8").read()[:400])

print("\n== calibration report: 15/15 and 4.7567 / 3.8582 ==")
txt = open(r"results/default-calibration-20261113/默认配置校准报告.md", encoding="utf-8").read()
for pat in ["15/15", "4.7567", "3.8582", "[1, 1]", "[1,1]"]:
    i = txt.find(pat)
    print(f"'{pat}' found at {i}", (txt[max(0,i-120):i+160].replace(chr(10),' ') if i>=0 else ""))

print("\n== predictor_v2 summary spot check ==")
pv = json.load(open(r"results/predictor_v2/summary.json", encoding="utf-8"))
print("keys:", list(pv.keys())[:10])
psr = pv.get("per_seed_records")
if psr:
    df = pd.DataFrame(psr)
    print(df.columns.tolist())
    sub = df[(df["layer"] if "layer" in df.columns else df.iloc[:,0]).astype(str).str.contains("onehot", case=False)] if "layer" in df.columns else df
    # rf onehot rows
    m = df.apply(lambda r: "onehot" in str(r.values) and "random_forest" in str(r.values), axis=1)
    if m.any():
        print(df[m].to_string())
