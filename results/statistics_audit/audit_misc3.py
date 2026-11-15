import pandas as pd, numpy as np, json, os
from scipy.stats import spearmanr

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

def proposals(path, methods=("llm", "knowledge")):
    out = []
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "llm_response" and e.get("method") in methods:
            resp = e.get("response", {})
            txt = resp.get("text") or ""
            try:
                parsed = json.loads(txt)
                props = parsed.get("proposals", [])
            except Exception:
                props = resp.get("proposals", [])
            out.extend(props)
    return out

allv1, allv3 = [], []
for s in (11, 22, 33):
    allv1 += proposals(f"results/seed-{s}/run/events.jsonl")
    allv3 += proposals(f"results/prompt-v3/seed-{s}/run/events.jsonl")
comb = allv1 + allv3
def vars_of(props):
    out = []
    for p in props:
        v = p.get("variant") or p.get("variants") or p.get("sequence_4mer")
        out.append(v)
    return out
vv = vars_of(comb)
print("total proposal entries:", len(comb), "with variant field:", sum(1 for x in vv if x))
uniq = {x for x in vv if x}
print("unique variants in v1+v3 proposals:", len(uniq))
# per version
uv1 = {x for x in vars_of(allv1) if x}; uv3 = {x for x in vars_of(allv3) if x}
print("v1 unique:", len(uv1), "v3 unique:", len(uv3), "union:", len(uv1 | uv3))
# first_answer stats: first attempt only per method-round? 修订版: 首答高分重合统计只包括LLM/knowledge 36 method-rounds
print("sample proposal keys:", list(comb[0].keys()))

# ---- ablation token totals ----
print("\n== ablation tokens ==")
sm = json.load(open(r"results/feedback-ablation/summary.json", encoding="utf-8"))
tok = sum(a.get("total_tokens", 0) for a in sm["arms"])
req = sum(a.get("requests", 0) for a in sm["arms"])
print("ablation arms tokens:", tok, "requests:", req)

# v3 fill requests/tokens: v3 runs (prompt-v3) usage: 33 req, 329745 tok total for llm+knowledge.
# README says 本阶段新增补种子23次请求，消融11次，共34次、344,597 tokens
print("v3 total (33 req, 329745 tok) + ablation (", req, "req,", tok, "tok ) =", 33 + req, "req", 329745 + tok, "tok")

# ---- G2 test spearman recompute ----
print("\n== G2 test spearman recomputed from llm-test-predictions.csv ==")
for s in (11, 22, 33):
    df = pd.read_csv(f"results/v2_experiments/G2/seed-{s}/run/llm-test-predictions.csv")
    cols = df.columns.tolist()
    rho = spearmanr(df[cols[-2]], df[cols[-1]]).statistic
    print(s, cols, "rho=", round(rho, 7))

# ---- v1 llm test spearman recompute ----
print("\n== v1 llm test spearman recomputed ==")
for s in (11, 22, 33):
    df = pd.read_csv(f"results/seed-{s}/run/llm-test-predictions.csv")
    cols = df.columns.tolist()
    rho = spearmanr(df[cols[-2]], df[cols[-1]]).statistic
    print(s, cols, "rho=", round(rho, 7))

# ---- all 48 positive check ----
pos, tot = 0, 0
for s in (11, 22, 33):
    for m in ("random", "model", "llm", "knowledge"):
        df = pd.read_csv(f"results/seed-{s}/run/{m}-test-predictions.csv")
        cols = df.columns.tolist()
        rho = spearmanr(df[cols[-2]], df[cols[-1]]).statistic
        tot += 1; pos += rho > 0
print("\nv1 all-methods test spearman positive:", pos, "/", tot)

# ---- default calibration 15/15 ----
print("\n== default-calibration confirmation ==")
import glob
for p in glob.glob(r"results/default-calibration-20261113/*.json") + glob.glob(r"results/default-calibration-20261113/*.md"):
    print(p)
