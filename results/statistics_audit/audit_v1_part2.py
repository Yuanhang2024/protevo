# Subagent I: verify initial-best improvement, tokens, selection metrics, v3 counts
import pandas as pd, numpy as np, json, os

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== initial best & final cumulative best per method-seed ==")
for s in (11, 22, 33):
    rm = pd.read_csv(f"results/seed-{s}/run/round_metrics.csv")
    cols = rm.columns.tolist()
    if s == 11:
        print("round_metrics cols:", cols)
    for m in ["random", "model", "llm", "knowledge"]:
        sub = rm[rm.method == m] if "method" in rm.columns else rm
        # final cumulative best
        cb = sub[sub["round"] == sub["round"].max()]
        print(s, m, cb.to_dict("records"))

print("\n== parse usage totals (v1) ==")
grand = 0
for s in (11, 22, 33):
    u = pd.read_csv(f"results/seed-{s}/run/usage.csv")
    for _, row in u.iterrows():
        try:
            j = json.loads(row["usage"])
            t = j.get("total_tokens")
            if t is None and "prompt_tokens" in j and "completion_tokens" in j:
                t = j.get("prompt_tokens", 0) + j.get("completion_tokens", 0)
            grand += t or 0
        except Exception as e:
            print("parse fail", s, e)
    print(s, "running total:", grand)
print("v1 total tokens:", grand)

print("\n== v3 (prompt-v3) counts ==")
v3recs = []
for s in (11, 22, 33):
    p = f"results/prompt-v3/seed-{s}/run/recommendations.csv"
    if os.path.exists(p):
        v3recs.append(pd.read_csv(p))
v3 = pd.concat(v3recs, ignore_index=True)
print("v3 recs:", len(v3), "unique:", v3["variant"].nunique(), "methods:", v3.groupby("method").size().to_dict())
v3_tok, v3_req = 0, 0
for s in (11, 22, 33):
    p = f"results/prompt-v3/seed-{s}/run/usage.csv"
    if os.path.exists(p):
        u = pd.read_csv(p)
        v3_req += len(u)
        for _, row in u.iterrows():
            try:
                j = json.loads(row["usage"])
                t = j.get("total_tokens") or (j.get("prompt_tokens", 0) + j.get("completion_tokens", 0))
                v3_tok += t or 0
            except Exception:
                pass
print("v3 requests:", v3_req, "tokens:", v3_tok)

print("\n== combined v1+v3 ==")
v1 = pd.concat([pd.read_csv(f"results/seed-{s}/run/recommendations.csv") for s in (11, 22, 33)], ignore_index=True)
comb = pd.concat([v1[["variant"]], v3[["variant"]]], ignore_index=True)
print("combined recs:", len(comb), "unique:", comb["variant"].nunique())
print("combined method-rounds: 36+18 =", 36 + len(v3.groupby(["seed", "method", "round"])))
print("combined tokens:", grand + v3_tok, "requests:", 31 + v3_req)

print("\n== v1 score events candidate entries (1408 claim) ==")
# count candidate entries in events.jsonl for llm/knowledge v1
import collections
cnt = 0
for s in (11, 22, 33):
    ev = open(f"results/seed-{s}/run/events.jsonl", encoding="utf-8").read().strip().splitlines()
    for line in ev:
        try:
            e = json.loads(line)
        except Exception:
            continue
        kind = e.get("kind") or e.get("event") or ""
        if "candidate" in str(kind).lower():
            c = e.get("candidates") or e.get("payload", {}).get("candidates")
            if isinstance(c, list):
                cnt += len(c)
print("v1 candidate entries from events (llm+knowledge only... all methods):", cnt)

print("\n== selection.json per-seed predictor metrics ==")
for s in (11, 22, 33):
    sel = json.load(open(f"results/seed-{s}/selection/selection.json"))
    comp = sel.get("comparison")
    # find selected family/leaf + val metrics
    sr = sel.get("selection_rule")
    spec = sel.get("spec")
    print(s, "selected:", json.dumps({k: v for k, v in sel.items() if k in ("config",)}, default=str)[:200])
    # print comparison rows
    if isinstance(comp, list):
        df = pd.DataFrame(comp)
        best = df.sort_values(df.filter(like="spearman").columns[0] if len(df.filter(like="spearman").columns) else 0, ascending=False).head(3)
        print(df.head(10).to_string())
    else:
        print("comparison type:", type(comp))
    break
