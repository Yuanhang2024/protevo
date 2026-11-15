import pandas as pd, numpy as np, json, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== count proposals in llm_response events, v1 llm+knowledge ==")
tot, per = 0, {}
for s in (11, 22, 33):
    for line in open(f"results/seed-{s}/run/events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "llm_response" and e.get("method") in ("llm", "knowledge"):
            resp = e.get("response", {})
            txt = resp.get("text") or ""
            try:
                parsed = json.loads(txt)
                props = parsed.get("proposals", [])
            except Exception:
                props = resp.get("proposals", [])
            n = len(props)
            tot += n
            per[e["method"]] = per.get(e["method"], 0) + n
print("v1 llm/knowledge proposal entries:", tot, per)

print("\n== same for v3 ==")
tot3, per3 = 0, {}
for s in (11, 22, 33):
    p = f"results/prompt-v3/seed-{s}/run/events.jsonl"
    if not os.path.exists(p):
        print("missing", p); continue
    for line in open(p, encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "llm_response" and e.get("method") in ("llm", "knowledge"):
            resp = e.get("response", {})
            txt = resp.get("text") or ""
            try:
                parsed = json.loads(txt)
                props = parsed.get("proposals", [])
            except Exception:
                props = resp.get("proposals", [])
            tot3 += len(props)
            per3[e["method"]] = per3.get(e["method"], 0) + len(props)
print("v3 llm/knowledge proposal entries:", tot3, per3)
print("combined v1+v3:", tot + tot3)

print("\n== README v3 per-seed means ==")
v3 = pd.concat([pd.read_csv(f"results/prompt-v3/seed-{s}/run/recommendations.csv") for s in (11, 22, 33)], ignore_index=True)
for m in ("llm", "knowledge"):
    ps = v3[v3.method == m].groupby("seed")["actual_fitness"].mean()
    print(f"v3 {m}: per-seed {dict(ps.round(4))} mean={ps.mean():.4f} sd={ps.std(ddof=1):.4f}")
v1 = pd.concat([pd.read_csv(f"results/seed-{s}/run/recommendations.csv") for s in (11, 22, 33)], ignore_index=True)
for m in ("llm", "knowledge"):
    ps = v1[v1.method == m].groupby("seed")["actual_fitness"].mean()
    print(f"v1 {m}: per-seed {dict(ps.round(4))} mean={ps.mean():.4f} sd={ps.std(ddof=1):.4f}")

print("\n== README: 知识增强每轮有效候选 10-40, 基础Agent 39-40 ==")
rm = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
for m in ("llm", "knowledge"):
    v = rm[rm.method == m]["valid_proposal_count"]
    print(m, "valid_proposal_count min-max:", v.min(), "-", v.max())

print("\n== model_returned in v1 usage ==")
mods = set()
for s in (11, 22, 33):
    u = pd.read_csv(f"results/seed-{s}/run/usage.csv")
    mods |= set(u["model_returned"].dropna().unique())
print("v1 model_returned:", mods)

print("\n== feedback-ablation summary (README: 23+11=34 req, 344,597 tok) ==")
try:
    sm = json.load(open(r"results/feedback-ablation/summary.json", encoding="utf-8"))
    print(json.dumps(sm, ensure_ascii=False)[:900])
except Exception as ex:
    print("summary err", ex)
