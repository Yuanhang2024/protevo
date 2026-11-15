import pandas as pd, numpy as np, json, os, glob

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== 1408 hypothesis: valid/scored candidate counts v1+v3 (llm/knowledge) ==")
rm1 = pd.concat([pd.read_csv(f"results/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
rm3 = pd.concat([pd.read_csv(f"results/prompt-v3/seed-{s}/run/round_metrics.csv") for s in (11, 22, 33)], ignore_index=True)
for name, rm in (("v1", rm1), ("v3", rm3)):
    sub = rm[rm.method.isin(["llm", "knowledge"])]
    print(name, "valid_proposal_count sum:", sub["valid_proposal_count"].sum(),
          "score_count sum:", sub["model_score_count"].sum(),
          "requested sum:", sub["requested_proposal_count"].sum())
print("v1+v3 valid:", rm1[rm1.method.isin(['llm','knowledge'])]['valid_proposal_count'].sum() +
      rm3[rm3.method.isin(['llm','knowledge'])]['valid_proposal_count'].sum())
print("v1+v3 scored:", rm1[rm1.method.isin(['llm','knowledge'])]['model_score_count'].sum() +
      rm3[rm3.method.isin(['llm','knowledge'])]['model_score_count'].sum())
print("v1+v3 requested:", rm1[rm1.method.isin(['llm','knowledge'])]['requested_proposal_count'].sum() +
      rm3[rm3.method.isin(['llm','knowledge'])]['requested_proposal_count'].sum())

print("\n== v3 per-seed requests/tokens ==")
for s in (11, 22, 33):
    u = pd.read_csv(f"results/prompt-v3/seed-{s}/run/usage.csv")
    tok = 0
    for _, r in u.iterrows():
        j = json.loads(r["usage"])
        t = j.get("total_tokens") or (j.get("prompt_tokens", 0) + j.get("completion_tokens", 0))
        tok += t or 0
    print(s, "req:", len(u), "tok:", tok)

print("\n== default-calibration final summary ==")
fs = json.load(open(r"results/default-calibration-20261113/final-summary.json", encoding="utf-8"))
print(json.dumps(fs, ensure_ascii=False, indent=1)[:3500])
