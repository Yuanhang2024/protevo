# Independent verification of factorial_summary.csv (subagent I)
import pandas as pd, numpy as np, json, os

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

# G1: v1 llm runs = results/seed-{s}/run (method llm), predictor rf_onehot, prompt mode
# G2: v2_experiments/G2/seed-{s}/run (esm2 ridge, prompt)
# G3/G4: agent_log.jsonl + state.json (tool_calling)

rows = []
# --- G1 ---
for s in (11, 22, 33):
    rm = pd.read_csv(f"results/seed-{s}/run/round_metrics.csv")
    sub = rm[rm.method == "llm"]
    mrf = sub["mean_fitness"].mean()
    fcb = sub[sub["round"] == sub["round"].max()]["cumulative_best"].iloc[0]
    nh = sub["new_high_fitness_count"].sum()
    u = pd.read_csv(f"results/seed-{s}/run/usage.csv")
    u = u[u.method == "llm"]
    calls = len(u); tok = 0
    for _, r in u.iterrows():
        j = json.loads(r["usage"])
        t = j.get("total_tokens") or (j.get("prompt_tokens", 0) + j.get("completion_tokens", 0))
        tok += t or 0
    rows.append(("G1", s, fcb, mrf, nh, calls, tok))
# --- G2 ---
for s in (11, 22, 33):
    rm = pd.read_csv(f"results/v2_experiments/G2/seed-{s}/run/round_metrics.csv")
    mrf = rm["mean_fitness"].mean()
    fcb = rm[rm["round"] == rm["round"].max()]["cumulative_best"].iloc[0]
    nh = rm["new_high_fitness_count"].sum()
    u = pd.read_csv(f"results/v2_experiments/G2/seed-{s}/run/usage.csv")
    calls = len(u); tok = 0
    for _, r in u.iterrows():
        j = json.loads(r["usage"])
        t = j.get("total_tokens") or (j.get("prompt_tokens", 0) + j.get("completion_tokens", 0))
        tok += t or 0
    rows.append(("G2", s, fcb, mrf, nh, calls, tok))
# --- G3/G4 from state.json ---
for g in ("G3", "G4"):
    for s in (11, 22, 33):
        st = json.load(open(f"results/v2_experiments/{g}/seed-{s}/state.json", encoding="utf-8"))
        keys = list(st.keys())
        if g == "G3" and s == 11:
            print("state.json keys:", keys)
        rounds = st.get("rounds") or st.get("completed_rounds") or []
        # extract per-round recommendations
        per_round_means, new_highs, fcb, calls, tok = [], 0, None, 0, 0
        if isinstance(rounds, list):
            for rd in rounds:
                recs = rd.get("recommendations") or []
                fits = [x.get("actual_fitness") for x in recs if x.get("actual_fitness") is not None]
                if fits:
                    per_round_means.append(np.mean(fits))
                new_highs += rd.get("new_high_fitness_count", 0) if isinstance(rd.get("new_high_fitness_count"), int) else sum(1 for x in recs if x.get("is_new_high"))
        calls = st.get("api_calls") or st.get("usage", {}).get("requests") or None
        tok = st.get("total_tokens") or st.get("usage", {}).get("total_tokens")
        fcb = st.get("final_cumulative_best") or st.get("cumulative_best")
        rows.append((g, s, fcb, np.mean(per_round_means) if per_round_means else None,
                     new_highs if isinstance(new_highs, int) else None, calls, tok))

df = pd.DataFrame(rows, columns=["group", "seed", "fcb", "mean_round_fitness", "new_high", "api_calls", "tokens"])
pd.set_option("display.width", 200)
print(df.to_string())

# compare with factorial_summary.csv
fs = pd.read_csv("results/v2_experiments/factorial_summary.csv")
print("\n== published factorial_summary.csv (seed rows) ==")
print(fs[fs["status"] != "n=3"][["group", "seed", "final_cumulative_best", "mean_round_fitness",
                                  "total_new_high_fitness_count", "api_calls", "total_tokens"]].to_string())
print("\n== published mean rows ==")
print(fs[fs["status"] == "n=3"][["group", "final_cumulative_best", "mean_round_fitness",
                                  "total_new_high_fitness_count", "api_calls", "total_tokens"]].to_string())

# recompute means/SD (ddof=1) per group from published seed values and from my recomputation
for g in ("G1", "G2", "G3", "G4"):
    pub = fs[(fs.group == g) & (fs["status"] != "n=3")]
    for col, mycol in [("final_cumulative_best", "fcb"), ("mean_round_fitness", "mean_round_fitness"),
                       ("total_new_high_fitness_count", "new_high"), ("api_calls", "api_calls"), ("total_tokens", "tokens")]:
        pv = pub[col].astype(float).values
        print(f"{g} {col}: pub mean={pv.mean():.6f} sd={pv.std(ddof=1):.6f}")
    mine = df[df.group == g]
    print(f"{g} my  fcb={mine['fcb'].astype(float).mean():.6f}±{mine['fcb'].astype(float).std(ddof=1):.6f}",
          f"mrf={mine['mean_round_fitness'].astype(float).mean():.6f}±{mine['mean_round_fitness'].astype(float).std(ddof=1):.6f}",
          f"newhigh={mine['new_high'].astype(float).mean():.4f} calls={mine['api_calls'].astype(float).mean():.4f} tok={mine['tokens'].astype(float).mean():.1f}")
    print()
