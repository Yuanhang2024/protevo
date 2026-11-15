import json, os, numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
st = json.load(open(r"results/v2_experiments/G3/seed-11/state.json", encoding="utf-8"))
print("top keys:", list(st.keys()))
r0 = st["rounds"][0]
print("round keys:", list(r0.keys()))
print({k: (v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} len={len(v)}>") for k, v in r0.items()})
print("final_test_metrics:", st["final_test_metrics"])
rec0 = r0.get("recommendations", [])[0] if r0.get("recommendations") else {}
print("recommendation keys:", list(rec0.keys()) if rec0 else None)
# budget info
print("budget:", st.get("budget"))
# count tokens in agent_log
for g in ("G3", "G4"):
    for s in (11, 22, 33):
        log = open(rf"results/v2_experiments/{g}/seed-{s}/agent_log.jsonl", encoding="utf-8").read().strip().splitlines()
        events = [json.loads(l) for l in log]
        kinds = {}
        tok = 0
        ncalls = 0
        for e in events:
            kinds[e.get("kind")] = kinds.get(e.get("kind"), 0) + 1
            t = e.get("tokens_used")
            if isinstance(t, dict):
                tok += t.get("total_tokens") or (t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)) or 0
            if e.get("kind") == "api_call":
                ncalls += 1
        st2 = json.load(open(rf"results/v2_experiments/{g}/seed-{s}/state.json", encoding="utf-8"))
        rounds = st2["rounds"]
        nh = 0
        for rd in rounds:
            for rec in rd.get("recommendations", []):
                nh += 1 if rec.get("is_new_high") else 0
        best = 0
        allfits = [rec.get("actual_fitness") for rd in rounds for rec in rd.get("recommendations", []) if rec.get("actual_fitness") is not None]
        print(g, s, "events:", kinds, "api_calls:", ncalls, "tokens:", tok, "new_high:", nh, "max_fitness:", max(allfits), "n_recs:", len(allfits))
