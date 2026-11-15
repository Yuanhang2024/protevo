import json, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
st = json.load(open(r"results/v2_experiments/G3/seed-11/state.json", encoding="utf-8"))
print("round1 metrics:", json.dumps(st["rounds"][0]["metrics"], indent=1)[:900])
print("\nround statuses:", [(r["round"], r["status"]) for r in st["rounds"]])

log = open(r"results/v2_experiments/G3/seed-11/agent_log.jsonl", encoding="utf-8").read().strip().splitlines()
events = [json.loads(l) for l in log]
api = [e for e in events if e.get("kind") == "api_call"]
print("\napi_call event keys:", list(api[0].keys()))
print("sample api_call:", json.dumps(api[0], ensure_ascii=False)[:600])
tc = [e for e in events if e.get("kind") == "tool_call"]
print("\ntool_call event keys:", list(tc[0].keys()))
print("sample tool_call:", json.dumps(tc[0], ensure_ascii=False)[:500])
print("\nauto flags on tool_calls:", [(e.get("call_id"), e.get("tool_name"), e.get("auto")) for e in tc][:20])

tot = 0
for e in api:
    tu = e.get("tokens_used")
    if isinstance(tu, dict):
        tot += tu.get("total_tokens") or (tu.get("prompt_tokens", 0) + tu.get("completion_tokens", 0)) or 0
    elif isinstance(tu, (int, float)):
        tot += tu
print("G3/11 total tokens from api_call events:", tot)

for g in ("G3", "G4"):
    for s in (11, 22, 33):
        st2 = json.load(open(rf"results/v2_experiments/{g}/seed-{s}/state.json", encoding="utf-8"))
        nh = sum(r["metrics"].get("new_high_fitness_count", 0) for r in st2["rounds"] if isinstance(r.get("metrics"), dict))
        log2 = [json.loads(l) for l in open(rf"results/v2_experiments/{g}/seed-{s}/agent_log.jsonl", encoding="utf-8").read().strip().splitlines()]
        tok = 0
        for e in log2:
            if e.get("kind") == "api_call":
                tu = e.get("tokens_used")
                if isinstance(tu, dict):
                    tok += tu.get("total_tokens") or (tu.get("prompt_tokens", 0) + tu.get("completion_tokens", 0)) or 0
                elif isinstance(tu, (int, float)):
                    tok += tu
        print(g, s, "new_high_total:", nh, "tokens:", tok)
