import json, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

def first_answer_proposals(path):
    tot, rounds = 0, 0
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "llm_response" and e.get("method") in ("llm", "knowledge") and e.get("attempt", 1) == 1:
            resp = e.get("response", {})
            txt = resp.get("text") or ""
            try:
                parsed = json.loads(txt)
                props = parsed.get("proposals", [])
            except Exception:
                props = resp.get("proposals", [])
            tot += len(props)
            rounds += 1
    return tot, rounds

g1, r1 = 0, 0
g3, r3 = 0, 0
for s in (11, 22, 33):
    a, b = first_answer_proposals(f"results/seed-{s}/run/events.jsonl")
    g1 += a; r1 += b
    a, b = first_answer_proposals(f"results/prompt-v3/seed-{s}/run/events.jsonl")
    g3 += a; r3 += b
print("v1 first-answer proposals:", g1, "in", r1, "method-round responses")
print("v3 first-answer proposals:", g3, "in", r3, "method-round responses")
print("combined:", g1 + g3, "in", r1 + r3, "method-rounds")
