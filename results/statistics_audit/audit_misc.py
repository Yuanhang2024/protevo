import pandas as pd, numpy as np, json, os

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)

print("== count candidate entries in v1 events (llm/knowledge only) ==")
tot = 0
per_method = {}
for s in (11, 22, 33):
    for line in open(f"results/seed-{s}/run/events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e.get("kind") == "candidates_selected":
            m = e.get("method")
            cands = e.get("candidates") or e.get("payload", {}).get("candidates") or []
            n = len(cands) if isinstance(cands, list) else 0
            tot += n
            per_method[m] = per_method.get(m, 0) + n
        if e.get("kind") == "llm_response" and tot == 0:
            print("llm_response sample keys:", list(e.keys()))
print("total candidates_selected entries:", tot, per_method)

# if zero, inspect one candidates_selected event
for line in open(r"results/seed-11/run/events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("kind") == "candidates_selected":
        print("candidates_selected keys:", list(e.keys()))
        print(json.dumps(e)[:700])
        break
for line in open(r"results/seed-11/run/events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("kind") == "llm_response":
        print("\nllm_response keys:", list(e.keys()))
        print(json.dumps(e)[:700])
        break
