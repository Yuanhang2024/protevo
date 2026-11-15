import json, os
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
for f in ["rank_batch1.json", "rank_batch2.json", "rank_batch3.json", "summary.json", "free_recall.json"]:
    p = rf"results/critique-followup-20261113/zero_context/{f}"
    d = json.load(open(p, encoding="utf-8"))
    s = json.dumps(d, ensure_ascii=False, indent=1)
    print("=====", f, "=====")
    print(s[:2600])
    print()

print("===== no_feedback/summary.json =====")
d = json.load(open(r"results/critique-followup-20261113/no_feedback/summary.json", encoding="utf-8"))
print(json.dumps(d, ensure_ascii=False, indent=1)[:3000])
