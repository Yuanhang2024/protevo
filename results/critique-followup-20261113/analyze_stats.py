# 批判性审视跟进统计：首答候选与高分变体重合率、假设锚点转移成功率。
# 只读取已保存的实验记录与官方S1，不发起任何模型请求。
import argparse
from itertools import groupby
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from protevo.statistical_definitions import DEFINITION_VERSION, anchor_weighted_summary, threshold_prevalence

AMINO = set("ACDEFGHIKLMNPQRSTVWY")
AA4 = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY]{4})\b")

# 4字母、全由氨基酸字母组成的常见英文词，避免把正文当锚点。
STOP = set("""WHEN WHAT HIGH SAME READ GAIN MATE LINE WELL LESS MAIN MEAN NEAR WIDE WILL
THAT THIS THEN THEY WITH HAVE HERE EACH BOTH SAID MAKE TAKE GAVE GIVE TIDE SIDE
RISE RIDE WAGE WAVE PAGE PANE PINE DINE WINE TWIN TWICE WIRES WIRED OTHER THEIR
THEM THESE THOSE FROM FORM TEAM TEAR TERM TEST TEXT THAT THAN SOME SITE SINS
SING RANG RAIN HAIR HAND HANG HARD HATE HEAD HEAL HEAR HEAT HELD HELL HELP HENS
HERD HERO HIDE HIKE HILL HINT HIRE HOLD HOLE HOME HOPE HOST HOUR HUGE HUNT HURT
IDEA INTO IRON ITEM JOINS JUST KEEN KEEP KEPT KIND KNEW KNOW LACK LADY LAID LAIR
LAND LANE LAST LATE LEAD LEAN LEFT LEND LENT LESS LIES LIFT LIKE LIME LIND LINE
LINK LIST LIVE LOAD LOAN LOCK LOSE LOSS LOST LOUD LOVE LUCK MADE MAIL MAID MALE
MANY MAPS MARE MARK MASS MAST MATE MEAL MEAT MEET MELT MEND MENU MERE MESS MILD
MILE MILK MIND MINE MINT MISS MIST MODE MOOD MOON MORE MOST MOTH MOVE MUCH MULE
MUST NAIL NAME NAVY NEAR NEAT NECK NEED NEST NEWS NEXT NICE NINE NONE NOON NOSE
NOTE NOUN OATH OBEY ODDS ONCE ONLY ONTO OPEN ORAL OURS OVAL OVEN OVER PACE PACK
PAID PAIL PAIN PAIR PALE PALM PAPA PARK PART PASS PAST PATH PEAK PEAL PEAR PEAS
PEEK PEER PENS PENT PERK PEST PICK PIER PINE PINK PINT PIPE PITS PLAN PLAY PLEA
PLOT PLOW PLUG PLUS POEM POET POLL POND POOL POOR PORT POSE POST POUR PRAM PRAY
PREP PREY PROM PROP PROS PROVE PULL PUMP PUNK PURE PUSH QUIT QUITE RACE RACK
RAFT RAGE RAIL RAKE RAMP RANK RANT RARE RASH RATE RAVE READ REAL REAR REED REEF
REEK REIN RELY RENT REST RICE RICH RIDE RIFT RIME RING RINK RIOT RISE RISK RITE
ROAD ROAM ROAR ROCK RODE ROLE ROLL ROOF ROOM ROOT ROPE ROSE ROUT ROVE RUBY RUDE
RUIN RULE RUNG RUSH RUST SAFE SAGE SAID SAIL SAKE SALE SALT SAME SAND SANE SANG
SANK SAVE SCAR SEAL SEAM SEAT SEED SEEK SEEM SEEN SEES SELF SELL SEND SENT SETS
SEWN SHAG SHAM SHED SHIM SHIN SHIP SHOD SHOE SHOP SHOT SHOW SHUN SHUT SICK SIDE
SIFT SIGH SIGN SILK SILL SILO SINK SINS SIRE SITS SITE SIZE SKID SKIM SKIN SKIP
SKIS SLAB SLAM SLAP SLAT SLAV SLAW SLED SLEW SLID SLIM SLIP SLIT SLOB SLOE SLOG
SLOT SLOW SLUG SLUM SMUG SNAG SNAP SNIK SNOB SNOT SNOW SNUB SNAP SOAK SOAP SOAR
SOCK SODA SOFA SOFT SOIL SOLD SOLE SOLO SOME SONG SOON SOOT SORE SORT SOUL SOUP
SOUR SOWN SPAN SPAR SPAS SPAT SPAY SPED SPIN SPIT SPOT SPRY SPUD SPUN SPUR STAB
STAG STAR STAY STEM STEP STEW STIR STOP STOW STUB STUD STUN SUCH SUCK SUIT SULK
SURE SWAB SWAG SWAM SWAN SWAP SWAT SWAY SWIG SWIM TAIL TAKE TALC TALE TALK TALL
TAME TANK TAPE TAPS TARE TARP TART TASK TEAK TEAL TEAM TEAR TEEM TEEN TELL TEMP
TEND TENT TERM TEST TEXT THAN THAT THAW THEE THEM THEN THEY THIN THIS THUS TIDE
TIDY TIED TIER TIES TILE TILL TILT TIME TINE TINS TINT TINY TIPS TIRE TOAD TOGO
TOIL TOLD TOLL TOMB TONE TONS TOOK TOOL TORE TORN TOSS TOTE TOTS TOUR TOUT TOWN
TRAG TRAM TRAP TRAY TREE TRIM TRIO TRIP TROD TROT TRUE TSAR TUBA TUBE TUCK TUFT
TUNA TUNE TURF TURN TUSK TWIG TWIN TWIT TYPE UNIT UPON URGE USED USER VAIL VAIN
VARY VASE VAST VEAL VEER VEIL VEIN VEND VENT VERB VERY VEST VETO VIAL VIBE VICE
VIEW VILE VINE VISA VOID VOLT VOTE WACK WADE WAGE WAIF WAIL WAIT WAKE WALK WALL
WAND WANE WANT WARD WARE WARM WARN WART WARY WASH WASP WAVE WAVY WAYS WEAK WEAN
WEAR WEED WEEK WEEP WELD WELL WENT WEPT WERE WEST WHAT WHEN WHET WHEY WHIM WHIP
WHIR WHIT WHIZ WICK WIDE WIFE WIGS WILD WILE WILL WILT WIMP WIND WINE WING WINK
WINS WIPE WIRE WIRY WISE WISH WISS WITH WITS WOKE WOLF WOMB WONT WOOD WOOL WORD
WORE WORK WORM WORN WRAP WREN YAKS YARD YARN YAWN YEAR YELL YOGA YOUR ZEAL ZERO
ZEST ZINC ZONE""".split())


def load_global_fitness(source=None):
    """官方S1全表：149,361条唯一变体与实测fitness。"""
    source = Path(source) if source is not None else ROOT / "data" / "raw" / "elife-16965-supp1-v4.xlsx"
    table = pd.read_excel(source, sheet_name="Supplementary Dataset 1")
    variants = table["Variants"].astype(str).str.upper().str.strip()
    fitness = pd.to_numeric(table["Fitness"], errors="coerce")
    keep = variants.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{4}") & fitness.notna() & np.isfinite(fitness) & (fitness >= 0)
    frame = pd.DataFrame({"variant": variants[keep], "fitness": fitness[keep].astype(float)})
    frame = frame.groupby("variant", as_index=False)["fitness"].first()
    assert len(frame) == 149361, f"unexpected S1 size {len(frame)}"
    return frame


def first_answer_variants(events_path):
    """每个方法轮次纠正前的首次原始回答；无法解析的回答保留为零候选记录。"""
    events_path = Path(events_path)
    if not events_path.exists():
        return []
    rows, seen = [], set()
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            if obj.get("kind") != "llm_response" or obj.get("attempt") != 1:
                continue
            group = (obj["method"], int(obj["round"]))
            if group in seen:
                raise ValueError("A method-round must have one saved first response")
            seen.add(group)
            proposals, parsed = [], False
            try:
                body = json.loads(obj["response"]["text"])
                if isinstance(body, dict) and isinstance(body.get("proposals"), list):
                    proposals, parsed = body["proposals"], True
            except (ValueError, TypeError, KeyError):
                pass
            variants = [row.get("variant") for row in proposals if isinstance(row, dict)]
            rows.append({"method": group[0], "round": group[1], "variants": variants,
                         "raw_candidate_entries": len(proposals), "parsed": parsed})
    return rows


def overlap_stats(global_fit, runs):
    """按方法轮次保留首答条目；均匀概率由实际达到S1分位阈值的比例给出。"""
    upper = threshold_prevalence(global_fit.fitness, .99)
    lower = threshold_prevalence(global_fit.fitness, .95)
    gf = global_fit.set_index("variant").fitness.to_dict()
    top10 = set(global_fit.sort_values(["fitness", "variant"], ascending=[False, True]).head(10).variant)
    per_run, grouped = [], {}
    for run_label, rows in runs:
        seen = set()
        for row in rows:
            group = (row["method"], int(row["round"]))
            if group in seen:
                raise ValueError("Duplicate source-method-round first response")
            seen.add(group)
            variants = [variant for variant in row["variants"] if isinstance(variant, str) and variant in gf]
            n = len(variants)
            hit99 = sum(gf[variant] >= upper["threshold"] for variant in variants)
            hit95 = sum(gf[variant] >= lower["threshold"] for variant in variants)
            per_run.append({"run": run_label, "method": group[0], "round": group[1],
                            "first_answer_n": n, "raw_candidate_entries": row.get("raw_candidate_entries", len(row["variants"])),
                            "excluded_entries": row.get("raw_candidate_entries", len(row["variants"])) - n,
                            "in_top1pct": int(hit99), "in_top5pct": int(hit95),
                            "in_global_top10": sum(variant in top10 for variant in variants)})
            bucket = grouped.setdefault(group[0], {"rounds": 0, "n": 0, "upper": 0, "lower": 0, "raw": 0})
            bucket["rounds"] += 1
            bucket["n"] += n
            bucket["raw"] += per_run[-1]["raw_candidate_entries"]
            bucket["upper"] += hit99
            bucket["lower"] += hit95
    summary = {}
    for method, value in grouped.items():
        n = value["n"]
        summary[method] = {"method_round_records": value["rounds"], "raw_candidate_entries": value["raw"],
                           "first_answer_candidates": n, "top1pct_rate": value["upper"] / n if n else None,
                           "uniform_expectation_top1pct": upper["uniform_ge_threshold_probability"],
                           "top5pct_rate": value["lower"] / n if n else None,
                           "uniform_expectation_top5pct": lower["uniform_ge_threshold_probability"]}
    return pd.DataFrame(per_run), summary


def known_variant_sets(seed_dir, recs):
    """已测资格集合：初始训练加调用方提供的此前完整轮次记录。"""
    known = set(pd.read_csv(seed_dir / "train.csv").variant.astype(str))
    for row in recs:
        known.add(row["variant"])
    return known


def first_prompt_visibility(events_path):
    """仅提取首次请求中显式带实测值的变体ID，不返回请求正文。"""
    result = {}
    events_path = Path(events_path)
    if not events_path.is_file():
        return result
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") != "llm_request" or event.get("attempt") != 1:
                continue
            group = (event["method"], int(event["round"]))
            if group in result:
                raise ValueError("A method-round must have one saved first request")
            context = event.get("context", {})
            known = context.get("known_data", {})
            records = [row for field in ("top_variants", "low_fitness_examples", "near_median_examples")
                       for row in known.get(field, [])]
            records.extend(context.get("recent_round_feedback", []))
            visible = set()
            for row in records:
                if not isinstance(row, dict) or not isinstance(row.get("variant"), str):
                    continue
                value = row.get("actual_fitness", row.get("fitness"))
                try:
                    measured = value is not None and np.isfinite(float(value))
                except (ValueError, TypeError):
                    measured = False
                if measured:
                    visible.add(row["variant"])
            result[group] = visible
    return result


def summarize_anchors(pairs, scope):
    summary = {}
    for (label, method), group in pairs.groupby(["source", "method"], sort=True):
        summary[f"{label}:{method}"] = {"eligibility_scope": scope, **anchor_weighted_summary(group.to_dict("records"))}
    summary["ALL"] = {"eligibility_scope": scope, **anchor_weighted_summary(pairs.to_dict("records"))}
    return summary


def anchor_stats(global_fit, sources, prompt_visibility=None):
    """按轮前已测资格筛选锚点，并另记其是否在首次请求中显式显示测量值。"""
    gf = global_fit.set_index("variant").fitness.to_dict()
    population = global_fit.fitness.to_numpy(dtype=float)
    pair_rows = []
    for label, recs, seed_dir in sources:
        recs_sorted = sorted(recs, key=lambda row: (row["method"], int(row["round"]), int(row.get("rank", 0))))
        for method, method_rows in groupby(recs_sorted, key=lambda row: row["method"]):
            known = set(pd.read_csv(Path(seed_dir) / "train.csv").variant.astype(str))
            for number, grouped_rows in groupby(method_rows, key=lambda row: int(row["round"])):
                round_rows = list(grouped_rows)
                visible = prompt_visibility.get(label, {}).get((method, number)) if prompt_visibility is not None else None
                for row in round_rows:
                    text = f"{row.get('hypothesis') or ''} {row.get('rationale') or ''}"
                    tokens = [token for token in AA4.findall(text.upper())
                              if token not in STOP and token != row["variant"] and token in known]
                    for anchor in list(dict.fromkeys(tokens))[:3]:
                        if row["variant"] in gf and anchor in gf:
                            candidate_fitness, anchor_fitness = float(gf[row["variant"]]), float(gf[anchor])
                            pair_rows.append({"source": label, "method": method, "round": number,
                                              "variant": row["variant"], "anchor": anchor,
                                              "candidate_fitness": candidate_fitness, "anchor_fitness": anchor_fitness,
                                              "success": int(candidate_fitness >= anchor_fitness),
                                              "random_ge_anchor": float(np.mean(population >= anchor_fitness)),
                                              "measurement_history_eligible": True,
                                              "prompt_visible": anchor in visible if visible is not None else None})
                known.update(row["variant"] for row in round_rows)
    columns = ["source", "method", "round", "variant", "anchor", "candidate_fitness", "anchor_fitness", "success", "random_ge_anchor",
               "measurement_history_eligible", "prompt_visible"]
    pairs = pd.DataFrame(pair_rows, columns=columns)
    return pairs, summarize_anchors(pairs, "measurement_history_eligible")


def recompute(output, root=ROOT, s1=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing statistics are preserved")
    global_fit = load_global_fitness(s1 or root / "data/raw/elife-16965-supp1-v4.xlsx")
    upper, lower = threshold_prevalence(global_fit.fitness, .99), threshold_prevalence(global_fit.fitness, .95)
    quantiles = {"top1pct_fitness": upper["threshold"], "top5pct_fitness": lower["threshold"],
                 "top1pct_population": upper, "top5pct_population": lower,
                 "global_top10": global_fit.sort_values(["fitness", "variant"], ascending=[False, True]).head(10).to_dict("records"),
                 "true_top_ties": "variant ID ascending"}
    runs, prompt_visibility = [], {}
    for seed in (11, 22, 33):
        for profile, path in (("v1", root / f"results/seed-{seed}/run/events.jsonl"),
                              ("v3", root / f"results/prompt-v3/seed-{seed}/run/events.jsonl")):
            if path.exists():
                rows = [row for row in first_answer_variants(path) if row["method"] in ("llm", "knowledge")]
                runs.append((f"{profile}-seed{seed}", rows))
                prompt_visibility[f"{profile}-seed{seed}"] = first_prompt_visibility(path)
    per_run, overlap_summary = overlap_stats(global_fit, runs)
    sources = []
    for profile, path in (("v1", root / "reports/all_recommendations.csv"),
                          ("v3", root / "reports/validation-stage/v3_recommendations.csv")):
        recommendations = pd.read_csv(path).to_dict("records")
        for seed in (11, 22, 33):
            rows = [row for row in recommendations if int(row["seed"]) == seed]
            saved = root / f"results/prompt-v3/seed-{seed}/run/recommendations.csv"
            if profile == "v3" and saved.exists():
                rows = pd.read_csv(saved).to_dict("records")
            sources.append((f"{profile}-seed{seed}", rows, root / f"data/processed/seed-{seed}"))
    pairs, anchor_summary = anchor_stats(global_fit, sources, prompt_visibility)
    visible_pairs = pairs.loc[pairs.prompt_visible.eq(True)].copy()
    visible_summary = summarize_anchors(visible_pairs, "prompt_visible_in_first_request_with_measurement")
    summary = {"definition_version": DEFINITION_VERSION, "model_requests": 0,
               "s1_quantiles": quantiles, "first_answer_overlap": overlap_summary,
               "first_answer_records": {"method_round_records": len(per_run),
                                        "candidate_entries": int(per_run.first_answer_n.sum()) if len(per_run) else 0,
                                        "raw_candidate_entries": int(per_run.raw_candidate_entries.sum()) if len(per_run) else 0,
                                        "scope": "first response in every method-round, before repair"},
               "anchor_transfer": anchor_summary, "anchor_transfer_prompt_visible": visible_summary,
               "anchor_visibility": {"measurement_history_eligible": "initial observations and earlier complete rounds only",
                                     "prompt_visible": "explicit individual measured values in saved first request",
                                     "known_visibility_pairs": int(pairs.prompt_visible.notna().sum()),
                                     "unknown_visibility_pairs": int(pairs.prompt_visible.isna().sum()),
                                     "prompt_visible_pairs": len(visible_pairs)},
               "historical_predictor_top_k_definition": "stable input-order ties; historical values unchanged"}
    output.mkdir(parents=True, exist_ok=False)
    per_run.to_csv(output / "first_answer_overlap.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "anchor_transfer.csv", index=False, encoding="utf-8-sig")
    visible_pairs.to_csv(output / "anchor_transfer_prompt_visible.csv", index=False, encoding="utf-8-sig")
    (output / "stats_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline statistics from saved first answers and selected recommendations")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    report = recompute(options.output, options.project_root, options.s1)
    print(json.dumps({"output": str(options.output), "definition_version": report["definition_version"],
                      "first_answer": report["first_answer_records"], "anchors": report["anchor_transfer"]["ALL"], "model_requests": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
