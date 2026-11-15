# run_cross_protein.py — 跨蛋白验证：换蛋白零代码（换 CSV 即用）
# 两个 ProteinGym 数据集（GFP Sarkisyan 2016 / 泛素 RL40A Roscoe 2014）上运行 configure_protein
#
# 模式：
#   notation — 直接用记谱格式 dms.csv（RF/Ridge 选型）
#   full     — 从记谱 + WT 重构全长序列 CSV，走 full 格式路径（可加 --esm 启用 esm2_ridge/esm2_mlp 候选）
#   indel    — indel 记谱库（composition_ridge + 可选 esm2_ridge/esm2_mlp 序列级选型）
# 用法：
#   .venv\Scripts\python.exe run_cross_protein.py --mode notation
#   D:\Anaconda3\envs\openfold\python.exe run_cross_protein.py --mode full --esm
#
# 嵌入磁盘缓存：--embed-cache（默认开）把 ESM 嵌入落 results/cross_protein/embed_cache/<name>_<mode>
# （经 configure_protein 的 esm_cache_dir 透传 EmbeddingService(cache_dir=...)）；
# 同名同 mode 复测时嵌入秒级命中，GFP 级数据集首跑约 20 分钟、此后缓存复用。--no-embed-cache 关闭。
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
from protevo.protein_config import configure_protein, parse_mutation_notation  # noqa: E402

DATASETS = {
    "gfp": PROJECT / "data/external/gfp_sarkisyan",
    "ubiq": PROJECT / "data/external/ubiq_roscoe_2014",
    "blat": PROJECT / "data/external/blat_ecolx_gonzalez_2019",
}
OUT_DIR = PROJECT / "results/cross_protein"


def reconstruct_full(dms: pd.DataFrame, wt: str) -> pd.Series:
    """记谱 + WT → 全长突变序列（记谱的野生型字母逐一与 WT 对账，错配即报错）"""
    seqs = []
    for v in dms["variant"]:
        s = list(wt)
        for wt_aa, pos, mt in parse_mutation_notation(v):
            if s[pos - 1] != wt_aa:
                raise ValueError(f"变体 {v} 位点 {pos} 的 WT 字母 {wt_aa} 与参考序列 {s[pos-1]} 不符")
            s[pos - 1] = mt
        seqs.append("".join(s))
    return pd.Series(seqs, index=dms.index)


def run_one(name: str, dms_dir, mode: str, use_esm: bool, embed_cache_dir=None, pool="recommended") -> dict:
    meta = json.loads((dms_dir / "meta.json").read_text(encoding="utf-8"))
    wt = meta["wt_sequence"]
    dms = pd.read_csv(dms_dir / "dms.csv")

    if mode == "full":
        csv_path = OUT_DIR / f"{name}_full.csv"
        pd.DataFrame({"variant": reconstruct_full(dms, wt), "fitness": dms["fitness"]}).to_csv(csv_path, index=False)
    else:  # notation / indel：记谱 CSV 直接配置，格式由 configure_protein 自动检测
        csv_path = dms_dir / "dms.csv"

    t0 = time.time()
    cfg = configure_protein(
        str(csv_path),
        reference_sequence=wt,
        higher_is_better=meta.get("higher_is_better", True),
        use_esm=use_esm,
        esm_cache_dir=str(embed_cache_dir) if embed_cache_dir else None,
        predictor_pool=pool,
    )
    elapsed = round(time.time() - t0, 1)

    return {
        "dataset": meta.get("proteingym_assay", name),
        "mode": mode,
        "n_variants": len(dms),
        "n_positions": len(cfg.mutable_positions),
        "variant_format": cfg.variant_format,
        "reference_source": cfg.reference_source,
        "fitness_higher_is_better": cfg.fitness_higher_is_better,
        "predictor": cfg.predictor_info.get("selected"),
        "val_spearman": cfg.predictor_info.get("validation_spearman"),
        "all_candidates": cfg.predictor_info.get("all_candidates"),
        "seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["notation", "full", "indel"], required=True)
    parser.add_argument("--esm", action="store_true", help="启用 ESM 嵌入候选 esm2_ridge/esm2_mlp（需 GPU 环境）")
    parser.add_argument("--embed-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="ESM 嵌入磁盘缓存 results/cross_protein/embed_cache/<name>_<mode>（默认开，复测提速；--no-embed-cache 关闭）")
    parser.add_argument("--pool", choices=["recommended", "full"], default="recommended",
                        help="预测器候选池：recommended=先验推荐集（默认，先验只作推荐不作拦截）；full=全部表示层面可行组合（如 onehot_mlp 入池）")
    parser.add_argument("--datasets", default="gfp,ubiq", help="内置数据集（%s），逗号分隔" % ",".join(DATASETS))
    parser.add_argument("--csv", help="任意数据集直通模式：dms.csv 路径（同级需有 meta.json），绕过内置注册表")
    parser.add_argument("--name", help="--csv 模式下的数据集标签（默认取 CSV 所在目录名）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []

    if args.csv:
        dms_dir = Path(args.csv).resolve().parent
        entries = [(args.name or dms_dir.name, dms_dir)]
    else:
        entries = []
        for name in args.datasets.split(","):
            name = name.strip()
            if name not in DATASETS:
                parser.error(f"未知数据集 {name!r}；内置: {sorted(DATASETS)}，任意数据集请用 --csv <dms.csv 路径>")
            entries.append((name, DATASETS[name]))

    for name, dms_dir in entries:
        cache_dir = (OUT_DIR / "embed_cache" / f"{name}_{args.mode}") if args.embed_cache else None
        rec = run_one(name, dms_dir, args.mode, args.esm, cache_dir, args.pool)
        summary = [r for r in summary if not (r["dataset"] == rec["dataset"] and r["mode"] == rec["mode"])]
        summary.append(rec)
        print(f"[{name}/{args.mode}] {rec['predictor']} valρ={rec['val_spearman']} ({rec['seconds']}s)")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
