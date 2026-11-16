# evolve.py — 蛋白无关的虚拟定向进化总装入口：configure_protein + run_agent_round 一条命令串多轮
"""One-command multi-round virtual directed evolution over a variant-fitness CSV.

虚拟 oracle：CSV 全量 fitness 作为查表回填池——每轮提交的变体按变体串精确匹配回填"实测"分
（蛋白无关，四种变体格式通用）。真湿实验场景请逐轮调用 run_agent_round 手动喂养实测数据。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import MissingCredentials
from .agent_v2 import build_openai_client, run_agent_round
from .protein_config import configure_protein
from .schema import ExperimentConfig
from .tools import AgentBudget

# 列名自动识别候选，与 configure_protein 内置规则保持一致
_VARIANT_COLUMNS = ("variant", "Variants", "mutations", "seq", "sequence")
_FITNESS_COLUMNS = ("fitness", "Fitness", "score", "activity", "value")


def _detect_columns(df):
    """识别 variant/fitness 两列（规则同 configure_protein：候选名优先，否则首个字符串/数值列）。"""
    variant_col = next((name for name in _VARIANT_COLUMNS if name in df.columns), None)
    if variant_col is None:
        object_columns = df.select_dtypes(include=["object"]).columns
        if len(object_columns) == 0:
            raise ValueError(f"CSV 列 {df.columns.tolist()} 中找不到变体列（期望字符串列）")
        variant_col = object_columns[0]
    fitness_col = next((name for name in _FITNESS_COLUMNS if name in df.columns), None)
    if fitness_col is None:
        number_columns = df.select_dtypes(include=[np.number]).columns
        if len(number_columns) == 0:
            raise ValueError(f"CSV 列 {df.columns.tolist()} 中找不到 fitness 列（期望数值列）")
        fitness_col = number_columns[0]
    return variant_col, fitness_col


def _oracle_fitness(oracle, sequences):
    """逐条查 oracle 池回填实测分；未命中记 None（蛋白无关，变体串精确匹配）。"""
    return [None if sequence not in oracle else float(oracle[sequence]) for sequence in sequences]


def _best_record(known, higher_is_better):
    """累计最优（按 fitness 方向）；round=0 表示最优来自初始训练子集。"""
    index = known.fitness.idxmax() if higher_is_better else known.fitness.idxmin()
    row = known.loc[index]
    tag = str(row.mutations)
    return {"round": int(tag.rsplit("_", 1)[1]) if tag.startswith("round_") else 0,
            "variant": str(row.variant), "fitness": float(row.fitness)}


def _write_per_round_csv(rows, path):
    pd.DataFrame(rows, columns=["round", "variant", "reason", "predicted_fitness", "measured_fitness"]) \
        .to_csv(path, index=False)


def run_evolution(csv_path, rounds=3, per_round=10, score_budget=40, initial_train=2000,
                  reference_sequence=None, higher_is_better=True, use_esm=False,
                  esm_model_path=None, seed=42, out_dir=None, client=None, model=None,
                  retrain_each_round=False, on_round=None,
                  predictor_pool=None, extra_candidates=None, model_kwargs=None,
                  val_fraction=None, language="zh") -> dict:
    """跑完整多轮虚拟定向进化，返回汇总 dict（与落盘的 summary.json 同构）。

    种子化随机划分 initial_train 条为初始已知，其余为虚拟 oracle 池；每轮提交结果查表回填
    并入 known、从 eligible 移除。out_dir 内产出：initial_train.csv / per_round.csv /
    summary.json / agent_round{r}.jsonl。

    retrain_each_round：预测器协议，两模式均在 summary.json 顶层记
    predictor_protocol = "fixed" | "refit_per_round"。
      - False（默认，fixed）：预测器为初始训练子集版本，轮间不重训（与 GB1 v2 协议一致
        的简化）；
      - True（refit_per_round）：每轮开始前以累计 known（初始训练 + 此前各轮回填）重新调
        configure_protein 选型训练（random_state=seed 保持链路可复现，use_esm 等配置沿用
        首次），再以新 config 的 predict_fitness 进入该轮——对齐报告协议"各轮只用初始训练
        和本方法此前获得的反馈拟合预测器"；第 1 轮的累计 known 即初始训练子集，沿用初始
        config。每轮实际使用的 predictor_info（选型与 val Spearman）记入 summary 的 rounds
        明细，重训训练集落盘 known_before_round{r}.csv 留档。

    on_round：可选回调 on_round(轮摘要 dict)，每轮结束（round_summary 组装完成后）同步
    调用一次，供交互前端（TUI 等）做"每轮一行"准实时展示；回调抛出的任何异常一律吞掉，
    不影响进化主流程。默认 None 行为与原先完全一致。

    predictor_pool / extra_candidates / model_kwargs / val_fraction：研究阶段 configure_protein
    的配置覆写透传（如 TUI open --pool full 的池选择），初始选型与 --retrain 的每轮重配共用；
    None（默认）不传，保持 configure_protein 自身的默认行为。summary 的 config.predictor
    记录实际采用的池（pool mode 与候选清单）。

    language：Agent 提示语言（"zh" 中文 | "en" 英文，默认 "zh"），透传 run_agent_round →
    build_system_prompt；只影响发往 LLM 的系统提示语言，不影响预测器与结果落盘。

    中断保护：任一轮内抛异常（如外部 LLM 服务 429）时不丢已完成轮次——落盘当前
    per_round.csv 与 summary.json（status="interrupted"，含 error 摘要与已完成轮数）后
    返回该部分结果汇总（不再向上抛），调用方可据 status 区分完整/中断运行。
    当前不支持自动续跑；中断后部分结果已保存（status=interrupted），建议从新 out_dir
    重新运行或手动使用已完成轮次的数据。
    """
    csv_path = Path(csv_path)
    # 预算硬校验 k <= proposal_count == score_budget 在 schema；运行期每轮评分上限由
    # AgentBudget(max_score_calls=score_budget) 真正执行
    ecfg = ExperimentConfig(rounds=rounds, k=per_round, proposal_count=score_budget, score_budget=score_budget)
    # 凭据前置：无注入 client 时先构建 OpenAI 客户端，缺 .env 凭据立即失败而非训练后中途崩溃
    if client is None:
        client, model = build_openai_client(ecfg)
    elif not model:
        raise ValueError("model is required when injecting a client")

    df = pd.read_csv(csv_path)
    variant_col, fitness_col = _detect_columns(df)
    variants = df[variant_col].astype(str).values
    fitness = pd.to_numeric(df[fitness_col], errors="raise").values
    if not 1 <= initial_train < len(variants):
        raise ValueError(f"initial_train={initial_train} 越界：需 1 <= initial_train < {len(variants)}（CSV 行数）")

    # 种子化随机划分：initial_train 条为初始已知，其余为虚拟 oracle 池
    shuffled = np.random.RandomState(seed).permutation(len(variants))
    known = pd.DataFrame({"variant": [str(variants[i]) for i in shuffled[:initial_train]],
                          "fitness": [fitness[i] for i in shuffled[:initial_train]],
                          "mutations": "initial_train"}).drop_duplicates("variant", keep="first")
    oracle = {str(variants[i]): float(fitness[i]) for i in shuffled[initial_train:]}
    eligible = set(oracle) - set(known.variant)

    if out_dir is None:
        out_dir = Path("results/evolve") / f"{csv_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_initial = len(known)

    # configure_protein 只吃文件路径：初始训练子集落盘 out_dir/initial_train.csv（兼作留档）
    train_csv = out_dir / "initial_train.csv"
    known[["variant", "fitness"]].to_csv(train_csv, index=False)
    esm_kwargs = {} if esm_model_path is None else {"esm_model_path": esm_model_path}
    # 研究阶段配置覆写（open 会话的池选择等）：None 不传，保持 configure_protein 默认行为
    config_overrides = {}
    if predictor_pool is not None:
        config_overrides["predictor_pool"] = predictor_pool
    if extra_candidates is not None:
        config_overrides["extra_candidates"] = extra_candidates
    if model_kwargs is not None:
        config_overrides["model_kwargs"] = model_kwargs
    if val_fraction is not None:
        config_overrides["val_fraction"] = val_fraction

    def _fit_predictor(train_csv_path):
        """初始训练与轮间重训共用同一套 configure_protein 配置（random_state=seed，链路可复现；
        池/超参等覆写一并透传，保证 open 的选择延续到每轮重配）。"""
        return configure_protein(str(train_csv_path), reference_sequence=reference_sequence,
                                 auto_select_predictor=True, random_state=seed,
                                 higher_is_better=higher_is_better, use_esm=use_esm,
                                 **esm_kwargs, **config_overrides)

    config = _fit_predictor(train_csv)
    initial_config = config  # 顶层 config 快照始终取初始选型；轮间重训版本只记入各轮 rounds 明细

    rows, round_summaries = [], []
    best = _best_record(known, higher_is_better)

    def _summary_payload(status, error=None):
        """汇总 dict（与落盘 summary.json 同构）。status 枚举 completed / interrupted；
        interrupted 时 rounds 只含已完成轮次，error 记失败摘要（类型/消息/失败轮次/已完成轮数）。"""
        payload = {
            "csv_path": str(csv_path), "out_dir": str(out_dir),
            "predictor_protocol": "refit_per_round" if retrain_each_round else "fixed",
            "status": status,
            "config": {"rounds": rounds, "per_round": per_round, "score_budget": score_budget,
                       "initial_train": initial_train, "seed": seed, "higher_is_better": higher_is_better,
                       "use_esm": use_esm, "reference_sequence": reference_sequence,
                       "rows_total": int(len(variants)), "initial_known": int(n_initial),
                       "oracle_pool": len(oracle), "variant_format": initial_config.variant_format,
                       "mutable_positions": list(initial_config.mutable_positions),
                       "predictor": dict(initial_config.predictor_info)},
            "rounds": round_summaries, "cumulative_best": best, "total_measured": len(known),
            "note": ("虚拟进化：实测分由 CSV fitness 查表回填；预测器每轮开始前以累计 known"
                     "（初始训练 + 此前各轮回填）重新选型训练（refit_per_round，各轮只用初始训练"
                     "和本方法此前获得的反馈拟合预测器）" if retrain_each_round else
                     "虚拟进化：实测分由 CSV fitness 查表回填；预测器为初始训练子集版本，轮间不重训"
                     "（与 GB1 v2 协议一致的简化）")}
        if error is not None:
            payload["error"] = error
            payload["note"] += (f"；第 {error['failed_round']} 轮中断（{error['type']}），"
                                "仅含已完成轮次的部分结果")
        return payload

    def _write_summary(payload):
        (out_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str),
            encoding="utf-8")

    for r in range(1, rounds + 1):
        try:
            known_before, eligible_before = len(known), len(eligible)
            if retrain_each_round and r > 1:
                # 轮间重训（refit_per_round）：以累计 known（初始训练 + 此前各轮回填）重新选型训练，
                # 训练集落盘留档；此后各轮均以新 config 的 predict_fitness 进入该轮
                retrain_csv = out_dir / f"known_before_round{r}.csv"
                known[["variant", "fitness"]].to_csv(retrain_csv, index=False)
                config = _fit_predictor(retrain_csv)
                print(f"   第 {r} 轮预测器重训（refit_per_round）：known={known_before} → "
                      f"{config.predictor_info.get('selected')} "
                      f"(val ρ={config.predictor_info.get('validation_spearman')})")
            print(f"——— 虚拟进化第 {r}/{rounds} 轮：known={known_before}，eligible={eligible_before} ———")
            record, toolbox = run_agent_round(
                r, known, eligible, config.predict_fitness, ecfg, client=client, model=model,
                log_path=out_dir / f"agent_round{r}.jsonl",
                budget=AgentBudget(max_score_calls=score_budget), protein_config=config, language=language)

            submission = toolbox.submission
            sequences = submission["sequences"]
            measured = _oracle_fitness(oracle, sequences)
            unresolved = [s for s, value in zip(sequences, measured) if value is None]
            if unresolved:
                print(f"⚠️  第 {r} 轮提交的 {len(unresolved)} 条变体不在 oracle 池中，实测分记 None：{unresolved}")
            for sequence, reason, predicted, value in zip(sequences, submission["reasons"],
                                                          submission["predicted_fitness"], measured):
                rows.append({"round": r, "variant": sequence, "reason": reason,
                             "predicted_fitness": predicted, "measured_fitness": value})

            # 回填：命中项并入 known（未命中无实测值，不伪造测量）；全部从 eligible 移除避免重复推荐
            hits = [(s, value) for s, value in zip(sequences, measured) if value is not None]
            if hits:
                known = pd.concat([known, pd.DataFrame(
                    [{"variant": s, "fitness": value, "mutations": f"round_{r}"} for s, value in hits])],
                    ignore_index=True)
            eligible -= set(sequences)
            best = _best_record(known, higher_is_better)
            measured_values = [value for value in measured if value is not None]
            round_summary = {
                "round": r, "status": record["status"], "stop_reason": record["stop_reason"],
                "known_before": known_before, "eligible_before": eligible_before,
                "submitted": len(sequences), "unresolved": unresolved,
                "measured_mean": float(np.mean(measured_values)) if measured_values else None,
                "measured_best": (max(measured_values) if higher_is_better else min(measured_values))
                                 if measured_values else None,
                "cumulative_best": dict(best),
                "score_calls_used": record["score_calls_used"], "max_score_budget": score_budget,
                "api_calls_used": record["api_calls_used"], "total_tokens_used": record["total_tokens_used"],
                "log_path": record["log_path"]}
            if retrain_each_round:
                # refit_per_round：记该轮实际使用的预测器选型与 val Spearman（fixed 模式见顶层 config.predictor）
                round_summary["predictor_info"] = dict(config.predictor_info)
            round_summaries.append(round_summary)
            if on_round is not None:
                # 准实时回调：异常吞掉（展示层故障不应中断进化主流程）
                try:
                    on_round(dict(round_summary))
                except Exception:
                    pass
            _write_per_round_csv(rows, out_dir / "per_round.csv")
            mean_text = "n/a" if not measured_values else f"{np.mean(measured_values):.4g}"
            print(f"第 {r} 轮完成：status={record['status']}，提交 {len(sequences)} 条，"
                  f"实测均值={mean_text}，累计最优={best['variant']}({best['fitness']:.4g})")
        except Exception as exc:
            # 中断保护（如外部 LLM 服务 429）：落盘已完成轮次的部分结果再返回 interrupted 汇总，
            # 不向上抛——调用方（TUI/CLI）据 status 区分完整/中断运行并保留可导出目录
            error = {"type": type(exc).__name__, "message": str(exc), "failed_round": r,
                     "completed_rounds": len(round_summaries)}
            _write_per_round_csv(rows, out_dir / "per_round.csv")
            partial = _summary_payload("interrupted", error)
            _write_summary(partial)
            print(f"⚠️  第 {r} 轮中断（{error['type']}: {exc}）："
                  f"已完成 {len(round_summaries)} 轮的部分结果已落盘 {out_dir}")
            return partial

    summary = _summary_payload("completed")
    _write_summary(summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m protevo.evolve",
        description="一行命令跑完整多轮虚拟定向进化（CSV 全量 fitness 作为查表回填 oracle）")
    parser.add_argument("csv", help="变体-适应度 CSV（variant+fitness 两列，列名自动识别）")
    parser.add_argument("--rounds", type=int, default=3, help="进化轮数（默认 3）")
    parser.add_argument("--per-round", type=int, default=10, help="每轮提交变体数（默认 10）")
    parser.add_argument("--reference", default=None, help="参考序列（记谱/突变串格式建议提供 WT）")
    parser.add_argument("--esm", action="store_true", help="预测器候选池加入 ESM-2 嵌入（需 torch）")
    parser.add_argument("--out", default=None, help="输出目录（默认 results/evolve/<csv名>_<时间戳>/）")
    args = parser.parse_args(argv)
    try:
        summary = run_evolution(args.csv, rounds=args.rounds, per_round=args.per_round,
                                reference_sequence=args.reference, use_esm=args.esm, out_dir=args.out)
    except MissingCredentials as exc:
        print(f"错误：{exc}", file=sys.stderr)
        print("缺少 LLM 凭据：请在项目根目录 .env 配置 OPENAI_API_KEY / OPENAI_MODEL，"
              "或先运行 python -m protevo.setup_assistant --doctor 查看其中的 llm_credentials 检查。",
              file=sys.stderr)
        return 2
    if summary.get("status") == "interrupted":
        error = summary.get("error") or {}
        print(f"中断：第 {error.get('failed_round')} 轮失败（{error.get('type')}: {error.get('message')}）；"
              f"已完成 {len(summary['rounds'])} 轮的部分结果已写入 {summary['out_dir']}"
              "（per_round.csv / summary.json / agent_round*.jsonl）", file=sys.stderr)
        return 1
    print(f"完成：结果已写入 {summary['out_dir']}（per_round.csv / summary.json / agent_round*.jsonl）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
