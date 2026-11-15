# 工具层：把实验历史、知识检索、候选打分、预算查询与最终提交封装为LLM可自主调用的工具。
"""OpenAI function-calling schemas and a stateful executor for the autonomous evolution agent.

protein_config=None 时为 GB1 旧路径（供已发布实验复现）；给定时校验/描述文本全部由配置派生。"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re

import numpy as np

from .agent import summarize
from .protein_config import parse_mutation_notation
from .schema import POSITIONS, ExperimentConfig, KnowledgeBase, variant_allowed
from .variant_notation import validate_variant

TOOL_NAMES = ("predict_fitness", "query_knowledge", "get_experiment_history", "submit_recommendations", "check_budget")

# ---- ProteinConfig驱动的描述文本派生；config=None时完全不参与，走GB1旧路径 ----

def _wt_residue_at(config, index):
    """取第index个可变位点的野生型残基：优先wt_by_position，缺失时回退参考序列。"""
    position = config.mutable_positions[index]
    wt = config.wt_by_position.get(position)
    if not wt and index < len(config.reference_sequence):
        wt = config.reference_sequence[index]
    return wt or "X"


def _notation_example(config):
    """生成形如V39D,G40E的记谱示例：取前两个可变位点，残基换成字母表里的另一个字母。"""
    parts = []
    for index, position in enumerate(config.mutable_positions[:2]):
        wt = _wt_residue_at(config, index)
        mut = next((letter for letter in config.aa_alphabet if letter != wt), "X")
        parts.append(f"{wt}{position}{mut}")
    return ",".join(parts)


def _schema_variant_text(config):
    """按config.variant_format派生工具描述中与蛋白相关的文本（位点数、位点、格式与示例）。"""
    count = len(config.mutable_positions)
    positions = ",".join(str(position) for position in config.mutable_positions)
    budget_note = ("Each newly scored eligible variant consumes one unit of the per-round score budget; "
                   "invalid, already measured or previously scored variants consume nothing. "
                   "Predicted fitness is an estimate, not a measurement.")
    if config.variant_format == "indel_notation":
        anchor = config.mutable_positions[0] if config.mutable_positions else 1
        wt = config.wt_full_sequence[anchor - 1] if config.wt_full_sequence else "X"
        return {
            "predict_description": (f"Score indel mutation-notation variants on the {len(config.wt_full_sequence)}-residue "
                                    'reference: substitutions "V39D", deletions "V39del"/"V39_G41del", insertions '
                                    '"V39_G40insAA", duplications "V39dup", wild type "WT". '
                                    f"{budget_note}"),
            "sequences_description": f'Indel-notation variants to score, e.g. ["{wt}{anchor}del", "WT"]',
            "submit_description": "Final recommended indel-notation variants, in preference order"}
    if config.variant_format == "mutation_notation":
        example = _notation_example(config)
        return {
            "predict_description": (f'Score mutation-notation variants (e.g. "{example}") over the {count} mutable '
                                    f"positions ({positions}) with the local fitness predictor. {budget_note}"),
            "sequences_description": f'Mutation-notation variants to score, e.g. ["{example}", "WT"]',
            "submit_description": "Final recommended mutation-notation variants, in preference order"}
    if config.variant_format == "mutation_string":
        reference = config.reference_sequence
        alt = next((letter for letter in config.aa_alphabet if letter != reference[:1]), reference[:1]) + reference[1:]
        return {
            "predict_description": (f"Score {count}-letter mutation-string variants ({count} mutable positions "
                                    f"{positions}) with the local fitness predictor. {budget_note}"),
            "sequences_description": f'{count}-letter variants to score, e.g. ["{reference}", "{alt}"]',
            "submit_description": f"Final recommended {count}-letter variants, in preference order"}
    length = len(config.reference_sequence)
    return {
        "predict_description": (f"Score full-length sequence variants ({length} residues, mutable positions "
                                f"{positions}) with the local fitness predictor. {budget_note}"),
        "sequences_description": f"Full-length sequences to score ({length} residues each)",
        "submit_description": "Final recommended full-length sequence variants, in preference order"}


# 传给chat.completions.create的tools参数；格式遵循OpenAI function calling官方规范。
# config=None返回GB1旧版schema（与模块常量TOOL_SCHEMAS逐字节一致）；给定时仅覆写
# predict_fitness/submit_recommendations中GB1特定的描述文本，蛋白无关工具原样保留。
def build_tool_schemas(config=None):
    schemas = [
        {"type": "function", "function": {
            "name": "predict_fitness",
            "description": ("Score four-letter GB1 variants (positions 39,40,41,54) with the local fitness predictor. "
                            "Each newly scored eligible variant consumes one unit of the per-round score budget; "
                            "invalid, already measured or previously scored variants consume nothing. "
                            "Predicted fitness is an estimate, not a measurement."),
            "parameters": {"type": "object", "properties": {
                "sequences": {"type": "array", "items": {"type": "string"},
                              "description": "Four-letter variants to score, e.g. [\"VDGV\", \"IDGV\"]"}},
                "required": ["sequences"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "query_knowledge",
            "description": ("Search the sourced directed-evolution knowledge rule base by keywords "
                            "(amino-acid properties, charge, epistasis, ...). Returns the matching rules with their "
                            "citable rule ids, or the list of available rule ids when nothing matches."),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Free-text keywords describing the design question"}},
                "required": ["query"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "get_experiment_history",
            "description": ("Read the current experiment state: known measurements with per-position residue statistics, "
                            "top variants, previously recommended variants with their measured feedback, and the variants "
                            "already scored this round. Call this first to ground the round in evidence."),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "submit_recommendations",
            "description": ("Submit the final recommendations for this round and end the round. Provide exactly the "
                            "required number of unique, eligible, unmeasured variants with one concise scientific "
                            "reason each. This call is terminal; validate your candidates before submitting."),
            "parameters": {"type": "object", "properties": {
                "sequences": {"type": "array", "items": {"type": "string"},
                              "description": "Final recommended four-letter variants, in preference order"},
                "reasons": {"type": "array", "items": {"type": "string"},
                            "description": "One reason per variant, same length as sequences"}},
                "required": ["sequences", "reasons"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "check_budget",
            "description": ("Report the remaining per-round budgets: LLM API calls, total tokens and candidate scoring "
                            "calls. Check this before large scoring requests or when unsure how much room is left."),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    ]
    if config is not None:
        text = _schema_variant_text(config)
        for schema in schemas:
            function = schema["function"]
            if function["name"] == "predict_fitness":
                function["description"] = text["predict_description"]
                function["parameters"]["properties"]["sequences"]["description"] = text["sequences_description"]
            elif function["name"] == "submit_recommendations":
                function["parameters"]["properties"]["sequences"]["description"] = text["submit_description"]
    return schemas


TOOL_SCHEMAS = build_tool_schemas(None)


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentBudget:
    """Per-round hard budgets shared by the agent loop and the check_budget tool."""
    max_api_calls: int = 15
    max_total_tokens: int = 50000
    max_score_calls: int = 40
    api_calls_used: int = 0
    total_tokens_used: int = 0
    score_calls_used: int = 0

    def remaining_api_calls(self):
        return max(0, self.max_api_calls - self.api_calls_used)

    def remaining_tokens(self):
        return max(0, self.max_total_tokens - self.total_tokens_used)

    def remaining_score_calls(self):
        return max(0, self.max_score_calls - self.score_calls_used)

    def consume_api_call(self, total_tokens):
        self.api_calls_used += 1
        self.total_tokens_used += max(0, int(total_tokens))

    def consume_scores(self, count):
        self.score_calls_used += count

    # 三个上限任何一个耗尽都终止循环；返回值作为stop_reason写入日志和提交记录。
    def exhausted_reason(self):
        if self.api_calls_used >= self.max_api_calls:
            return "max_api_calls"
        if self.total_tokens_used >= self.max_total_tokens:
            return "max_total_tokens"
        if self.score_calls_used >= self.max_score_calls:
            return "max_score_calls"
        return None

    def snapshot(self):
        return {"max_api_calls": self.max_api_calls, "api_calls_used": self.api_calls_used,
                "api_calls_remaining": self.remaining_api_calls(),
                "max_total_tokens": self.max_total_tokens, "total_tokens_used": self.total_tokens_used,
                "tokens_remaining": self.remaining_tokens(),
                "max_score_calls": self.max_score_calls, "score_calls_used": self.score_calls_used,
                "score_calls_remaining": self.remaining_score_calls()}


# ProteinConfig驱动的变体校验：合法返回None，非法返回英文原因（与旧路径错误风格一致）。
# mutation_string按可变位点数校验长度；mutation_notation用parse_mutation_notation校验且位点须已声明；
# full按参考序列长度校验；三种格式的残基都必须落在config.aa_alphabet内。
def _config_variant_error(sequence, protein_config):
    if not isinstance(sequence, str):
        return "variant must be a string"
    if protein_config.variant_format == "indel_notation":
        # indel 记谱校验统一走 variant_notation：WT 坐标界、锚点字母、重叠检测、插入锚点相邻性
        return validate_variant(sequence, protein_config.wt_full_sequence, protein_config.aa_alphabet)
    alphabet = set(protein_config.aa_alphabet)
    if protein_config.variant_format == "mutation_notation":
        try:
            triples = parse_mutation_notation(sequence)
        except ValueError:
            return "not valid mutation notation (expected tokens like A12C, separated by commas or spaces)"
        declared = set(protein_config.mutable_positions)
        for wild_type, position, mutant in triples:
            if position not in declared:
                return f"position {position} is not one of the mutable positions {sorted(declared)}"
            if wild_type not in alphabet or mutant not in alphabet:
                return f"residues must come from the configured amino-acid alphabet {protein_config.aa_alphabet}"
        return None
    expected_length = (len(protein_config.reference_sequence) if protein_config.variant_format == "full"
                       else len(protein_config.mutable_positions))
    if len(sequence) != expected_length:
        return (f"length must be {expected_length} to match the "
                f"{'reference sequence' if protein_config.variant_format == 'full' else 'mutable positions'}")
    if set(sequence) - alphabet:
        return f"residues must come from the configured amino-acid alphabet {protein_config.aa_alphabet}"
    return None


# protein_config 路径的已知数据摘要：位点列表与各位点残基分布完全由配置驱动，不含任何 GB1 字面量。
# 残基按 config.variant_format 解析（与编码器同源）：mutation_notation 用记谱解析（未突变位点记
# 野生型残基），mutation_string 按位取字符，full 取 mutable 位置字符；indel 无固定位点残基，不输出分布。
def _config_known_summary(known, top_n, config):
    top = known.sort_values(["fitness", "variant"], ascending=[False, True]).head(top_n)
    summary = {"n_known": len(known), "fitness_mean": float(known.fitness.mean()),
               "best_fitness": float(known.fitness.max()),
               "top_variants": top[["variant", "fitness", "mutations"]].to_dict(orient="records"),
               "interpretation": "Marginal residue statistics are observations in sampled backgrounds, not causal substitution effects."}
    if config.variant_format == "indel_notation":
        return summary
    residues = [config._extract_residues(str(variant)) for variant in known.variant]
    positions = []
    for index, position in enumerate(config.mutable_positions):
        grouped = (known.assign(residue=[row[index] for row in residues])
                   .groupby("residue").fitness.agg(["count", "mean", "max"]))
        positions.append({"position": position, "residue_statistics": grouped.reset_index().to_dict(orient="records")})
    summary["positions"] = positions
    return summary


# 工具持有本轮全部可见状态；agent循环只负责消息编排，评分与预算的真实计数都发生在这里。
class ToolBox:
    """Stateful executor exposing the five agent tools over shared experiment state."""

    def __init__(self, known, eligible, predict_fn, config, knowledge=None, budget=None, feedback_history=(),
                 protein_config=None):
        self.known = known
        self.measured = set(known.variant)
        self.eligible = set(eligible)
        self.predict_fn = predict_fn
        self.config = config
        # protein_config=None走GB1旧路径（逐字节兼容）；给定时变体校验与round_context均由它派生。
        self.protein_config = protein_config
        self.knowledge = knowledge
        self.budget = AgentBudget() if budget is None else budget
        self.feedback_history = [dict(record) for record in feedback_history]
        self.score_cache = {}
        self.submission = None
        self._handlers = {"predict_fitness": self._predict_fitness,
                          "query_knowledge": self._query_knowledge,
                          "get_experiment_history": self._get_experiment_history,
                          "submit_recommendations": self._submit_recommendations,
                          "check_budget": self._check_budget}

    def required_count(self):
        return min(self.config.k, len(self.eligible))

    def ranked_scored(self):
        ranked = sorted(self.score_cache, key=lambda v: (-self.score_cache[v], v))
        return [{"rank": index, "variant": variant, "predicted_fitness": self.score_cache[variant]}
                for index, variant in enumerate(ranked, 1)]

    # arguments接受JSON字符串或已解析对象；所有失败以带error键的字典返回，绝不中断agent循环。
    def execute(self, name, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return {"error": f"tool arguments are not valid JSON: {arguments!r}"}
        if not isinstance(arguments, dict):
            return {"error": "tool arguments must be a JSON object"}
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(**arguments)
        except Exception as exc:
            return {"error": f"{type(exc).__name__} while executing {name}: {exc}"}

    def _predict_fitness(self, sequences):
        if not isinstance(sequences, list) or not sequences:
            return {"error": "sequences must be a nonempty list of variant strings"}
        errors, requested, seen = [], [], set()
        for sequence in sequences:
            if self.protein_config is not None:
                invalid_reason = _config_variant_error(sequence, self.protein_config)
                if invalid_reason is not None:
                    errors.append({"sequence": sequence, "reason": invalid_reason})
                    continue
            elif not isinstance(sequence, str) or len(sequence) != 4 or set(sequence) - set(self.config.allowed_amino_acids):
                errors.append({"sequence": sequence, "reason": "not a four-letter variant over the allowed amino acids"})
                continue
            if sequence in seen:
                continue
            seen.add(sequence)
            if sequence in self.measured:
                errors.append({"sequence": sequence, "reason": "already measured in known data; choose an unmeasured candidate"})
                continue
            # variant_allowed内嵌GB1四位点约束，仅旧路径使用；config路径已由配置校验覆盖。
            if self.protein_config is None and not variant_allowed(sequence, self.config):
                errors.append({"sequence": sequence, "reason": "violates the configured mutation constraints"})
                continue
            if sequence not in self.eligible:
                errors.append({"sequence": sequence, "reason": "not in the eligible unmeasured candidate pool"})
                continue
            requested.append(sequence)
        fresh = [sequence for sequence in requested if sequence not in self.score_cache]
        room = self.budget.remaining_score_calls()
        to_score, deferred = fresh[:room], fresh[room:]
        if to_score:
            error = self._bulk_score(to_score)
            if error:
                return {"error": error}
        ranked = sorted((sequence for sequence in requested if sequence in self.score_cache),
                        key=lambda v: (-self.score_cache[v], v))
        result = {"scored": [{"rank": index, "variant": variant, "predicted_fitness": self.score_cache[variant]}
                             for index, variant in enumerate(ranked, 1)],
                  "remaining_score_budget": self.budget.remaining_score_calls(),
                  "notes": "Predicted fitness comes from the local predictor and is an estimate, not a measurement."}
        if errors:
            result["errors"] = errors
        if deferred:
            result["not_scored"] = [{"sequence": sequence, "reason": "per-round score budget exhausted"}
                                    for sequence in deferred]
        return result

    def _bulk_score(self, sequences):
        """批量调用 predict_fn：成功消耗评分预算并写入缓存返回 None，失败返回错误且不消耗。"""
        values = self.predict_fn(sequences)
        if not isinstance(values, (list, tuple)) or len(values) != len(sequences):
            return "predictor must return exactly one score per requested sequence"
        try:
            scores = [float(value) for value in values]
        except (TypeError, ValueError):
            return "predictor returned nonnumeric scores"
        if not all(np.isfinite(scores)):
            return "predictor returned nonfinite scores"
        # 只有真实完成的批量预测才消耗预算；失败请求不计费。
        self.budget.consume_scores(len(sequences))
        self.score_cache.update(zip(sequences, scores))
        return None

    # 提交层硬不变量（protein_config 路径）：提交序列中的每条候选必须已有预测分。
    # 存在合格未测但未评分的变体时先批量补齐（正常消耗评分预算）；预算不足则返回拒绝原因，
    # 由调用方拒绝该提交（LLM 可改用 predict_fitness 或换已评分候选重试）。
    # GB1 旧路径（protein_config=None）不补分，行为与已发布复现完全一致。
    def ensure_scored(self, sequences):
        missing = [sequence for sequence in sequences if sequence not in self.score_cache]
        if self.protein_config is None or not missing:
            return None
        room = self.budget.remaining_score_calls()
        if len(missing) > room:
            return (f"提交前必须完成评分，剩余评分额度不足（还需评 {len(missing)} 条，剩余 {room} 条），"
                    "请缩减批次或先用 predict_fitness")
        return self._bulk_score(missing)

    def _query_knowledge(self, query):
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a nonempty string"}
        if self.knowledge is None:
            return {"query": query, "matched_rules": [],
                    "note": "No knowledge base is configured for this agent."}
        tokens = [token for token in re.split(r"[^0-9a-z-]+", query.lower()) if len(token) > 2]
        matched = []
        for rule in self.knowledge.rules:
            text = " ".join([rule.id, rule.scope, rule.condition, rule.action, rule.source, rule.explanation]).lower()
            if any(token in text for token in tokens):
                matched.append(rule.model_dump())
        if not matched:
            return {"query": query, "matched_rules": [],
                    "available_rule_ids": [rule.id for rule in self.knowledge.rules],
                    "note": "No rule matched; try other keywords or inspect the available rule ids."}
        return {"query": query, "matched_rules": matched}

    def _get_experiment_history(self):
        # config路径的参考序列、位点、已知摘要与合法未测示例全部由ProteinConfig派生；旧路径保持GB1硬编码值不变。
        if self.protein_config is not None:
            reference_variant = self.protein_config.reference_sequence
            positions = list(self.protein_config.mutable_positions)
            known_data = _config_known_summary(self.known, self.config.summary_top, self.protein_config)
            # 让模型直接看到合法未测候选的原生形态（确定性字典序前30条），避免反复请求已测或库外变体
            round_context = {"reference_variant": reference_variant, "positions": positions,
                             "eligible_candidate_count": len(self.eligible),
                             "eligible_total": len(self.eligible),
                             "eligible_examples": sorted(self.eligible)[:30],
                             "eligibility_note": ("Eligible candidates are unmeasured variants that appear in the "
                                                  "dataset pool; write them in exactly the native variant format "
                                                  "shown in eligible_examples."),
                             "required_recommendations": self.required_count()}
        else:
            reference_variant, positions = "VDGV", list(POSITIONS)
            known_data = summarize(self.known, self.config.summary_top)
            round_context = {"reference_variant": reference_variant, "positions": positions,
                             "eligible_candidate_count": len(self.eligible),
                             "required_recommendations": self.required_count()}
        return {"round_context": round_context,
                "known_data": known_data,
                "previous_recommendations": self.feedback_history,
                "scored_variants_this_round": self.ranked_scored(),
                "notes": ("Marginal residue statistics are observations in sampled backgrounds, not causal "
                          "substitution effects; feedback lists only variants this method already assayed.")}

    def _submit_recommendations(self, sequences, reasons):
        if self.submission is not None:
            return {"status": "rejected", "error": "recommendations were already submitted for this round"}
        required = self.required_count()
        if not isinstance(sequences, list) or not isinstance(reasons, list):
            return {"status": "rejected", "error": "sequences and reasons must be lists"}
        if len(sequences) != len(reasons):
            return {"status": "rejected", "error": "sequences and reasons must have the same length"}
        if len(sequences) != required:
            return {"status": "rejected", "error": f"submit exactly {required} recommendations, got {len(sequences)}"}
        for sequence, reason in zip(sequences, reasons):
            if not isinstance(sequence, str) or sequence not in self.eligible:
                return {"status": "rejected", "error": f"variant {sequence!r} is not in the eligible unmeasured candidate pool"}
            if not isinstance(reason, str) or not reason.strip():
                return {"status": "rejected", "error": f"reason for {sequence!r} must be a nonempty string"}
        if len(set(sequences)) != len(sequences):
            return {"status": "rejected", "error": "recommendations must be unique"}
        # 硬不变量：提交前所有候选必须已有预测分——未评分变体在此自动补齐，预算不足则拒绝提交
        error = self.ensure_scored(sequences)
        if error is not None:
            return {"status": "rejected", "error": error}
        return self._record_submission(sequences, reasons, budget_exhausted=False, stop_reason=None, auto=False)

    def _check_budget(self):
        return {**self.budget.snapshot(), "submitted": self.submission is not None,
                "notes": "All budgets are hard per-round limits; exhausting any of them triggers an automatic submission."}

    # LLM主动提交与预算耗尽后的自动提交共用同一份记录结构，便于下游统一解析。
    def _record_submission(self, sequences, reasons, budget_exhausted, stop_reason, auto):
        self.submission = {"sequences": list(sequences), "reasons": list(reasons),
                           "predicted_fitness": [self.score_cache.get(sequence) for sequence in sequences],
                           "budget_exhausted": budget_exhausted, "stop_reason": stop_reason, "auto": auto,
                           "submitted_at": _now()}
        return {"status": "submitted", "budget_exhausted": budget_exhausted, "stop_reason": stop_reason, "auto": auto,
                "recommendations": [{"variant": sequence, "reason": reason,
                                     "predicted_fitness": self.score_cache.get(sequence)}
                                    for sequence, reason in zip(sequences, reasons)]}

    def force_submit(self, sequences, reasons, stop_reason):
        """Record the automatic submission used when a budget is exhausted; inputs come from trusted internal state."""
        if self.submission is not None:
            return {"status": "rejected", "error": "recommendations were already submitted for this round"}
        # 自动提交与 LLM 提交走同一层补分：保证提交记录每条都带 predicted_fitness
        error = self.ensure_scored(sequences)
        if error is not None:
            return {"status": "rejected", "error": error}
        return self._record_submission(sequences, reasons, budget_exhausted=True, stop_reason=stop_reason, auto=True)


def sklearn_predictor(model):
    """Adapt a fitted protevo.models regressor to the predict_fn(sequences)->list[float] interface."""
    from .data import one_hot

    def predict(sequences):
        return [float(value) for value in model.predict(one_hot(sequences))]

    return predict
