# 二代智能体：LLM通过OpenAI function calling自主选择工具，循环至主动提交或预算耗尽。
"""Autonomous tool-calling agent loop with per-round hard budgets and full JSONL logging."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

from .agent import MissingCredentials, usage_counts
from .prompt_builder import build_system_prompt
from .tools import TOOL_NAMES, TOOL_SCHEMAS, AgentBudget, ToolBox, build_tool_schemas

DEFAULT_SYSTEM_PROMPT = """You are an autonomous scientific agent for GB1 directed evolution.
You design four-site variants (positions 39,40,41,54; reference residues VDGV) to maximize IgG-Fc binding fitness.
You decide the workflow yourself by calling tools; typical science loops inspect experiment history, query
knowledge rules, form hypotheses, score candidate variants, then refine. Never invent measurements: predicted
fitness is an estimate with epistatic uncertainty, and previously measured variants cannot be recommended again.
Budgets are finite; check_budget before large scoring requests. End the round only by calling
submit_recommendations with exactly the required number of variants and one concise reason each."""

NO_TOOL_CALLS_NUDGE = ("You replied without calling a tool. Continue the design loop by calling one of the available "
                       "tools; when your analysis is complete, call submit_recommendations.")


# 兼容SDK对象与测试注入的SimpleNamespace；只保留协议必需字段。
def _assistant_message_dict(message):
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    else:
        data = {"role": getattr(message, "role", None) or "assistant", "content": getattr(message, "content", None)}
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            data["tool_calls"] = [{"id": getattr(call, "id", None), "type": getattr(call, "type", None) or "function",
                                   "function": {"name": getattr(call.function, "name", None),
                                                "arguments": getattr(call.function, "arguments", "")}}
                                  for call in calls]
    data.setdefault("role", "assistant")
    if not data.get("tool_calls"):
        data.pop("tool_calls", None)
    if data.get("content") is None:
        data.pop("content", None)
    return data


def _usage_dict(usage):
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}
    if hasattr(usage, "model_dump"):
        return usage_counts(usage.model_dump(mode="json"))
    keys = ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens")
    data = {key: getattr(usage, key, None) for key in keys}
    return usage_counts({key: value for key, value in data.items() if value is not None})


def _loggable_arguments(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return {"raw": raw, "parse_error": True}
    return raw


class ToolCallingAgent:
    """Runs one autonomous round: LLM response -> tool calls -> tool results -> ... -> submission or budget stop."""

    def __init__(self, toolbox, client=None, model=None, system_prompt=None, log_path=None, environ=None,
                 tool_schemas=None):
        self.toolbox = toolbox
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        # 蛋白无关路径注入 build_tool_schemas(protein_config)；默认与模块常量 TOOL_SCHEMAS 一致（GB1 旧路径）
        self.tool_schemas = tool_schemas or TOOL_SCHEMAS
        self.log_path = Path(log_path) if log_path is not None else None
        if client is None:
            client, model = build_openai_client(toolbox.config, environ)
        elif not model:
            raise ValueError("model is required when injecting a client")
        self.client = client
        self.model = model
        self._record_count = 0

    # 每条记录一行JSON并fsync，与storage.event相同的持久化语义；字段以任务规定的schema为准。
    def _log(self, record):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **record},
                          ensure_ascii=False, allow_nan=False, default=str)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _next_call_id(self, round_number):
        self._record_count += 1
        return f"r{round_number}-{self._record_count:03d}"

    def run_round(self, round_number, task_message):
        budget = self.toolbox.budget
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task_message}]
        self._log({"kind": "round_start", "round": round_number, "call_id": self._next_call_id(round_number),
                   "tool_name": None, "arguments": None,
                   "result": {"known_count": len(self.toolbox.measured), "eligible_count": len(self.toolbox.eligible),
                              "required_recommendations": self.toolbox.required_count(), "budget": budget.snapshot()},
                   "tokens_used": 0, "context_length": len(messages), "model": self.model})
        tool_sequence = []
        stop_reason = None
        while self.toolbox.submission is None:
            # 每次API调用前检查硬预算；耗尽即跳出并自动提交当前最佳。
            stop_reason = budget.exhausted_reason()
            if stop_reason:
                break
            context_length = len(messages)
            context_chars = sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=list(messages), tools=self.tool_schemas, tool_choice="auto")
            except Exception as exc:
                self._log({"kind": "api_error", "round": round_number, "call_id": self._next_call_id(round_number),
                           "tool_name": None, "arguments": None,
                           "result": {"error_type": type(exc).__name__, "error": str(exc)},
                           "tokens_used": 0, "context_length": context_length, "model": self.model})
                raise
            usage = _usage_dict(getattr(response, "usage", None))
            budget.consume_api_call(usage["total_tokens"])
            choice = response.choices[0]
            assistant = _assistant_message_dict(choice.message)
            calls = assistant.get("tool_calls") or []
            self._log({"kind": "api_call", "round": round_number, "call_id": self._next_call_id(round_number),
                       "tool_name": None,
                       "arguments": {"tools": list(TOOL_NAMES), "tool_choice": "auto"},
                       "result": {"finish_reason": getattr(choice, "finish_reason", None),
                                  "tool_call_count": len(calls), "content": assistant.get("content")},
                       "tokens_used": usage["total_tokens"], "context_length": context_length,
                       "context_chars": context_chars, "model": self.model,
                       "model_returned": getattr(response, "model", None), "response_id": getattr(response, "id", None),
                       "usage": usage})
            messages.append(assistant)
            if not calls:
                # 没有工具调用的纯文本回复：追加提示让LLM继续，预算上限保证循环必然终止。
                messages.append({"role": "user", "content": NO_TOOL_CALLS_NUDGE})
                continue
            for call in calls:
                name = call["function"]["name"]
                result = self.toolbox.execute(name, call["function"]["arguments"])
                tool_sequence.append(name)
                self._log({"kind": "tool_call", "round": round_number, "call_id": self._next_call_id(round_number),
                           "tool_name": name, "arguments": _loggable_arguments(call["function"]["arguments"]),
                           "result": result, "tokens_used": 0, "context_length": len(messages), "model": self.model})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False, allow_nan=False, default=str)})
        if self.toolbox.submission is None:
            stop_reason = stop_reason or budget.exhausted_reason() or "unknown_termination"
            self._auto_submit(round_number, stop_reason, messages, tool_sequence)
        status = "auto_submitted" if self.toolbox.submission["auto"] else "submitted"
        record = {"round": round_number, "status": status, "budget_exhausted": bool(stop_reason),
                  "stop_reason": stop_reason, "submission": self.toolbox.submission,
                  "tool_call_sequence": tool_sequence, "api_calls_used": budget.api_calls_used,
                  "total_tokens_used": budget.total_tokens_used, "score_calls_used": budget.score_calls_used,
                  "log_path": str(self.log_path) if self.log_path is not None else None}
        self._log({"kind": "round_end", "round": round_number, "call_id": self._next_call_id(round_number),
                   "tool_name": None, "arguments": None,
                   "result": {"status": status, "budget_exhausted": record["budget_exhausted"],
                              "stop_reason": stop_reason, "tool_call_sequence": tool_sequence},
                   "tokens_used": budget.total_tokens_used, "context_length": len(messages), "model": self.model})
        return record

    # 预算耗尽时提交当前最佳：已评分候选按预测分排序，不足k个用确定性回补填满。
    def _auto_submit(self, round_number, stop_reason, messages, tool_sequence):
        toolbox = self.toolbox
        required = toolbox.required_count()
        ranked = toolbox.ranked_scored()[:required]
        sequences = [row["variant"] for row in ranked]
        reasons = [f"Auto-submitted after {stop_reason}: rank {row['rank']} by predicted fitness "
                   f"({row['predicted_fitness']:.4g})." for row in ranked]
        if len(sequences) < required:
            fill = sorted(toolbox.eligible - set(sequences))[:required - len(sequences)]
            if toolbox.protein_config is not None:
                # 蛋白无关路径：确定性回补候选先补评分再进提交（与提交层同一套 ensure_scored）；
                # 剩余评分额度不足时截断到可负担数量，保证自动提交的每条推荐都带预测分。
                fill = fill[:toolbox.budget.remaining_score_calls()]
                toolbox.ensure_scored(fill)
                sequences += fill
                for variant in fill:
                    score = toolbox.score_cache.get(variant)
                    detail = f"predicted fitness {score:.4g}" if score is not None else "no predictor evidence"
                    reasons.append(f"Auto-submitted after {stop_reason}: deterministic fallback from the eligible pool "
                                   f"({detail}).")
            else:
                sequences += fill
                reasons += [f"Auto-submitted after {stop_reason}: unscored deterministic fallback from the eligible pool "
                            "(no predictor evidence)." for _ in fill]
        result = toolbox.force_submit(sequences, reasons, stop_reason)
        tool_sequence.append("submit_recommendations")
        self._log({"kind": "tool_call", "round": round_number, "call_id": self._next_call_id(round_number),
                   "tool_name": "submit_recommendations", "arguments": {"sequences": sequences, "reasons": reasons},
                   "result": result, "tokens_used": 0, "context_length": len(messages), "model": self.model,
                   "auto": True, "budget_exhausted": True, "stop_reason": stop_reason})


def build_openai_client(config, environ=None):
    """Create an OpenAI chat client from project environment variables; no implicit credentials or network here."""
    if environ is None:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
        environ = os.environ
    model = environ.get("OPENAI_MODEL", "").strip()
    if not model or not environ.get("OPENAI_API_KEY"):
        raise MissingCredentials("Set project OPENAI_API_KEY and OPENAI_MODEL to run the tool-calling agent")
    from openai import OpenAI
    client = OpenAI(api_key=environ["OPENAI_API_KEY"], base_url=environ.get("OPENAI_BASE_URL") or None,
                    timeout=config.llm_timeout, max_retries=0)
    return client, model


def default_task_message(round_number, toolbox, protein_config=None):
    """轮次开场白。protein_config=None 时为 GB1 旧文案（逐字节不变）；给定配置时派生蛋白无关开场。"""
    budget = toolbox.budget
    if protein_config is not None:
        if protein_config.variant_format == "indel_notation":
            design = (f"design substitution and insertion/deletion variants on the "
                      f"{len(protein_config.wt_full_sequence)}-residue reference sequence")
        else:
            design = f"design variants over the {len(protein_config.mutable_positions)} configured mutable positions"
        return (f"Directed evolution round {round_number}: {design} to maximize fitness. "
                f"Eligible unmeasured candidates: {len(toolbox.eligible)}. "
                f"Submit exactly {toolbox.required_count()} recommendations this round. Hard budgets: "
                f"{budget.max_score_calls} candidate scorings, {budget.max_api_calls} LLM calls, "
                f"{budget.max_total_tokens} tokens. Work autonomously with the tools; end by calling "
                f"submit_recommendations.")
    return (f"Directed evolution round {round_number}: design GB1 four-site variants (positions 39,40,41,54; "
            f"reference VDGV) to maximize binding fitness. Eligible unmeasured candidates: {len(toolbox.eligible)}. "
            f"Submit exactly {toolbox.required_count()} recommendations this round. Hard budgets: "
            f"{budget.max_score_calls} candidate scorings, {budget.max_api_calls} LLM calls, "
            f"{budget.max_total_tokens} tokens. Work autonomously with the tools; end by calling "
            f"submit_recommendations.")


def run_agent_round(round_number, known, eligible, predict_fn, config, knowledge=None, client=None, model=None,
                    system_prompt=None, feedback_history=(), log_path=None, budget=None, task_message=None,
                    environ=None, protein_config=None):
    """Assemble toolbox plus agent and run one autonomous round; returns (round record, toolbox).

    protein_config 给定时走蛋白无关路径：ToolBox 校验、工具 schema、系统提示与轮次开场白全部由该
    配置派生（system_prompt/task_message 显式传入时优先）；None 时保持 GB1 旧路径逐字节不变。
    """
    toolbox = ToolBox(known=known, eligible=eligible, predict_fn=predict_fn, config=config, knowledge=knowledge,
                      budget=budget, feedback_history=feedback_history, protein_config=protein_config)
    if system_prompt is None and protein_config is not None:
        system_prompt = build_system_prompt(protein_config)
    agent = ToolCallingAgent(toolbox, client=client, model=model, system_prompt=system_prompt,
                             log_path=log_path, environ=environ,
                             tool_schemas=build_tool_schemas(protein_config) if protein_config is not None else None)
    record = agent.run_round(round_number,
                             task_message or default_task_message(round_number, toolbox, protein_config))
    return record, toolbox
