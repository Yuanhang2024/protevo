# GB1 Agent v2 —— LLM 自主 Tool-Calling 架构说明

> 对应代码：`protevo/tools.py`（工具层）+ `protevo/agent_v2.py`（智能体循环）+ `tests/test_agent_v2.py`（mock 测试）
> 创建日期：2026-11-14。原 v1 代码（`protevo/agent.py` + `protevo/experiment.py`）一字未动，保留新旧对比。

---

## 1. 背景与动机

v1 架构中，流程控制完全由 Python 硬编码：

```
Python 固定循环 { 组装上下文 → LLM 一次性生成 JSON 候选 → Python 打分 → Python 选 Top-k → Python 查标签 }
```

LLM 只决定"候选内容"这一件事：它看不到预测分数，不能先查知识再设计，不能根据评分结果调整策略，
何时结束也由 Python 决定。流程控制权完全在程序手中。

v2 将流程控制权交给 LLM：Python 只提供工具和预算护栏，LLM 自主决定调用顺序、迭代次数和提交时机。

## 2. 总体架构

```
                         ┌──────────────────────────────────────────────┐
                         │              ToolCallingAgent                │
   system prompt ──────► │  while submission is None:                  │
   task message         │    ① 预算检查（3 项硬上限，耗尽→自动提交）      │
   tools=TOOL_SCHEMAS   │    ② chat.completions.create(messages,tools) │
                         │    ③ response.message.tool_calls            │
                         │    ④ 逐个执行 → role:"tool" 消息回传          │
                         │    ⑤ 无 tool_calls → 追加提示继续             │
                         │  直到 LLM 调 submit_recommendations 或预算耗尽 │
                         └───────────────┬──────────────────────────────┘
                                         │ execute(name, arguments)
                         ┌───────────────▼──────────────────────────────┐
                         │                 ToolBox                      │
                         │  predict_fitness / query_knowledge           │
                         │  get_experiment_history / check_budget       │
                         │  submit_recommendations (+force_submit)      │
                         │  状态: known, eligible, score_cache,          │
                         │        submission, AgentBudget               │
                         └──────────────────────────────────────────────┘
                                         │ 每次 API 调用与 tool call
                         ┌───────────────▼──────────────────────────────┐
                         │   JSONL 日志（append + fsync，逐条落盘）       │
                         └──────────────────────────────────────────────┘
```

OpenAI function calling 协议（已核实官方格式）：

- 请求：`messages` + `tools=[{type:"function", function:{name, description, parameters}}]` + `tool_choice="auto"`
- 响应：`message.tool_calls=[{id, type:"function", function:{name, arguments(JSON 字符串)}}]`
- 回传：`messages.append({role:"tool", tool_call_id: id, content: result_json})`

## 3. 五个 Tools

| 工具 | 参数 | 返回 | 预算效应 |
|---|---|---|---|
| `predict_fitness` | `sequences: list[str]` | `{scored:[{rank,variant,predicted_fitness}], errors, not_scored, remaining_score_budget}` | 每个新评分的合格候选消耗 1 次评分预算；已评分走缓存免费；非法/已测/不合格不消耗 |
| `query_knowledge` | `query: str` | `{query, matched_rules}`；无命中时附 `available_rule_ids` | 无 |
| `get_experiment_history` | 无 | `{round_context, known_data(含逐位置残基统计), previous_recommendations, scored_variants_this_round}` | 无 |
| `submit_recommendations` | `sequences: list[str], reasons: list[str]` | `{status, recommendations, budget_exhausted}` | 终止信号：调用成功即结束本轮 |
| `check_budget` | 无 | `{api_calls_remaining, tokens_remaining, score_calls_remaining, ...}` | 无 |

关键约束（在工具内强制执行，坏输入以 `error` 字段返回、绝不中断循环）：

- `predict_fitness` 只接受合格且未测量的候选池变体；评分预算不足时截断并把余量列入 `not_scored`；
- `submit_recommendations` 要求恰好 `k` 个（`min(k, 可用候选数)`）唯一、合格、未测量的变体，`sequences` 与 `reasons` 等长；
  重复提交被拒绝；
- 已知数据摘要复用 v1 的 `summarize()`（含"边际统计非因果效应"的解释），反馈历史只包含本方法已测变体，无隐藏标签泄漏。

## 4. 预算硬限制与自动提交

每轮三项上限（默认值即任务规定）：`max_api_calls=15`、`max_total_tokens=50000`、`max_score_calls=40`。

- 检查点：**每次 API 调用之前**检查三项计数，任一耗尽立即跳出循环；
- token 计数来自每次响应的 usage（复用 v1 `usage_counts()` 的兼容归一化；usage 缺失按 0 计并打标）；
- 评分预算在 `predict_fitness` 内部硬性截断，物理上不可能超限；
- LLM 纯文本回复（无 tool_calls）时追加一条 user 提示让其继续，API 预算保证循环必然终止。

**自动提交语义**（`budget_exhausted=True`）：按预测分降序取当前已评分前 `k` 个作为"当前最佳"；
若一个都没评过，用排序后的合格池前 `k` 个做确定性回补（reason 中如实标注 unscored fallback）。
LLM 主动提交则 `budget_exhausted=False`。轮次记录里 `status` 区分 `submitted` / `auto_submitted`，
`stop_reason` 记录 `max_api_calls` / `max_total_tokens` / `max_score_calls`。

## 5. 全量 JSONL 日志（可复现性核心）

每次 API 调用、每次 tool call、每轮起止各写一行（append + flush + fsync）。统一字段：

| 字段 | 含义 |
|---|---|
| `timestamp` | UTC ISO 时间 |
| `call_id` | 轮内单调编号，如 `r2-007` |
| `round` | 轮次号 |
| `tool_name` | 工具名；API 调用记录为 `null` |
| `arguments` | tool call 的已解析参数；API 调用记录为 `{tools, tool_choice}` |
| `result` | 工具返回 / API 调用摘要（finish_reason、tool_call_count、assistant 正文） |
| `tokens_used` | 该 API 调用的 total tokens（tool call 恒为 0） |
| `context_length` | 发送时的消息条数 |
| `model` | 请求模型名（API 记录另含 `model_returned`、`response_id`、`usage`、`context_chars`） |

记录种类（`kind`）：`round_start`（含预算快照与候选池规模）→ 交替的 `api_call` / `tool_call` →
`round_end`（含 tool 调用序列）；异常路径写 `api_error`；自动提交的 tool_call 记录额外带
`auto: true, budget_exhausted: true, stop_reason`。

## 6. v1 与 v2 对比

| 维度 | v1（agent.py + experiment.py） | v2（tools.py + agent_v2.py） |
|---|---|---|
| 流程控制 | Python 固定步骤 | LLM 自主决定工具调用序列 |
| LLM 输出形式 | 单次 JSON 提案批 | 多轮 function calling |
| LLM 能否看到预测分 | 否（打分在 Python 内部） | 是（可评分→看结果→调整再提交） |
| 知识使用 | 整块塞进上下文 | LLM 按需检索 |
| 结束时机 | Python 决定 | LLM 主动提交或预算耗尽自动提交 |
| 预算 | 请求次数/token 两项 | API 次数 + token + 评分次数三项硬上限 |
| 日志 | events.jsonl（请求/响应/阶段） | 逐 API 调用 + 逐 tool call 的统一 schema |
| 候选校验 | collect_proposals 事后校验 | 工具入参即时校验，错误回传给 LLM 自纠 |

## 7. 接入方式

```python
from protevo.agent_v2 import run_agent_round, sklearn_predictor
from protevo.experiment import known_for          # v1 的轮间反馈组装可直接复用
from protevo.tools import AgentBudget

record, toolbox = run_agent_round(
    round_number=2,
    known=known_for(dataset, method_state),   # DataFrame: variant/fitness/mutations
    eligible=eligible_set,                    # 未测量且合格的候选变体集合
    predict_fn=sklearn_predictor(model),      # 或 ESM-2 预测器，接口 list[str] -> list[float]
    config=config,                            # ExperimentConfig（用其 k 与突变约束）
    knowledge=load_knowledge("configs/knowledge.json"),  # 可选
    feedback_history=previous_round_feedback, # 可选：[{round, recommendations:[{variant, actual_fitness, ...}]}]
    log_path="results/v2/agent-r2.jsonl",
    budget=AgentBudget(max_api_calls=15, max_total_tokens=50000, max_score_calls=40),
    # client/model 省略时从 .env 的 OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL 构建
)
# record: {status, budget_exhausted, stop_reason, submission{sequences, reasons,
#          predicted_fitness, budget_exhausted, auto}, tool_call_sequence, 用量, log_path}
```

每轮结束后按 v1 流程把 `record["submission"]["sequences"]` 送入 `HiddenOracle` 取真实反馈，
追加进 `feedback_history` 进入下一轮（与 G 组 2×2 因子实验的接线方式一致）。

## 8. 测试策略（无真实 API）

`tests/test_agent_v2.py`（13 个用例）使用脚本化 mock 客户端：注入只实现
`chat.completions.create(**kwargs)` 的对象，按脚本弹出预构造响应（含 `tool_calls`、usage），
并记录每次请求用于协议断言。覆盖：

- 5 个 schema 的 OpenAI 格式合法性；
- 各工具的评分/缓存/预算截断/知识匹配/历史汇总/提交校验/坏参数以错误值返回；
- 端到端 happy path：多工具调用 → 评分 → 主动提交；消息协议（tool_call_id 对应、tools 参数、
  上下文增长 2→5→7）；JSONL 记录种类顺序与全部必需字段；
- 三种预算耗尽各自触发自动提交（含纯文本回复的 nudge 循环与确定性回补路径）。

## 9. 设计决策记录

1. **错误以值返回**：工具内一切校验失败返回 `{error: ...}` 并回传给 LLM 自行纠正，agent 循环永不因坏输入崩溃；
2. **缓存评分免费**：同一变体重复评分走缓存，避免 LLM 浪费预算；
3. **预算检查在调用前**：保证"最多 15 次调用"是严格的，而不是超了才发现；
4. **自动提交保留可用信息**：已评分按预测分排序，未评分如实标注，绝不伪造证据；
5. **评分预算与 legacy `score_budget` 解耦**：v2 预算是 Agent 自治护栏，实验接线时由调用方对齐两者；
6. **默认 system prompt 内置**：完整科学提示词由 Wave 3F 的 `prompts/scientist_v2.txt` 提供，
   `system_prompt` 参数可直接替换。
