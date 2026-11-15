# 统计分析方案（v2）— GB1 Agent 升级实验

> 配套配置：`configs/experiment_v2.json`
> 统计口径基线：`configs/statistical-definitions-v2.json`（2026-11-14 冻结，本方案全部指标沿用其定义，不再重复定义 fitness / top-k overlap / quality / completion / usage 口径）
> 状态：预注册文档。数据分析前（子代理 H 开工前）冻结，不得事后修改检验族与阈值。

---

## 0. 范围与总原则

1. 本方案覆盖三类分析：
   - **修正项**：对旧数据（首版实验，RF + Prompt 架构）的统计口径修正与重新检验；
   - **新增对比项**：2×2 因子设计（评分器 × Agent 架构）的新实验；
   - **Agent 自主性指标**：基于 tool call 日志的过程性分析。
2. 全部结论措辞遵守两条硬规则：
   - **R1**：新 2×2 实验每组仅 3 个种子，所有组间数字一律标注"**描述性统计，不构成显著性判断**"；
   - **R2**：任何位置不得使用"显著优于 / significantly better"等表述，**除非**对应检验经多重比较校正后 p < 0.05（双侧），并同时给出效应量与置信区间。n=3 的 2×2 主比较默认不满足该条件。
3. 独立重复单元的认定沿用 statistical-definitions-v2 `quality.sampling_units`：**独立单元是 state 和独立生成的 episode/trajectory；proposal 与 repair 不是独立重复**。所有 bootstrap、SE、检验的样本量都按此口径计数。
4. 缺失数据处理沿用 `usage.missing` 口径：缺失记为 unknown，不得填 0；整 episode 成本含未知调用时报 unknown 并给出已知和与缺失计数。
5. 负结果也是结果（AGENTS.md 执行原则 8）：ESM-2 若不优于 RF，如实报告并分析原因，不得选择性呈现。

---

## 1. 修正项（对旧数据）

旧数据范围：首版实验全部已归档结果（historical method-rounds = 36，见 `first_response_enrichment.historical_method_rounds`）。

### 1.1 Spearman 方向修正

**问题**：旧报告把 raw position 的负 Spearman 误读为"无排序能力"。

**修正口径**（与 `statistical-definitions-v2.json` `ranking` 节一致，此处为执行说明）：

- 排序输出为 best-first（越靠前越好），`position` 数值越小越好；
- `raw_position_correlation = spearman(position, measured_fitness)`：**负值代表方向正确**（排在前面的实测 fitness 更高）；
- 报告一律使用换算后的高分方向：`reported_high_score_correlation = spearman(-position, measured_fitness)`，**正值 = 排序更好**；
- 预测器评估同理：fitness `higher_is_better = true`，验证集/测试集 Spearman(预测分, 实测 fitness) 应为正；旧 RF 基线参考值 ≈ +0.54（换算后）。

**执行动作**：旧数据所有 Spearman 值按上述规则统一重算符号后重报；文字结论中所有"无排序能力 / 方向错误"类表述逐条改写；新旧口径一致性列入子代理 I 审计清单（"新旧口径一致？"项）。

### 1.2 Friedman 检验 + Nemenyi 事后比较（多方法整体比较）

- **适用场景**：≥ 3 种方法在同一批配对单元上的指标比较（如：随机突变 / 模型直接推荐 / Prompt Agent / 知识增强 Agent，或新旧四组 G1–G4 的探索性分析）。
- **设计**：随机化区组设计。区组（block）= 同一配对评估单元（同 seed × 同 episode/state 下各方法的得分）；处理（treatment）= 方法。
- **检验**：Friedman 检验（`scipy.stats.friedmanchisquare`），统计量 χ²_F，df = k − 1（k 为方法数），双侧 α = 0.05。
- **事后比较**：整体显著时做 Nemenyi 事后检验（`scikit-posthocs.posthoc_nemenyi_friedman`），临界差

  CD = q_α · sqrt(k(k+1) / (6N))，N 为区组数，

  绘制平均秩次 + CD 横杠的 critical difference diagram。
- **样本量门槛**：N（区组数）≥ 10 时可作推断性报告；否则降级为描述性并注明（n=3 seeds 的 2×2 主比较即属此类）。
- **旧数据应用**：旧数据存在 method-round 级配对（36 轮），以 method-round 为区组可作推断性 Friedman + Nemenyi；结果与措辞按 R2 规则执行。

### 1.3 Wilcoxon 符号秩配对检验 + BH FDR 校正

- **适用场景**：两两方法配对比较（Friedman 显著后，或单独一对预先声明的对比）。
- **配对单元**：同 1.2，配对差 = 同一区组内两方法指标之差。
- **检验**：双侧 Wilcoxon 符号秩检验（`scipy.stats.wilcoxon`）；有效配对数 n < 25 且无严重并列时用精确分布（`mode="exact"`）；并列差值的处理方式（drop vs Pratt）在结果表脚注注明。
- **多重比较校正**：同一检验族内（例如 4 方法两两 6 对，或多个指标 × 多对比较的全集）统一做 Benjamini–Hochberg FDR 校正（q = 0.05），同时报告原始 p、校正后 p 和族大小。
- **措辞联动（R2）**：仅 BH 校正后 p < 0.05 的对比可写"显著"；其余一律写"在本样本量下未检测到差异"或直接报描述性数字。

### 1.4 效应量 Cohen's d + 95% bootstrap CI

- **配对设计效应量**：d_z = mean(diff) / sd(diff)（配对差标准化），报告时明确标注为 d_z（配对），不与独立样本 d 混用。
- **95% CI**：对配对差做 bootstrap；**重采样单元 = 独立重复单元**（state / episode，见第 0 节第 3 条），绝不可按 proposal 行重采样（会虚假放大 n）；B = 10,000 次（固定随机种子，如 20261114），percentile 法 2.5%–97.5%。
- **一致性检查**：d_z 的 CI 是否含 0 应与 Wilcoxon 结论方向一致；不一致时以精确检验为准，并在脚注说明（bootstrap 在极小 n 下不稳定属预期）。
- **呈现**：效应量表与 p 值表并列；n 很小时 CI 必然很宽，如实呈现宽度，不删减。

---

## 2. 新增对比项：2×2 因子设计

### 2.1 设计

| 组别 | 评分器（predictor） | Agent 模式（agent_mode） | 目的 | 数据来源 |
|---|---|---|---|---|
| G1 | RF + one-hot | prompt（旧架构） | 基线 | 复用旧实验数据 |
| G2 | ESM-2 | prompt（旧架构） | 隔离 ESM-2 因子 | 新跑 |
| G3 | RF + one-hot | tool_calling（新架构） | 隔离 Tool-Calling 因子 | 新跑 |
| G4 | ESM-2 | tool_calling（新架构） | 联合效果 | 新跑 |

- 种子：**[11, 22, 33]**（以本方案与 `configs/experiment_v2.json` 为准；AGENTS.md 子代理 E 任务书中"11/22/23"为笔误，数据划分一律用 11/22/33）。
- 每组每种子 rounds = 3，score_budget = 40；新增实验量 = 3 seeds × 3 组（G2–G4）× 3 rounds = 27 轮；G1 复用 `results/seed-{s}/run/`。
- 预算（tool_calling 模式）：max_api_calls = 15/轮、max_tokens = 50,000/轮（`configs/experiment_v2.json` `budget`，scope = per_round）；超限自动提交当前最佳并标记 `budget_exhausted`。

### 2.2 指标

主指标（沿用 statistical-definitions-v2 口径，不重新定义）：
- trajectory mean measured fitness（`quality.trajectory_mean`）；
- cumulative best（`quality.cumulative_best`，含初始已知测量，天然单调）；
- top-k overlap（`top_k_overlap`，分母 = min(k, 评估集大小)）；
- 完成率（`completion.reported_as`）与失败数**并排报告**，质量指标标注"conditional on completion"。

辅助指标：token 消耗与耗时（`usage` 口径）、首轮富集（`first_response_enrichment`，描述性）。

### 2.3 分析与措辞规则

- **每格 n = 3 个种子 → 全部组间比较为描述性统计，明确标注"描述性统计，不构成显著性判断"**（规则 R1）。
- 描述性因子分解（只报数字与方向，不做检验）：
  - ESM-2 主效应（描述性）= mean(G2, G4) − mean(G1, G3)
  - Tool-Calling 主效应（描述性）= mean(G3, G4) − mean(G1, G2)
  - 交互（描述性）= (G4 − G3) − (G2 − G1)
- 呈现格式：4 格表 = 跨种子 mean ± SD（n=3），另附每种子 × 每组的明细表（episode 级均值，标注独立单元数）。
- **"显著优于"禁用**：除非某对比经 1.3 节全流程（配对 Wilcoxon + BH 校正）得到校正后 p < 0.05（规则 R2）。默认预期：n=3 不满足 → 全文使用"在本样本量下 G4 均值高于 G1（描述性）"式表述。
- 敏感性分析（可选、预声明）：若以 episode/state 为配对单元且族已预注册，可补充推断性检验；无论结果如何，主文结论仍以描述性口径呈现，检验结果放附录。

---

## 3. Agent 自主性指标（仅 tool_calling 组 G3/G4；G1/G2 为 prompt 模式，流程固定，作构造性对照）

### 3.1 Tool 调用序列多样性（不同 Agent run 之间的调用顺序差异）

- **序列定义**：一次 run 内按时间序的 tool-name 序列（如 `get_experiment_history → query_knowledge → predict_fitness → ...`）。
- **指标**（每组内跨 run 计算）：
  1. 序列类型数 / 总 run 数（= 1 表示所有 run 完全同序）；
  2. 两两 run 间的归一化 Levenshtein 距离（对 tool-name 字符串序列计算，除以较长序列长度）的组内均值与范围；
  3. tool 相邻二元组（bigram）分布的经验熵（bits），熵越高行为越不固定。
- **对照**：prompt 模式（G1/G2）流程由 Python 硬编码，序列恒定 → 多样性恒为 0（构造性下界，无需统计检验即可陈述）。
- **解读边界**：多样性高低本身不等于好坏，只作为"自主性存在"的证据；与质量指标的关系只做描述性并列。

### 3.2 LLM 主动选择工具 vs 被动等待比例

- **决策点定义**：一次 LLM API 回合，其 choice space 含 ≥ 2 个可选动作（任一 tool + submit/stop，或 ≥ 2 个 tool）。
- **主动（proactive）**：LLM 在决策点上自主选择调用哪个工具、调用参数、或决定提交/结束（含自主调用 `submit_recommendations` 与自主调用 `check_budget`）。
- **被动（passive）**：该回合框架只提供唯一动作（强制回填、固定轮询），LLM 无实际选择权。
- **指标**：每 run 的 主动决策点数 / 总决策点数；组内跨 run 汇总（mean ± SD，n = run 数）。G3/G4 预期接近 1；显著低于 1 说明架构仍有隐藏的流程强制，需在报告中定位是哪个环节。
- **判定依据完全来自日志字段**（`decision_point`、`choice_space`、`forced`、`initiation`，见第 4 节），不靠事后人工读对话判定。

### 3.3 "LLM 看到评分后是否改变策略"的证据（tool call 序列中的策略转折点）

- **策略标签（strategy_tag）**：由预设规则从 (tool_name, arguments) 映射，日志写入时自动打标。示例规则：
  - `predict_fitness` 且候选集中在位点集 S → 标签 `explore:S` / `exploit:S`（S 为 4 位点子集，exploit = 复用历史高分位点）；
  - `query_knowledge` 按查询主题分类（理化性质 / 突变规则 / 历史 / 图谱）；
  - `get_experiment_history` / `submit_recommendations` / `check_budget` 各自固定标签。
- **转折点定义（预设、规则化）**：紧跟在 `score_query` 结果事件之后的第一个决策点 t，若
  1. tool 名改变，或
  2. 同 tool 但位点焦点集 Jaccard 重叠 < 0.5，或
  3. explore ↔ exploit 切换，
  则记为一次策略转折（`strategy_changed = true`）。
- **指标**：
  1. 每 run 转折点数；
  2. 组内 ≥ 1 次转折的 run 比例；
  3. 转折前后提交候选的预测分 / 实测 fitness 分布变化（描述性）；
  4. 附 2–3 条典型转折的完整 tool call 轨迹作为案例展示。
- **证据分级**：规则判定为主证据；LLM 在 arguments/reasons 里的自述理由为辅助证据，**不得单独作为"改变策略"的依据**。

---

## 4. 需新收集的数据字段（tool call 日志格式）

格式：**JSONL**，每次 run 一个文件，每行一个事件。这是可复现性的核心保障（首版即因缺少这些记录受限），也是第 3 节全部指标的唯一数据源。字段分为公共信封（每行必有）+ 各事件专有字段。`R` = 必需，`O` = 可选（无值时置 null，不得省略键）。

### 4.1 公共信封（所有事件行）

| 字段 | 类型 | 级别 | 说明 |
|---|---|---|---|
| `ts` | string (ISO 8601 含时区) | R | 事件时间戳 |
| `run_id` | string | R | 格式 `v2-{group}-s{seed}-r{round}-e{episode}` |
| `group` | "G1"\|"G2"\|"G3"\|"G4" | R | 因子组别 |
| `seed` | int | R | 11/22/33 |
| `round` | int | R | 1–3 |
| `episode` | int | R | 同 seed 同 round 内的重复编号 |
| `predictor` | "rf"\|"esm2" | R | 本 run 实际使用的评分器 |
| `agent_mode` | "prompt"\|"tool_calling" | R | 本 run 实际架构 |
| `event` | string | R | 事件类型（见 4.2–4.9） |
| `call_index` | int | R | run 内单调递增序号 |

### 4.2 `run_start`

| 字段 | 级别 | 说明 |
|---|---|---|
| `config_path` / `config_hash` | R / R | `configs/experiment_v2.json` 路径与内容 hash |
| `env.python` / `env.packages_hash` | R / R | Python 版本、requirements 锁定 hash |
| `env.model_requested` / `env.model_returned` | R / R | LLM 型号（请求 vs 实际返回） |
| `env.predictor_checkpoint_hash` | R | RF joblib 或 ESM-2 checkpoint hash（G3/G4 必填；G1/G2 若适用） |
| `env.esm2_layer` | O | ESM-2 组：选定的嵌入层号 |

### 4.3 `llm_request`（一次完整 API 调用）

| 字段 | 级别 | 说明 |
|---|---|---|
| `request_id` | R | 对应 API 调用 |
| `model_requested` / `model_returned` / `api_mode` | R | 同 AGENTS.md bonus 口径 |
| `context_length` | R | 输入消息总 token |
| `usage.prompt_tokens` / `usage.cached_input_tokens` / `usage.completion_tokens` / `usage.reasoning_output_tokens` | R / O / R / O | 按 `usage` 口径：cached 计一次、reasoning 不重复计 |
| `latency_ms` | R | 全程生成耗时 |
| `finish_reason` | R | stop / tool_calls / length / error |
| `tools_offered` | list | 本次向 LLM 开放的工具名列表 |
| `decision_point` | bool | 是否构成 3.2 节定义的决策点 |
| `choice_space` | list | 实际可选动作（tools + submit/stop），判定 proactive 用 |
| `followed_score_event` | bool | 本回合是否紧跟 score_query 结果之后（3.3 节转折点判定用） |

### 4.4 `tool_call`

| 字段 | 级别 | 说明 |
|---|---|---|
| `call_id` | R | OpenAI tool_call id |
| `tool_name` | R | 5 工具之一（见下） |
| `arguments` | object | 已解析的参数 JSON（非字符串） |
| `initiation` | "llm"\|"framework" | 谁发起：LLM 自主 vs 框架注入 |
| `forced` | bool | 是否唯一可用动作（被动） |
| `strategy_tag` / `prev_strategy_tag` / `strategy_changed` | R / R / R | 3.3 节规则自动打标结果 |

工具名固定为子代理 B 定义的 5 个：`predict_fitness`、`query_knowledge`、`get_experiment_history`、`submit_recommendations`、`check_budget`。

### 4.5 `tool_result`

| 字段 | 级别 | 说明 |
|---|---|---|
| `call_id` / `tool_name` | R | 回填对应 |
| `result_summary` | R | 结构化摘要（如返回条数、top 项），避免整包膨胀 |
| `result_hash` | R | 完整结果内容的 hash（全量结果落盘单独文件时关联用） |
| `error` / `error_message` | R / O | 失败标记 |

### 4.6 `score_query`（predict_fitness 触发的预算消耗）

| 字段 | 级别 | 说明 |
|---|---|---|
| `candidate_ids` | list | 本次送评变体 ID |
| `predicted_scores` | list | 预测分 |
| `score_calls_used` | int | 本次消耗计数 |
| `budget_remaining.api_calls` / `.tokens` / `.score_calls` | R / R / R | 三类预算余量快照 |

### 4.7 `submit`

| 字段 | 级别 | 说明 |
|---|---|---|
| `sequences` / `reasons` | R / R | 提交序列与理由 |
| `predicted_scores` | R | 提交时预测分 |
| `selected_top_k` | list | 最终 Top-k ID |

### 4.8 `budget_event`

| 字段 | 级别 | 说明 |
|---|---|---|
| `kind` | R | "warning" \| "exhausted" |
| `budget_snapshot` | R | 触发时三类预算余量 |

### 4.9 `run_end`

| 字段 | 级别 | 说明 |
|---|---|---|
| `status` | R | "completed" \| "budget_exhausted" \| "error" |
| `totals.api_calls` / `.tokens` / `.score_calls` / `.wall_time_s` | R | 汇总（按 `usage.reported_known_sum` 口径附未知计数） |
| `final_metrics` | O | trajectory mean / cumulative best / top-k overlap 快照 |

### 4.10 示例行（`tool_call`）

```json
{"ts": "2026-11-15T10:23:41+08:00", "run_id": "v2-G4-s11-r1-e0", "group": "G4", "seed": 11, "round": 1, "episode": 0, "predictor": "esm2", "agent_mode": "tool_calling", "event": "tool_call", "call_index": 6, "call_id": "call_abc123", "tool_name": "predict_fitness", "arguments": {"sequences": ["VDGV", "IDGV", "VDAV"]}, "initiation": "llm", "forced": false, "strategy_tag": "exploit:{39,41}", "prev_strategy_tag": "explore:{39,40,41,54}", "strategy_changed": true}
```

### 4.11 与旧字段的最小兼容

子代理 B 任务书的最小字段 `{timestamp, call_id, tool_name, arguments, result, tokens_used, context_length}` 全部被本 schema 覆盖（映射：timestamp→ts，result→tool_result.result_summary/result_hash，tokens_used→llm_request.usage，context_length→llm_request.context_length），即按本 schema 记录自动满足最小要求。

---

## 5. 预注册决策规则（防 p-hacking）

1. 检验族在数据收集前固定：旧数据 = {Friedman 族（多方法）} ∪ {两两 Wilcoxon 族}；新 2×2 = 描述性，默认不进检验族。
2. α = 0.05，双侧；FDR 阈值 q = 0.05。
3. 主指标在跑实验前锁定为第 2.2 节所列；不得事后更换或增补后只报有利项。
4. bootstrap 随机种子固定 20261114，B = 10,000。
5. 所有分析脚本与原始日志一并归档至 `results/v2_experiments/`，供子代理 H（从数据独立计算）与子代理 I（黑盒审计）双盲交叉验证。
