# scientist_v2.txt 设计说明（Tool-Calling 模式系统提示）

> 配套文件：`prompts/scientist_v2.txt`（799 个非空白字符，其中汉字 552，落在 500–800 字要求区间）
> 服务对象：`gb1/agent_v2.py` 的 `ToolCallingAgent`；工具行为定义见 `gb1/tools.py`

## 一、定位与注入方式

本提示是二代智能体（Tool-Calling 架构）的系统提示，经 `system_prompt` 参数注入：

```python
from gb1.agent_v2 import run_agent_round
from pathlib import Path

system_prompt = Path("prompts/scientist_v2.txt").read_text(encoding="utf-8")
record, toolbox = run_agent_round(..., system_prompt=system_prompt)
```

与每轮任务消息（`default_task_message`）的分工：**系统提示只写稳定的角色、工具定位、推理模式与纪律**；轮次号、合格候选数、要求数量、三项预算的具体数值（15 次 API / 50K token / 40 次评分）由任务消息携带。因此提示中刻意不写死任何数字，配置改动后无需改提示。

与 `prompts/` 现有文件的关系：`legacy-v1.txt`、`scientific-v2.txt`、`scientific-v3.txt` 服务于旧 JSON-schema 模式（`gb1/agent.py`：LLM 只输出候选 JSON，流程由 Python 硬编码）；`scientist_v2.txt` 是 Tool-Calling 模式的系统提示，"v2" 指二代智能体，命名与 `scientific-v2.txt`（旧模式第二版提示）刻意区分，避免混淆。

## 二、任务要求逐条落实

| 要求 | 提示中的落实 |
| --- | --- |
| 告知 5 个 Tool 及功能 | 第二段编号清单，每个工具一句话功能定位 |
| 鼓励科学推理流程 | 第三段"形成假设 → 查询知识与数据 → 设计候选 → 预评估 → 依据结果修正假设 → 提交" |
| 不限定调用顺序 | 清单前明示"调用时机与顺序完全由你自主决定，下列编号仅为列举"；循环明示"参考模式而非规定流程" |
| 预算意识 | 第四段整段：三项硬限制、耗尽即自动提交、大批量评分前先 check_budget、留意返回的剩余预算、为提交保留至少一次调用 |
| 可先查知识库或先看历史 | 第三段显式列出两种合法开局 |
| 评分后可改变策略 | 第三段"如果 predict_fitness 的结果与预期相悖，鼓励你修正假设、更换或补充候选并再次评估" |
| 提交需给理由 | 工具 5 描述"每条附一句科学理由"（与 `submit_recommendations` 的 reasons 参数校验一致） |
| 中文书写 | 全文中文，仅工具名、位点数字、氨基酸串保留英文 |

## 三、关键设计决策

**1. 工具清单只写功能定位，不复制参数 schema。** 参数与调用规范已通过 `tools=TOOL_SCHEMAS` 以 OpenAI function calling 原生格式下发给模型，系统提示里重复只会浪费 token。清单补充了 schema 描述里没有强调的行为后果（如"只有新评分的合格未测变体消耗评分预算"），这是 LLM 做批量评分决策时最需要知道的计费规则。

**2. 示例循环必须显式降级为"参考模式"。** 这是本提示最重要的一句。LLM 指令遵循倾向很强：若给出编号流程而不加豁免声明，模型会把它当作必须执行的脚本，27 轮实验的 tool 调用序列会高度同质化——升级到 Tool-Calling 的核心目的（让 LLM 自主决策流程）就落空了，子代理 C 定义的自主性指标（调用序列多样性、评分后策略改变）也拿不到证据。因此做了三重强化：清单前的"顺序由你决定"、循环后的"参考模式而非规定流程"、以及显式列出两种不同开局（先知识库 / 先历史数据）来示范"顺序不唯一"。

**3. 显式许可"看到评分后改策略"。** 同样是自主性的关键证据点。若不写这句话，模型倾向于把首轮设计当作承诺，评分后只做排序不做修正。写明"与预期相悖时可修正假设、更换或补充候选并再次评估"，给了模型偏离初始方案的合法通道。

**4. 预算提醒三处落点。** 对应 `AgentBudget` 的三条硬预算与耗尽自动提交逻辑：(a) 大批量评分前先 `check_budget`（对应 schema 描述的建议）；(b) 留意 `predict_fitness` 结果里的 `remaining_score_budget` 字段（结果自带，无需额外调用）；(c) 始终为最终提交保留至少一次 API 调用——最劣情形是评分耗尽评分预算后没有调用次数提交，只能靠自动提交兜底，这是要在提示层面主动规避的。注意提示只把自动提交作为预算风险告知，不把它描述成可依赖的退出方式，防止模型"躺平"等自动提交。

**5. 科学纪律条目与工具返回的 notes 对齐。** "边际残基统计是观察而非因果"来自 `get_experiment_history` 的 notes；"预测是估计而非实测"来自 `predict_fitness` 的 notes；"单点有利不保证组合有利"强调的上位性效应。提前在系统层写明，减少模型与工具返回注释间的认知冲突，也压低幻觉分数与虚构规则编号的概率。

**6. 不写死预算数值、不写死 k。** 见第一节数值分工；提交数量要求由模型从任务消息和 `get_experiment_history` 的 `required_recommendations` 读取，提示只说"恰好要求数量"。

## 四、与 tools.py 行为的一致性核对

| 提示表述 | tools.py 依据 |
| --- | --- |
| 只有新评分的合格未测变体消耗评分预算 | `_predict_fitness`：仅 `to_score`（新且合格）计费，失败请求不计费 |
| 预测是估计而非实测 | 结果 `notes` 字段同文 |
| 无匹配时列出可用规则编号 | `_query_knowledge` 返回 `available_rule_ids` |
| 提交需恰好要求数量、互异、合格、未测 | `_submit_recommendations` 四条校验，逐条对应 |
| submit 结束本轮 | 提交成功后循环条件 `submission is None` 退出 |
| 任一预算耗尽触发自动提交 | `exhausted_reason` + `_auto_submit` |

## 五、给后续子代理的验证建议

子代理 G 跑实验时，本提示的效果可从 JSONL 日志直接检验：

1. **开局多样性**：各轮首个工具调用是否在 `get_experiment_history` 与 `query_knowledge` 间分化（而非全部同一入口）；
2. **策略修正证据**：`predict_fitness` 之后是否出现重新 `query_knowledge` 或修改假设的再评估（而非直接提交）；
3. **主动提交率**：`auto=False` 的提交占比，以及被拒绝的 submit（数量/资格错误）次数；
4. **预算行为**：大批量评分前是否有 `check_budget`，是否出现预算耗尽导致的自动提交。

若 27 轮调用序列几乎一致，应优先排查模型是否仍把示例循环当脚本执行，再考虑加强第三节的豁免措辞。
