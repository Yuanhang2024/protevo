# 统计修订与补充实验实施说明

本文件为实施分工。原始文件已按明确名单备份到本目录originals，原始实验响应与数据不覆盖。正式科学定义、报告与解释由Lead撰写。

## 顺序

1. 先修正统计定义、计算、报告及可复算资料，运行测试并形成correction-status.json（status=passed）。
2. 在新目录预先冻结补充实验参数和随机顺序；不读取新结果选模型/状态。
3. 先执行无LLM强模型对照；反馈实验需要Lead明确允许后才调用既有项目API。
4. 结果作为预试或正式研究的范围如实标注，不预设正收益，不覆盖历史、不自动推送仓库。

## 代码分工

- 统计代码：只改results/critique-followup-20261113/{zero_context.py,analyze_stats.py}、tools/calibration-report/statistics_report.py，新增纯函数模块gb1/statistical_definitions.py及对应test。历史JSON/CSV不覆盖，重算输出新目录。
- 强模型对照：只写tools/supplement_20261114/strong_baselines.py与对应tests；不得改现有核心/data/prompts。
- 反馈预试：只写tools/supplement_20261114/feedback_pilot.py及对应tests，与strong作者协调可复用其prepare函数但不同时修改同一文件；不得改主插件/原实验核心/历史NoFeedback脚本。
- Lead：指标定义、主/验证报告生成器、Markdown与README、PDF导出、统计检查和实验允许状态。

## 统一统计规则

- best-first位置p越小越优；高分方向score=-p。同时保留position-vs-fitness原符号和score-vs-fitness符号，不能把字段名仍称rank-position却悄悄翻号。
- 排序回答必须为输入全集的无重复排列。新排名评估真值Top-k并列用变体ID，与模型排序无关。历史模型评估固定输入顺序并列规则保留其版本标记，不静默改历史值。
- 首答是每个方法/轮次首次原始回答，不是仅实验第1轮；现有1408条来自36方法轮次。全局高分阈值用S1实际分位数，均匀比例按实际达到阈值的条目数计算。
- 锚点统计同时给候选—锚点对加权与候选记录等权；候选group=(source,method,round,variant)。一轮全部完成后才把该轮选中变体加入可见known集合。
- 完成率报告完成数/计划数。全0/全1二元样本的经验bootstrap区间不可当总体概率区间，lower/upper置null并给reason，观察比例保留。两臂完成差退化时同理。
- 不从三状态近零差推无效应，不从高分重合确定记忆机制，不把高分选择同均匀S1尾部比较解释为锚点因果效应。
- 用量缺失为未知；可保留reported-known sum，但完整成本平均/中位数不能把缺失当0。
- 禁止显式hash/checksum；禁止读取凭据到工具输出；不发起任何未经允许的实验模型请求。

## 强模型对照

优先复用历史seed11/22/33的data/processed/seed-*/与results/seed-*/selection/selection.json，检查相同manifest/config。比较3轮、每轮10标签，初始2000、val1000、最终test5000不进入迭代：

- uniform_random：无标签均匀选样。
- full_pool_frozen：初始选定预测器仅拟合初始2000一次，各轮对完整剩余候选池预测后Top10。
- full_pool_updated：同一选定spec，每轮只用本臂此前10/20个新观测重训，再全池Top10。

对照按相同新增标签预算30，不限制全池预测次数为40，准确记录score_count/CPU时间。隐藏标签仅选定Top10写入记录后通过HiddenOracle读取，各臂反馈隔离。保存全部候选ID/预测（可压缩CSV）、选中、实测、训练ID、选择spec与版本。训练标签/预测重算校验可以在纯分析阶段进行，不能预先提供给选样函数。

新状态1201/1202/1203/1204可供反馈准备使用；须先声明后生成，不以fitness好坏挑选或剔除状态。基础配置遵循现有推荐科学参数C40/k10，模型spec由各状态验证集规则选定。

## 反馈单步预试接口（先实现/测试，禁止自行启动API）

默认4个新状态×3次独立生成重复×2种上下文=24个生成臂，最多48次调用（每臂一次设计加一次纠正）。选择器P0/P1对同一生成批次回放，形成48个选择结果；这是固定状态一步机制比较，不当作完整多轮效果。最终参数及预算由Lead冻结，可缩小。

每状态：train2000/val1000/test5000/候选其余；按验证选spec，训练P0。用P0全池贪心预先选10个共同前缀，先保存选择/预测，再查询标签。训练P1=train+prefix10。记录这10标签是共同状态成本，不混入后续组间结果。

上下文C0：只显示初始2000的观测摘要，无新增prefix标签、残差或其聚合信息。
上下文C1：显示初始+prefix的更新摘要与全部10个prefix的实际/原预测/残差及原选择假设。
两组均显示相同round=2，拥有相同候选资格和已测排除名单。可将所有2010已测ID作为共同forbidden/observed ID名单，说明部分已测值不展示。不能通过C0 top/low/median/positions/fitness_mean/best/feedback_summary暴露prefix标签。LLM不知道当前使用P0或P1选择器。

生成预算和规则：每臂最多2次；first_n，仅原始顺序前请求N条；require_full_batch=True；40条不足即标失败、不用程序补齐；纠正使用本状态config.seed及预先分配repair设计种子，两个C臂配对保留相同设计seed，但不宣称API随机性因此配对。修复输入的known表仅需ID；两臂传相同排除ID，不能在修复中恢复C0隐藏标签。保持v3原提示。

生成完40条后分别用P0/P1对相同40条评分，各自选Top10、保存选择后查询原S1；记录两选择器重复/不同的实际ID。候选生成不重复调用以挑高分。处理失败/不明网络结果：保留，停止该臂，绝不自动重发未知调用。每次dispatch前持久化；恢复只继续未派发臂。

使用现有OpenAIProposer/项目连接；prepare和public-preflight不调用generate，只返回非秘密model_requested/api_mode/参数，绝不显示BaseURL/Key或dotenv内容。必须绑定预先声明模型ID，API返回错误仅记录异常类别与数值status，不记录可能含认证信息的原始异常。不得更改DSH profile或安装新Host helper。

必须记录：输入完整payload（无认证）、prompt、参数、时间、调用序号、raw public response、usage、accepted/rejected、全部P0/P1评分、Top10、选后标签、训练ID与模型spec、状态和错误码。每次请求有唯一operationId/episodeId，unknown不自动重试；只将固定已完成state prefix用于两C臂，不互传本次选择标签。

主结果：每状态固定选择器P0下，先平均同条件独立生成重复的Top10均值，再取C1-C0，最后状态等权。P1下的C效应及交互作用为次要；任何失败单独给n/N和完成条件下描述，不删除失败后宣称因果收益。预试不作确认性显著性结论，先检查生成波动/日志/预算可行性。

需要fake-proposer离线测试：C0不泄漏新增观测、两组资格相同、同一候选能回放两模型、种子/round正确、选后查询、完整记录可重建、duplicate operation不重发、unknown停止、缺usage阻止新派发、C40/k10失败保留。新输出必须空目录或严格恢复同计划，不能覆盖历史。
