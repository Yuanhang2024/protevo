# 统计声明独立审计报告（子代理 I）

- 审计日期：2026-11-14
- 审计者：子代理 I（与子代理 H 独立运行，未参考 H 的中间结果；`results/statistics_computed/statistics.json` 在审计清单内，其数值一律从原始数据独立复算）
- 审计对象：
  1. `reports/实验结果_修订版.md`
  2. `reports/实验结果.md`（旧版）
  3. `README.md` 结果段（"首轮v1结果"、"固定版本验证与详细反馈消融"、"六项扩展实验"等含数字段落）
  4. `results/v2_experiments/factorial_summary.csv`
  5. `results/statistics_computed/statistics.json`
- 复算脚本：本目录 `audit_*.py`（共 14 个，全部可直接用 `.venv\Scripts\python.exe` 重跑）
- 原始数据源：`results/seed-{11,22,33}/run/*`、`results/prompt-v3/*`、`results/v2_experiments/G{2,3,4}/*`、`results/feedback-ablation/*`、`results/critique-followup-20261113/*`、`results/predictor_v2/summary.json`、`results/bonus-learning/*`、`results/bonus-graph/*`、`results/default-calibration-20261113/*`、`data/raw/elife-16965-supp1-v4.xlsx`（149,361 条唯一变体已重新解析确认）

## 审计清单结论总表

| # | 项目 | 结论 | 说明 |
| --- | --- | --- | --- |
| 1 | 每个数字能从原始 CSV 复算（抽验 ≥20%） | ✓ | 实际复算覆盖率远超 20%（接近 100%，见下文逐项）；唯一不能从原始预测复算的是 G3/G4 的 test 指标（原始逐点预测未持久化，文件中已如实标注来源为 state.json） |
| 2 | Spearman 方向正确（高分=正） | ✓ | 零数据排序、预测器方向、逐轮位置相关三处方向全部独立复算正确 |
| 3 | "显著"都有 p<0.05 支撑 | ⚠ | 报告文本本身不主张显著性（描述性口径，✓）；但 statistics.json 中 24 个 Wilcoxon 检验里有 3 个的 p_raw 违反其预注册规则（见问题 A），若按规则用并列校正 p，BH24 校正后显著数由 4 变为 0 |
| 4 | 样本量 n 与数据行数一致 | ✓ | 360=4×3×30；36 轮；v3=180；区块=9；factorial 12 个种子行=4×3；全部与文件行数一致 |
| 5 | SD vs SE 标注正确 | ✓ | 所有 "±" 均为样本标准差（ddof=1），复算全部一致 |
| 6 | CI 方法正确 | ✓ | 分块百分位 bootstrap（B=10,000、固定种子），独立重跑落在蒙特卡洛误差内；校准报告双层 bootstrap 单位交代清楚 |
| 7 | 多重比较已校正 | ✓（附问题 A 传导） | BH-24 全族 + 每指标族内 BH-6、Friedman 跨 4 指标 BH、Nemenyi CD 均复核无误 |
| 8 | 结论未超出数据范围 | ✓ | 修订版措辞（"仅描述""尚未证明""无法区分来源"）与数据支持度相称；README 有修订警示框 |
| 9 | 新旧口径一致 | ✓ | 新旧报告核心表数值逐位一致；修订版只改解释不改数 |
| 10 | 效应量方向与文字一致 | ✓ | 图谱负面结果、ESM2<RF、tool_calling>prompt、消融正差、自适应−冻结 −0.008 等方向全部与数据一致 |
| 11 | Agent 自主性声明有 tool call 日志证据 | ✓ | v1 events.jsonl 无 LLM 工具事件（纯程序流水线）；G3/G4 agent_log.jsonl 含 api_call(5 工具、tool_choice=auto) 与 auto=true 强制提交事件，与声明一致 |
| 12 | 事实性声明（GP/CUDA 等）正确 | ✓ | 本机实测 sklearn GaussianProcessRegressor 纯 CPU 拟合通过；gpt-6-luna、49 tokens 连接检查、149,361 条 S1 均核实 |

## 发现的问题（✗/⚠ 项详述）

### 问题 A（⚠，statistics.json）：new_high_count 三个 Wilcoxon 检验的 p_raw 违反其自身预注册规则，BH24 显著性结论不稳健

- **报告原文声明**（`results/statistics_computed/statistics.json`，part1_3_wilcoxon）：
  - `new_high_count / model - random`：`mode: "normal approx w/ continuity (ties or zeros present; zero diffs dropped)"`，`p_raw: 0.00390625`，`significant_BH_q0.05: true`
  - `new_high_count / llm - random`：同上 mode，`p_raw: 0.00390625`，显著
  - `new_high_count / knowledge - random`：同上 mode，`p_raw: 0.00390625`，显著
  - 汇总字段：`n_significant_after_BH24: 4`
- **实际数据/计算**：三组配对差为整数且 |差| 存在大量并列（model−random 的 |diff| = [1,1,1,2,3,4,1,4,5]，9 个值只有 5 个唯一值）。按该文件自己声明的规则（`footnote_zero_diffs`：有并列 |diffs| 时应使用带连续性校正的正态近似），scipy 复算 p 应为 **0.008492 / 0.009030 / 0.008849**；而 0.00390625 = 2×2⁻⁹ 恰是无并列时精确检验的 p。即 mode 标签与 p 值互相矛盾，p 取的是在并列数据上不保守的"精确"值。
- **差异大小和影响**：
  - 直接差异：p 从 ~0.0085–0.0090 报告为 0.0039（约 2.2–2.3 倍）。
  - 传导到 BH24：用并列校正 p 替换后重算 BH，**24 个检验中 0 个** 在 q<0.05 下显著（最小 BH 值 ≈0.054），而非报告的 4 个。唯一在两种口径下都存活的是 `mean_fitness / knowledge - random`（精确 p=0.0078125，无并列，精确模式合法；BH24=0.046875<0.05）——该条不受影响。
  - 族内（每指标 6 个）BH 校正下，三个 new_high_count 检验的 BH6 ≈ 0.018–0.026，仍 <0.05；结论对校正族的选择敏感。
  - 效应量（mean_diff、d_z、W）与描述统计不受影响；Friedman 主结果（chi2、BH 后 p）为秩检验，不受此问题影响。
- **建议**：对这三个检验改用并列校正 p（或精确置换分布含并列修正），并同步更新 `n_significant_after_BH24`；报告文本未引用该显著数，无需改报告。

### 问题 B（⚠，statistics.json）：`pairs_beyond_CD` 的配对命名易误读

- **报告原文**：`"pair": "random>llm", "mean_rank_diff": -1.9444`（mean_fitness、max_fitness、new_high_count 三处）。
- **实际情况**：该文件的平均秩采用"秩 1 = 最小 fitness"约定（scipy friedmanchisquare 内部约定），llm 平均秩 3.2778 > random 1.3333，即 **llm 显著优于 random**。`random>llm` 字样按字面读会得出相反结论；负的 mean_rank_diff 也说明该标签并非"左侧优于右侧"。
- **差异大小和影响**：数值本身（|Δrank|=1.9444 > CD=1.5635）与方向均正确，纯命名/可读性问题；但这是唯一一处可能被直接误读为"random 优于 llm"的表述，建议改为 `llm>random` 或注明"高秩=高fitness=优"。

### 问题 C（⚠，非错误，提示）：Nemenyi CD 与 p 的分布约定需要随文说明

- `CD_alpha0.05 = 1.563463` 对应 q=2.569（Demšar 2006 表值；等于 studentized range ppf(0.95,k=4,∞)=3.633 除以 √2）。若有人直接用 `scipy.stats.studentized_range.ppf(0.95,4,inf)` 会得到 CD=2.211，误判为"无显著配对"。
- 独立复算确认：全部 6 个 Nemenyi p 值在公式 `p = 1 − sr.cdf(|Δrank|·√2/se, k=4, ∞)`（scikit_posthocs 的 tukey 约定）下逐位复现（如 random-llm：我的 0.007637 vs 发布 0.007635）。CD 与 p 同一约定，内部自洽。建议在文件中注明该约定，避免复现者用错分位数。

### 问题 D（⚠，范围声明）：G3/G4 的 test 指标不可从原始逐点预测复算

- `factorial_summary.csv` / statistics.json 中 G3/G4 的 `test_spearman/test_mse/test_top10_hit_rate` 来源为 `state.json final_test_metrics`，原始 5,000 点预测未持久化（文件已如实标注）。我核对 state.json 存储值与 CSV 一致（G3: 0.5363/0.5195/0.5412；G4: 0.4352/0.4156/0.4110 等），但无法像 G1/G2 那样从 `*-test-predictions.csv` 重算 Spearman。属可核查性缺口而非数值错误；G1/G2 的同名指标我已从原始预测 CSV 独立重算并逐位一致。

## 逐项复算明细（证据）

### 1. v1 核心表（两份报告 + README）

从 `results/seed-{11,22,33}/run/recommendations.csv`（360 行）与 `round_metrics.csv`（36 行）重算：

- 每轮均值表 12 个单元格（random 0.134/0.037/0.352 … knowledge 0.658/2.558/1.328）与 ±SD（ddof=1）**全部逐位一致**。
- 三轮新增最高：random 3.241±1.931、model 4.029±2.521、llm 5.123±0.818、knowledge 5.010±0.151，**一致**。
- 超过初始最佳（final cumulative_best vs initial best 4.1880/4.9530/4.5224）：random 1/3（种子33）、model 1/3（33）、llm 1/3（11）、knowledge 2/3（11、33），**一致**。
- README 汇总值：全部推荐均值 0.174/0.386/2.084/1.515、高于 WT(>1) 计数 4/90、8/90、48/90、40/90，**一致**。
- 预测模型选择表：种子 11/22/33 选 RF（叶 1/1/3），val Spearman 0.5441/0.5234/0.5465、val MSE 0.0929/0.0950/0.1299 —— 从各 `selection/model_comparison.csv` 复算**一致**，且三个种子的 RF 行确为验证 Spearman 最优。
- 用量：31 次请求、297,662 tokens（逐条解析 usage JSON 求和）、连接检查 49 tokens（api_check.json）、model_returned=gpt-6-luna，**全部一致**。

### 2. 零实验数据排序（修订版"修正方向"声明）

从 `data/raw/elife-16965-supp1-v4.xlsx` 重新解析 149,361 条唯一变体，对三个 rank_batch 文件独立计算：

| 批次 | 原始 spearman(position, fitness) | 转高分方向 | Top-10 交集 |
| --- | ---: | ---: | ---: |
| 1 | −0.2665 | +0.2665 | 3 |
| 2 | −0.2688 | +0.2688 | 4 |
| 3 | −0.1819 | +0.1819 | 2 |

与修订版表逐位一致；**方向修正正确**（负的位置相关=高分排前）。自由列举 10 个变体中 2 个入全局前十（WWLG、FWLG）、7 个入前一百，**一致**（全局前十名单已独立重建核对）。

### 3. 反馈消融与自适应/冻结

- `results/feedback-ablation/paired_results.csv`：full−withheld = +0.14054/+0.83342/+0.50324，均值 +0.4924，**与 README/修订版一致**。
- `results/critique-followup-20261113/no_feedback/summary.json`：三状态差 +0.1847/+0.1206/−0.3290，均值 −0.0079 ≈ **−0.008**，一致。
- README"本阶段 23+11=34 次、344,597 tokens"：v3 补种子 22/33 = 23 次、229,796 tokens（逐条解析 usage）+ 消融 11 次、114,801 tokens（summary.json arms 求和）= 34 次、344,597 tokens，**一致**。

### 4. 运行数量（v1/v3/合并）

- v1：360 条、274 唯一变体、36 方法轮次 ✓；v3：180 条、143 唯一变体、18 方法轮次、33 次、329,745 tokens ✓；合并 540/381/54/64/627,407 ✓（全部从原始文件复算）。
- "首答 36 方法轮次、1,408 候选条目"：仅统计 attempt=1 的 llm/knowledge 首答，v1 688 + v3 720 = **1,408**，分布 36 个方法轮次 ✓（含纠正重试的全量口径为 1,607，报告选用的"首答"口径定义明确、数字正确）。

### 5. factorial_summary.csv（全部 17 行 × 7 数值列）

- G1 从 v1 llm 的 round_metrics/usage 重算、G2 从 `G2/seed-*/run/*` 重算：final_cumulative_best、mean_round_fitness、new_high、api_calls、total_tokens **全部一致**。
- G3/G4 从 `agent_log.jsonl`（api_call 计数 10/13/12 与 10/13/11、tokens 逐条求和）与 `state.json`（rounds 指标求和 new_high 30/29/30、29/30/30；累计最优=max actual_fitness）重算：**全部一致**。
- test 指标：G1 从 `llm-test-predictions.csv` 重算 Spearman 0.53182/0.52601/0.54544 ✓；G2 重算 0.48634/0.46874/0.43140 ✓；G3/G4 见问题 D。
- 四格均值/SD（如 G1 mrf 2.083973±1.794927）与描述性主效应（tool_calling +2.286783、esm2 −0.859805、交互 +1.060529）用未舍入值复算**一致**。

### 6. statistics.json 抽验（复算而非对照）

- part1_1 全部 36 个单元的位置 Spearman：逐方法 mean_raw（+0.151364/−0.453988/−0.276768/−0.061279）、换向后均值、raw 负数单元数（3/8/8/4）、总计 23、overall 0.160168 —— **全部一致**。
- Friedman 四个指标 chi2/p（11.561798/0.00904542；11.582278/0.00896009；3.139535/0.37060795；19.071429/0.00026426）与 9 区块描述统计 —— **全部一致**；Nemenyi 见问题 C（已复现）。
- Wilcoxon：mean_fitness 全 6 对、max_fitness llm−random、cumulative_best model−random 的 W/p/d_z **逐位一致**（含精确模式合法性检查——这些指标的 |diffs| 无并列）；new_high_count 三对见问题 A。
- BH：对发布的 24 个 p_raw 重算 BH24，最大绝对差 4.5×10⁻⁹ ✓；Friedman 跨 4 指标 BH（0.01206056/0.01206056/0.00105703/0.37060795）✓。
- Bootstrap CI：同种子(20261114)、B=10,000、区块重抽样独立重跑 knowledge−random（mean_fitness）得 [0.4488, 2.3196] vs 发布 [0.4557, 2.3153]，蒙特卡洛误差内一致 ✓。
- 方向检查 `48/48 正`：从 `predictor_v2/summary.json` per_seed_records（16 配置×3 种子=48 行）复算 test_spearman 全正 ✓；rf onehot val 0.539155±0.008497 ✓。
- part3（G3/G4 自主性）：抽查 G3 种子 11 —— 轮状态 {submitted:1, auto_submitted:2}、工具序列首项、api_call=10、auto=true 强制提交位于 r2-010/r3-010，与文件记载**一致**（Levenshtein/熵等派生量未逐项重算，原始日志存在且抽查吻合）。

### 7. README 其余数字段

- v3 三种子均值：基础 4.060±0.288、知识 3.586±0.452（v1 对应 2.084±1.795、1.515±1.018）✓。
- 有效候选范围：知识 10–40、基础 39–40 ✓。
- 校准：确认阶段 15/15（两臂）、所选 4.757[4.498,5.067] vs 参考 3.858[3.540,4.136]（README 的 4.7567/3.8582 为未舍入值）✓；应报计数）。
- 扩展：108 轮、1,080 条推荐（seeds 44/55/66）✓；mean_plus_std 3.4787 / greedy 3.4194（90 条池化均值逐位复算）✓；图谱 6 条件、60 条推荐、11 次请求、142,187 tokens ✓；"图谱组合均值低于静态规则"在 3 个种子方向全部成立 ✓。

### 8. 事实性声明

- GP/CUDA：修订版称"GP 需要 CUDA 不正确"——**核实为真**：本机（无 GPU 配置变更）用项目 venv 的 sklearn GaussianProcessRegressor 纯 CPU 拟合成功。原错误声明的出处（bonus 文档现行版本）已不含该句，修订版的更正记录无需再改。
- 数据源 149,361 条唯一变体、参考序列第二位 Q、位点 39/40/41/54（VDGV）：原始 xlsx 复核 ✓。
- `gpt-6-luna`（model_returned）✓；`49 tokens` 连接检查 ✓。

## 总体结论

- **数值层面**：五份审计对象中可从原始数据复算的数字，除问题 A 所列三个 p 值外**全部复现**（抽验比例实际接近全量）。报告文本（修订版、旧版、README）没有发现任何无法复现或方向错误的数字。
- **解释层面**：修订版的谨慎措辞与数据支持度匹配；README 的警示框正确引导读者。
- **需要行动的仅一处**：statistics.json 的三个 new_high_count p_raw（问题 A）应按其预注册规则改为并列校正值，并同步更新 `n_significant_after_BH24`（4 → 0）；该修正不改变任何报告文本结论，但会使"经 BH 校正后 4 项显著"这一中间统计结论失效（族内 BH6 口径下三项仍显著，需明确所报口径）。
- 次要建议：问题 B 的配对命名、问题 C 的分布约定注释、问题 D 的 G3/G4 预测持久化。
