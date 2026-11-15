# GB1 Scientific Agent — 整体架构设计
> 2026-11-14 · 整合现有全部模块

---

## 一、系统总览（五层架构）

```
┌──────────────────────────────────────────────────────────────────┐
│                    ⑤ 评 估 层 (Evaluation)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Friedman  │  │ Wilcoxon │  │ 因子分解  │  │ 自主性指标     │  │
│  │ +Nemenyi  │  │ +BH FDR  │  │ 2×2 设计 │  │ tool日志分析   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       └─────────────┴──────┬──────┴────────────────┘          │
│                     statistics_computed.csv                     │
├──────────────────────────────────────────────────────────────────┤
│                  ④ Agent 层 (Tool-Calling)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Agent Loop: while not (submit or budget_exhausted)     │   │
│  │    LLM 分析状态 → 自主选择 Tool → 执行 → 看结果 → 继续   │   │
│  │                                                          │   │
│  │  ┌────────────┐ ┌───────────┐ ┌────────────────────┐   │   │
│  │  │predict_    │ │query_     │ │get_experiment_     │   │   │
│  │  │fitness()   │ │knowledge()│ │history()           │   │   │
│  │  └─────┬──────┘ └─────┬─────┘ └─────────┬──────────┘   │   │
│  │        │              │                 │               │   │
│  │  ┌─────┴──────┐ ┌────┴─────┐ ┌────────┴──────────┐    │   │
│  │  │submit_     │ │check_    │ │ [可扩展 Tool]      │    │   │
│  │  │recommen-   │ │budget()  │ │ switch_predictor  │    │   │
│  │  │dations()   │ │          │ │ propose_experiment│    │   │
│  │  └────────────┘ └──────────┘ └───────────────────┘    │   │
│  │                                                          │   │
│  │  预算: 15调用/轮 · 50K tokens/轮 · 40评分/轮              │   │
│  │  日志: agent_log.jsonl (每次API调用+工具执行)             │   │
│  └──────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                 ③ 配 置 层 (Auto-Configure)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  configure_protein(data.csv)                            │   │
│  │  ├── 自动检测变体格式 (全长序列 / 突变串)                  │   │
│  │  ├── 自动检测突变位点 (比较差异位置)                       │   │
│  │  ├── 自动推断野生型 (各位置最常见残基)                     │   │
│  │  ├── 自动训练 + 比较预测器 (RF / Ridge / MLP / ESM-2)   │   │
│  │  ├── [缓存] 同数据+配置 → 直接加载模型                    │   │
│  │  └── 输出 ProteinConfig 对象                             │   │
│  │                                                          │   │
│  │  ProteinConfig:                                          │   │
│  │    reference_sequence: str     ← 参考序列                │   │
│  │    mutable_positions: [int]    ← 可突变位点              │   │
│  │    predictor: model            ← 选好的预测器            │   │
│  │    predictor_info: dict        ← 选型元数据              │   │
│  │    encode_variant() -> vec     ← 特征编码               │   │
│  │    predict_fitness() -> list   ← 预测接口               │   │
│  └──────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                  ② 预测 层 (Predictor Pool)                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐  │
│  │RF +    │ │Ridge + │ │MLP +   │ │ESM-2   │ │ESM-2 8M    │  │
│  │one-hot │ │one-hot │ │one-hot │ │L33 emb │ │L6 emb      │  │
│  │(80d)   │ │(80d)   │ │(80d)   │ │(1280d) │ │(320d)      │  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────────┘  │
│       ✓0.588    △0.52     △0.55     △0.537     △0.535        │
│       (val Spearman @ 50k training)                             │
│                                                                 │
│  选择规则: 任务结构驱动                                          │
│  → 位点身份+上位性主导 → RF one-hot                            │
│  → 突变分散+上下文重要 → ESM-2 embedding                       │
│  → 样本少(<5k) → 简单模型                                     │
│  → 样本多(>50k) → 神经网络收益增大                              │
├──────────────────────────────────────────────────────────────────┤
│                  ① 数 据 层 (Data & Cache)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  data/                                                   │   │
│  │  ├── processed/all_variants.csv     ← 原始数据           │   │
│  │  ├── processed/seed-{11,22,33}/     ← 划分               │   │
│  │  ├── esm2_embeddings/               ← PLM 嵌入缓存       │   │
│  │  │   ├── esm2_pooled_L20.npy        (149k, 1280)        │   │
│  │  │   ├── esm2_pooled_L33.npy        (149k, 1280)        │   │
│  │  │   ├── esm2_fullseq_L33.npy       (149k, 1280)        │   │
│  │  │   ├── esm2_8m_mutpos.npy         (149k, 320)         │   │
│  │  │   ├── esm2_per_residue_L33.npy   (149k, 4, 1280)     │   │
│  │  │   └── esm2_pairwise_dot.npy      (149k, 6)           │   │
│  │  └── cache/                                                │   │
│  │      ├── models/{protein}/{version}/  ← 缓存模型          │   │
│  │      └── configs/                     ← 缓存配置          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## 二、执行流程

```
用户输入: data.csv (variant + fitness)
    │
    ▼
[③ 配置层] configure_protein("data.csv")
    │  自动: 格式→位点→参考→编码→训练→选模
    │  缓存: 同数据+配置 → 跳过训练
    ▼
ProteinConfig 对象
    │
    ▼
[④ Agent层] run_agent_round(config, system_prompt, budget)
    │  Agent 循环:
    │    1. LLM 读取 get_experiment_history()
    │    2. LLM 可选: query_knowledge("疏水残基有哪些")
    │    3. LLM 生成候选 → predict_fitness(sequences)
    │    4. LLM 看到评分 → [可改变策略] → 重新生成
    │    5. LLM 满意 → submit_recommendations(seqs, reasons)
    │    6. 系统回填真实 fitness → 更新 history
    │    7. 下一轮 (或 LLM 决定停止)
    │  全程: agent_log.jsonl 记录
    ▼
推荐结果 + 日志
    │
    ▼
[⑤ 评估层] 统计计算 + 审计
    │  Friedman + Wilcoxon + 因子分解 + 自主性指标
    ▼
最终报告
```

## 三、换蛋白的完整操作

```python
# 蛋白 A (GB1)
config_gb1 = configure_protein("gb1_data.csv")
agent_gb1 = ProteinEvolutionAgent(config_gb1)
result_gb1 = agent_gb1.run(rounds=3, budget=40)

# 蛋白 B (GFP) — 只需换 CSV
config_gfp = configure_protein("gfp_data.csv")  # 自动适配
agent_gfp = ProteinEvolutionAgent(config_gfp)
result_gfp = agent_gfp.run(rounds=3, budget=40)

# 两个实验的日志、结果、模型完全独立
# Agent 代码零修改
```

## 四、预测器选择决策树（从消融数据提炼）

```
任务特征分析
    │
    ├── 突变位点 < 10 且集中于局部区域?
    │   ├── 是 → RF + one-hot (0.588)
    │   │        理由: 身份特征+交互主导, 树模型天然匹配
    │   └── 否 → 上下文信息重要
    │       ├── 训练样本 < 5k?
    │       │   ├── 是 → Ridge + ESM-2 8M嵌入 (简单模型防过拟合)
    │       │   └── 否 → MLP + ESM-2 650M嵌入 (0.546, 还在提升)
    │       │            模型规模: 8M ≈ 650M → 用 8M (更小更快)
    │       └── 需要交互信号?
    │           └── pairdot_only (6d) 作为补充特征
    │
    └── 多目标?
        └── [未实现] 多目标优化框架
```

## 五、文件结构（最终版）

```
gb1-evolution/
├── gb1/                          # 核心包
│   ├── protein_config.py         # ③ 自动配置层
│   ├── tools.py                  # ④ Agent 工具定义
│   ├── agent_v2.py               # ④ Tool-Calling Agent
│   ├── data.py                   # ① 数据处理
│   ├── models.py                 # ② 预测器实现
│   └── ...
├── configs/
│   ├── experiment.json           # v1 实验配置
│   └── experiment_v2.json        # v2 实验配置 (ESM-2 + Tool)
├── data/
│   ├── esm2_embeddings/          # ① PLM 嵌入缓存
│   └── cache/                    # 模型+配置缓存 (待实现)
├── results/
│   ├── predictor_v2/             # ② 预测器消融结果
│   │   ├── full_ablation_runs.csv       (243组)
│   │   └── sample_size_ablation.csv     (样本量消融)
│   ├── v2_experiments/           # ④ 因子实验结果
│   │   ├── factorial_summary.csv        (G1-G4)
│   │   └── G{2,3,4}/seed-{s}/           (完整日志)
│   ├── statistics_computed/      # ⑤ 统计计算
│   └── statistics_audit/         # ⑤ 统计审计
├── prompts/
│   └── scientist_v2.txt          # Agent 系统提示
├── docs/
│   ├── statistics_plan.md        # 预注册统计方案
│   └── agent_v2_architecture.md  # Agent 架构文档
├── archive/v1/                   # 旧架构归档
├── embed_gb1.py                  # ESM-2 嵌入提取
├── run_predictor_v2.py           # 预测器对比
├── run_factorial.py              # 因子实验
├── run_sample_ablation.py        # 样本量消融
├── AGENTS.md                     # 项目总控文档
└── README.md
```
