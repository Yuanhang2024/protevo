# 发布前审查报告 — 数据集钉死（代码 / Prompt / 路径与目录名）

> 审查日期：2026-11-15 · 范围：全部代码、Prompt 文本、硬编码路径与目录命名
> 结论先行：**活路径（通用框架层）零数据集钉死**；残留钉死全部位于按设计冻结的 GB1 实验设施层（可复现性要求），本报告逐条在案。

---

## 一、代码字符串审查（活路径模块）

扫描模式：`GB1 | VDGV | IgG | 39,40 | Sarkisyan | Roscoe | BLAT | GFP`（非注释行）。

| 模块 | 命中 | 判定 |
|---|---|---|
| `variant_notation.py` | **0** | ✅ 干净 |
| `prompt_builder.py` | **0** | ✅ 干净（系统提示全部由 ProteinConfig 派生） |
| `embedding_service.py` | **0** | ✅ 干净 |
| `protein_config.py` | 2 | ✅ 均为文档示例/提示文案（docstring 用法示例 `reference_sequence="VDGV"`、均匀库警告"如 GB1: 'VDGV'"），非逻辑钉死 |
| `tools.py` | 4 | ✅ L85/L91/L348 为 `config=None` 旧路径默认文本（逐字节向后兼容，测试钉住）；L2 模块 docstring 已改为双层定位表述 |
| `agent_v2.py` | 6 | ✅ 全部为 `DEFAULT_SYSTEM_PROMPT` 与 `default_task_message` 的 None 分支默认值（GB1 旧路径，`tests/test_agent_v2_universal.py` 验证注入路径零 GB1、旧路径逐字节恒等） |

**验证测试**：`tests/test_agent_v2_universal.py`（mock 端到端：LLM 收到的 tools/系统提示/开场白均无 GB1）、`tests/test_tools_config_driven.py`、`tests/test_cross_protein_agent_wiring.py`、`tests/test_indel_agent_wiring.py`。

## 二、Prompt 审查

| 文件 | 判定 |
|---|---|
| `prompts/scientist_v2.txt` / `scientific-v3.txt` / `legacy-v1.txt` / `v2_design_notes.md` | **按设计冻结**——已发布实验（v1/v2/v3、2×2 因子）的提示记录，改写即伪造实验条件 |
| `protevo/prompts.py`（v1 prompt profiles） | 同上，v1 实验设施 |
| `protevo/prompt_builder.py` | 通用层，0 钉死，由配置参数化生成 |

## 三、路径与目录名审查

| 位置 | 判定 / 处置 |
|---|---|
| `run_cross_protein.py` DATASETS 注册表（gfp/ubiq/blat） | ⚠️ 曾为活路径钉死 → **已修**：新增 `--csv <dms.csv> [--name X]` 任意数据集直通模式；注册表降级为三个内置示例的便捷项 |
| `data/external/{gfp_sarkisyan, ubiq_roscoe_2014, blat_ecolx_gonzalez_2019}` | ✅ 数据目录名即数据内容（诚实命名），非代码钉死；meta.json 记录出处 |
| `results/` 历史目录（`hidden/` `selection/` `data/` `round-*/` `mean_plus_std/` `std_only/` `run-11|22|33/` 等） | v1/v2 实验原始输出（GB1 特定），**不入发布包**；保留在工作区 |
| `results/predictor_v2/best_model.joblib`（47MB）等二进制工件 | 不入包（消融 CSV 随包，模型可由脚本复训） |
| 根目录 `run_*.py`、`greedy_baseline.py`、`embed_gb1.py` 内部路径（seed-11/22/33、all_variants.csv、esm2_embeddings） | **按设计冻结**——实验脚本即实验记录的一部分 |
| `tests/` fixture 路径（ubiq/blat external、`metric-correction-20261114/`、`tools/calibration-report/*.py`） | ✅ 示例数据/测试依赖随包分发；集成测试对 shipped 示例数据运行属正当引用 |
| 包名 `gb1` → `protevo` | ✅ **已完成改名**（v0.2.0）：包目录/全部 import/`python -m` 调用/包根目录/文档路径；保留项为数据格式标识符（`gb1-proposals-*`、`gb1-research-session/1`、记录内 `source/` 快照命名等）与冻结层代码内容中的 GB1 事实。`embed_gb1.py` 文件名保留——GB1 实验脚本属冻结层 |
| `.env`（API 密钥） | **绝不入包**（打包脚本只复制具名条目） |

## 四、冻结层钉死清单（按设计保留，计数）

`schema.py`(5) `data.py`(6) `agent.py`(4) `prompts.py`(1) `research_session.py`(8) `cli.py`(1) `statistical_definitions.py`(1)；`experiment.py` `storage.py` `aggregate.py` `models.py` `feedback_ablation.py` `compare_prompts.py` 均为 0。

定位声明：这些模块构成 **GB1 实验设施层**——已发布实验（243 组消融、2×2 因子、Greedy 基线、双盲统计）的可复现性依赖其原样存在。`protevo/__init__.py`（v0.2.0）docstring 已明确双层定位：通用框架层 + GB1 实验设施层。

## 五、发布包排除清单

`.venv/` · `__pycache__/` · `archive/v1/` · 旧交互插件的源码目录（不随本包发布）· `data/esm2_embeddings/`（12.7GB）· `tools/` 的 test-output 嵌套快照与 quick_start 深树（仅保留测试依赖的子目录与顶层脚本）· `results/` 大体积原始输出（hidden/selection/data/round-*/mean_plus_std 等，仅保留具名小文件）· `HANDOFF.md` `AGENTS.md`（内部工作区文档）· `.env` 及全部点文件 · **插件域测试**（`tests/test_plugin_worker.py` + `tests/test_configuration_upgrade.py`，共 98 项——测的是 旧插件目录 的 worker 协议，不随本包发布）

包含：`data/raw/`（1PGA 参考结构等 GB1 原始数据，研究诊断测试依赖）。

## 六、复验方式与实测结果

```bash
cd <发布包根>
python -m pytest tests -q
python run_cross_protein.py --mode notation --csv data/external/gfp_sarkisyan/dms.csv --name gfp
```

**实测（2026-11-15，86.1 MB 包）**：`397 passed + 5 skipped, 0 failed`。
对账：工作区全套件 500 项（496 过 + 4 GPU 跳）− 历史插件域测试 98 项（不随本包发布）= 包内 402 项，全部通过。


## 七、protevo 改名验证（2026-11-15）

- import 残留扫描：`from gb1|import gb1|"gb1.` 在全部 .py 中 **0 命中**
- 保留的小写 `gb1`：仅数据格式标识符（prompt 版本串、research-session 格式串、导出记录 source/ 快照命名）与 `embed_gb1.py` 等冻结层实验文件名
- 包根目录 `release/protevo`（原 gb1-evolution）；包内全套件重跑 **397 passed + 5 skipped, 0 failed**

- 2026-11-15 晚间追加功能（硬件自适应 device=auto、setup_assistant 幂等+doctor、evolve 一行入口）后包内实测：426 passed + 11 skipped（10 GPU 门控、1 缺公开输入），0 failed

## 八、esm2_mlp 入池复测附记（2026-11-15）

- **候选池扩充**：use_esm 路径（full 格式与 indel 序列级两处）新增 esm2_mlp——`MLPRegressor(hidden_layer_sizes=(256,64), alpha=1e-4, early_stopping=True, random_state=42, max_iter=500)`，与 243 组预测器消融总榜第二的 ESM+MLP 同规格；与 esm2_ridge 共享同一嵌入矩阵（嵌入只算一次，GPU/嵌入不可用时两候选一并优雅跳过）。one-hot×MLP 刻意不入池：消融中 one-hot 特征上 MLP 三连败（2K 0.344 / 10K 0.309 / 50K 0.556，vs RF 0.543/0.589/0.655），依据注明于 `protein_config.py` 候选池定义处。
- **嵌入磁盘缓存**：`run_cross_protein.py --embed-cache`（默认开，`results/cross_protein/embed_cache/<name>_<mode>`），经 `configure_protein` 新增的 `esm_cache_dir` 参数透传 `EmbeddingService(cache_dir=...)`——侵入最小的实现路径；缓存命中后嵌入零推理（ubiq 复测端到端 25.8s，余为 checkpoint 加载与 sklearn 训练）。
- **三数据集复测**（openfold GPU）：esm2_ridge 三处逐位复现旧值（GFP 0.6275 / 泛素 0.4826 / BLAT 0.6568）；GFP 与泛素选型不变（ridge 0.8652 / random_forest 0.7849，esm2_mlp 分别 0.7765 / 0.6357 仍不敌 one-hot 系）；**BLAT indel 选型由 esm2_ridge 变为 esm2_mlp 0.7552**（ESM 家族内部头升级，README"四次全对"结论不变——协议仍选中可用池中最优）。`results/cross_protein/summary.json` 已按同名同 mode 覆盖更新。
- 包内全套件重跑：426 passed + 12 skipped（新增 1 项 GPU 门控测试 `test_esm_pool_contains_ridge_and_mlp`，零回归）。
