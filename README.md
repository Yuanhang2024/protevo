# protevo — Universal Protein Directed-Evolution Agent Framework

**English version: [README_EN.md](README_EN.md)**

以 LLM Tool-Calling 智能体为核心、蛋白无关的定向进化框架：换一个 CSV 即换一个蛋白。
GB1（蛋白 G B1 结构域，IgG-Fc 结合）是本框架的第一个完整实证案例（243 组预测器消融、
2×2 因子实验、双盲统计）；另在 ProteinGym 的 GFP / 泛素 / TEM-1（indel）三个真实数据集上
以零代码改动完成跨蛋白复测。

## 一键启动

```bash
# Windows：双击 run.bat，或命令行运行
run.bat

# Linux / macOS
sh run.sh
```

脚本自动完成：检测 Python 3.10+ → 创建 .venv → 安装依赖（首次数分钟，后续跳过）→ 进入命令行交互台。
环境变量 `PROTEVO_PY` 可指定解释器路径。TUI 为纯标准库实现，三平台同一入口；Windows 已实测，Linux/macOS 依赖全部跨平台、按同流程运行。

## 快速上手

```python
from protevo.protein_config import configure_protein
from protevo.agent_v2 import run_agent_round

config = configure_protein("data/external/ubiq_roscoe_2014/dms.csv",
                           reference_sequence=<WT>)   # 自动检测格式/位点/方向并选型预测器
config.predict_fitness(["I3V", "K6R"])                # 原格式进、原标度出
```

```bash
# 任意数据集直通（--csv 模式，无需改代码）
python run_cross_protein.py --mode notation --csv data/external/gfp_sarkisyan/dms.csv --name gfp

# 不确定本机硬件/环境？让 LLM 辅助配置（无 LLM 凭据时输出确定性规则计划）
python -m protevo.setup_assistant data/external/blat_ecolx_gonzalez_2019/dms.csv --ask "这台机器该怎么配？"
```

### 一行命令跑完整多轮虚拟定向进化

```bash
python -m protevo.evolve data/external/ubiq_roscoe_2014/dms.csv --rounds 2 --reference <WT>
# 结果落 results/evolve/<数据集名>_<时间戳>/：per_round.csv / summary.json / agent_round*.jsonl
```

虚拟进化 = 用 CSV 全量 fitness 做回填 oracle：每轮提交的变体按原表精确匹配回填"实测"分，
无需湿实验即可复盘完整多轮闭环。真湿实验场景请逐轮调用 `run_agent_round` 在轮间手动喂养实测数据。

预测器协议两模式（summary.json 顶层 `predictor_protocol`）：默认 `fixed`——固定初始训练子集版本、轮间不重训；`run_evolution(..., retrain_each_round=True)` 为 `refit_per_round`——每轮开始前以累计 known（初始训练 + 此前各轮回填）重训选型（各轮只用初始训练和本方法此前获得的反馈拟合预测器），每轮 `predictor_info` 记入 rounds 明细。

## 命令行交互（TUI）

`python -m protevo.tui` 一条命令启动命令式交互台（stdlib `cmd.Cmd`，零新交互依赖；少量
ANSI 颜色，非终端自动关闭）。**覆盖**（数据与配置 / 实验 /
可视化 / 回放与导出四面板 → 四组命令），可随时 `setup` 重跑引导或 `--no-wizard` 跳过。

首次启动自动检测包根 `.env`：凭据不全且在真实终端时进入三步引导——① Base URL（OpenAI
官方 / DeepSeek 预填 / 自定义兼容端点三选）② API Key（getpass 静默输入）③ 模型名（按端点
建议默认值）——汇总确认后写入 `.env`（只追加/更新这三个键，其余行不动），并提示可用
`doctor` 的 llm_credentials 项复核。任一步可 skip：**CPU 预测器路径无需凭据**（datasets /
open / predict / plot / export 全离线），仅 run（Agent 进化）与 ask（LLM 顾问）需要。
stdin 非 tty（管道/测试）时不阻塞：打印一行凭据说明直接进 REPL。

| 命令 | 说明 |
|---|---|
| `datasets` | 列出 4 个内置示例数据集（gb1 / ubiq / gfp / blat） |
| `open <csv\|内置名> [--ref WT] [--esm] [--pool recommended\|full] [--hrb higher\|lower] [--lang zh\|en]` | 配置数据集并打印配置卡（格式/位点/参考来源含不可靠警示/方向/选型与 val ρ/全部候选） |
| `info` / `status` | 当前配置卡 / 会话总览（数据·预测器·凭据·最近运行） |
| `run [--rounds 3] [--per-round 10] [--budget 40] [--retrain]` | 多轮虚拟进化；每轮一行准实时摘要（轮次/提交/回填均值/累计最佳/预算余量），结束打轨迹小表 |
| `model [名称]` | 查看/设置本会话 LLM 模型名（持久化用 setup） |
| `lang [zh\|en]` | 查看/切换 Agent 提示语言（`open <csv> --lang en` 切换 Agent 提示语言；仅影响发往 LLM 的系统提示，界面文本保持中文） |
| `plot <scatter\|trajectory\|heatmap\|length\|budget>` | matplotlib 出 PNG 至输出目录 + 终端一行 ASCII 摘要（缺 matplotlib 自动降级仅 ASCII）；heatmap 面向替换库、length 面向 indel 库，用错互给指引 |
| `log [轮次]` | 回放 agent_round*.jsonl：默认逐轮摘要；指定轮逐条 kind/tool/参数（含 submit 变体+理由） |
| `export [--out DIR]` | 导出 per_round.csv / summary.json / 全部适用图并打印清单 |
| `plan` / `doctor [--execute]` | 数据探测+规则计划 / 环境体检（可自动修复） |
| `ask <问题>` | LLM 顾问（无凭据打印降级说明） |
| `predict SEQ...` | 当前预测器打分（原格式进、原始标度出） |
| `setup` / `help` / `quit` | 重跑三步引导 / 命令表 / 退出（EOF 同效） |

Agent 提示语言可选中英文：`open <csv> --lang en` 切换 Agent 提示语言（或 `lang [zh|en]` 随时切换；
仅影响发往 LLM 的系统提示，TUI 界面文本保持中文）。

示例会话（`open → run → plot → export`）：

```
protevo> open ubiq                       # 内置泛素库：自动检测格式/位点/方向并选型
protevo (ubiq|RF ρ=0.78)> run --rounds 2 # 每轮一行摘要 + 轨迹小表（需 .env 凭据）
protevo (ubiq|RF ρ=0.78)> plot trajectory# 轨迹 PNG 落最近输出目录 + ASCII 摘要
protevo (ubiq|RF ρ=0.78)> log 1          # 逐条回放第 1 轮工具调用与提交理由
protevo (ubiq|RF ρ=0.78)> export         # per_round.csv / summary.json / 全部适用图打包
```

## 支持的变体格式（自动检测）

| 格式 | 示例 | 说明 |
|---|---|---|
| `full` | 全长序列 CSV | 可变位点全量扫描检测 |
| `mutation_string` | `VDGV` | GB1 式突变串 |
| `mutation_notation` | `V39D,G40E` / `K3R:V55A` | ProteinGym/MaveDB 记谱（冒号分隔原生兼容） |
| `indel_notation` | `V39del` / `V39_G40insAA` / `V39dup` | HGVS 子集；序列级预测器（composition / ESM-2） |

## 跨蛋白实证（`results/cross_protein/summary.json`）

| 蛋白 | 库结构 | 自动选型 | val Spearman |
|---|---|---|---|
| GB1（Wu 2016，149,361） | 4 位点替换枚举 | one-hot + RF | 0.6554（50k 训练） |
| GFP（Sarkisyan 2016，51,714） | 233 位点替换 | one-hot + Ridge | 0.8652（esm2_mlp 0.7765） |
| 泛素 RL40A（Roscoe 2014，1,380） | 75 位点单替换 | one-hot + RF | 0.7849（esm2_mlp 0.6357） |
| TEM-1 BLAT（Gonzalez 2019，4,751） | 单点 indel | **ESM-2 嵌入 + MLP** | 0.7552（esm2_ridge 0.6568；composition 基线 0.0323） |

选型协议在四种库结构上四次全部选对（候选池已含 243 组消融总榜第二的 esm2_mlp，
与 esm2_ridge 共享同一嵌入矩阵、嵌入只算一次）：替换库上 one-hot 系三连胜——esm2_mlp
在 GFP/泛素仅 0.7765/0.6357，仍不敌 one-hot 系；indel 库上 ESM 翻盘且最优头由 Ridge
升级为 MLP——固定位置编码失效处恰是 PLM 嵌入用武之地。

## 结构

```
protevo/                通用框架层 + GB1 实验设施层（见 protevo/__init__.py 的双层定位）
  protein_config.py      一行配置入口（四格式/方向归一化/选型/ESM 可选）
  variant_notation.py    indel 记谱核心（解析/校验/重构）
  prompt_builder.py      配置驱动的系统提示模板（zh/en）
  tools.py               ToolBox + build_tool_schemas(protein_config)
  agent_v2.py            Tool-Calling 循环（protein_config 注入，GB1 旧路径逐字节兼容）
  embedding_service.py   按需 ESM-2 嵌入（LRU+落盘缓存，惰性 torch，device=auto 硬件自适应）
  setup_assistant.py     LLM 辅助环境配置（确定性探测 + 受约束菜单 + 守卫校验）
  tui.py                 命令行交互台（cmd.Cmd，首次启动引导 + 四面板全功能，替代图形插件）
tests/                   536 项测试（520 过 + 16 跳：GPU 门控与可选数据）；交互界面为内置 TUI，零外部插件依赖
data/external/           三个 ProteinGym 数据集（MIT，出处见各 meta.json）
docs/release_audit.md    发布前审查报告（数据集钉死逐条在案）
results/                 跨蛋白结果 + GB1 消融/因子/统计汇总（大体积原始日志未随包）
```

> 包已于 v0.2.0 由 `gb1` 更名为 `protevo`（本包）。GB1 特定实验设施
> （schema/data/agent/research_session）按可复现性要求冻结原样，`docs/release_audit.md` 有完整清单。
> 交互能力由内置 TUI 提供（`python -m protevo.tui`），零外部插件依赖。

## 环境与测试

```bash
pip install -r requirements.txt      # CPU 全功能
python -m pytest tests -q            # 预期全绿（GPU 门控自动跳过）
```

LLM Agent 与 setup_assistant 的 LLM 顾问层需要 `.env`：`OPENAI_API_KEY` / `OPENAI_MODEL`
（可选 `OPENAI_BASE_URL`，任何 OpenAI 兼容端点均可）。无凭据时 setup_assistant 退化为确定性规则计划。

### 硬件探针（无 torch 也不报错）

```python
from protevo.embedding_service import hardware_probe
print(hardware_probe())
# {'torch': True, 'cuda': True, 'backend': 'CUDA 12.8',
#  'device_name': 'NVIDIA GeForce RTX 5090 Laptop GPU', 'vram_gb': 23.9, 'transformers': True}
```

### ESM-2 嵌入路径（use_esm=True，可选增强）

`configure_protein(..., use_esm=True)` 会在候选池中加入 ESM-2 嵌入预测器 esm2_ridge（Ridge）与
esm2_mlp（MLP，与 243 组消融总榜第二同规格；两候选共享同一嵌入矩阵，嵌入只算一次；可选
`esm_cache_dir` 嵌入落盘复用——`run_cross_protein.py` 的 `--embed-cache` 默认开启）
（torch/transformers/checkpoint 不可用时打印说明并一并优雅跳过，不影响 CPU 路径）。三步就绪：

**1. 按平台安装 torch（另需 transformers）**

```bash
# CPU（无独显，或仅想跑通管线）
pip install torch transformers

# NVIDIA CUDA（cu126 仅为示例——请按显卡驱动版本选择，见 https://pytorch.org）
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers

# AMD ROCm（ROCm 下 PyTorch 将 hip 设备映射为 cuda，设备名仍为 "cuda"，代码零改动）
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers
```

**2. 准备 ESM-2 checkpoint**

自动（推荐）——首次调用下载约 2.5 GB 到默认缓存 `~/.cache/protevo/esm2_t33_650M`，
已就绪则直接复用、不重复下载：

```python
from protevo.embedding_service import ensure_esm_checkpoint
from protevo.protein_config import configure_protein

model_path = ensure_esm_checkpoint()
config = configure_protein("gb1.csv", use_esm=True, esm_model_path=model_path)
```

国内网络较慢时走 hf-mirror 镜像（huggingface_hub 原生尊重 `HF_ENDPOINT` 环境变量）：

```bash
set HF_ENDPOINT=https://hf-mirror.com        # Linux/macOS: export HF_ENDPOINT=https://hf-mirror.com
```

或手动下载后把目录传给 `esm_model_path`（直连与镜像两式）：

```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir D:/models/esm2_t33_650M_UR50D
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir D:/models/esm2_t33_650M_UR50D
```

**3. 开启 ESM 并运行**

```python
config = configure_protein("my_data.csv", use_esm=True)
# esm_device 默认 "auto"：检测到 CUDA（含 ROCm，hip 已映射为 cuda）自动用 GPU，否则回退 CPU；
# 也可显式指定 esm_device="cuda" / "cpu"（显式 "cuda" 在无 GPU 环境会得到明确报错）
```

### LLM 辅助环境配置（setup_assistant）

`python -m protevo.setup_assistant <data.csv> [--reference WT] [--ask 问题] [--execute]`：
探测硬件（torch/CUDA/ROCm/显存/checkpoint）与数据（格式/均匀库护栏/indel），LLM 在**受约束菜单**
内给出配置选择（设备/蛋白模型 650M|8M/torch 安装命令/checkpoint 动作），守卫层逐字段校验——
与硬件或数据事实矛盾的选择（如无 GPU 选 cuda、低显存选 650M、突变串数据开 ESM）自动覆写并记录。
科学选型（RF/Ridge/MLP/ESM 谁好）始终由验证 Spearman 协议决定，LLM 只配环境不选模型。

选型候选池可放开（先验只作推荐不作拦截）：默认 `predictor_pool="recommended"` 为内证与
文献校准的推荐集；`predictor_pool="full"` 放开该格式下全部表示层面可行组合（如 onehot_mlp），
`extra_candidates=["onehot_mlp"]` 可细粒度追加，`run_cross_protein.py --pool full` 同义透传。
表示层面不可行的组合（indel 上的 one-hot、突变串/记谱上的 ESM/composition）仍排除，
原因记于 `predictor_info["pool"]["excluded"]`——这是表示事实，不是先验。

- **已配置短路（幂等）**：`--doctor` 通过或 `--execute` 成功后写入 `~/.cache/protevo/setup_state.json`，之后默认运行直接打印"已配置"摘要并退出（不发 LLM 调用）。
- **`--doctor` 环境体检**：Python/核心依赖/torch 构建与平台匹配/transformers/checkpoint 权重/嵌入与预测器冒烟/LLM 凭据逐项检查并给 fix 命令；配 `--execute` 自动补下载缺失 checkpoint 后复检。
- **指纹自动失效**：状态文件记录硬件指纹（backend/cuda/设备名/显存 ±1GB 容差），换卡或驱动变化后自动视为未配置、重走正常流程（`--force` 可手动强制重新配置）。

## License

MIT（见 LICENSE）。数据集出处与许可见 `data/external/*/meta.json`（ProteinGym v1.3, Zenodo 10.5281/zenodo.15293562）。
