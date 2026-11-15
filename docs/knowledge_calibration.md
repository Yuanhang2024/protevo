# 模型选型校准知识（外部文献核实版）

供 `protevo/setup_assistant.py` 的 LLM 系统提示词与人工查阅使用。知识块原文注入
`configure_with_llm` 的 system prompt（常量 `MODEL_SELECTION_KNOWLEDGE`），置信级别与
限定词不得删改。**用途限定：知识只用于解释与菜单内选择建议，与硬件/数据探测事实矛盾时
以守卫校验为准**——菜单与守卫逻辑不因知识块改变。

判定级别：**支持**（[文献强支持]）/ **部分支持**（[文献部分支持]）/ **仅内证**（[仅内证]）。

---

## 条目与判定

### 1. 位点特异小库 vs 全蛋白 DMS 的编码-模型选择 —— 判定：支持 [文献强支持]

> 位点特异组合库、样本≤5万、插值任务：one-hot+线性/树模型可与冻结PLM嵌入相当甚至更强（FLIP/PLMFit 在GB1/AAV证实）；但全蛋白多位点、高N的DMS设定下PLM嵌入监督明显更强，勿外推。

来源：
- FLIP：<https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1>
- PLMFit：<https://academic.oup.com/bib/article/26/4/bbaf381/8219059>
- Bryant/Sinai AAV：<https://www.nature.com/articles/s41587-020-00793-4>

### 2. 冻结嵌入+浅层头是默认起点；微调"非必需但常有增益" —— 判定：部分支持 [文献部分支持]

> 冻结PLM嵌入+浅层头是标准低成本做法、可作默认起点；系统对比显示任务微调（全参/LoRA）几乎总能再提升，"微调非必需"应读作"非必需但常有增益"。

来源：
- PLMFit：<https://academic.oup.com/bib/article/26/4/bbaf381/8219059>
- Çelik 2025：<https://www.biorxiv.org/content/10.1101/2024.10.22.619563v1>

### 3. 浅层 MLP 头优于线性头 —— 判定：仅内证 [仅内证]

> "浅层MLP头优于线性头"是本项目观察；公开基准同测两类头但无一致赢家，应作为待调超参而非默认假设。

来源：本项目内部证据（无专属外部文献）；公开基准同测两类头但无一致赢家，可对照：
- FLIP：<https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1>
- PLMFit：<https://academic.oup.com/bib/article/26/4/bbaf381/8219059>

### 4. 监督小数据下中小 PLM 够用；零样本打分规模收益大 —— 判定：支持 [文献强支持]

> 监督小数据下中小PLM（150M–650M）可与大模型相当甚至反超；零样本打分则不然：ESM-2 8M→650M规模收益大（Spearman 0.23 vs 0.42），两设定不可混用。

来源：
- ProteinGym：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/>
- Schmirler 2024：<https://www.nature.com/articles/s41467-024-51844-2>

### 5. 池化方式应显式消融（mean vs mut-mean） —— 判定：支持 [文献强支持]

> 嵌入提取应显式消融：全序列mean-pooling与突变区域池化（mut-mean）在FLIP中并列使用、各有适用，无普适赢家。

来源：
- FLIP：<https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1>

### 6. sklearn 树模型仅 CPU —— 判定：支持 [文献强支持]

> sklearn随机森林/树模型仅CPU（Array API GPU名单不含树模型）；GPU树需RAPIDS cuML或XGBoost/LightGBM另建依赖栈。

来源：
- sklearn Array API：<https://scikit-learn.org/stable/modules/array_api.html>
- cuML：<https://developer.nvidia.com/cuml>

### 7. ESM-2 长度上限 1022 残基 —— 判定：支持 [文献强支持]

> ESM-2位置嵌入表1024 token（含<cls>/<eos>），氨基酸上限1022；esm-extract默认1022截断，超长需自行分块且超出预训练范围。

来源：
- fair-esm：<https://github.com/facebookresearch/esm>

### 8. indel 变长序列的表征兼容性 —— 判定：支持 [文献强支持]

> PLM嵌入与自回归打分天然兼容indel变长序列，固定位置one-hot/BLOSUM62/MSA不能；但ESM masked-marginal零样本打分同样不适用于indel，且indel整体仍难于替换（最佳零样本Spearman~0.47）。

来源：
- ProteinGym：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/>
- fair-esm：<https://github.com/facebookresearch/esm>

---

## 与 setup_assistant 的关系

- 注入位置：`protevo/setup_assistant.py` 常量 `MODEL_SELECTION_KNOWLEDGE`，经
  `_SYSTEM_PROMPT_TEMPLATE` 的 `{knowledge}` 段进入 system prompt。
- 科学选型（RF/Ridge/MLP/ESM 谁好）仍由 `configure_protein` 的种子化验证 Spearman 协议
  决定；LLM 与本知识块均只配环境与算法可用性，不替数据选模型。
