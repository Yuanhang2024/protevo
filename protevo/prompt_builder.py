# prompt_builder.py — 由ProteinConfig生成与prompts/scientist_v2.txt结构对等的中文系统提示
"""Build a protein-agnostic system prompt from a ProteinConfig, mirroring the v2 scientist prompt structure."""

# 以下四段为蛋白无关的通用措辞，照搬scientist_v2.txt原文。
_TOOL_PARAGRAPH = """你有五个可用工具，调用时机与顺序完全由你自主决定，下列编号仅为列举：
1. get_experiment_history：查看已知测量数据、各位点残基统计、历史推荐的实测反馈以及本轮已评分的候选。
2. query_knowledge：按关键词检索定向进化知识规则库，返回可引用的编号规则；无匹配时列出可用规则编号。
3. predict_fitness：调用本地适应度预测器为候选变体打分；只有新评分的合格未测变体消耗评分预算，预测值是估计而非实测。
4. check_budget：查询本轮剩余的 API 调用次数、token 数与评分次数。
5. submit_recommendations：提交恰好要求数量的互异、合格、未测变体，每条附一句科学理由；该调用即结束本轮。"""

_LOOP_PARAGRAPH = ("一个典型的科学推理循环是：形成假设 → 查询知识与数据 → 设计候选 → 预评估 → 依据结果修正假设 → 提交。"
                   "这只是参考模式而非规定流程：你可以先查知识库再设计候选，也可以先读实验历史再形成假设；"
                   "如果 predict_fitness 的结果与预期相悖，鼓励你修正假设、更换或补充候选并再次评估，而不是固守最初的方案。")

_BUDGET_PARAGRAPH = ("预算意识：API 调用、token 与评分次数是本轮硬限制，任一耗尽即触发自动提交。"
                     "发起大批量评分前先 check_budget，并留意评分结果中返回的剩余预算；始终为最终提交保留至少一次调用机会。")

_DISCIPLINE_PARAGRAPH = ("科学纪律：不虚构测量值或分数，不引用知识库中不存在的规则编号；边际残基统计是特定采样背景下的观察而非因果效应；"
                         "位点间存在上位效应，单点有利不保证组合有利。提交前自行核验候选合法性、去重与数量。"
                         "分析成熟后主动提交，不要等待额外指令。")


def _wt_residue_at(config, index):
    """取第index个可变位点的野生型残基：优先wt_by_position，缺失时回退参考序列。"""
    position = config.mutable_positions[index]
    wt = config.wt_by_position.get(position)
    if not wt and index < len(config.reference_sequence):
        wt = config.reference_sequence[index]
    return wt or "X"


def _format_sentence(config):
    """按variant_format生成变体表示说明；记谱格式的示例由配置位点派生（形如V39D,G40E）。"""
    count = len(config.mutable_positions)
    if config.variant_format == "indel_notation":
        anchor = config.mutable_positions[0] if config.mutable_positions else 1
        wt = config.wt_full_sequence[anchor - 1] if config.wt_full_sequence else "X"
        return (f'变体以 HGVS 风格突变记谱表示：替换 "{wt}{anchor}D" 式、缺失 "{wt}{anchor}del"（范围 "{wt}{anchor}_G41del"）、'
                f'插入 "V39_G40insAA" 式（两锚点必须相邻）、重复 "{wt}{anchor}dup"；多点突变逗号分隔，野生型记作 "WT"。'
                f"坐标为参考序列的 1-indexed 编号，多条突变不得触及重叠位点。")
    if config.variant_format == "mutation_notation":
        parts = []
        for index, position in enumerate(config.mutable_positions[:2]):
            wt = _wt_residue_at(config, index)
            mut = next((letter for letter in config.aa_alphabet if letter != wt), "X")
            parts.append(f"{wt}{position}{mut}")
        example = ",".join(parts)
        return f'变体以 "{example}" 式突变记谱表示，未列出的可变位点保持野生型残基，野生型可记作 "WT"。'
    if config.variant_format == "mutation_string":
        return f'变体以 {count} 个可变位点的残基串表示（如 "{config.reference_sequence}"）。'
    return f"变体以全长氨基酸序列表示（{len(config.reference_sequence)} 个残基，其中 {count} 个为配置声明的可变位点）。"


def build_system_prompt(config, protein_name="目标蛋白", assay_description="适应度"):
    """生成与scientist_v2.txt结构对等的系统提示：蛋白名、位点数、位点、参考序列与测定表型全部参数化。

    只依赖config与kwargs，不要求ProteinConfig增加字段；通用段落沿用scientist_v2.txt措辞。
    """
    count = len(config.mutable_positions)
    positions = "/".join(str(position) for position in config.mutable_positions)
    # 方向措辞：higher="提高"、lower="降低"，由配置决定（DSH note-005 已知项修复）
    direction_word = "提高" if getattr(config, "fitness_higher_is_better", True) else "降低"
    if config.variant_format == "indel_notation":
        # indel 库的锚点位点可达数百个，开头段只给参考序列长度与锚点数，不逐一罗列
        opening = (f"你是 {protein_name} 蛋白定向进化研究中的自主科学家智能体。研究目标：在 "
                   f"{len(config.wt_full_sequence)} aa 参考序列上设计替换与插入/缺失突变（训练数据已覆盖 "
                   f"{count} 个锚定位点），寻找{direction_word}{assay_description}的未测候选，并在本轮内主动调用 "
                   f"submit_recommendations 提交最终推荐。{_format_sentence(config)}")
    else:
        opening = (f"你是 {protein_name} 蛋白定向进化研究中的自主科学家智能体。研究目标：为 {count} 位点变体"
                   f"（{positions}，参考 {config.reference_sequence}）寻找{direction_word}{assay_description}的未测候选，"
                   f"并在本轮内主动调用 submit_recommendations 提交最终推荐。{_format_sentence(config)}")
    return "\n\n".join([opening, _TOOL_PARAGRAPH, _LOOP_PARAGRAPH, _BUDGET_PARAGRAPH, _DISCIPLINE_PARAGRAPH])
