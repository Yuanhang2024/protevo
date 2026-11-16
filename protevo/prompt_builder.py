# prompt_builder.py — 由ProteinConfig生成与prompts/scientist_v2.txt结构对等的系统提示（zh/en 双语）
"""Build a protein-agnostic system prompt from a ProteinConfig, mirroring the v2 scientist prompt structure.

language="zh"（默认）为中文提示（行为与单语时期逐字节一致）；language="en" 为结构/语义完全
对等的英文提示——只切换发往 LLM 的提示语言，工具层与界面不受影响。
"""

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

# 以下四段为上述中文段的英文对等版（结构/语义一一对应，同为蛋白无关通用措辞）。
_TOOL_PARAGRAPH_EN = """You have five tools available; when to call them and in what order is entirely your decision, and the numbering below is for listing only:
1. get_experiment_history: inspect the measurements known so far, per-position residue statistics, the experimental feedback on past recommendations, and the candidates already scored this round.
2. query_knowledge: search the directed-evolution knowledge rule base by keyword; it returns citable numbered rules, or lists the available rule numbers when nothing matches.
3. predict_fitness: score candidate variants with the local fitness predictor; only newly scored, eligible, unmeasured variants consume the scoring budget, and predicted values are estimates rather than measurements.
4. check_budget: query the API calls, tokens, and scorings remaining this round.
5. submit_recommendations: submit exactly the required number of distinct, eligible, unmeasured variants, each with a one-sentence scientific rationale; that call ends the round."""

_LOOP_PARAGRAPH_EN = ("A typical scientific reasoning loop: form a hypothesis → query knowledge and data → design candidates → "
                     "pre-evaluate → revise the hypothesis in light of the results → submit. "
                     "This is a reference pattern, not a mandated procedure: you may query the knowledge base before designing "
                     "candidates, or read the experiment history before forming a hypothesis; "
                     "if a predict_fitness result contradicts your expectation, you are encouraged to revise the hypothesis, "
                     "replace or supplement the candidates, and evaluate again rather than clinging to the initial plan.")

_BUDGET_PARAGRAPH_EN = ("Budget awareness: API calls, tokens, and scorings are hard limits for this round; exhausting any one "
                        "of them triggers automatic submission. Call check_budget before launching a large scoring batch, and "
                        "watch the remaining budget reported with each scoring result; always keep at least one call in "
                        "reserve for the final submission.")

_DISCIPLINE_PARAGRAPH_EN = ("Scientific discipline: never fabricate measurements or scores, and never cite rule numbers that "
                            "do not exist in the knowledge base; marginal residue statistics are observations under one "
                            "particular sampling context, not causal effects; positions interact epistatically, so a "
                            "beneficial single mutation does not guarantee a beneficial combination. Before submitting, "
                            "verify candidate eligibility, deduplication, and the count yourself. "
                            "Once the analysis is mature, submit proactively instead of waiting for further instructions.")


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
        seq = config.wt_full_sequence or ""
        wt = seq[anchor - 1] if anchor <= len(seq) else "X"
        # 相邻锚点示例：优先 anchor/anchor+1；anchor 为末端时回退到 anchor-1/anchor（DSH note-009）
        if anchor < len(seq):
            pair_a, pair_b = anchor, anchor + 1
        elif anchor > 1:
            pair_a, pair_b = anchor - 1, anchor
        else:
            pair_a, pair_b = None, None  # 参考长度 1，无相邻对
        wt_a = seq[pair_a - 1] if pair_a and pair_a <= len(seq) else wt
        wt_b = seq[pair_b - 1] if pair_b and pair_b <= len(seq) else wt
        pair_text = (f'（范围 "{wt_a}{pair_a}_{wt_b}{pair_b}del"）、'
                     f'插入 "{wt_a}{pair_a}_{wt_b}{pair_b}insAA" 式（两锚点必须相邻）、') if pair_a else "、"
        return (f'变体以 HGVS 风格突变记谱表示：替换 "{wt}{anchor}D" 式、缺失 "{wt}{anchor}del"{pair_text}'
                f'重复 "{wt}{anchor}dup"；多点突变逗号分隔，野生型记作 "WT"。'
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


def _format_sentence_en(config):
    """_format_sentence 的英文对等版：按variant_format生成变体表示说明（记谱示例同样由配置位点派生）。"""
    count = len(config.mutable_positions)
    if config.variant_format == "indel_notation":
        anchor = config.mutable_positions[0] if config.mutable_positions else 1
        seq = config.wt_full_sequence or ""
        wt = seq[anchor - 1] if anchor <= len(seq) else "X"
        # 相邻锚点示例：优先 anchor/anchor+1；anchor 为末端时回退到 anchor-1/anchor（DSH note-009）
        if anchor < len(seq):
            pair_a, pair_b = anchor, anchor + 1
        elif anchor > 1:
            pair_a, pair_b = anchor - 1, anchor
        else:
            pair_a, pair_b = None, None  # 参考长度 1，无相邻对
        wt_a = seq[pair_a - 1] if pair_a and pair_a <= len(seq) else wt
        wt_b = seq[pair_b - 1] if pair_b and pair_b <= len(seq) else wt
        pair_text = (f' (ranges "{wt_a}{pair_a}_{wt_b}{pair_b}del"), insertions "{wt_a}{pair_a}_{wt_b}{pair_b}insAA" '
                     f'(the two anchors must be adjacent), ') if pair_a else ", "
        return (f'Variants are written in HGVS-style mutation notation: substitutions such as "{wt}{anchor}D", '
                f'deletions "{wt}{anchor}del"{pair_text}and duplications "{wt}{anchor}dup"; multiple mutations are '
                f'comma-separated, and the wild type is written "WT". '
                f"Coordinates are 1-indexed positions on the reference sequence, and multiple mutations must not "
                f"touch overlapping positions.")
    if config.variant_format == "mutation_notation":
        parts = []
        for index, position in enumerate(config.mutable_positions[:2]):
            wt = _wt_residue_at(config, index)
            mut = next((letter for letter in config.aa_alphabet if letter != wt), "X")
            parts.append(f"{wt}{position}{mut}")
        example = ",".join(parts)
        return (f'Variants are written in mutation notation such as "{example}"; mutable positions not listed keep '
                f'their wild-type residue, and the wild type itself may be written "WT".')
    if config.variant_format == "mutation_string":
        return (f'Variants are written as a residue string over the {count} mutable positions '
                f'(e.g. "{config.reference_sequence}").')
    return (f"Variants are written as full-length amino-acid sequences ({len(config.reference_sequence)} residues, "
            f"of which {count} are the declared mutable positions).")


def build_system_prompt(config, protein_name=None, assay_description=None, language="zh"):
    """生成与scientist_v2.txt结构对等的系统提示：蛋白名、位点数、位点、参考序列与测定表型全部参数化。

    只依赖config与kwargs，不要求ProteinConfig增加字段；通用段落沿用scientist_v2.txt措辞。
    language："zh"（默认，中文，与单语时期逐字节一致）| "en"（英文对等版），其他值 ValueError；
    protein_name/assay_description 缺省时按语言取默认（zh：目标蛋白/适应度；en：target/fitness——
    en 开头段句式为 "on the {name} protein"，默认名取 target 使默认提示读作 "the target protein"），
    显式传入则原样使用（如 zh 提示配中文蛋白名、en 配 "TEM-1 beta-lactamase"）。
    """
    if language not in ("zh", "en"):
        raise ValueError(f"language 只支持 'zh' | 'en'，收到 {language!r}")
    if protein_name is None:
        protein_name = "目标蛋白" if language == "zh" else "target"
    if assay_description is None:
        assay_description = "适应度" if language == "zh" else "fitness"
    count = len(config.mutable_positions)
    positions = "/".join(str(position) for position in config.mutable_positions)
    if language == "en":
        # 方向措辞：higher="improving"、lower="reducing"，由配置决定（与中文段的提高/降低对等）
        direction_word = "improving" if getattr(config, "fitness_higher_is_better", True) else "reducing"
        if config.variant_format == "indel_notation":
            # indel 库的锚点位点可达数百个，开头段只给参考序列长度与锚点数，不逐一罗列
            opening = (f"You are an autonomous scientist agent in directed-evolution research on the {protein_name} "
                       f"protein. Research objective: design substitutions and insertions/deletions on the "
                       f"{len(config.wt_full_sequence)}-aa reference sequence (the training data already covers "
                       f"{count} anchor positions), find unmeasured candidates for {direction_word} "
                       f"{assay_description}, and proactively call submit_recommendations within this round to "
                       f"submit your final recommendations. {_format_sentence_en(config)}")
        else:
            opening = (f"You are an autonomous scientist agent in directed-evolution research on the {protein_name} "
                       f"protein. Research objective: for the {count}-position variants ({positions}; reference "
                       f"{config.reference_sequence}), find unmeasured candidates for {direction_word} "
                       f"{assay_description}, and proactively call submit_recommendations within this round to "
                       f"submit your final recommendations. {_format_sentence_en(config)}")
        return "\n\n".join([opening, _TOOL_PARAGRAPH_EN, _LOOP_PARAGRAPH_EN,
                            _BUDGET_PARAGRAPH_EN, _DISCIPLINE_PARAGRAPH_EN])
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
