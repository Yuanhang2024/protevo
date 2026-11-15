"""Update current report wording while preserving archived experimental records."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path, pairs):
    content = path.read_text(encoding='utf-8')
    for old, new in pairs:
        if content.count(old) != 1:
            raise ValueError(f'Expected one current text match in {path.name}')
        content = content.replace(old, new, 1)
    path.write_text(content, encoding='utf-8')


replace(ROOT / 'build_report.py', [
    ('关键依赖已固定，当前完整测试套件268项通过；数据划分与真实调用另有运行记录。', '关键依赖已固定；软件检查与实验重复分别记录，当前修订验证见配套统计定义v2资料。'),
    ('复现：按项目README安装requirements，依次运行prepare、select、run、finalize、plot、aggregate。独立API需自备密钥；重新调用LLM可能产生不同输出，原响应已保留。代码与精选结果已发布于GitHub私有仓库 github.com/Yuanhang2024/gb1-scientific-agent（MIT许可证）。后续阶段已完成v3提示三种子验证、多模型比较与默认值校准，结果见配套验证报告与本仓库校准报告。',
     '复现：按README安装依赖，运行prepare、select、run、finalize、plot、aggregate。API重跑产生新随机输出。仓库github.com/Yuanhang2024/gb1-scientific-agent提供代码和精选结果；选中结果可复算，历史过程复核还需对应完整记录。数据、模型配置与计算定义见配套资料。'),
    ('后续对照（2026-11-13）：零上下文时模型可回忆出全局前十中的2个，但对随机批次排序Spearman为−0.18至−0.27，说明预训练记忆了该公开数据的高分区域而非整表；首答候选42%—53%落入全局前1%，优势含记忆成分。同预算无反馈对照的均值差为+0.185、+0.121、−0.329，反馈无独立可测增益。资源限制：一周内单机CPU完成，ESM/GP需CUDA栈未验证，主实验用CPU可复现的one-hot与树集成（见仓库critique-followup结果）。',
     '后续观测按定义v2复算：best-first位置越小越优，高分方向得分为负位置，三批Spearman为+0.2665、+0.2688、+0.1819。高分先验来源尚未分离。自适应减冻结流程的三状态均值差为+0.1846、+0.1206、−0.3290；这不是独立反馈效应或无效应证明。主实验使用CPU上的one-hot与树模型，ESM/GP未作同条件性能比较。'),
])

replace(ROOT / 'build_validation_report.py', [
    ('GB1 科学智能体 · 固定版本验证与反馈消融 · 2026-11-09', 'GB1 科学智能体 · 固定v3观测与详细反馈展示 · 统计定义v2'),
    ('这些原则经本项目实际验证，不直接沿用文献中的性能数字。', '本实现借鉴这些方法组织提示和记录；其性能作用由本项目对照另行评估。'),
    ('完整离线测试{test_count}项通过；真实请求、候选来源、数据范围和配对条件另行核查。', '本次修订记录的软件检查共{test_count}项通过；该数量不是实验重复数。真实请求、候选来源和配对条件另行核查。'),
    ('复现与交付：项目README记录依赖、数据获取和命令；本阶段独立入口ablate_feedback.py提供prepare/run/compare。所有提示、请求和原始响应保留，旧结果未覆盖。API重跑可能产生不同输出；代码与精选结果已发布于GitHub私有仓库。本报告附统计摘要、推荐CSV及paired_results.csv。',
     '复现与交付：README记录安装及命令；ablate_feedback.py提供prepare/run/compare。本地保存完整提示、响应和配对状态，历史比较器需要这些文件及预测器。发布的推荐表与配对表可复算汇总，不能据此恢复未发布过程。API重跑是新实验，不会补回缺失的历史输出。'),
])

replace(ROOT / 'reports/实验结果_修订版.md', [
    ('原报告存在统计解释错误（排序方向）、结论强度过高（无显著性检验）、反馈收益证据不足、Agent 自主性范围描述不准确等问题。',
     '本版区分排名位置与高分方向、描述性结果与效应证明、详细展示与整套流程更新、候选生成与执行控制。'),
    ('- 实验轮次：36（v1+v3），推荐记录：360+180=540，跨种子/方法唯一推荐序列：274。\n- 正式 API 请求：31；服务返回总用量：297,662 tokens。',
     '- v1：36个方法轮次、360条推荐、274种唯一变体；正式生成31次，报告297,662 tokens。\n- v3：18个方法轮次、180条推荐、143种唯一变体；正式生成33次，报告329,745 tokens。\n- 两版本合计：54个方法轮次、540条推荐、381种唯一变体；生成64次，报告627,407 tokens。\n- 首答高分重合统计只包括两版本中的LLM/knowledge，共36个方法轮次，不包括random/model；三个数据种子不因版本合并而变成六个新状态。'),
    ('**结论**：反馈的独立收益**尚未证明**，需补做 C0P0/C1P0/C0P1/C1P1 四格对照实验。',
     '**结论**：反馈的独立收益**尚未证明**。补充研究先冻结比较目标；上下文新观测与预测器更新可用C0P0/C1P0/C0P1/C1P1分别检验，单步预试与完整多轮效果分开报告。'),
])

replace(ROOT / 'reports/validation-stage/验证结果.md', [
    ('此结果支持在这三个固定状态中保留详细反馈。样本为三个固定状态；两条件的请求随机性与上下文长度随实验设计变化，withheld仍保留更新后的已知数据统计。',
     '三个固定状态的详细展示观测差均为正。每状态每条件仅一次生成，输入长度与内容同时变化，withheld仍保留更新后的已知数据统计；该结果尚不足以证明独立、稳定的反馈收益。'),
    ('- 完整离线测试268项通过；新模块独立于固定v3运行核心，没有修改既有提示或实验结果。',
     '- 软件验证与实验重复分别记录。历史提示和实验响应保留；本次统计修订测试结果见统计定义v2的验证资料。'),
])

replace(ROOT / 'README.md', [
    ('结论强度过高（无显著性检验）', '描述性结果与效应证明未充分区分'),
    ('零上下文记忆检验、首答高分重合率与同预算无反馈对照的完整记录在该目录。',
     '零实验数据提示的排序、高分重合和历史自适应/冻结流程汇总在该目录。冻结流程只有已选结果和用量，缺少完整生成与评分过程；不将其称为独立反馈收益检验。'),
    ('本阶段新增补种子23次请求，消融11次，共34次、344,597 tokens，低于36次上限。当前完整离线测试268项通过，真实结果另有独立核对。',
     '本阶段新增补种子23次请求，消融11次，共34次、344,597 tokens，低于36次上限。实验重复单位为三个固定状态；软件检查数量按具体执行版本记录。'),
    ('若只复算现有统计，可对 `results/feedback-ablation` 执行 `compare`，不调用模型。',
     '若本地保留完整manifest、六组状态与预测器，可对 `results/feedback-ablation` 执行 `compare`；只有发布配对表时只能复算表内均值与差值，不能直接运行依赖完整状态的历史比较器。'),
    ('完整离线测试268项通过。主动选样采用主动学习策略', '该扩展的算法与测量统计按原记录报告，软件测试结果单独记载。主动选样采用主动学习策略'),
])
print('Updated current report builders and explanatory text; archived originals and raw experimental records retained.')
