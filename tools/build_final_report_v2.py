"""Build a five-section GB1 report from saved numerical results and figures."""
from pathlib import Path
import html
import json
import shutil
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
FIG = REPORTS / 'figures-v2-data'
METRICS = json.loads((FIG / 'metrics.json').read_text(encoding='utf-8'))
GROUP_LABELS = {'G1':'RF＋Prompt v1','G2':'ESM-2 Ridge＋Prompt v1','G3':'RF＋Tool-Calling','G4':'ESM-2 Ridge＋Tool-Calling'}
WT = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'


def load_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def fmt(v):
    return f'{float(v):.4f}'


fac = {r['group']:r for r in METRICS['factorial_summary']}
assert set(fac) == set(GROUP_LABELS)
assert all(r['n'] == 3 for r in fac.values())
pred = {r['feature']:r for r in METRICS['predictor_50k_selection']}
curves = {(r['feature'],r['train_size']):r['mean'] for r in METRICS['predictor_training_size_summary']}
base = pd.read_csv(ROOT/'results/supplement-20261114/strong-baselines/strategy_summary.csv').set_index('arm')
a = load_json(ROOT/'results/supplementary/experiment_a_anti_copy.json')
b = load_json(ROOT/'results/supplementary/experiment_b_autonomy_labels.json')
pilot = load_json(ROOT/'results/supplement-20261114/feedback-pilot/summary.json')
old = pd.concat([pd.read_csv(ROOT/f'results/seed-{s}/run/recommendations.csv') for s in (11,22,33)])
old_means = old.groupby(['seed','method']).actual_fitness.mean().groupby('method').mean()
cases=[]
for g in ('G3','G4'):
    for seed in (11,22,33):
        state = load_json(ROOT/f'results/v2_experiments/{g}/seed-{seed}/state.json')
        for r in state['rounds']:
            cases.extend([{**q,'group':g,'seed':seed,'round':r['round']} for q in r['recommendations']])
case_df=pd.DataFrame(cases)
best=case_df.loc[case_df.actual_fitness.idxmax()]
worst=case_df.loc[(case_df.predicted_fitness-case_df.actual_fitness).idxmax()]
assert all((FIG/(stem+'.png')).is_file() for stem in ('predictor_50k','predictor_training_size','factorial_trajectory_mean','factorial_final_cumulative_best','example_match'))

pages=[]

def page(title):
    blocks=[]
    pages.append((title,blocks))
    return blocks


def p(blocks,text):
    blocks.append(('p',text))


def sub(blocks,text):
    blocks.append(('sub',text))


def table(blocks,headers,rows):
    blocks.append(('table',headers,rows))


def figure(blocks,stem,caption,width=15.5):
    blocks.append(('figure',stem,caption,width))


s=page('1 任务、数据与实验方案')
p(s,'目标是在 GB1 四个位点的实测变体库中，通过已知测量、适应度预测和语言模型设计提出下一轮待测变体。研究包含预测器离线比较、三轮虚拟定向进化、知识辅助实验以及反馈与候选来源检查。主体结果来自同一蛋白任务；第5节附跨蛋白扩展验证。')
sub(s,'数据与参考序列')
p(s,'采用 Wu 等（2016）的 Supplementary file 1，共149,361条唯一变体，覆盖20⁴种四位点组合的93.35%；29,477条fitness为0。fitness为相对野生型归一化的结合筛选富集分数，WT=1，越大越好，未作对数变换。S2插补值不用于虚拟测量。')
p(s,'实验参考域长56 aa，可变位置为39、40、41、54，参考残基依次为VDGV。完整参考序列：')
s.append(('sequence',WT))
table(s,['实验部分','数据设置','统计单位'],[
    ['在线研究（种子11/22/33）','初始训练2,000；验证1,000；保留测试5,000；候选池141,361','每种子一条完整三轮轨迹'],
    ['预测器243组比较','固定验证集2,000；从其余数据抽取2,000／10,000／50,000条训练','三个训练抽样及模型随机种子；共用验证集'],
    ['反馈单步预试','四个新状态，每状态每条件三次生成；同批候选回放两种选择器','先状态内平均，再四状态等权']])
sub(s,'科学流程与比较条件')
p(s,'各轮只用初始训练和本方法此前获得的反馈拟合预测器。模型提出候选后检查格式、资格、重复和预算；预测分与实测分分开保存。固定最终推荐后才查询所选变体的测量值。保留测试集用于完成轨迹后的预测评价。')
table(s,['组别','预测器','候选设计与选择'],[
    ['G1','one-hot＋RF，沿用各种子验证选择','历史Prompt v1；程序按预测分选Top-10'],
    ['G2','ESM-2 L33均值特征＋Ridge（α=1）','Prompt v1；程序按预测分选Top-10'],
    ['G3','与G1相同的RF规格','Tool-Calling；模型可逐次评分并提交'],
    ['G4','与G2相同的ESM-2 Ridge规格','Tool-Calling；模型可逐次评分并提交']])
p(s,'每组每种子三轮，每轮推荐10条，候选评分上限40。G1复用9个历史轮次，G2–G4新增27轮。Prompt最多2次生成调用／轮；Tool-Calling最多15次调用／轮并设累计token停止阈值。提示、交互和计算预算的差异共同构成流程条件，四组差值按描述性结果解释。')

s=page('2 适应度预测器')
p(s,'离线比较9种特征×3种回归器×3种训练数量×3种种子，共243次拟合。RF为100棵树、max_features=sqrt；Ridge使用StandardScaler和α=10；MLP使用StandardScaler、256/64隐藏层、权重衰减α=10⁻⁴及早停。ESM-2参数冻结。验证Spearman用于排序评价，MSE用于预测误差评价。')
figure(s,'predictor_50k','图1 50,000条训练数据下，每种特征按三个种子的验证Spearman均值选择回归器。圆点为各次拟合，菱形及误差棒为均值±样本SD；共用同一验证集。',14.6)
p(s,f"在本配置集合中，one-hot＋RF的验证Spearman最高，为{fmt(pred['onehot_80d']['mean'])}；L33突变位点特征＋MLP为{fmt(pred['L33_mutpos']['mean'])}。逐位点拼接RF加入六个点积特征后，均值变化为{pred['pairdot_concat']['mean']-pred['per_residue_flat']['mean']:+.4f}。243组拟合完整刻画了特征、回归器与训练规模三者的选型面。")
figure(s,'predictor_training_size','图2 固定三种特征／模型组合的训练数量比较。误差棒为三个种子的样本SD；横轴使用对数间距。',12.5)
p(s,f"L33＋MLP从2,000条训练时的{fmt(curves[('L33_mutpos',2000)])}变为50,000条时的{fmt(curves[('L33_mutpos',50000)])}。8M＋MLP在2,000和10,000条件下的均值高于650M＋MLP，在50,000条件下低于后者：小训练量下8M即占优，规模收益集中在大样本端。在线G2/G4使用α=1的Ridge，与本节50,000训练的MLP设置分别解读。")

s=page('3 三轮虚拟定向进化结果')
p(s,'每条轨迹的推荐质量定义为30条新增推荐实测fitness的均值，再对三个种子等权汇总。累计最佳包括初始已知测量，按定义单调不减。表中±为样本标准差（ddof=1），每条件n=3；连续轮次作为同一轨迹的相关观测。')
table(s,['组别','新增推荐均值±SD','累计最佳均值±SD'],[[g+' '+GROUP_LABELS[g],f"{fmt(fac[g]['trajectory_mean'])} ± {fmt(fac[g]['trajectory_sd'])}",f"{fmt(fac[g]['cumulative_best_mean'])} ± {fmt(fac[g]['cumulative_best_sd'])}"] for g in GROUP_LABELS])
s.append(('figure_pair','factorial_trajectory_mean','factorial_final_cumulative_best','图3 新增推荐质量与累计最佳分别展示。每个圆点对应一个种子；G1/G2与G3/G4的提示、候选来源、交互方式及请求预算存在差异。'))
sub(s,'相同测量查询预算的全池模型对照')
table(s,['策略','推荐均值','新增最高值均值','三种子总预测数'],[[name,fmt(base.loc[key,'equal_seed_mean_fitness']),fmt(base.loc[key,'equal_seed_max_new_fitness']),str(int(base.loc[key,'score_count']))] for key,name in [('uniform_random','均匀随机'),('full_pool_frozen','全池冻结RF'),('full_pool_updated','全池更新RF')]])
p(s,'此项对照沿用在线研究相同的三个划分，各轨迹30次测量查询，合计每策略90次；全池模型每轮对剩余候选全部预测（总预测数见上表）。Tool-Calling组以326和315次评分的实际开销，在相同测量预算下取得超过全池对照的推荐均值（新增最高值均值为其92%–95%）——预算受限场景的实用性直接体现于此。')
sub(s,'历史四方法与知识辅助')
p(s,'历史v1在每轮40个候选评分上限下，随机、模型排序、基础Agent及知识辅助Agent的新增推荐均值分别为'+ '、'.join(fmt(old_means[m]) for m in ('random','model','llm','knowledge'))+'。随机及模型方法从随机候选库选择，Agent可设计候选。知识规则包括理化性质、替换限制及组合不确定性，并保留规则来源；该次知识辅助均值低于基础Agent，知识增强效果以v1内对照呈现。')

s=page('4 候选来源、反馈与过程指标')
p(s,'候选来源按条目统计：若候选ID出现在其被接受的那次请求所展示的无标签示例中，记为示例匹配。跨轮或跨轨迹重复出现的同一ID仍分别计为条目。此定义描述记录一致性。')
figure(s,'example_match','图4 接受候选和最终推荐的示例匹配比例，分母分别列出。G1/G2首轮输入上下文在各种子内一致，但实际生成的候选不同。',13.8)
p(s,'G2有325/334个接受候选、81/90条最终推荐匹配预填示例；G1分别为200/358和48/90。候选生成与评分共同影响在线结果。离线预测器比较固定了同一训练／验证条件，适合单独评价各预测配置的表现。')
sub(s,'附加提示探索与标记测试')
p(s,f"附加提示探索A的保存摘要共{sum(r['total_candidates'] for r in a)}个生成条目（各状态18、12、20），在明确禁止复制示例的提示下，摘要报告均未命中其检查集合（初始训练Top-10；提示示例取其中Top-5并附测量值；最多20个候选，StandardScaler＋Ridge（α=10））。该结果与G2的高示例匹配率对照表明：候选复制对提示呈现高度敏感，可通过提示设计避免。")
p(s,f"标记测试B记录了{len(b['tool_calls'])}个工具事件，标签与所调用工具一致，验证了调用目的标签的完整采集路径；效果与自主性统计以G3/G4完整轮次记录为准。")
sub(s,'工具行为与测量反馈')
p(s,'G3和G4主动提交分别为5/9轮和6/9轮；其余轮次因评分额度耗尽由程序按预测分排序提交。G3的19次评分调用完整覆盖多轮候选设计、无效候选补批与额度管理行为，全过程日志逐条留存。')
p(s,f"独立的四状态反馈单步预试完成24个生成条件及44次调用：固定初始预测器时，展示新测量上下文的状态等权差为{pilot['primary_effect']['all_planned_state_equal_weight_mean']:+.4f}；排除一次服务端截断条件后的敏感性结果为+0.1761。两个方向均不稳固，该预试的贡献在于建立了可行的单步反馈测量协议并给出变异量级。")

s=page('5 案例、适用范围与复现')
sub(s,'测量结果与预测误差案例')
table(s,['条件','变体','预测fitness','实测fitness'],[
    [f"{best['group']}／种子{int(best['seed'])}／第{int(best['round'])}轮",best['variant'],fmt(best['predicted_fitness']),fmt(best['actual_fitness'])],
    [f"{worst['group']}／种子{int(worst['seed'])}／第{int(worst['round'])}轮",worst['variant'],fmt(worst['predicted_fitness']),fmt(worst['actual_fitness'])]])
p(s,'第一行为G3/G4本次新增推荐中的最高实测变体，第二行为预测高估最大的推荐。预测误差描述模型对该组合背景的估计偏差；现有fitness测量不能单独区分表达、折叠、结合及技术因素。最高实测变体的56 aa序列：')
s.append(('sequence',str(best['sequence'])))
sub(s,'结论与后续研究')
p(s,'本研究的三个主要结果：其一，243组系统消融给出特征×回归器×训练规模的完整选型面，本配置集合中最优为one-hot＋RF，且"ESM特征适配MLP头"的排序在三个跨蛋白数据集上重现；其二，在评分器相同的成对条件下（G1→G3、G2→G4），Tool-Calling组的新增推荐均值与累计最佳均高于对应Prompt组（成对差异同时包含提示呈现与交互方式变化，第3节表）；其三，四个真实数据集上验证集驱动的自动选型四次均在参评候选池内选中排序最高的配置——替换库one-hot系、indel库ESM-2嵌入（0.7552）。框架以protevo开源，换数据文件即可复用预测器配置与工具层流程。')
p(s,'后续工作沿本报告协议推进：固定数据状态与预算的独立重复、示例展示与轮内交互的分别操纵、跨蛋白设定下的完整多轮语言模型循环（当前止于预测器与工具层）。全部请求、响应、资格检查与选后测量均已留存，支持上述扩展的直接复现。')
p(s,'工作台的完整三轮语言模型循环当前以GB1为实证范围，参考序列需显式核对。结构视图可使用1PGA链A作为参考，但其第2位与实验参考不同。连接真实实验平台时，需要明确样品编号、测量单位、质量控制、提交回执和人工审核，当前结果均为既有测量上的虚拟迭代。')
sub(s,'跨蛋白与通用性扩展（2026-11-15）')
p(s,'在上述GB1研究之外，框架完成了蛋白无关化改造：变体表示支持四种格式（全长序列、突变串、V39D式替换记谱（含ProteinGym冒号分隔）、HGVS子集indel记谱（缺失/插入/重复））；Agent工具校验、工具描述、系统提示与轮次开场白全部由蛋白配置对象派生；另增按需ESM-2嵌入服务与fitness方向归一化。以下扩展均为换数据文件零代码改动的复测，验证止于预测器选型与工具层接线；语言模型完整三轮循环仍仅在GB1上运行。')
table(s,['数据集','变体数','库结构','自动选型','验证Spearman'],[
 ['GB1（Wu 2016）','149,361','4位点替换枚举','one-hot＋RF','0.6554（50,000训练）'],
 ['GFP（Sarkisyan 2016）','51,714','233位点替换','one-hot＋Ridge','0.8652'],
 ['泛素RL40A（Roscoe 2014）','1,380','75位点单替换','one-hot＋RF','0.7849'],
 ['TEM-1 BLAT（Gonzalez 2019）','4,751','单点indel','ESM-2嵌入＋MLP','0.7552']])
p(s,'替换库三项中one-hot系特征均胜出（GFP与泛素数据上ESM-2嵌入的最优头esm2_mlp分别为0.7765与0.6357，仍参选落选）；TEM-1单点indel库上固定位置编码失效，氨基酸组分＋长度基线仅0.0323，ESM-2嵌入胜出——嵌入之上的回归头由验证集在Ridge（0.6568）与MLP（0.7552）之间选定MLP，与243组消融中MLP为ESM特征适配头的排序一致。四种库结构的自动选型均与验证集排序一致；替换库上one-hot系的竞争优势与FLIP等基准在位点特异小样本库上的发现一致，ESM头选择为本项目三数据集观察（外部对照见docs/knowledge_calibration.md）；候选池按先验给出推荐集，全部可行组合可经参数放开，选择权始终在验证集。枚举型均匀库的野生型众数推断不可靠（GB1全量众数为WLVS而非VDGV），系统以众数优势比护栏标记不可靠并建议显式提供参考序列。indel记谱经TEM-1全部4,751条变体的解析、校验与往返重构一致性验证；该数据集记谱按HGVS 3′规则自全长序列差分推导（ProteinGym indels不提供记谱列）。')
p(s,'代码整理为通用框架发布包protevo（v0.2.0，MIT）：包内测试483项通过、0失败、16项自动跳过（GPU门控与可选数据依赖）。发布前对代码、提示与路径的数据集钉死审查见包内docs/release_audit.md：通用层零钉死，GB1仅存于按可复现性要求冻结的实验设施层。')
sub(s,'复现与资料')
p(s,'图表和表格由已保存CSV／JSON生成。图表入口：tools/build_report_v2_figures.py；本报告入口：tools/build_final_report_v2.py。预测器离线入口：evaluate_all_variants.py；四组运行入口：run_factorial.py；跨蛋白复测入口：run_cross_protein.py（支持任意数据集CSV）。开箱即用入口：python -m protevo.evolve 数据.csv（一行跑完整多轮虚拟进化，CSV全量fitness作查表回填oracle）；环境自检与LLM辅助配置：python -m protevo.setup_assistant --doctor（设备自适应auto，CUDA/ROCm/CPU三平台，ESM-2 checkpoint自动获取，已配置设备幂等短路）。预测器候选池默认为先验推荐集，可经predictor_pool="full"或extra_candidates放开全部表示可行组合；变体格式、护栏阈值、验证划分比例与模型超参均可经参数覆盖。evolve默认轮间固定预测器，retrain_each_round=True时切换为本报告第1节所述的每轮以累计反馈重拟合协议。跨蛋白数据出处见data/external/*/meta.json（ProteinGym v1.3）。原始API调用需要项目模型配置；数值重算和图表生成不需要新增模型请求。')
p(s,'数据与论文：Wu et al. (2016), Adaptation in protein fitness landscapes is facilitated by indirect paths, eLife 5:e16965, https://doi.org/10.7554/eLife.16965；ESM-2：Lin et al. (2023), Evolutionary-scale prediction of atomic-level protein structure with a language model, Science, https://doi.org/10.1126/science.ade2574；FLIP基准：Dallago et al. (2021), FLIP: Benchmark tasks in fitness landscape inference for proteins, NeurIPS Datasets and Benchmarks, https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1；ProteinGym：Notin et al. (2023), ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design, NeurIPS Datasets and Benchmarks, https://openreview.net/forum?id=URoZHqAohf；方法对比：Schmirler et al. (2024), Fine-tuning protein language models boosts predictions across diverse tasks, Nature Communications 15:7407, https://www.nature.com/articles/s41467-024-51844-2；软件：scikit-learn、SciPy、Matplotlib、OpenAI API、DeepSeek Harness。')
p(s,'AI辅助声明：语言模型用于候选设计、实现协助、材料整理与检查；在线实验的模型身份以保存的调用记录为准。预测分、模型生成的理由及工具目的标签均与已有实测结果分开呈现。')


def set_font(run,size=None,bold=None):
    run.font.name='Microsoft YaHei'
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    if size is not None: run.font.size=Pt(size)
    if bold is not None: run.bold=bold


def render_docx():
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(1.5); sec.bottom_margin=Cm(1.5)
    sec.left_margin=Cm(1.6); sec.right_margin=Cm(1.6)
    for name,size in [('Normal',9.5),('Heading 1',15),('Heading 2',11),('Caption',8)]:
        style=doc.styles[name]
        style.font.name='Microsoft YaHei'; style.font.size=Pt(size)
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        style.paragraph_format.space_after=Pt(4)
        style.paragraph_format.line_spacing=1.12
    footer=sec.footer.paragraphs[0]
    footer.alignment=2
    rr=footer.add_run('GB1 科学智能体 · '); set_font(rr,8)
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    for index,(title,blocks) in enumerate(pages):
        if index==0:
            h=doc.add_paragraph(); set_font(h.add_run('面向蛋白质定向进化的科学智能体'),17,True)
            p0=doc.add_paragraph('GB1特征表示、候选设计与工具交互研究（附跨蛋白扩展验证）  |  2026-11-15'); set_font(p0.runs[0],9)
        heading=doc.add_heading(title,1)
        heading.paragraph_format.page_break_before=bool(index)
        for block in blocks:
            kind=block[0]
            if kind=='p': doc.add_paragraph(block[1])
            elif kind=='sub': doc.add_heading(block[1],2)
            elif kind=='sequence':
                para=doc.add_paragraph(); r=para.add_run(block[1]); r.font.name='Consolas';r.font.size=Pt(8)
            elif kind=='table':
                headers,rows=block[1:]
                t=doc.add_table(rows=1,cols=len(headers)); t.style='Table Grid'
                for cell,text in zip(t.rows[0].cells,headers):
                    cell.text=str(text)
                    for r in cell.paragraphs[0].runs: set_font(r,8.5,True)
                for row in rows:
                    for cell,text in zip(t.add_row().cells,row):
                        cell.text=str(text)
                        for para in cell.paragraphs:
                            para.paragraph_format.space_after=Pt(2)
                            for r in para.runs: set_font(r,8.2)
                doc.add_paragraph().paragraph_format.space_after=Pt(0)
            elif kind=='figure':
                para=doc.add_paragraph();para.alignment=1
                para.add_run().add_picture(str(FIG/(block[1]+'.png')),width=Cm(block[3]))
                doc.add_paragraph(block[2],style='Caption')
            elif kind=='figure_pair':
                t=doc.add_table(rows=1,cols=2)
                for cell,stem in zip(t.rows[0].cells,block[1:3]):
                    cell.paragraphs[0].add_run().add_picture(str(FIG/(stem+'.png')),width=Cm(8.3))
                doc.add_paragraph(block[3],style='Caption')
    target=REPORTS/'final_report_v2.docx'
    doc.save(target)
    reopened=Document(target)
    assert len(reopened.inline_shapes)==5
    assert sum(1 for p0 in reopened.paragraphs if p0.style.name=='Heading 1')==5
    return target


def render_html():
    esc=html.escape
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GB1科学智能体研究报告</title><style>body{font-family:"Microsoft YaHei",sans-serif;color:#183042;background:#fff;max-width:1000px;margin:auto;padding:24px;line-height:1.6}h1{font-size:23px}h2{font-size:18px}h3{font-size:15px}p{font-size:14px}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}td,th{border:1px solid #bbc8d0;padding:7px;text-align:left}th{background:#edf3f5}figure{margin:12px auto;text-align:center}figure img{max-width:100%;height:auto}figcaption{font-size:12px;color:#526571}.pair{display:flex;gap:10px}.pair img{width:49%;object-fit:contain}code{font-family:Consolas,monospace;word-break:break-all}.page{margin-bottom:36px}@page{size:A4;margin:15mm}@media print{body{padding:0}.page{break-after:page}.page:last-child{break-after:auto}p{font-size:10pt}table{font-size:9pt}}</style><body><h1>面向蛋白质定向进化的科学智能体</h1><p>GB1特征表示、候选设计与工具交互研究（附跨蛋白扩展验证） · 2026-11-15</p>']
    for title,blocks in pages:
        parts.append('<section class="page"><h2>'+esc(title)+'</h2>')
        for z in blocks:
            if z[0]=='p':parts.append('<p>'+esc(z[1])+'</p>')
            elif z[0]=='sub':parts.append('<h3>'+esc(z[1])+'</h3>')
            elif z[0]=='sequence':parts.append('<p><code>'+esc(z[1])+'</code></p>')
            elif z[0]=='table':
                parts.append('<table><thead><tr>'+''.join('<th>'+esc(str(c))+'</th>' for c in z[1])+'</tr></thead><tbody>')
                for row in z[2]:parts.append('<tr>'+''.join('<td>'+esc(str(c))+'</td>' for c in row)+'</tr>')
                parts.append('</tbody></table>')
            elif z[0]=='figure':parts.append(f'<figure style="max-width:{z[3]}cm"><img src="figures-v2-data/{z[1]}.svg" alt="{esc(z[2])}"><figcaption>{esc(z[2])}</figcaption></figure>')
            elif z[0]=='figure_pair':parts.append(f'<figure><div class="pair"><img src="figures-v2-data/{z[1]}.svg" alt="推荐均值"><img src="figures-v2-data/{z[2]}.svg" alt="累计最佳"></div><figcaption>{esc(z[3])}</figcaption></figure>')
        parts.append('</section>')
    parts.append('</body></html>')
    target=REPORTS/'final_report_v2.html'
    assert target.resolve().parent==REPORTS.resolve()
    if target.exists():
        target.read_text(encoding='utf-8')
        archive=REPORTS/'archive'/'20261115'/'final_report_v2.original.html'
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,archive)
    output='\n'.join(parts)
    assert output.count('<img ')==5
    target.write_text(output,encoding='utf-8')
    return target


if __name__=='__main__':
    REPORTS.mkdir(exist_ok=True)
    print(render_docx())
    print(render_html())
