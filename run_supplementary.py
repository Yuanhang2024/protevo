# 补充实验：修正候选来源混淆 + 严格自主性标记
# 解决问题 1（G2 预填候选）和问题 2（策略转折分类）
import json, time, sys, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import numpy as np

PROJECT = Path(str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT))
from protevo.tools import TOOL_SCHEMAS, ToolBox, AgentBudget
from protevo.agent_v2 import ToolCallingAgent

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

# === 实验 A：反复制 G2 ===
# 修改 prompt：明确禁止复制示例
ANTI_COPY_PROMPT = """你是一个蛋白质定向进化科学家。

当前任务：基于已有实验数据，为GB1蛋白设计下一轮突变候选。

**重要规则**：
- 你必须生成全新的候选序列，**严禁复制下方示例中的任何序列**
- 示例仅供参考格式，不代表推荐方向
- 每个候选必须是你的独立推理结果

可用的突变位点：39, 40, 41, 54（对应变体字符串的4个位置）
野生型：VDGV

以下是历史推荐示例（仅供格式参考，禁止复制）：
{examples}

当前已知数据摘要：
{history}

请生成最多{max_candidates}个新候选序列（4字符突变串）。
对每个候选给出科学理由（为什么这个突变可能有益）。
以JSON格式输出：
{{"candidates": [{{"sequence": "XXXX", "reason": "..."}}, ...], "hypothesis": "当前的突变假设"}}"""

# === 实验 B：严格自主性标记 ===
# 修改 agent_v2 使每次工具调用记录 reason
STRICT_AUTONOMY_PROMPT = """你是一个蛋白质定向进化科学家。

你可以使用以下工具：
1. predict_fitness(sequences) — 预测适应度
2. query_knowledge(query) — 查询知识库
3. get_experiment_history() — 获取实验历史
4. submit_recommendations(sequences, reasons) — 提交最终推荐
5. check_budget() — 查询剩余预算

**每次调用工具时，你必须在同一回复中说明调用原因**：
- "new_hypothesis" — 基于新的科学假设
- "explore" — 探索性查询（无特定假设）
- "budget_topup" — 补充评分额度
- "verify" — 验证之前的假设
- "final" — 最终提交

当前已知数据摘要：
{history}

请开始你的研究流程。自主决定使用哪些工具、何时使用。"""

def run_experiment_a():
    """实验A：反复制 G2（ESM-2 + Prompt + 禁止复制示例）"""
    print("=== 实验 A：反复制 G2 ===")
    results = []
    
    for seed in [11, 22, 33]:
        data_dir = PROJECT / f"data/processed/seed-{seed}"
        train = __import__('pandas').read_csv(data_dir / "train.csv")
        
        # 构建历史摘要（与 G1/G2 一致）
        top_variants = train.nlargest(10, "fitness")[["variant", "fitness"]]
        history = f"已知 {len(train)} 条数据，fitness > 1.0 的有 {len(train[train.fitness>1])} 条\n"
        history += f"Top-10 变体：{', '.join(top_variants.variant.tolist())}"
        
        # 获取一些示例（从预填候选中取，与 G2 一致）
        examples = "\n".join([
            f"  {v} (fitness={f:.3f})"
            for v, f in zip(top_variants.variant[:5], top_variants.fitness[:5])
        ])
        
        prompt = ANTI_COPY_PROMPT.format(
            examples=examples,
            history=history,
            max_candidates=20,
        )
        
        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a protein engineering expert. Respond in JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        
        # 解析响应
        import re
        text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                candidates = data.get("candidates", [])
                sequences = [c["sequence"] for c in candidates if "sequence" in c]
                hypothesis = data.get("hypothesis", "")
            except:
                sequences, hypothesis = [], ""
        else:
            sequences, hypothesis = [], ""
        
        # 检查与示例的重复
        example_set = set(top_variants.variant.tolist())
        novel_count = sum(1 for s in sequences if s not in example_set)
        
        # 用 ESM-2 预测器评分
        emb = np.load(PROJECT / "data/esm2_embeddings/esm2_pooled_L33.npy")
        all_var = __import__('pandas').read_csv(PROJECT / "data/processed/all_variants.csv")
        var2idx = {v: i for i, v in enumerate(all_var.variant)}
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        train_rows = [var2idx[v] for v in train.variant]
        X_tr = emb[train_rows]
        y_tr = train.fitness.values
        
        predictor = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=10))])
        predictor.fit(X_tr, y_tr)
        
        # 评分
        valid_seqs = [s for s in sequences if s in var2idx and len(s) == 4]
        if valid_seqs:
            test_rows = [var2idx[s] for s in valid_seqs]
            X_test = emb[test_rows]
            preds = predictor.predict(X_test)
            
            # 获取真实 fitness
            true_fitness = [all_var.fitness.iloc[var2idx[s]] for s in valid_seqs]
            
            result = {
                "seed": seed,
                "total_candidates": len(sequences),
                "novel_candidates": novel_count,
                "copy_rate": round(1 - novel_count / max(len(sequences), 1), 3),
                "mean_pred": round(float(np.mean(preds)), 4) if len(preds) > 0 else None,
                "mean_true_fitness": round(float(np.mean(true_fitness)), 4) if len(true_fitness) > 0 else None,
                "max_true_fitness": round(float(np.max(true_fitness)), 4) if len(true_fitness) > 0 else None,
                "hypothesis": hypothesis[:100],
            }
        else:
            result = {"seed": seed, "total_candidates": 0, "novel_candidates": 0, 
                     "copy_rate": None, "error": "no valid candidates"}
        
        results.append(result)
        print(f"  Seed {seed}: {len(sequences)} candidates, {novel_count} novel "
              f"(copy rate: {result.get('copy_rate', 'N/A')})")
    
    # 保存
    out_dir = PROJECT / "results/supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment_a_anti_copy.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 汇总
    novel_rates = [r.get("novel_candidates", 0) / max(r.get("total_candidates", 1), 1) for r in results if r.get("total_candidates", 0) > 0]
    copy_rates = [r.get("copy_rate", 1) for r in results if r.get("copy_rate") is not None]
    print(f"\n实验A汇总:")
    print(f"  G2原始 copy rate: ~97% (325/334)")
    print(f"  G2反复制 novel rate: {np.mean(novel_rates):.1%}" if novel_rates else "  No valid results")
    print(f"  G2反复制 copy rate: {np.mean(copy_rates):.1%}" if copy_rates else "")
    
    return results


def run_experiment_b():
    """实验B：严格自主性标记（Tool-Calling + reason字段）"""
    print("\n=== 实验 B：严格自主性标记 ===")
    
    # 使用 seed 11, 1 轮, Tool-Calling
    data_dir = PROJECT / "data/processed/seed-11"
    train = __import__('pandas').read_csv(data_dir / "train.csv")
    
    # 构建 history
    top = train.nlargest(10, "fitness")
    history_text = f"Known data: {len(train)} variants, {len(train[train.fitness>1])} with fitness>1.0\n"
    history_text += f"Top variants: {', '.join(top.variant.tolist())}"
    
    system_prompt = STRICT_AUTONOMY_PROMPT.format(history=history_text)
    
    # 运行 Tool-Calling 循环
    messages = [{"role": "system", "content": system_prompt}]
    
    tool_calls_log = []
    api_calls = 0
    max_calls = 15
    
    print("  Running Tool-Calling agent with reason tracking...")
    
    while api_calls < max_calls:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=4000,
            )
            api_calls += 1
            
            msg = response.choices[0].message
            
            # 记录
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    
                    # 尝试从回复文本中提取 reason
                    content = msg.content or ""
                    reason = "unclassified"
                    for r_tag in ["new_hypothesis", "explore", "budget_topup", "verify", "final"]:
                        if r_tag in content.lower():
                            reason = r_tag
                            break
                    
                    tool_calls_log.append({
                        "call_id": tc.id,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "reason": reason,
                        "api_call_number": api_calls,
                        "tokens_used": response.usage.total_tokens if response.usage else 0,
                    })
                    print(f"    [{api_calls}] {tool_name} (reason: {reason})")
                    
                    # 执行工具（简化版）
                    if tool_name == "submit_recommendations":
                        # 记录提交
                        tool_calls_log[-1]["result"] = "submitted"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "ok"})})
                        break
                    elif tool_name == "predict_fitness":
                        # 模拟评分
                        seqs = tool_args.get("sequences", [])
                        tool_calls_log[-1]["result"] = f"scored {len(seqs)} sequences"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"scores": [0.5]*len(seqs)})})
                    elif tool_name == "check_budget":
                        remaining = max_calls - api_calls
                        tool_calls_log[-1]["result"] = f"{remaining} calls remaining"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"remaining_calls": remaining})})
                    else:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"status": "ok"})})
            
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": None})
            
            if not msg.tool_calls:
                # LLM 没有调用工具，提示继续
                messages.append({"role": "user", "content": "Please continue using the available tools."})
                
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    # 统计
    reasons = [tc["reason"] for tc in tool_calls_log]
    from collections import Counter
    reason_dist = Counter(reasons)
    
    print(f"\n  实验B汇总 ({len(tool_calls_log)} tool calls):")
    for reason, count in reason_dist.most_common():
        print(f"    {reason}: {count}")
    
    # 保存
    out_dir = PROJECT / "results/supplementary"
    with open(out_dir / "experiment_b_autonomy_labels.json", "w", encoding="utf-8") as f:
        json.dump({"tool_calls": tool_calls_log, "reason_distribution": dict(reason_dist)}, f, indent=2, ensure_ascii=False)
    
    return tool_calls_log, reason_dist


if __name__ == "__main__":
    print("补充实验：修正候选来源混淆 + 严格自主性标记")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # 实验 A
    results_a = run_experiment_a()
    
    # 实验 B
    results_b, reason_dist = run_experiment_b()
    
    print("\n" + "="*60)
    print("补充实验完成")
    print(f"实验A: {len(results_a)} seeds, 结果保存至 results/supplementary/experiment_a_anti_copy.json")
    print(f"实验B: {len(results_b)} tool calls, 结果保存至 results/supplementary/experiment_b_autonomy_labels.json")
    print("="*60)
