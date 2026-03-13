# -*- coding: utf-8 -*-
"""
评测脚本 B: LLM-as-Judge 评测
用 DeepSeek V3 (或 GPT-4) 作为裁判，盲测对比不同模型的判案推理质量。

运行模式: 无卡（纯 API 调用）
用法: python eval_llm_judge.py --file_a eval_results_base.jsonl --file_b eval_results_sft.jsonl

依赖: pip install openai tqdm loguru python-dotenv
"""
import os
import json
import re
import argparse
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============ 配置 ============
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # DeepSeek V3 作为裁判

MAX_JUDGE = 100  # 最多盲测多少条（控制 API 成本）

JUDGE_PROMPT = '''你是一位法律评审专家，请评估以下两位"法官"对同一案件的判决意见。

<案件事实>
{fact}
</案件事实>

<真实判决>
罪名: {true_accusation}
刑期: {true_sentence}
</真实判决>

<法官A的判决>
{response_a}
</法官A的判决>

<法官B的判决>
{response_b}
</法官B的判决>

请从以下 4 个维度分别打分（1-5 分），并给出总评：
1. **罪名判定准确性**: 罪名是否正确、法律适用是否恰当
2. **推理逻辑严密性**: 推理过程是否有条理、证据引用是否充分
3. **量刑合理性**: 刑期判定是否合理、是否考虑了量刑情节
4. **格式规范性**: 输出是否结构化、清晰易读

请严格按以下 JSON 格式输出：
```json
{{
    "judge_a": {{"罪名准确": 分数, "推理逻辑": 分数, "量刑合理": 分数, "格式规范": 分数, "总分": 分数}},
    "judge_b": {{"罪名准确": 分数, "推理逻辑": 分数, "量刑合理": 分数, "格式规范": 分数, "总分": 分数}},
    "winner": "A 或 B 或 平局",
    "comment": "简要评价"
}}
```'''


def load_eval_results(filepath):
    """加载评测结果文件（跳过第一行汇总）"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过汇总行
            try:
                results.append(json.loads(line))
            except:
                pass
    return results


def judge_one(client, fact, true_acc, true_sentence, response_a, response_b):
    """用 LLM 裁判打分"""
    prompt = JUDGE_PROMPT.format(
        fact=fact[:1000],  # 截断过长的事实
        true_accusation=true_acc,
        true_sentence=true_sentence,
        response_a=response_a[:800],
        response_b=response_b[:800],
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            timeout=30,
        )
        content = response.choices[0].message.content
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"Judge 失败: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge 法律评测")
    parser.add_argument("--file_a", required=True, help="模型A的评测结果文件")
    parser.add_argument("--file_b", required=True, help="模型B的评测结果文件")
    parser.add_argument("--label_a", default="模型A", help="模型A的名称标签")
    parser.add_argument("--label_b", default="模型B", help="模型B的名称标签")
    parser.add_argument("--max_judge", type=int, default=MAX_JUDGE)
    parser.add_argument("--output", default="./eval_judge_results.jsonl")
    args = parser.parse_args()
    
    if not API_KEY:
        logger.error("请设置 DEEPSEEK_API_KEY")
        return
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    results_a = load_eval_results(args.file_a)
    results_b = load_eval_results(args.file_b)
    
    # 按 ID 对齐
    map_b = {r["id"]: r for r in results_b}
    pairs = []
    for ra in results_a:
        rb = map_b.get(ra["id"])
        if rb:
            pairs.append((ra, rb))
    
    pairs = pairs[:args.max_judge]
    logger.info(f"找到 {len(pairs)} 对可对比样本")
    
    # 盲测：随机打乱 A/B 顺序防止位置偏差
    import random
    random.seed(42)
    
    a_wins = 0
    b_wins = 0
    ties = 0
    a_scores = []
    b_scores = []
    
    with open(args.output, 'w', encoding='utf-8') as fout:
        for ra, rb in tqdm(pairs, desc="LLM 裁判评分"):
            # 随机决定 A/B 在 prompt 中的顺序（消除位置偏差）
            swap = random.random() > 0.5
            if swap:
                resp_first, resp_second = rb["response"], ra["response"]
            else:
                resp_first, resp_second = ra["response"], rb["response"]
            
            # 需要原始事实 —— 从评测数据中读取
            # (eval_baseline.py 没保存 fact，我们用 true_accusation 代替上下文)
            fact_text = f"（罪名: {ra['true_accusation']}）"
            
            verdict = judge_one(
                client, fact_text,
                ra["true_accusation"],
                f"{ra.get('true_months', '未知')} 个月",
                resp_first, resp_second,
            )
            
            if verdict:
                # 还原实际的 A/B
                winner = verdict.get("winner", "平局")
                if swap:
                    if winner == "A":
                        winner = "B"
                    elif winner == "B":
                        winner = "A"
                    verdict["judge_a"], verdict["judge_b"] = verdict["judge_b"], verdict["judge_a"]
                
                if winner == "A":
                    a_wins += 1
                elif winner == "B":
                    b_wins += 1
                else:
                    ties += 1
                
                score_a = verdict.get("judge_a", {}).get("总分", 0)
                score_b = verdict.get("judge_b", {}).get("总分", 0)
                a_scores.append(score_a)
                b_scores.append(score_b)
                
                fout.write(json.dumps({
                    "id": ra["id"],
                    "verdict": verdict,
                    "actual_winner": winner,
                }, ensure_ascii=False) + '\n')
    
    # 汇总
    logger.info("=" * 50)
    logger.info(f"LLM-as-Judge 盲测结果 ({len(pairs)} 对)")
    logger.info(f"  {args.label_a} 胜: {a_wins} ({a_wins/len(pairs)*100:.1f}%)")
    logger.info(f"  {args.label_b} 胜: {b_wins} ({b_wins/len(pairs)*100:.1f}%)")
    logger.info(f"  平局: {ties} ({ties/len(pairs)*100:.1f}%)")
    if a_scores:
        logger.info(f"  {args.label_a} 平均总分: {np.mean(a_scores):.2f}")
        logger.info(f"  {args.label_b} 平均总分: {np.mean(b_scores):.2f}")
    logger.info("=" * 50)


import numpy as np

if __name__ == "__main__":
    main()
