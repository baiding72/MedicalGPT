# -*- coding: utf-8 -*-
"""
评测脚本 C: 汇总所有评测结果，生成最终对比报告
将 eval_baseline.py 多次运行的结果汇总到一张表里。

用法: python eval_summary.py
"""
import json
import os
import glob
from loguru import logger


def load_summary(filepath):
    """读取评测结果文件的第一行（汇总行）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        return json.loads(first_line)


def main():
    # 自动发现所有评测结果文件
    result_files = sorted(glob.glob("./eval_results_*.jsonl"))
    
    if not result_files:
        logger.error("未找到任何 eval_results_*.jsonl 文件")
        logger.info("请先运行 eval_baseline.py 生成评测结果")
        return
    
    # 汇总
    print("\n" + "=" * 80)
    print(f"{'模型':40s} | {'罪名准确率':>10s} | {'刑期MAE(月)':>10s} | {'解析失败':>8s}")
    print("-" * 80)
    
    for f in result_files:
        try:
            summary = load_summary(f)
            model_name = os.path.basename(summary.get("model", f))
            acc = summary.get("accusation_accuracy", "N/A")
            mae = summary.get("sentence_mae_months", "N/A")
            fails = summary.get("parse_fail_count", "N/A")
            
            acc_str = f"{acc}%" if isinstance(acc, (int, float)) else str(acc)
            mae_str = f"{mae}" if isinstance(mae, (int, float)) else str(mae)
            
            print(f"{model_name:40s} | {acc_str:>10s} | {mae_str:>10s} | {fails:>8s}")
        except Exception as e:
            logger.warning(f"读取 {f} 失败: {e}")
    
    print("=" * 80)
    
    # Judge 结果
    judge_file = "./eval_judge_results.jsonl"
    if os.path.exists(judge_file):
        print("\n[LLM-as-Judge 盲测结果]")
        with open(judge_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        a_wins = sum(1 for l in lines if '"actual_winner": "A"' in l)
        b_wins = sum(1 for l in lines if '"actual_winner": "B"' in l)
        ties = len(lines) - a_wins - b_wins
        print(f"  模型A 胜: {a_wins}, 模型B 胜: {b_wins}, 平局: {ties}")
    
    print()


if __name__ == "__main__":
    main()
