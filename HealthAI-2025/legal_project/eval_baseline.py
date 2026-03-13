# -*- coding: utf-8 -*-
"""
评测脚本 A: 基座模型 Baseline 评测
用 CAIL2018 测试集评估模型的罪名预测准确率和刑期预测能力。
可同时用于评测 Base 模型、SFT 模型、DPO/GRPO 模型。

运行模式: 需要 GPU
用法: python eval_baseline.py --model_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct

依赖: pip install transformers torch tqdm loguru
"""
import os
import json
import re
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger

# ============ 配置 ============
EVAL_DATA = "./data/cail2018_target_3k.jsonl"  # 用 Step 0 分出来的 3K 目标集
MAX_EVAL = 500       # 评测多少条（控制时间）
MAX_NEW_TOKENS = 512 # 最大生成长度


SYSTEM_PROMPT = """你是一名资深刑事法官。请根据以下案件事实，给出你的判决结果。
请严格按照以下 JSON 格式输出，不要输出其他内容：
```json
{
    "accusation": "罪名",
    "sentence_months": 刑期月数(整数),
    "reasoning": "简要推理过程"
}
```"""


def load_model(model_path):
    """加载模型和 tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer


def generate_prediction(model, tokenizer, fact_text):
    """用模型生成一条预测"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"案件事实：\n{fact_text}"},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
    )
    
    # 只取生成的部分
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def parse_prediction(response):
    """从模型输出中提取罪名和刑期"""
    result = {"accusation": None, "sentence_months": None}
    
    # 尝试提取 JSON
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result["accusation"] = parsed.get("accusation", "").replace("罪", "")
            months = parsed.get("sentence_months")
            if months is not None:
                result["sentence_months"] = int(months)
            return result
        except (json.JSONDecodeError, ValueError):
            pass
    
    # JSON 解析失败，尝试用正则兜底
    acc_match = re.search(r'(?:罪名|指控|构成|犯)[：:]*\s*(.+?)(?:罪|[，,。\n])', response)
    if acc_match:
        result["accusation"] = acc_match.group(1).strip()
    
    months_match = re.search(r'(\d+)\s*(?:个月|月)', response)
    years_match = re.search(r'(\d+)\s*年', response)
    if months_match:
        result["sentence_months"] = int(months_match.group(1))
    elif years_match:
        result["sentence_months"] = int(years_match.group(1)) * 12
    
    return result


def evaluate(model_path, output_file=None):
    """主评测逻辑"""
    model, tokenizer = load_model(model_path)
    
    # 加载评测数据
    items = []
    with open(EVAL_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= MAX_EVAL:
                break
    logger.info(f"加载 {len(items)} 条评测数据")
    
    # 逐条评测
    results = []
    acc_correct = 0
    sentence_errors = []
    parse_fail = 0
    
    for item in tqdm(items, desc="评测中"):
        response = generate_prediction(model, tokenizer, item["fact"])
        pred = parse_prediction(response)
        
        # 罪名匹配（去掉"罪"字后比较）
        true_acc = item["accusation"][0] if item["accusation"] else ""
        true_acc_clean = true_acc.replace("罪", "")
        pred_acc_clean = (pred["accusation"] or "").replace("罪", "")
        
        acc_match = (pred_acc_clean in true_acc_clean) or (true_acc_clean in pred_acc_clean)
        if acc_match and pred_acc_clean:
            acc_correct += 1
        
        # 刑期误差
        true_months = item.get("sentence_months", None)
        pred_months = pred.get("sentence_months", None)
        if true_months is not None and pred_months is not None:
            sentence_errors.append(abs(true_months - pred_months))
        elif pred_months is None:
            parse_fail += 1
        
        results.append({
            "id": item.get("id"),
            "true_accusation": true_acc,
            "pred_accusation": pred.get("accusation"),
            "acc_correct": acc_match,
            "true_months": true_months,
            "pred_months": pred_months,
            "response": response[:500],  # 截断保存
        })
    
    # 汇总指标
    total = len(items)
    acc_rate = acc_correct / total * 100
    mae = np.mean(sentence_errors) if sentence_errors else float('nan')
    
    logger.info("=" * 50)
    logger.info(f"评测模型: {model_path}")
    logger.info(f"评测样本数: {total}")
    logger.info(f"罪名预测准确率: {acc_correct}/{total} = {acc_rate:.1f}%")
    logger.info(f"刑期预测 MAE: {mae:.1f} 个月 (有效样本 {len(sentence_errors)} 条)")
    logger.info(f"解析失败: {parse_fail} 条")
    logger.info("=" * 50)
    
    # 保存详细结果
    if output_file is None:
        model_name = os.path.basename(model_path.rstrip('/'))
        output_file = f"./eval_results_{model_name}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 第一行写汇总
        summary = {
            "model": model_path,
            "total": total,
            "accusation_accuracy": round(acc_rate, 2),
            "sentence_mae_months": round(mae, 2) if not np.isnan(mae) else None,
            "parse_fail_count": parse_fail,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    logger.info(f"详细结果保存到: {output_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法律大模型评测脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径，如 /root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--eval_data", type=str, default=EVAL_DATA,
                        help="评测数据路径")
    parser.add_argument("--max_eval", type=int, default=MAX_EVAL,
                        help="最多评测多少条")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径")
    
    args = parser.parse_args()
    EVAL_DATA = args.eval_data
    MAX_EVAL = args.max_eval
    
    evaluate(args.model_path, args.output)
