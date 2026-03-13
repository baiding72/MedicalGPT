# -*- coding: utf-8 -*-
"""
Step 5: 将蒸馏数据制作为 MedicalGPT 兼容的 SFT / DPO / GRPO 数据集
运行模式: 无卡
用法: python 5_make_datasets.py
"""
import json
import re
import os
import random
from collections import Counter
from tqdm import tqdm
from loguru import logger

# ============ 配置 ============
INPUT_FILE = "./data/distilled_r1.jsonl"
OUTPUT_DIR = "./data/output"

SFT_FILE = os.path.join(OUTPUT_DIR, "legal_sft.jsonl")
DPO_FILE = os.path.join(OUTPUT_DIR, "legal_dpo.jsonl")
GRPO_FILE = os.path.join(OUTPUT_DIR, "legal_grpo.jsonl")
EVAL_FILE = os.path.join(OUTPUT_DIR, "legal_eval.jsonl")

SFT_COUNT = 4500     # SFT 数据量
EVAL_COUNT = 500     # 评测集
GRPO_COUNT = 1000    # GRPO 数据量（从 SFT 中分出）

SYSTEM_PROMPT = '''你是一名资深刑事法官，请严格按以下要求分析案件：
1. 诊断犯罪类型（如有多个罪名，分点列出）
2. 详细说明定罪依据和量刑分析（分点说明）
返回严格符合以下JSON格式：
{
    "reasoning_content": "自然语言推理过程",
    "reason": "定罪依据和量刑分析分点说明",
    "accusation": "罪名",
    "sentence": "量刑建议"
}
注意：
- 使用标准法律术语
- 依据需结合案情中的事实、证据、量刑情节等要素
- 必须使用双引号，严格避免JSON格式错误'''


def load_distilled_data():
    """加载蒸馏结果"""
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 验证关键字段
            if not item.get("feature_content") or not item.get("reasoning_content"):
                continue
            if len(item["reasoning_content"]) < 50:
                continue
            data.append(item)
    
    logger.info(f"加载有效蒸馏数据: {len(data)} 条")
    return data


def make_sft_data(data):
    """制作 SFT 数据集（ShareGPT 格式，兼容 MedicalGPT）"""
    sft_data = []
    
    for item in data:
        # 构建 gpt 回复（JSON 字符串）
        gpt_response = json.dumps({
            "reasoning_content": item["reasoning_content"],
            "reason": item.get("reason", ""),
            "accusation": item.get("accusation", ""),
            "sentence": item.get("sentence", item.get("sentence_text", "")),
        }, ensure_ascii=False)
        
        sft_entry = {
            "conversations": [
                {"from": "human", "value": item["feature_content"]},
                {"from": "gpt", "value": gpt_response},
            ],
            "system": SYSTEM_PROMPT,
        }
        sft_data.append(sft_entry)
    
    return sft_data


def make_grpo_data(data):
    """制作 GRPO 数据集（question + answer 格式，兼容 MedicalGPT）"""
    grpo_data = []
    
    for item in data:
        # question = 原始案情事实
        # answer = 罪名 + 刑期
        accusation_str = item.get("accusation", "")
        if isinstance(item.get("original_accusation"), list):
            accusation_str = "、".join(item["original_accusation"])
        
        sentence_str = item.get("sentence_text", "")
        answer = f"{accusation_str}，{sentence_str}" if sentence_str else accusation_str
        
        # 用原始 fact 作为 question（更自然）
        question = item.get("original_fact", item["feature_content"])
        
        grpo_data.append({
            "question": question,
            "answer": answer,
        })
    
    return grpo_data


def make_dpo_placeholder(data):
    """制作 DPO 数据的 prompt 部分（rejected 需要 SFT 模型生成后补充）
    
    这里只生成 question + response_chosen，
    response_rejected 需要在 SFT 训练完成后用基座模型生成。
    """
    dpo_data = []
    
    for item in data:
        gpt_response = json.dumps({
            "reasoning_content": item["reasoning_content"],
            "reason": item.get("reason", ""),
            "accusation": item.get("accusation", ""),
            "sentence": item.get("sentence", item.get("sentence_text", "")),
        }, ensure_ascii=False)
        
        dpo_data.append({
            "question": item["feature_content"],
            "response_chosen": gpt_response,
            "response_rejected": "",  # 待 SFT 模型生成后填充
        })
    
    return dpo_data


def quality_check(sft_data):
    """数据质量检查"""
    errors = []
    
    for idx, item in enumerate(sft_data):
        try:
            response = item["conversations"][1]["value"]
            parsed = json.loads(response)
            
            if len(parsed.get("reasoning_content", "")) < 50:
                errors.append(f"样本 {idx}: 推理内容过短 ({len(parsed.get('reasoning_content', ''))} 字)")
            
            if not parsed.get("accusation"):
                errors.append(f"样本 {idx}: 缺少罪名")
                
        except json.JSONDecodeError as e:
            errors.append(f"样本 {idx}: JSON 解析失败 - {e}")
    
    if errors:
        logger.warning(f"发现 {len(errors)} 个质量问题:")
        for err in errors[:10]:
            logger.warning(f"  {err}")
        if len(errors) > 10:
            logger.warning(f"  ... 还有 {len(errors) - 10} 个")
    else:
        logger.info("质量检查通过，无异常")
    
    return errors


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    data = load_distilled_data()
    
    # 打乱
    random.seed(42)
    random.shuffle(data)
    
    # 分割
    eval_data = data[:EVAL_COUNT]
    train_data = data[EVAL_COUNT:]
    grpo_source = train_data[:GRPO_COUNT]
    
    logger.info(f"数据分割: 训练 {len(train_data)}, 评测 {len(eval_data)}, GRPO {len(grpo_source)}")
    
    # 制作各数据集
    sft_data = make_sft_data(train_data)
    grpo_data = make_grpo_data(grpo_source)
    dpo_data = make_dpo_placeholder(train_data[:5000])  # DPO 用前 5000 条
    eval_sft = make_sft_data(eval_data)
    
    # 质量检查
    quality_check(sft_data)
    
    # 保存 SFT
    with open(SFT_FILE, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"SFT 数据: {SFT_FILE} ({len(sft_data)} 条)")
    
    # 保存 GRPO
    with open(GRPO_FILE, 'w', encoding='utf-8') as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"GRPO 数据: {GRPO_FILE} ({len(grpo_data)} 条)")
    
    # 保存 DPO（不完整，需 SFT 后补充 rejected）
    with open(DPO_FILE, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"DPO 数据（待补充 rejected）: {DPO_FILE} ({len(dpo_data)} 条)")
    
    # 保存评测集
    with open(EVAL_FILE, 'w', encoding='utf-8') as f:
        for item in eval_sft:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"评测集: {EVAL_FILE} ({len(eval_sft)} 条)")
    
    # 统计
    acc_counter = Counter()
    for item in train_data:
        for acc in item.get("original_accusation", []):
            acc_counter[acc] += 1
    
    logger.info(f"\n========== 数据集统计 ==========")
    logger.info(f"SFT 训练集: {len(sft_data)} 条")
    logger.info(f"DPO 数据集: {len(dpo_data)} 条 (response_rejected 待填充)")
    logger.info(f"GRPO 数据集: {len(grpo_data)} 条")
    logger.info(f"评测集: {len(eval_sft)} 条")
    logger.info(f"覆盖罪名: {len(acc_counter)} 种")
    logger.info(f"Top-10 罪名: {acc_counter.most_common(10)}")
    logger.info(f"\n下一步:")
    logger.info(f"1. 将 {SFT_FILE} 复制到 MedicalGPT/data/finetune/")
    logger.info(f"2. 将 {GRPO_FILE} 复制到 MedicalGPT/data/grpo/")
    logger.info(f"3. 运行 SFT 训练")
    logger.info(f"4. SFT 完成后，用 SFT 模型生成 rejected，补充 {DPO_FILE}")
    logger.info(f"5. 运行 DPO 训练")
    logger.info(f"6. 运行 GRPO 训练")


if __name__ == "__main__":
    main()
