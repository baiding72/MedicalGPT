# -*- coding: utf-8 -*-
"""
Step 4: 调用 DeepSeek R1 蒸馏推理过程
运行模式: 无卡
用法: python 4_distill_r1.py

将 Step 3 筛选出的 2 万条高质量数据发给 DeepSeek R1，
让 R1 扮演法官进行推理，保存其推理过程（reasoning_content）。
"""
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

# ============ 配置 ============
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-reasoner"  # DeepSeek R1（带推理能力）

INPUT_FILE = "./data/filtered_top5k.jsonl"  # 注意这里文件名变了，由 step3 决定
OUTPUT_FILE = "./data/distilled_r1.jsonl"
MAX_WORKERS = 10  # R1 推理较慢，并发不要太高
MAX_SAMPLES = 5000

SYSTEM_PROMPT = '''你是一名资深刑事法官，请严格按以下要求分析案件：
1. 根据案情事实，逐步推理出被告人应当承担的刑事责任
2. 明确指出构成何种犯罪及依据
3. 分析量刑情节（从重/从轻/减轻）
4. 给出量刑建议

返回严格符合以下 JSON 格式：
{
    "reason": "1. [定罪依据分点说明]\\n2. [量刑情节分析]\\n3. ...",
    "accusation": "罪名（如有多个用序号分列）",
    "sentence": "量刑建议"
}

注意：
- 使用标准法律术语
- 推理必须基于案情中的事实和证据
- 必须使用双引号，严格避免 JSON 格式错误
- 如涉及多个罪名，分别说明定罪依据'''


# ============ 初始化 ============
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_deepseek_with_retry(messages):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4000,
        timeout=120,
    )


def process_one(item):
    """处理单条数据"""
    try:
        feature = item.get("feature_content", "")
        if not feature:
            return None
            
        # 提取 Ground Truth (GT)
        gt_accusation = "、".join(item.get("original_accusation", []))
        gt_sentence = item.get("sentence_text", "")
        
        # 构造带有 GT 的 Prompt
        system_prompt = f'''你是一名资深刑事法官。我会给你【案情事实】以及该案最终的【真实判决结果】（罪名和量刑）。
请你严格根据这些已知信息，逆向推导出严密、符合逻辑的法庭说理过程（为什么会这么判）。

【已知真实判决结果如下，你的推理必须最终导向这个结论】：
最终罪名: {gt_accusation}
最终量刑: {gt_sentence}

请按以下结构分析案件：
1. 提取案件事实：简述关键事实
2. 定罪分析：针对已知罪名 {gt_accusation}，结合事实和刑法条文说明为什么构成本罪
3. 量刑分析：提取从重/从宽情节，分析为什么最终判决是 {gt_sentence}

返回严格符合以下 JSON 格式：
{{
    "reason": "1. [案件事实简述]\\n2. [定罪分析]\\n3. [量刑分析]",
    "accusation": "{gt_accusation}",
    "sentence": "{gt_sentence}"
}}

注意：
- 推理过程必须严丝合缝地支撑已知的最终判决结果，不要给出不同的罪名或量刑。
- 必须使用双引号，严格避免 JSON 格式错误。'''
        
        response = call_deepseek_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": feature},
        ])
        
        content = response.choices[0].message.content
        # R1 的推理过程在 reasoning_content 字段
        reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''
        
        # 解析 JSON 输出
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            # 如果输出不是 JSON，直接用原文
            result = {
                "reasoning_content": reasoning,
                "reason": content,
                "accusation": "、".join(item.get("original_accusation", [])),
                "sentence": item.get("sentence_text", ""),
            }
        else:
            result = json.loads(json_match.group())
            result["reasoning_content"] = reasoning
        
        # 保留元数据
        result["id"] = item["id"]
        result["feature_content"] = feature
        result["original_accusation"] = item.get("original_accusation", [])
        result["sentence_months"] = item.get("sentence_months", 0)
        result["sentence_text"] = item.get("sentence_text", "")
        result["original_fact"] = item.get("original_fact", "")
        
        return result
        
    except Exception as e:
        logger.warning(f"ID {item.get('id', '?')} 失败: {e}")
        return None


def main():
    if not API_KEY:
        logger.error("请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    # 读取输入
    items = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    items = items[:MAX_SAMPLES]
    logger.info(f"待蒸馏: {len(items)} 条")
    
    # 断点续传
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
        logger.info(f"断点续传: 已处理 {len(processed_ids)} 条")
    
    items = [item for item in items if item["id"] not in processed_ids]
    logger.info(f"剩余待处理: {len(items)} 条")
    
    if not items:
        logger.info("全部处理完毕")
        return
    
    success = 0
    fail = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, item): item for item in items}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="R1 蒸馏"):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fout.flush()
                    success += 1
                else:
                    fail += 1
    
    logger.info(f"完成: 成功 {success}, 失败 {fail}")
    logger.info(f"保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
