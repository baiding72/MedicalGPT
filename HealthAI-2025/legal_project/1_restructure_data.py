# -*- coding: utf-8 -*-
"""
Step 1: 调用 DeepSeek V3 将 CAIL2018 原始数据改写为结构化案情
运行模式: 无卡
用法: python 1_restructure_data.py

注意: 
- 需要设置环境变量 DEEPSEEK_API_KEY
- 使用 DeepSeek V3 API（兼容 OpenAI 格式）
- 支持断点续传（已处理的 ID 会跳过）
"""
import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # 自动读取当前目录的 .env 文件

# 之后 os.environ.get("DEEPSEEK_API_KEY") 就能拿到值了


# ============ 配置 ============
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"  # DeepSeek API 地址
MODEL = "deepseek-chat"  # DeepSeek V3

INPUT_FILE = "./data/cail2018_clean.jsonl"
OUTPUT_FILE = "./data/cail2018_restructured.jsonl"
MAX_WORKERS = 50  # 并发数（大幅提升，DeepSeek V3 通常支持较高并发）
MAX_SAMPLES = 20000  # 修改为 2万条底池（省钱高效方案）

# ============ Prompt ============
SYSTEM_PROMPT = "你是一名专业的法律助手，擅长分析刑事案件并提取结构化信息。"

USER_PROMPT_TEMPLATE = '''
请仔细阅读以下刑事案件的事实描述，完成以下任务：

1. 提取案件的结构化信息（被告人信息、犯罪行为、关键证据、量刑情节等）
2. 基于事实和所犯罪名，给出简要的定罪推理过程

<案件事实>
{fact}
</案件事实>

<涉及罪名>
{accusation}
</涉及罪名>

<判决结果>
{sentence}
</判决结果>

请严格按照以下 JSON 格式输出，不要输出其他内容：
```json
{{
    "feature_content": "被告人: [姓名/称呼]\\n性别: [提取值或未知]\\n年龄: [提取值或未知]\\n犯罪行为概述: [一句话概括]\\n作案时间: [提取值或未知]\\n作案地点: [提取值或未知]\\n关键证据: [分点列出]\\n量刑情节: [自首/坦白/从犯/累犯/退赃等，如有]\\n前科记录: [如有]",
    "reason": "1. [基于事实的定罪推理，分点说明]\\n2. [关键证据与犯罪构成要件的对应]\\n3. [量刑情节分析]",
    "accusation": "[罪名]",
    "sentence": "[判决结果]",
    "score": "[0-5的信息完整度评分]"
}}
```
'''


# ============ 初始化 ============
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging

# 只有在遇到异常时才重试，最多试 3 次，等待时间指数增加
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def call_llm(user_content):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=1500,
        timeout=30,
    )
    return response

def process_one(item):
    """处理单条数据"""
    try:
        accusation_str = "、".join(item["accusation"])
        
        user_content = USER_PROMPT_TEMPLATE.format(
            fact=item["fact"],
            accusation=accusation_str,
            sentence=item["sentence_text"],
        )
        
        response = call_llm(user_content)
        
        content = response.choices[0].message.content
        
        # 提取 JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return None
        
        result = json.loads(json_match.group())
        
        # 保留原始标签
        result["id"] = item["id"]
        result["original_fact"] = item["fact"]
        result["original_accusation"] = item["accusation"]
        result["original_articles"] = item["relevant_articles"]
        result["sentence_months"] = item["sentence_months"]
        result["sentence_text"] = item["sentence_text"]
        
        return result
        
    except Exception as e:
        logger.warning(f"处理 ID {item.get('id', '?')} 失败: {e}")
        return None


def main():
    if not API_KEY:
        logger.error("请设置 DEEPSEEK_API_KEY 环境变量")
        logger.error("export DEEPSEEK_API_KEY='你的key'")
        return
    
    # 流式读取，只取前 MAX_SAMPLES 条（不把整个文件读进内存）
    items = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= MAX_SAMPLES:
                break
    logger.info(f"读取 {len(items)} 条数据（MAX_SAMPLES={MAX_SAMPLES}）")
    
    # 断点续传：加载已处理的 ID
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
        logger.info(f"断点续传：已处理 {len(processed_ids)} 条")
    
    # 过滤已处理的
    items_to_process = [item for item in items if item["id"] not in processed_ids]
    logger.info(f"待处理: {len(items_to_process)} 条")
    
    if not items_to_process:
        logger.info("所有数据已处理完毕")
        return
    
    # 多线程处理
    success = 0
    fail = 0
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, item): item for item in items_to_process}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM 改写"):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    fout.flush()
                    success += 1
                else:
                    fail += 1
    
    logger.info(f"处理完成: 成功 {success}, 失败 {fail}")
    logger.info(f"结果保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
