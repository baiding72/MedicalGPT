# -*- coding: utf-8 -*-
"""
Step 0: 预处理 CAIL2018 数据集
运行模式: 无卡
前置条件: 已通过 datasets 库下载并 save_to_disk 到 ./cail2018_data
用法: python 0_download_cail2018.py
"""
import json
import os
import random
from collections import Counter
from tqdm import tqdm
from loguru import logger
from datasets import load_from_disk

# ============ 配置 ============
DATASET_DIR = "./cail2018_data"          # save_to_disk 保存的路径
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cail2018_clean.jsonl")
TARGET_FILE = os.path.join(OUTPUT_DIR, "cail2018_target_3k.jsonl")  # 目标集（用于 Step 3 匹配）


# ============ 预处理 ============
def parse_record(data):
    """解析一条 CAIL2018 记录，统一为标准格式"""
    fact = data.get("fact", "")
    if not fact or len(fact) < 20:  # 过滤太短的
        return None

    # 提取标签（HuggingFace 版本字段已扁平化）
    accusations = data.get("accusation", [])
    articles = data.get("relevant_articles", [])
    criminals = data.get("criminals", [])
    money = data.get("punish_of_money", 0)
    imprisonment = data.get("imprisonment", 0)
    is_life = data.get("life_imprisonment", False)
    is_death = data.get("death_penalty", False)

    if not accusations:  # 没有罪名标注的跳过
        return None

    # 计算刑期（月）
    imprisonment_months = imprisonment if isinstance(imprisonment, (int, float)) else 0

    if is_death:
        sentence_text = "死刑"
        sentence_months = -2
    elif is_life:
        sentence_text = "无期徒刑"
        sentence_months = -1
    elif imprisonment_months > 0:
        years = int(imprisonment_months) // 12
        months = int(imprisonment_months) % 12
        if years > 0 and months > 0:
            sentence_text = f"有期徒刑{years}年{months}个月"
        elif years > 0:
            sentence_text = f"有期徒刑{years}年"
        elif months > 0:
            sentence_text = f"有期徒刑{months}个月"
        else:
            sentence_text = "免予刑事处罚"
        sentence_months = int(imprisonment_months)
    else:
        sentence_text = "免予刑事处罚"
        sentence_months = 0

    return {
        "fact": fact,
        "accusation": accusations,
        "relevant_articles": articles,
        "criminals": criminals,
        "sentence_text": sentence_text,
        "sentence_months": sentence_months,
        "punish_of_money": money,
    }


def preprocess():
    """从 save_to_disk 加载数据并预处理（流式处理，不占内存）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TEMP_FILE = os.path.join(OUTPUT_DIR, "_temp_all.jsonl")

    logger.info(f"正在从 {DATASET_DIR} 加载数据...")
    dataset = load_from_disk(DATASET_DIR)
    logger.info(f"数据集结构: {dataset}")

    # ===== 第一遍：流式解析，直接写文件（不存内存） =====
    splits_to_use = ["exercise_contest_train", "first_stage_train"]
    accusation_counter = Counter()
    sentence_dist = Counter()
    total_count = 0

    with open(TEMP_FILE, 'w', encoding='utf-8') as fout:
        for split_name in splits_to_use:
            if split_name not in dataset:
                logger.warning(f"跳过不存在的 split: {split_name}")
                continue
            split = dataset[split_name]
            n = len(split)
            logger.info(f"处理 {split_name}: {n} 条")

            # 按批次读取，比逐条迭代快很多
            BATCH = 10000
            for start in tqdm(range(0, n, BATCH), desc=f"解析 {split_name}", total=(n + BATCH - 1) // BATCH):
                end = min(start + BATCH, n)
                batch = split[start:end]  # HF Dataset 切片返回 dict of lists

                # 逐条处理这个 batch
                facts = batch.get("fact", [])
                accusations_list = batch.get("accusation", [])
                articles_list = batch.get("relevant_articles", [])
                criminals_list = batch.get("criminals", [])
                money_list = batch.get("punish_of_money", [0] * (end - start))
                imprisonment_list = batch.get("imprisonment", [0] * (end - start))
                life_list = batch.get("life_imprisonment", [False] * (end - start))
                death_list = batch.get("death_penalty", [False] * (end - start))

                for i in range(len(facts)):
                    record = {
                        "fact": facts[i],
                        "accusation": accusations_list[i] if i < len(accusations_list) else [],
                        "relevant_articles": articles_list[i] if i < len(articles_list) else [],
                        "criminals": criminals_list[i] if i < len(criminals_list) else [],
                        "punish_of_money": money_list[i] if i < len(money_list) else 0,
                        "imprisonment": imprisonment_list[i] if i < len(imprisonment_list) else 0,
                        "life_imprisonment": life_list[i] if i < len(life_list) else False,
                        "death_penalty": death_list[i] if i < len(death_list) else False,
                    }
                    parsed = parse_record(record)
                    if parsed:
                        parsed["id"] = total_count
                        fout.write(json.dumps(parsed, ensure_ascii=False) + '\n')
                        total_count += 1
                        for acc in parsed["accusation"]:
                            accusation_counter[acc] += 1
                        # 刑期统计
                        sm = parsed["sentence_months"]
                        if sm == -2: sentence_dist["死刑"] += 1
                        elif sm == -1: sentence_dist["无期徒刑"] += 1
                        elif sm == 0: sentence_dist["免予刑事处罚"] += 1
                        elif sm <= 12: sentence_dist["1年以下"] += 1
                        elif sm <= 36: sentence_dist["1-3年"] += 1
                        elif sm <= 120: sentence_dist["3-10年"] += 1
                        else: sentence_dist["10年以上"] += 1

    logger.info(f"有效数据: {total_count} 条")
    logger.info(f"罪名种类: {len(accusation_counter)} 种")
    logger.info(f"Top-10 罪名: {accusation_counter.most_common(10)}")
    logger.info(f"刑期分布: {dict(sentence_dist)}")

    # ===== 第二遍：用水塘抽样选 3000 条做目标集，其余做主数据集 =====
    logger.info("正在分割目标集 (3000条) 和主数据集...")

    # 水塘抽样选 3000 个行号
    random.seed(42)
    target_indices = set()
    reservoir = []
    K = 3000
    with open(TEMP_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < K:
                reservoir.append(i)
            else:
                j = random.randint(0, i)
                if j < K:
                    reservoir[j] = i
    target_indices = set(reservoir)
    logger.info(f"水塘抽样完成，选中 {len(target_indices)} 条作为目标集")

    # 扫第二遍，分流写入
    with open(TEMP_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_main, \
         open(TARGET_FILE, 'w', encoding='utf-8') as f_target:
        for i, line in enumerate(tqdm(fin, total=total_count, desc="分割数据")):
            if i in target_indices:
                f_target.write(line)
            else:
                f_main.write(line)

    logger.info(f"主数据集: {OUTPUT_FILE} ({total_count - len(target_indices)} 条)")
    logger.info(f"目标集: {TARGET_FILE} ({len(target_indices)} 条)")

    # 清理临时文件
    os.remove(TEMP_FILE)
    logger.info("临时文件已清理")


if __name__ == "__main__":
    preprocess()
