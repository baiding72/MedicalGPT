# -*- coding: utf-8 -*-
"""
Step 2: 使用本地 BGE 模型向量化 feature_content
运行模式: 开卡（需要 GPU）
用法: python 2_vectorize.py

替换了原版 HealthAI 的智谱 Embedding API，改用本地 BGE 模型（零成本）
"""
import json
import os
import numpy as np
from tqdm import tqdm
from loguru import logger

# ============ 配置 ============
RESTRUCTURED_FILE = "./data/cail2018_restructured.jsonl"   # Step 1 输出
TARGET_FILE = "./data/cail2018_target_3k.jsonl"            # Step 0 的目标集

RESTRUCTURED_VECTORS_FILE = "./data/restructured_vectors.npy"
RESTRUCTURED_IDS_FILE = "./data/restructured_ids.json"
TARGET_VECTORS_FILE = "./data/target_vectors.npy"
TARGET_IDS_FILE = "./data/target_ids.json"
TARGET_FEATURES_FILE = "./data/target_features.jsonl"       # 目标集也需要 feature_content
# 原来
# BGE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# 改成本地路径
BGE_MODEL_NAME = "/root/autodl-tmp/models/bge-large-zh-v1.5"

BATCH_SIZE = 64


def load_model():
    """加载 BGE 模型"""
    from sentence_transformers import SentenceTransformer
    logger.info(f"加载模型: {BGE_MODEL_NAME}")
    model = SentenceTransformer(BGE_MODEL_NAME)
    logger.info(f"模型加载完成，设备: {model.device}")
    return model


def vectorize_restructured(model):
    """向量化 Step 1 的结构化数据"""
    if os.path.exists(RESTRUCTURED_VECTORS_FILE):
        logger.info(f"已存在 {RESTRUCTURED_VECTORS_FILE}，跳过")
        return
    
    texts = []
    ids = []
    
    with open(RESTRUCTURED_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取结构化数据"):
            item = json.loads(line)
            feature = item.get("feature_content", "")
            if feature:
                texts.append(feature)
                ids.append(item["id"])
    
    logger.info(f"待向量化: {len(texts)} 条")
    
    # 批量编码
    vectors = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        normalize_embeddings=True,  # L2 归一化，后续直接 dot 就是 cosine
    )
    
    # 保存
    np.save(RESTRUCTURED_VECTORS_FILE, vectors)
    with open(RESTRUCTURED_IDS_FILE, 'w') as f:
        json.dump(ids, f)
    
    logger.info(f"向量保存到: {RESTRUCTURED_VECTORS_FILE}, shape={vectors.shape}")


def vectorize_target(model):
    """向量化目标集（用于 Step 3 匹配）
    
    目标集是 CAIL2018 的一个子集，代表"理想数据分布"。
    因为目标集没有经过 Step 1 改写，所以直接用 fact 字段向量化。
    """
    if os.path.exists(TARGET_VECTORS_FILE):
        logger.info(f"已存在 {TARGET_VECTORS_FILE}，跳过")
        return
    
    texts = []
    ids = []
    
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取目标集"):
            item = json.loads(line)
            # 目标集直接用 fact 字段
            text = item.get("fact", "")
            if text:
                texts.append(text)
                ids.append(item["id"])
    
    logger.info(f"目标集待向量化: {len(texts)} 条")
    
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    np.save(TARGET_VECTORS_FILE, vectors)
    with open(TARGET_IDS_FILE, 'w') as f:
        json.dump(ids, f)
    
    logger.info(f"目标向量保存到: {TARGET_VECTORS_FILE}, shape={vectors.shape}")


def main():
    # 检查依赖
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("请安装依赖: pip install sentence-transformers")
        return
    
    model = load_model()
    vectorize_restructured(model)
    vectorize_target(model)
    logger.info("向量化完成！")


if __name__ == "__main__":
    main()
