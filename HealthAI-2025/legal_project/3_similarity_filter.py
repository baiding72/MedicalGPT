# -*- coding: utf-8 -*-
"""
Step 3: 相似度匹配 + MMR 多样性筛选
运行模式: 开卡（大矩阵运算，GPU 更快；也可 CPU）
用法: python 3_similarity_filter.py

创新点: 在原版 HealthAI 的 Top-K 排序基础上，增加 MMR 多样性去重，
确保筛选出的数据覆盖更多案件类型，避免某类案件过度集中。
"""
import json
import os
import numpy as np
from tqdm import tqdm
from loguru import logger

# ============ 配置 ============
RESTRUCTURED_VECTORS_FILE = "./data/restructured_vectors.npy"
RESTRUCTURED_IDS_FILE = "./data/restructured_ids.json"
TARGET_VECTORS_FILE = "./data/target_vectors.npy"

RESTRUCTURED_FILE = "./data/cail2018_restructured.jsonl"
OUTPUT_FILE = "./data/filtered_top5k.jsonl"

TOP_K_MATCH = 5       # 每条数据匹配目标集的 Top-K
TOP_N_SELECT = 5000   # 最终筛选的数据量（蒸馏的高优底池）
MMR_LAMBDA = 0.7      # MMR 参数: 0=纯多样性, 1=纯相关性


def load_data():
    """加载向量和原始数据"""
    logger.info("加载向量...")
    query_vectors = np.load(RESTRUCTURED_VECTORS_FILE)
    target_vectors = np.load(TARGET_VECTORS_FILE)
    
    with open(RESTRUCTURED_IDS_FILE, 'r') as f:
        query_ids = json.load(f)
    
    # 加载原始数据（按 ID 索引）
    id_to_data = {}
    with open(RESTRUCTURED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            id_to_data[item["id"]] = item
    
    logger.info(f"查询向量: {query_vectors.shape}, 目标向量: {target_vectors.shape}")
    return query_vectors, target_vectors, query_ids, id_to_data


def compute_similarity_scores(query_vectors, target_vectors):
    """计算每条查询数据与目标集的 Top-K 平均相似度"""
    logger.info("计算相似度矩阵...")
    
    # 向量已经 L2 归一化，dot product = cosine similarity
    BATCH_SIZE = 5000
    n_queries = len(query_vectors)
    avg_scores = np.zeros(n_queries, dtype=np.float32)
    
    for start in tqdm(range(0, n_queries, BATCH_SIZE), desc="相似度计算"):
        end = min(start + BATCH_SIZE, n_queries)
        batch = query_vectors[start:end]
        
        sim = np.dot(batch, target_vectors.T)  # (batch, n_targets)
        
        # Top-K 平均分
        k = min(TOP_K_MATCH, sim.shape[1])
        top_k_vals = np.partition(sim, -k, axis=1)[:, -k:]
        avg_scores[start:end] = top_k_vals.mean(axis=1)
    
    return avg_scores


def mmr_select(query_vectors, avg_scores, n_select, lambda_param=0.7):
    """MMR (Maximal Marginal Relevance) 筛选
    
    在保证高相似度的同时，增加数据多样性。
    
    score(i) = λ * relevance(i) - (1-λ) * max_similarity_to_selected(i)
    
    为了效率，先取 Top-3N 候选，再在候选中做 MMR。
    """
    n_candidates = min(n_select * 3, len(avg_scores))
    
    # 先取 Top-3N 候选
    candidate_indices = np.argsort(-avg_scores)[:n_candidates]
    candidate_vectors = query_vectors[candidate_indices]
    candidate_scores = avg_scores[candidate_indices]
    
    # 归一化分数到 [0, 1]
    score_min, score_max = candidate_scores.min(), candidate_scores.max()
    if score_max > score_min:
        normalized_scores = (candidate_scores - score_min) / (score_max - score_min)
    else:
        normalized_scores = np.ones_like(candidate_scores)
    
    logger.info(f"MMR 筛选: 从 {n_candidates} 个候选中选 {n_select} 个 (λ={lambda_param})")
    
    selected = []
    selected_vectors = []
    remaining = list(range(n_candidates))
    
    # 第一个：选相关性最高的
    first = np.argmax(normalized_scores)
    selected.append(candidate_indices[first])
    selected_vectors.append(candidate_vectors[first])
    remaining.remove(first)
    
    for _ in tqdm(range(n_select - 1), desc="MMR 筛选"):
        if not remaining:
            break
        
        remaining_arr = np.array(remaining)
        remaining_vectors = candidate_vectors[remaining_arr]
        remaining_relevance = normalized_scores[remaining_arr]
        
        # 计算与已选集合的最大相似度
        selected_matrix = np.array(selected_vectors)
        sim_to_selected = np.dot(remaining_vectors, selected_matrix.T)  # (n_remaining, n_selected)
        max_sim = sim_to_selected.max(axis=1)
        
        # MMR 分数
        mmr_scores = lambda_param * remaining_relevance - (1 - lambda_param) * max_sim
        
        # 选最高的
        best_idx = np.argmax(mmr_scores)
        best_remaining_idx = remaining_arr[best_idx]
        
        selected.append(candidate_indices[best_remaining_idx])
        selected_vectors.append(candidate_vectors[best_remaining_idx])
        remaining.remove(best_remaining_idx)
    
    return selected


def main():
    query_vectors, target_vectors, query_ids, id_to_data = load_data()
    
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"已存在 {OUTPUT_FILE}，跳过相似度计算和 MMR 筛选")
        # 直接从文件加载 MMR 结果
        results = []
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
        avg_scores = compute_similarity_scores(query_vectors, target_vectors)
    else:
        # 计算相似度分数
        avg_scores = compute_similarity_scores(query_vectors, target_vectors)
        
        # MMR 筛选
        selected_indices = mmr_select(query_vectors, avg_scores, TOP_N_SELECT, MMR_LAMBDA)
        
        # 输出
        results = []
        for idx in selected_indices:
            data_id = query_ids[idx]
            if data_id in id_to_data:
                item = id_to_data[data_id]
                item["similarity_score"] = float(avg_scores[idx])
                results.append(item)
        
        # 保存
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"保存到: {OUTPUT_FILE}")
    
    # 统计罪名分布
    from collections import Counter
    acc_counter = Counter()
    for item in results:
        for acc in item.get("original_accusation", []):
            acc_counter[acc] += 1
    
    logger.info(f"筛选结果: {len(results)} 条")
    logger.info(f"覆盖罪名: {len(acc_counter)} 种")
    logger.info(f"Top-10 罪名: {acc_counter.most_common(10)}")
    
    # ========== 多样性对比分析 ==========
    topk_indices = np.argsort(-avg_scores)[:TOP_N_SELECT]
    topk_acc = Counter()
    for idx in topk_indices:
        data_id = query_ids[idx]
        if data_id in id_to_data:
            for acc in id_to_data[data_id].get("original_accusation", []):
                topk_acc[acc] += 1
    
    def diversity_metrics(counter, label):
        """计算多样性指标"""
        counts = np.array(list(counter.values()), dtype=float)
        total = counts.sum()
        probs = counts / total
        
        # 1. Shannon 熵（越高 = 分布越均匀）
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts))  # 完全均匀时的最大熵
        evenness = entropy / max_entropy if max_entropy > 0 else 0  # Pielou 均匀度指数 [0,1]
        
        # 2. Gini 系数（越低 = 分布越均匀，0=完全平等，1=完全不平等）
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumulative = np.cumsum(sorted_counts)
        gini = 1 - 2 * cumulative.sum() / (n * total) + 1 / n
        
        # 3. 尾部覆盖：最少的 20% 罪名占了多少样本
        bottom_20_n = max(1, int(len(sorted_counts) * 0.2))
        bottom_20_share = sorted_counts[:bottom_20_n].sum() / total * 100
        
        # 4. 头部集中度：最多的 1 种罪名占比
        top1_share = counts.max() / total * 100
        
        # 5. 最小类别样本数
        min_count = int(counts.min())
        
        logger.info(f"[{label}] 覆盖罪名: {len(counter)} 种")
        logger.info(f"[{label}] Shannon 熵: {entropy:.4f} (最大={max_entropy:.4f}, 均匀度={evenness:.4f})")
        logger.info(f"[{label}] Gini 系数: {gini:.4f} (越低越均匀)")
        logger.info(f"[{label}] Top-1 罪名占比: {top1_share:.1f}%")
        logger.info(f"[{label}] 尾部 20% 罪名样本占比: {bottom_20_share:.2f}%")
        logger.info(f"[{label}] 最少类别样本数: {min_count}")
        
        return {"entropy": entropy, "evenness": evenness, "gini": gini, 
                "top1_share": top1_share, "bottom_20_share": bottom_20_share, "min_count": min_count}
    
    logger.info("=" * 60)
    logger.info("多样性对比分析: 纯 Top-K vs MMR")
    logger.info("=" * 60)
    topk_m = diversity_metrics(topk_acc, "Top-K")
    mmr_m = diversity_metrics(acc_counter, " MMR ")
    
    logger.info("-" * 60)
    logger.info(f"Shannon 熵提升: {mmr_m['entropy'] - topk_m['entropy']:+.4f}")
    logger.info(f"Gini 系数下降: {mmr_m['gini'] - topk_m['gini']:+.4f} (负数=更均匀)")
    logger.info(f"Top-1 集中度变化: {mmr_m['top1_share'] - topk_m['top1_share']:+.1f}%")
    logger.info(f"尾部 20% 罪名覆盖提升: {mmr_m['bottom_20_share'] - topk_m['bottom_20_share']:+.2f}%")
    logger.info("=" * 60)
    
    # ========== 类内语义多样性分析（核心验证） ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("类内语义多样性分析: 同一罪名下，案件之间的平均相似度")
    logger.info("(越低 = 案件越多样化，说明 MMR 有效做了语义去重)")
    logger.info("=" * 60)
    
    # 构建 ID → 向量索引 的映射
    id_to_vec_idx = {qid: i for i, qid in enumerate(query_ids)}
    
    # 构建 Top-K 的 ID 集合
    topk_id_set = set()
    for idx in topk_indices:
        topk_id_set.add(query_ids[idx])
    
    # 构建 MMR 的 ID 集合
    mmr_id_set = set()
    for item in results:
        mmr_id_set.add(item["id"])
    
    # 按罪名分组（取 Top-5 大罪名来分析，小罪名样本太少不具统计意义）
    top_accusations = [acc for acc, _ in acc_counter.most_common(5)]
    
    def compute_intra_sim(id_set, accusation):
        """计算某个罪名下被选中样本的平均内部相似度"""
        # 找到属于该罪名且被选中的样本的向量索引
        vec_indices = []
        for data_id in id_set:
            if data_id in id_to_data:
                accs = id_to_data[data_id].get("original_accusation", [])
                if accusation in accs and data_id in id_to_vec_idx:
                    vec_indices.append(id_to_vec_idx[data_id])
        
        if len(vec_indices) < 2:
            return None, len(vec_indices)
        
        # 随机采样最多 200 个（避免超大矩阵）
        if len(vec_indices) > 200:
            np.random.seed(42)
            vec_indices = list(np.random.choice(vec_indices, 200, replace=False))
        
        vecs = query_vectors[vec_indices]
        # 计算 pairwise cosine (已归一化，dot = cosine)
        sim_matrix = np.dot(vecs, vecs.T)
        # 取上三角（排除对角线的自身=1.0）
        n = len(vecs)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(upper_tri.mean()), len(vec_indices)
    
    logger.info(f"{'罪名':30s} | {'Top-K 类内相似度':>16s} | {'MMR 类内相似度':>14s} | {'差值':>8s}")
    logger.info("-" * 80)
    
    for acc in top_accusations:
        topk_sim, topk_n = compute_intra_sim(topk_id_set, acc)
        mmr_sim, mmr_n = compute_intra_sim(mmr_id_set, acc)
        
        if topk_sim is not None and mmr_sim is not None:
            diff = mmr_sim - topk_sim
            logger.info(f"{acc:30s} | {topk_sim:.4f} (n={topk_n:4d}) | {mmr_sim:.4f} (n={mmr_n:4d}) | {diff:+.4f}")
    
    logger.info("-" * 80)
    logger.info("差值为负 → MMR 选出的同类案件之间更不相似 → 语义去重有效")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
