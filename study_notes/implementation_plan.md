# 法律大模型全链路训练项目 — 实施计划

## 项目目标

基于 CAIL2018/2024 法律数据，利用 HealthAI-2025 的数据构造方法生成高质量训练数据，通过 MedicalGPT 的 SFT → DPO → GRPO 三阶段训练 Qwen2.5-7B，打造一个具备法律推理能力的垂直领域大模型。项目面向简历/实习，需兼顾工作量、效果和创新点。

## 整体流程

```
CAIL2018 (268万条刑事文书)
    ↓ Phase 1: 数据构造（HealthAI 方法）
高质量法律 SFT/DPO/GRPO 数据
    ↓ Phase 2: SFT 训练
    ↓ Phase 3: DPO 对齐
    ↓ Phase 4: GRPO 推理强化
    ↓ Phase 5: 评测对比
法律推理模型 + 实验报告
```

---

## Phase 1: 数据构造（3-4 天）

### 1.1 获取与预处理 CAIL2018 数据

- **数据源**: [CAIL2018-Small](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip)（~19.6 万条）或 HuggingFace 镜像
- **格式**: JSON，每行一条，含 `fact`, `meta.accusation`, `meta.relevant_articles`, `meta.term_of_imprisonment`
- **预处理**: 过滤空数据、去重、统一编码

### 1.2 Step 1: LLM 改写为结构化案情

- **改造** [1.批量推理构造数据集.py](file:///Users/baiding/Desktop/HealthAI-2025/1.批量推理构造数据集.py)
- **输入**: CAIL2018 的 `fact` + `meta`
- **API**: DeepSeek V3 API（替换智谱 GLM-4，更便宜）
- **输出格式**:
```json
{
  "feature_content": "案件类型: 刑事\n被告人: 张某\n罪行描述: ...\n关键证据: ...\n量刑情节: ...",
  "reason": "1. 被告人供述...\n2. 证据显示...",
  "accusation": "盗窃罪",
  "sentence": 12
}
```
- **数量**: 处理 ~10 万条（从 19.6 万中筛出有效数据）

### 1.3 Step 2: 向量化

- **改造** [2.向量化构造的数据集.py](file:///Users/baiding/Desktop/HealthAI-2025/2.向量化构造的数据集.py)
- **替换**: 智谱 Embedding → 本地 `BAAI/bge-large-zh-v1.5`（零成本）
- **向量化对象**: `feature_content` 字段
- **同时向量化目标集**: 从 CAIL2018 测试集中取 3000 条作为目标分布

### 1.4 Step 3: 相似度匹配与筛选

- **复用** [3.相似度匹配（矩阵计算法）.py](file:///Users/baiding/Desktop/HealthAI-2025/3.相似度匹配（矩阵计算法）.py)
- **创新点: MMR 多样性筛选** — 在 Top-K 排序后增加 MMR 去重，确保案件类型覆盖均衡
- **输出**: 筛选出 2 万条高质量样本

### 1.5 Step 4: DeepSeek R1 蒸馏

- **改造** [4.排序、筛选、批量推理.py](file:///Users/baiding/Desktop/HealthAI-2025/4.排序、筛选、批量推理.py)
- **API**: DeepSeek R1 批量推理
- **Prompt**: 要求 R1 扮演法官，输出推理过程 + 罪名判定 + 量刑依据
- **输出**: 2 万条带 `reasoning_content` 的数据

### 1.6 Step 5: 制作三种训练数据集

**SFT 数据**（改造 [5.制作微调数据.py](file:///Users/baiding/Desktop/HealthAI-2025/5.制作微调数据.py)）:
```json
{"conversations": [
  {"from": "human", "value": "案件类型: 刑事\n被告人: ..."},
  {"from": "gpt", "value": "{\"reasoning_content\": \"...\", \"accusation\": \"盗窃罪\", \"legal_basis\": \"...\", \"sentence\": \"有期徒刑1年\"}"}
], "system": "你是一名资深法官..."}
```
- 数量: ~1.8 万条（留 2000 条做评测）

**DPO 数据**:
```json
{
  "question": "案件类型: 刑事\n被告人: ...",
  "response_chosen": "(R1 蒸馏的高质量回答)",
  "response_rejected": "(SFT 模型或基座模型的低质量回答)"
}
```
- 数量: ~5000 条（SFT 训练完后用 SFT 模型生成 rejected）

**GRPO 数据**:
```json
{"question": "被告人王某于2019年3月盗窃他人财物价值3万元...", "answer": "盗窃罪，有期徒刑2年"}
```
- 数量: ~5000 条（从 CAIL2018 直接转换，question=fact, answer=accusation+sentence）

---

## Phase 2: SFT 训练（1 天）

### 环境与注意事项
- **平台**: AutoDL，RTX 3090 24GB × 1 或 H800
- **模型**: `Qwen/Qwen2.5-7B-Instruct`
- **⚠️ 避坑指南 (参考实战经验)**: AutoDL 系统盘通常只有 30GB，SFT 缓存很容易把系统盘占满导致训练崩溃。必须在环境变量中设置 `export HF_HOME=/root/autodl-tmp/hf_cache`，将模型和数据集缓存转移到数据盘。

### 训练命令
```bash
python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file_dir ./data/finetune \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --use_peft True --lora_rank 16 --lora_alpha 32 \
    --fp16 --gradient_checkpointing True \
    --template_name qwen \
    --output_dir outputs-sft-legal
```

### 产出
- LoRA 权重 → 合并为 `merged-sft-legal/`

---

## Phase 3: DPO 对齐（1 天）

### 构造 rejected 数据与避坑指南
- 用 SFT 模型对 5000 条 prompt 生成回答作为 rejected
- R1 蒸馏回答（带完整 `<think>`）作为 chosen
- 过滤质量差距不明显的对（创新点: 质量差距自动筛选）
- **⚠️ 避坑指南 (参考实战经验)**: 开源的偏好数据常存在“标签反转”（chosen 不如 rejected）的问题，导致 DPO 训练推理效果大幅下降。我们用 R1 重构 chosen 并且用自己的 SFT 输出作为 rejected，从根本上解决了这个痛点，避免白费算力。

### 训练命令
```bash
python dpo_training.py \
    --model_name_or_path ./merged-sft-legal \
    --template_name qwen \
    --train_file_dir ./data/reward \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --use_peft True --lora_rank 16 --lora_alpha 32 \
    --fp16 --gradient_checkpointing True \
    --output_dir outputs-dpo-legal
```

### 产出
- 合并为 `merged-dpo-legal/`

---

## Phase 4: GRPO 推理强化（1 天）

### 要点与避坑指南
- 用 CAIL2018 的 fact → accusation+sentence 作为 QA 对
- 自定义三个奖励函数（参考实战经验）：
  1. `accusation_reward()`: 罪名是否预测正确（语义相似度匹配或精确匹配）
  2. `format_reward()`: 输出是否符合 `<think>...</think><answer>...</answer>` 格式（格式规范奖励极为关键）
  3. `length_penalty()`或困惑度惩罚: 避免 reward hacking（模型通过无意义长篇大论骗高分）
- GRPO 不需要 RM，直接用规则奖励训练，并在训练过程中密切观察实时输出和 Eval 曲线，及时调整奖励权重。

### 训练命令
```bash
python grpo_training.py \
    --model_name_or_path ./merged-dpo-legal \
    --train_file_dir ./data/grpo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_steps 200 \
    --use_peft True --lora_rank 16 --lora_alpha 32 \
    --fp16 --gradient_checkpointing True \
    --output_dir outputs-grpo-legal
```

### 创新点
- 设计法律领域专用的奖励函数（罪名匹配 + 法条引用 + 格式合规）

---

## Phase 5: 评测对比（1 天）

### 评测维度

| 指标 | 方法 | 数据 |
|------|------|------|
| 罪名预测准确率 | 精确匹配 / F1 | CAIL2018 测试集 2000 条 |
| 量刑预测 MAE | 月数绝对误差 | 同上 |
| 推理质量 | LLM-as-Judge (GPT-4打分) | 抽样 200 条 |
| 格式合规率 | 正则检查 | 全量 |

### 对比实验表

| 模型 | 罪名 Acc | 刑期 MAE | 推理质量 | 格式合规 |
|------|---------|---------|---------|---------|
| Qwen2.5-7B-Instruct (基座) | - | - | - | - |
| + SFT (普通数据) | - | - | - | - |
| + SFT (蒸馏+筛选数据) | - | - | - | - |
| + DPO | - | - | - | - |
| + GRPO | - | - | - | - |

### 消融实验
- 数据量消融: 2K vs 5K vs 10K vs 20K 条 SFT 数据
- 筛选方法消融: 随机采样 vs Top-K vs MMR

---

## 三大创新点总结

1. **MMR 多样性数据筛选**: 在向量相似度排序基础上引入 MMR 算法，确保训练数据的案件类型覆盖均衡
2. **强弱模型差距驱动的 DPO 数据构造**: 利用 R1 与 SFT 模型的输出差异自动生成偏好对，设置质量差距阈值过滤噪声
3. **法律领域专用 GRPO 奖励函数**: 设计罪名匹配奖励 + 法条引用奖励 + 格式合规奖励的组合奖励机制

---

## 预算估算

| 项目 | 费用 |
|------|------|
| DeepSeek V3 API (10万条改写) | ¥30-60 |
| DeepSeek R1 API (2万条蒸馏) | ¥100-200 |
| AutoDL RTX 3090 (训练+评测 ~20h) | ¥30-40 |
| BGE 向量化 (本地) | ¥0 |
| **合计** | **¥160-300** |

---

## 时间线

| 天数 | 任务 |
|------|------|
| Day 1 | 下载 CAIL2018，改写 Step 1 脚本，提交批量改写任务 |
| Day 2 | 向量化 + 相似度筛选 + MMR 筛选 |
| Day 3 | 提交 R1 蒸馏任务，制作 SFT 数据 |
| Day 4 | SFT 训练 + 合并模型 |
| Day 5 | 生成 rejected 数据 + DPO 训练 |
| Day 6 | 制作 GRPO 数据 + GRPO 训练 |
| Day 7 | 全量评测 + 消融实验 + 撰写报告 |

---

## Verification Plan

### 数据质量验证
- 每个 Step 完成后抽样 10 条检查格式和内容
- SFT 数据: 运行 [validate_jsonl.py](file:///Users/baiding/Desktop/MedicalGPT/validate_jsonl.py) 验证格式
- DPO 数据: 检查 chosen/rejected 是否确实有质量差距
- GRPO 数据: 检查 question/answer 字段完整性

### 训练验证
- 每个训练阶段检查 loss 曲线是否下降
- SFT 后用 [inference.py](file:///Users/baiding/Desktop/MedicalGPT/inference.py) 手动测试 5 个法律问题
- DPO 后对比 SFT 模型和 DPO 模型对同一问题的回答
- GRPO 后检查模型是否学会了 `<think>` 格式输出

### 评测验证
- 在 2000 条测试集上运行自动评测脚本
- 用 LLM-as-Judge 抽样 200 条评分
- 生成最终对比表格
