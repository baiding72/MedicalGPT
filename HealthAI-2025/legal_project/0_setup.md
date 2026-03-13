# AutoDL 环境配置指南

## 策略

| 步骤 | 模式 | 原因 |
|------|------|------|
| Step 0: 下载 CAIL2018 | 无卡 | 纯下载 |
| Step 1: DeepSeek V3 改写 | 无卡 | 只发 API 请求 |
| **Step 2: BGE 向量化** | **开卡** | 需要 GPU |
| **Step 3: 相似度匹配** | **开卡** | 大矩阵运算（也可 CPU，但 GPU 更快） |
| Step 4: R1 蒸馏 | 无卡 | 只发 API 请求 |
| Step 5: 制作数据集 | 无卡 | 纯 JSON 处理 |
| 训练 (SFT/DPO/GRPO) | **开卡** | 需要 GPU |

## 第一步：创建 AutoDL 实例

1. 登录 https://www.autodl.com
2. 创建实例：
   - **镜像**: `PyTorch 2.1.0` + `Python 3.10` + `CUDA 12.1`
   - **GPU**: 先选"无卡模式"（做 Step 0-1）
   - **数据盘**: ≥ 50GB（数据 + 模型全放这里，关机不丢）
3. 创建后，复制 SSH 登录命令

## 第二步：SSH 登录 & 初始环境

```bash
# SSH 登录（用 AutoDL 给你的命令）
ssh -p 端口号 root@region-xxx.autodl.pro

# 切到数据盘（关机不丢数据）
cd /root/autodl-tmp

# 克隆项目
git clone https://github.com/yuandaxia2001/HealthAI-2025.git
cd HealthAI-2025

# 创建法律项目目录
mkdir -p legal_project/data

# 安装基础依赖（无卡模式也能装）
pip install tqdm loguru requests openai numpy
```

## 第三步：下载 CAIL2018 数据集

```bash
cd /root/autodl-tmp/HealthAI-2025/legal_project/data

# 方法 1：从 HuggingFace 下载（推荐）
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('thunlp/PromptCBLUE', 'CAIL-2018')
# 如果上面不行，试：
# ds = load_dataset('china-ai-law-challenge/cail2018')
print(ds)
"

# 方法 2：直接下载 JSON 文件
# CAIL2018-Small（~196K 条，小的够用）
wget https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip
unzip CAIL2018_ALL_DATA.zip

# 如果上面链接失效，用 GitHub:
git clone --depth 1 https://github.com/thunlp/CAIL.git
ls CAIL/
```

> ⚠️ 如果下载遇到问题，AutoDL 自带学术资源加速：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 第四步：配置 API Key

```bash
# 编辑环境变量
echo 'export DEEPSEEK_API_KEY="你的key"' >> ~/.bashrc
source ~/.bashrc
```

DeepSeek API Key 获取：https://platform.deepseek.com/api_keys

## 第五步：切换开卡/无卡模式

- **无卡 → 开卡**：AutoDL 控制台 → 实例 → "更换配置" → 选择 GPU
- **开卡 → 无卡**：AutoDL 控制台 → 实例 → "更换配置" → 选择"无卡模式"
- 切换后实例会重启，但 `/root/autodl-tmp` 数据不丢

## 执行顺序

```
[无卡] python 0_download_cail2018.py    # 下载数据
[无卡] python 1_restructure_data.py      # LLM 改写（发 API）
[开卡] python 2_vectorize.py             # BGE 向量化
[开卡] python 3_similarity_filter.py     # 相似度匹配 + MMR
[无卡] python 4_distill_r1.py            # R1 蒸馏（发 API）
[无卡] python 5_make_datasets.py         # 制作 SFT/DPO/GRPO 数据
[开卡] 训练 ...                          # SFT/DPO/GRPO
```
