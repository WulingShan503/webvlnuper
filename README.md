# webvlnuper

本项目针对多模态AI智能体在复杂网页环境中的决策效率问题，通过复现并改进WebVLN （视觉语言导航）基线模型，设计了一种基于大语言模型的两阶段候选动作筛选机制，有效压缩了高噪声的候选决策空间，提升了模型在复杂UI环境中的导航准确率与路径效率。

> 论文：《基于视觉感知的网页自动导航模型优化》（对外经济贸易大学，2026-3）
> 基线：Chen Q. et al. *WebVLN: Vision-and-Language Navigation on Websites*, AAAI 2024, 38(2): 1165-1173.

## 核心思路

WebVLN 基线模型在每一步决策时面对页面上全部可点击元素（平均约 45 个，最多可达 100 个）。
轨迹分析表明，模型倾向于点击导航栏、侧边栏中的高频通用链接，而非与指令语义最相关的内容链接，
即**候选噪声是性能的主要瓶颈**。

本项目提出两阶段候选动作筛选机制：

```
原始候选 (~45)
   │
   ├─ 阶段一：规则过滤 (RuleFilter)        去重 / 区域剪枝 / 关键词黑名单
   │                                       → ~22 个候选（压缩 51.1%）
   │
   ├─ 阶段二：LLM 语义排序 (LLMRanker)     候选文本化 → GPT-3.5-Turbo 相关性排序
   │                                       → Top-k 子集（k=5 最优）
   │
   └─ 送入 WebVLN-Net 决策层
```

## 主要结果

| 模型      | Val SR(%) | Test SR(%) | SPL(%) | TL   |
| --------- | --------- | ---------- | ------ | ---- |
| Baseline  | 38.35     | 34.21      | 38.30  | 3.99 |
| LLM-top8  | 38.67     | 34.58      | 38.55  | 3.85 |
| LLM-top5  | **39.12** | **35.03**  | 39.01  | 3.72 |
| LLM-top3  | 38.90     | 34.76      | 38.88  | 3.65 |

候选压缩率 (CR) 与召回保持率 (RR)：Top-5 在压缩 35.5% 候选的同时保持 93.5% 的召回。

## 目录结构

```
configs/                 实验配置（YAML）
webvln/
  screening/             第四章：两阶段候选动作筛选（本文核心贡献）
    serializer.py          4.2  结构化候选 → 语义文本描述
    rule_filter.py         4.4  阶段一：规则过滤
    llm_ranker.py          4.3  阶段二：LLM 语义排序 + Top-k
    cache.py               响应缓存（降低 API 调用 52%）
    pipeline.py            两阶段流水线编排
    metrics.py             4.5  CR / RR 指标
  models/                第三章：WebVLN-Net 基线复现
  data/                  WebVLN-v1 数据集加载与特征
  train/                 训练 / 评测循环
tests/                   单元测试
```

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

论文实验环境：Ubuntu 20.04 LTS，Python 3.8，PyTorch 1.13，NVIDIA RTX 3090 (24GB)。

## 状态

代码库正在按论文章节逐步搭建，当前进度见 [ROADMAP.md](ROADMAP.md)。
