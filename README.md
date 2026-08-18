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
    candidate.py           4.1  候选动作数据结构
    serializer.py          4.2  结构化候选 → 语义文本描述
    rule_filter.py         4.4  阶段一：规则过滤
    llm_ranker.py          4.3  阶段二：LLM 语义排序 + Top-k
    prompts.py             4.3  提示词模板
    llm_backend.py         LLM 调用后端（OpenAI / 测试替身）
    cache.py               响应缓存（降低 API 调用 52%）
    metrics.py             4.5  CR / RR 指标
    pipeline.py            4.6  两阶段流水线编排
    area.py                页面区域推断（官方数据无区域标注）
    adapter.py             模拟器候选格式对接
    integration.py         接入官方 env.py 的入口
    config.py              YAML 配置装配
  models/                第三章：WebVLN-Net 基线复现
    config.py              超参集中定义（附 PAPER_CONFIG 对照）
    language.py            3.2  语言编码器
    candidate_encoder.py   3.3  候选特征编码
    attention.py           3.5  注意力构件
    cross_modal.py         3.4/3.5 状态递归与跨模态层
    action.py              3.6  动作预测
    answering.py           3.7  回答头
    losses.py              3.8  损失函数
    webvln_net.py          主模型串联
  data/                  WebVLN-v1 数据集加载与特征
  train/                 训练 / 评测循环
tests/                   单元测试（101 个）
```

## 接入基线模型

筛选须插在官方 `r2r_src/env.py` 的 `make_candidate` **之前**——该函数会为每个
候选查三张特征表并拼成 4864 维向量，先筛选才能省下被剔除候选的计算。
在 `R2RBatch._get_obs` 中改一行即可：

```python
from webvln.screening import build_screener, screen_state

screener = build_screener(path="configs/screening.yaml")   # __init__ 中构建一次

for i, (feature, state) in enumerate(self.env.getStates()):
    item = self.batch[i]
    state = screen_state(state, item, screener)            # 新增此行
    candidate_feature = self.make_candidate(feature, state)
```

`screen_state` 返回结构相同、仅候选变少的 state，`make_candidate` 及其下游无需改动。
`[EOA]` 停止动作始终保留，不参与筛选。设 `screening.enabled: false` 即退回基线路径。

单独使用筛选模块：

```python
from webvln.screening import Candidate, ElementType, PageArea, TwoStageScreener

screener = TwoStageScreener(ranker=...)
out = screener.screen("What material is this shirt made of?", candidates)
print(out.kept_indices)          # 保留候选的原始下标
print(screener.stats())          # CR / RR / API 调用与缓存命中
```

## 测试

```bash
pip install pytest pyyaml
python -m pytest tests
```

第四章筛选模块与第三章配置共 110 个测试，不依赖 torch 与 API key 即可运行。
第三章其余模块（跨模态层、回答头、损失）需 torch，相应测试待补。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

论文实验环境：Ubuntu 20.04 LTS，Python 3.8，PyTorch 1.13，NVIDIA RTX 3090 (24GB)。

## 与官方实现的差异

复现第三章时核对了官方 [WebVLN](https://github.com/WebVLN/WebVLN) 的 `r2r_src/`，
发现四处与论文正文描述不一致。代码**以官方实现为准**（论文报告的 SR / SPL 由该代码产出），
论文写法保留在注释与 `PAPER_CONFIG` 中：

| 项 | 论文 | 官方实现 |
| --- | --- | --- |
| 候选特征 | 768+2048+2048=4864 维拼接 | 三段各占一个 token，每段 512 维 |
| 动作 logits | softmax over M 候选 | 跨模态注意力分数对头取平均 |
| 层数 | 跨模态 4 层 / 回答头 4 层 | `vl_layers=2` / `qa_layers=2` |
| 学习率 | 1e-4 | 1e-5（`run/train.bash`） |

候选特征的组织方式带来一处实现细节：注意力分数长度为 `3n+1`，
而动作空间为 `n+1`，两者需按步长 3 归约（见 `action.py:pool_token_logits`）。

## 状态

代码库正在按论文章节逐步搭建，当前进度见 [ROADMAP.md](ROADMAP.md)。
