# 开发路线图

按论文章节顺序逐步搭建。每一步完成后单独提交。

## 阶段 0：项目骨架
- [x] README / ROADMAP / .gitignore / requirements
- [x] 包结构与配置系统

## 阶段 1：第四章 —— 候选动作筛选（核心贡献，优先实现）
- [x] 4.1 候选动作数据结构 (`Candidate`)
- [x] 4.2 候选序列化：结构化特征 → 语义文本描述
- [x] 4.4 阶段一：规则过滤（去重 / 区域剪枝 / 关键词黑名单）
- [x] 4.3 阶段二：LLM 语义排序 + Top-k 选择
- [x] 响应缓存
- [x] 4.5 CR / RR 指标
- [x] 4.6 两阶段流水线编排
- [x] 与模拟器候选格式对接（`adapter.py`，对齐官方 `map.json` 字段）
- [x] 页面区域推断（`area.py`，官方数据无区域标注）
- [x] 接入官方 `env.py` 的入口（`integration.py`）
- [x] YAML 配置装配（`config.py`）

## 阶段 2：第三章 —— WebVLN-Net 基线复现

> 已核对官方 `r2r_src/` 实现，四处与论文描述不一致（候选特征组织方式、
> 动作 logits 来源、层数、学习率），实现以官方为准并在注释中标注论文写法。
> 详见 `webvln/models/config.py` 与各模块 docstring。

- [x] 结构与训练超参集中定义（`config.py`，含 `PAPER_CONFIG` 对照）
- [x] 3.2 语言编码器（BERT-base，[CLS] Q [SEP] D [SEP]）
- [x] 3.3 候选特征编码（官方三 token 形式 + 论文 4864 拼接形式）
- [x] 3.4 状态 token 与递归更新（`cross_modal.py`）
- [x] 3.5 跨模态 Transformer（`attention.py` / `cross_modal.py`）
- [x] 3.6 动作预测（`action.py`，含 token→候选归约与教师动作）
- [x] 3.7 回答头（`answering.py`，自回归解码 + 贪心生成）
- [x] 3.8 损失函数（`losses.py`，式 3.8.1 / 3.8.2）
- [x] 主模型串联（`webvln_net.py`）
- [ ] 补齐依赖 torch 的单元测试（本机无 torch，待环境具备）

## 阶段 3：数据与训练
- [x] WebVLN-v1 数据集加载（8,960 / 1,262 / 4,603）
      - `episode.py` episode 数据结构（可转回官方字典格式）
      - `text.py` 指令 / 答案编码（对齐官方 maxInput 50、答案 40）
      - `dataset.py` 划分加载、`_enc.json` 缓存、批次回绕取样、规模自检
- [x] 特征提取（ResNet152）与导航图
      - `features.py` 三张官方 pkl 的统一查询 + `ResNet152Extractor` 重抽路径
      - `graph.py` `map.json` / `shortest_paths.json`、教师动作定位、候选规模自检
- [x] 导航环境与 rollout（`train/env.py`、`train/rollout.py`，均不依赖 torch）
      - 教师动作按最短路径重算，偏离 ground-truth 后仍有效
      - 筛选后教师动作重新对齐下标；目标被筛掉时落到 [EOA]
- [x] 训练循环（`train/trainer.py`、`train/batching.py`，AdamW / 200k iters）
      - mix 展开为 sample + teacher 两趟，损失累加后一次反传（官方做法）
      - 验证调度 140,000 后每 1,000 步，按 Best Score = SR + WUPS0.9 选模型
      - 张量构造需 torch，仅做静态校验；调度与长度计算已单测
- [x] 评测指标（SR / OSR / SPL / TL / WUPS）
      - `eval/metrics.py` 式 (5.1.1) SPL、OSR、Best Score = SR + WUPS0.9
      - `eval/wups.py` 词项集合版 + 官方逐字符版（论文数字由后者产出）

## 阶段 4：实验复现

> 实验由用户自行运行（需 torch 与 API key）。代码负责配置装配、
> 结果汇总与与论文数字的自动比对；论文数字集中在 `experiments/reference.py`。

- [x] 论文数字与内部不一致的记录（`reference.py`，含 5 处 KNOWN_INCONSISTENCIES）
- [x] 5.2 基线复现（`baseline_config`，关闭筛选）
- [x] 5.3 Top-k 消融（`screening_config_for_k`，k ∈ {3,5,8}）
- [x] 5.4 CR / RR 分析（`rule_filter_only_config` + `screening_summary_row`）
- [x] 5.5 两阶段有效性验证（`single_stage_config` 对照）
- [x] 实验驱动脚本（`runner.py`，`python -m webvln.experiments.runner`）
