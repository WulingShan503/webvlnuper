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
- [ ] WebVLN-v1 数据集加载（8,960 / 1,262 / 4,603）
- [ ] 特征提取（ResNet152）
- [ ] rollout 采样与训练循环（AdamW, lr 1e-4, 200k iters）
- [ ] 评测指标（SR / OSR / SPL / TL / WUPS）

## 阶段 4：实验复现
- [ ] 5.2 基线复现实验
- [ ] 5.3 Top-k 消融
- [ ] 5.4 CR / RR 分析
- [ ] 5.5 两阶段有效性验证
