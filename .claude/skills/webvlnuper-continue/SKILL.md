---
name: webvlnuper-continue
description: Resume building the WebvlnUper thesis codebase (WebVLN baseline reproduction + LLM two-stage candidate screening). Use when continuing work on this repo, picking the next ROADMAP item, running its tests, or committing an increment. Triggers on "继续", "下一步", "接着做", "webvln", "筛选", "第三章", "第四章".
---

# WebvlnUper 续作流程

毕业论文《基于视觉感知的网页自动导航模型优化》的配套代码库。
论文在 `毕业论文.pdf`，基线在 `00105-AAAI24.ChenQ.pdf`。

## 工作约定（用户明确要求）

- **只写代码与文档**，不跑实验、不调真实 API。实验数据由用户自己产出。
- **小步推进**：一轮做完一个 ROADMAP 项即可，不要一次铺开太大工程量。
- 每轮结束：跑测试 → 更新 `ROADMAP.md` → 提交 → 更新记忆与本 skill。
- 提交信息用中文，标注对应论文章节。git 身份已配在仓库本地。

## 环境

这台机器**没有系统 Python**。用项目内的嵌入式解释器：

```bash
cd "D:/WebvlnUper" && .tools/py/python.exe -m pytest tests
```

`.tools/` 已在 `.gitignore` 中。装包用 `.tools/py/python.exe -m pip install <pkg>`。
直接 `-c` 执行脚本时需 `sys.path.insert(0,'.')`（embeddable 版忽略 `PYTHONPATH`）。

读论文只能用 `pdftotext -layout`；中文字形是 CID 内嵌字体提取不出来，
**只有英文、公式、表格数字、参考文献可读**，中文正文靠章节号与英文关键词定位。

## 开工步骤

1. 读 `ROADMAP.md`，找第一个未打勾项。
2. 若涉及官方数据格式，参考 `webvln/screening/adapter.py` 顶部的格式说明，
   或 shallow clone https://github.com/WebVLN/WebVLN 查 `r2r_src/`。
3. 写实现 + 单元测试，跑通全部测试。
4. 更新 `ROADMAP.md` 打勾、必要时更新 `README.md` 目录结构。
5. 提交，然后更新 `~/.claude/projects/D--WebvlnUper/memory/` 与本 skill。

## 代码风格

这份代码要给论文评审看，所以：

- 模块 docstring 开头标注论文小节号，如 `"""4.4 阶段一：规则过滤。"""`。
- 注释解释**为什么**这样设计，不要复述代码在做什么。
  例：`# 返回 UNKNOWN 而非默认 MAIN，是为了让区域剪枝对无证据的候选保持中立`
- 公式实现处引用论文编号，如 `式 (4.5.1)`、`式 (4.4.1)`。
- 中文注释与 docstring；标识符用英文。

## 已完成（第四章，核心贡献）

`webvln/screening/` 共 12 个模块，101 个测试全部通过：

| 模块 | 论文 | 要点 |
|---|---|---|
| `candidate.py` | 4.1 | `index` 全程不变，用于映射回特征张量的行 |
| `serializer.py` | 4.2 | 复刻论文模板，innerText 截断 100 字符 |
| `rule_filter.py` | 4.4 | 去重 / 区域剪枝 / href 黑名单 + 过滤至空的回填保护 |
| `llm_ranker.py` | 4.3 | 式 (4.3.2)，三层 JSON 解析回退 |
| `prompts.py` | 4.3 | 提示词模板 |
| `llm_backend.py` | — | `OpenAIBackend` / `ScriptedBackend`（测试替身） |
| `cache.py` | 5.5 | JSONL 持久化，对应 API 调用降低 52% |
| `metrics.py` | 4.5 | 式 (4.5.1) CR、RR |
| `pipeline.py` | 4.6 | 式 (4.4.1) 两阶段编排 |
| `area.py` | 4.4 | 区域推断，官方数据无标注，DOM > href > 锚文本 |
| `adapter.py` | — | 官方候选字典 → `Candidate` |
| `integration.py` | — | `screen_state`，插在 `make_candidate` 之前 |
| `config.py` | — | YAML 装配，支持单阶段对照与 Top-k 消融 |

## 已完成（第三章，基线复现）

`webvln/models/` 共 10 个模块。**本机无 torch，这些模块只做了静态校验**
（语法 + 内部导入可解析），没有运行时验证——用户已同意"只写代码不跑验证"。

| 模块 | 论文 | 要点 |
|---|---|---|
| `config.py` | — | 超参集中定义，附 `PAPER_CONFIG` 记录论文写法 |
| `language.py` | 3.2 | `[CLS] Q [SEP] D [SEP]`，语言只在初始化时编码一次 |
| `candidate_encoder.py` | 3.3 | 官方三 token 形式 + 论文 4864 拼接形式 |
| `attention.py` | 3.5 | forward 一律返回注意力分数（动作 logits 要用） |
| `cross_modal.py` | 3.4/3.5 | 状态递归；语言侧交叉注意力跳过第 0 位 |
| `action.py` | 3.6 | token(3n+1) → 动作(n+1) 归约、掩码、教师动作 |
| `answering.py` | 3.7 | 自回归解码，因果掩码注册为 buffer |
| `losses.py` | 3.8 | 式 (3.8.1)/(3.8.2)，导航损失按和累计再除 batch |
| `webvln_net.py` | 3.1 | 主模型串联，**不含 rollout 循环**（属训练逻辑） |

**与论文的四处差异**已记在 `[[webvln-official-model-vs-thesis]]` 与 README 的
"与官方实现的差异"表格里，实现以官方为准。

## 下一步：阶段 3 数据与训练

按 `ROADMAP.md`：数据集加载（8,960 / 1,262 / 4,603）→ ResNet152 特征提取
→ rollout 与训练循环 → 评测指标（SR / OSR / SPL / TL / WUPS）。

关键数字：AdamW lr 1e-5（官方）/ 1e-4（论文），weight decay 1e-2，
梯度裁剪 1.0，batch 4（官方）/ 8（论文），200,000 迭代，
140,000 后每 1,000 步验证；Best Score = SR + WUPS0.9；maxAction 10；
featdropout / dropout 均 0.4；feedback 用 `mix`（teacher 与 argmax 混合）。

数据路径：`shortest_paths.json`、`map.json`、`text_feats.pkl`、
`img_feats.pkl`、`screenshot_crop_feats.pkl`（官方 Google Drive 下载）。

rollout 关键逻辑在官方 `agent.py:253` 起，教师动作在 `_teacher_action`，
候选特征铺排在 `_candidate_variable`。

## 易错点

第四章（筛选）：
- `[EOA]` 停止动作与可点击候选并列，**不能**被筛掉。
- 官方 `text` 字段是**列表**不是字符串。
- 跳过 `[EOA]` 时不能让后续候选下标前移，否则与 `make_candidate` 行号错位。
- 解析失败的 LLM 结果**不入缓存**，否则该页面永久退化为不筛选。
- 不要就地修改模拟器的 `state['candidate']`，模拟器内部持有同一对象。

第三章（模型）：
- 注意力分数长度是 `3n+1`（每候选 3 个 token），动作空间是 `n+1`，
  必须按步长 3 归约，否则 argmax 落在错误候选上。
- 跨模态层的交叉注意力要传 `lang_feats[:, 1:, :]`（跳过第 0 位），
  第 0 位已被状态 token 占据，不跳过会让状态对自己做注意力。
- 导航损失用 `reduction="sum"` 跨步累计再除 batch；逐步取均值会摊薄
  长轨迹样本的权重。
- `length2mask` 返回 **True 表示 PAD**（与官方一致），语义反了会屏蔽掉
  所有真实候选。
- [EOA] 的特征是**全零**，其 logit 完全由注意力从状态 token 学得。
