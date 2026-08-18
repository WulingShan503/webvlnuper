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

## 下一步：第三章 WebVLN-Net 基线复现

按 `ROADMAP.md` 阶段 2，建议顺序：语言编码器 → 候选特征编码（4864 维）
→ 状态 token 递归 → 跨模态 Transformer（4 层 8 头）→ 动作预测 → 回答头 → 损失。

关键数字：BERT-base 12 层 768 维；候选特征 768+2048+2048=4864；
跨模态 Transformer 4 层 8 头，FFN 3072；AdamW lr 1e-4，weight decay 1e-2，
梯度裁剪 1.0，batch 8，200,000 迭代，140,000 后每 1,000 步验证；
Best Score = SR + WUPS0.9。数据集 8,960 / 1,262 / 4,603。

## 易错点

- `[EOA]` 停止动作与可点击候选并列，**不能**被筛掉。
- 官方 `text` 字段是**列表**不是字符串。
- 跳过 `[EOA]` 时不能让后续候选下标前移，否则与 `make_candidate` 行号错位。
- 解析失败的 LLM 结果**不入缓存**，否则该页面永久退化为不筛选。
- 不要就地修改模拟器的 `state['candidate']`，模拟器内部持有同一对象。
