"""论文第五章报告的数字。

集中一处记录，供复现结果自动比对——手工核对四张表容易看错行。
数据全部取自论文正文与表格，来源在各常量的注释里标了表号。

**这些是论文的既定成果，不是本代码跑出来的**。代码的作用是佐证方法，
复现时的实际数字与此处有偏差属正常（随机种子、API 响应波动、
GPT-3.5-Turbo 版本变化都会影响结果）。
"""

from __future__ import annotations

from typing import Any, Dict, List

#: 表 5.1（第 37 页）：基线复现。WebVLN 原文 Val SR 39.46 / Test SR 34.76。
#: 三行分别为原文报告、官方权重复现、本文从头训练。
REFERENCE_BASELINE: Dict[str, Dict[str, float]] = {
    # Chen et al. AAAI 2024 表 2 的 "Ours" 行
    "paper_original": {
        "SR": 39.46, "OSR": 39.54, "SPL": 39.46, "WUPS0.9": 24.26, "WUPS0.0": 31.87,
    },
    # 用官方 best_val 权重评测
    "official_ckpt": {
        "SR": 39.22, "OSR": 39.22, "SPL": 39.22, "TL": 3.50,
        "WUPS0.9": 24.43, "WUPS0.0": 31.93,
    },
    # 本文从头训练 200,000 迭代
    "reproduced": {
        "SR": 38.35, "OSR": 38.35, "SPL": 38.30, "TL": 3.43,
        "WUPS0.9": 23.13, "WUPS0.0": 30.84,
    },
}

#: 表 5.2（第 38 页）：Top-k 消融。TL 越小越好，SR / SPL 越大越好。
REFERENCE_ABLATION: Dict[str, Dict[str, float]] = {
    "baseline":  {"val_SR": 38.35, "test_SR": 34.21, "SPL": 38.30, "TL": 3.99},
    "llm_top8":  {"val_SR": 38.67, "test_SR": 34.58, "SPL": 38.55, "TL": 3.85},
    "llm_top5":  {"val_SR": 39.12, "test_SR": 35.03, "SPL": 39.01, "TL": 3.72},
    "llm_top3":  {"val_SR": 38.90, "test_SR": 34.76, "SPL": 38.88, "TL": 3.65},
}

#: 表 5.3（第 39 页）：CR / RR。在 500 步样本上统计。
#: ``avg_candidates`` 为该配置下送入决策层的平均候选数。
REFERENCE_SCREENING: Dict[str, Dict[str, float]] = {
    "baseline": {"avg_candidates": 12.4, "CR": 0.0,  "RR": 100.0},
    "llm_top8": {"avg_candidates": 8.0,  "CR": 35.5, "RR": 96.8},
    "llm_top5": {"avg_candidates": 5.0,  "CR": 59.7, "RR": 93.5},
    "llm_top3": {"avg_candidates": 3.0,  "CR": 75.8, "RR": 88.2},
}

#: 4.4 / 5.4 节正文：阶段一自身的效果（在全量候选上统计，非表 5.3 的 500 步样本）。
REFERENCE_STAGE_ONE: Dict[str, float] = {
    "avg_before": 45.0,
    "avg_after": 22.0,
    "CR": 51.1,
    "RR": 98.2,
}

#: 表 5.4（第 40 页）：两阶段有效性。单阶段直接把 45 个候选交给 LLM。
#: ``api_cost`` 为相对开销，两阶段为单阶段的 0.48（即降低 52%）。
REFERENCE_TWO_STAGE: Dict[str, Dict[str, float]] = {
    "single_stage": {"SR": 39.71, "avg_candidates_to_llm": 45.0, "api_cost": 1.00},
    "two_stage":    {"SR": 39.67, "avg_candidates_to_llm": 22.0, "api_cost": 0.48},
}

#: 5.5 节正文：提示词 token 量的估算，用于解释 API 开销下降 52%。
REFERENCE_TOKEN_COST: Dict[str, float] = {
    "tokens_per_candidate": 50.0,
    "single_stage_tokens": 2250.0,  # 45 × 50
    "two_stage_tokens": 1100.0,     # 22 × 50
}


#: 论文内部前后不一致之处。复现时若发现数字对不上，先查这里。
#:
#: 记录它们不是要挑错，而是因为代码要给评审看：若代码默默按某一处的数字
#: 写死断言，另一处的读者会认为实现有误。
KNOWN_INCONSISTENCIES: List[Dict[str, Any]] = [
    {
        "topic": "Top-k 的 SR 数值",
        "table": "表 5.2（第 38 页）",
        "text": "5.3 节正文（第 38 页）",
        "detail": (
            "表 5.2 记 Val SR：top5 39.12 / top3 38.90 / top8 38.67；"
            "正文却说「k=5 时 Val SR 从 38.35% 提升至 39.67%」，"
            "并称 k=3 为 38.92%、k=8 为 39.12%。"
            "正文的 39.67 与表 5.4 单阶段/两阶段的 39.67 相同，"
            "疑为正文误引了表 5.4 的数字。"
        ),
        "adopted": "以表 5.2 为准（表格是结果的正式呈现）",
    },
    {
        "topic": "Test SR 数值",
        "table": "表 5.2",
        "text": "5.3 节正文",
        "detail": (
            "表 5.2 记 Test SR top5 35.03；正文写 35.12%（另一处又写 35.24%）。"
        ),
        "adopted": "以表 5.2 的 35.03 为准",
    },
    {
        "topic": "基线平均候选数",
        "table": "表 5.3 记 12.4",
        "text": "4.1 / 5.4 节正文记「平均约 45 个」",
        "detail": (
            "两者不矛盾但口径不同：45 是全量候选的平均，"
            "12.4 是表 5.3 那 500 步样本上、经阶段一之后的平均。"
            "CR 一栏可验证这一点：1 - 8/12.4 = 35.5%，"
            "1 - 5/12.4 = 59.7%，1 - 3/12.4 = 75.8%，与表中三个 CR 完全吻合，"
            "说明表 5.3 的 CR 是相对 12.4 而非 45 计算的。"
        ),
        "adopted": "CR 按表 5.3 的口径（相对 12.4）计算",
    },
    {
        "topic": "RR 与 CR 的行对应",
        "table": "表 5.3",
        "text": "5.4 节正文第 2、3 点",
        "detail": (
            "表 5.3 记 Top5 RR 93.5%；正文第 2 点说「Top-5 保留 88.9%」、"
            "第 3 点又说「Top5 达 89.3%」。三个数字互不相同。"
            "正文第 3 点的 Top3 76.5% / Top8 94.1% 也与表中 88.2 / 96.8 不符。"
        ),
        "adopted": "以表 5.3 为准",
    },
    {
        "topic": "学习率",
        "table": "—",
        "text": "3.9 / 5.1 节写 1e-4",
        "detail": "官方 run/train.bash 为 1e-5，论文报告的分数由官方代码产出。",
        "adopted": "代码用 1e-5，论文写法保留在 PAPER_CONFIG",
    },
]
