"""第五章：实验复现。

把论文报告的数字集中为可比对的数据（``reference.py``），
并提供各实验的驱动逻辑（``ablation.py``）：

    5.2  基线复现        对照 WebVLN 原文与本文复现的差距
    5.3  Top-k 消融      k ∈ {3, 5, 8} 的 SR / SPL / TL
    5.4  CR / RR 分析    压缩率与召回保持率的权衡
    5.5  两阶段有效性    单阶段（仅 LLM）与两阶段的 SR 与 API 开销对比

运行实验需要 torch 与 API key，由用户自行执行；本模块只负责
配置装配、结果汇总与与论文数字的比对。
"""

from webvln.experiments.ablation import (
    TOPK_VALUES,
    ExperimentResult,
    baseline_config,
    compare_with_reference,
    format_table,
    rule_filter_only_config,
    screening_config_for_k,
    screening_summary_row,
    single_stage_config,
)
from webvln.experiments.reference import (
    KNOWN_INCONSISTENCIES,
    REFERENCE_ABLATION,
    REFERENCE_BASELINE,
    REFERENCE_SCREENING,
    REFERENCE_STAGE_ONE,
    REFERENCE_TOKEN_COST,
    REFERENCE_TWO_STAGE,
)

__all__ = [
    "ExperimentResult",
    "KNOWN_INCONSISTENCIES",
    "REFERENCE_ABLATION",
    "REFERENCE_BASELINE",
    "REFERENCE_SCREENING",
    "REFERENCE_STAGE_ONE",
    "REFERENCE_TOKEN_COST",
    "REFERENCE_TWO_STAGE",
    "TOPK_VALUES",
    "baseline_config",
    "compare_with_reference",
    "format_table",
    "rule_filter_only_config",
    "screening_config_for_k",
    "screening_summary_row",
    "single_stage_config",
]
