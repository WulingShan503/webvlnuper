"""第四章：基于大语言模型的候选动作筛选优化。

WebVLN 基线在每步决策时面对页面上全部可点击元素（平均约 45 个，最多 100 个），
其中大量为导航栏、页脚等与指令无关的通用链接。本模块通过两阶段筛选压缩决策空间：

    阶段一 (rule_filter)  规则过滤：去重、区域剪枝、关键词黑名单
    阶段二 (llm_ranker)   LLM 语义排序：候选文本化后按与指令的相关性排序，取 Top-k
"""

from webvln.screening.cache import RankCache
from webvln.screening.candidate import Candidate, ElementType, PageArea
from webvln.screening.llm_backend import OpenAIBackend, ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker, RankResult
from webvln.screening.metrics import ScreeningMetrics, compression_ratio
from webvln.screening.pipeline import ScreeningOutput, TwoStageScreener
from webvln.screening.rule_filter import FilterStats, RuleFilter
from webvln.screening.serializer import serialize, serialize_all

__all__ = [
    "Candidate",
    "ElementType",
    "PageArea",
    "serialize",
    "serialize_all",
    "RuleFilter",
    "FilterStats",
    "LLMRanker",
    "RankResult",
    "RankCache",
    "OpenAIBackend",
    "ScriptedBackend",
    "ScreeningMetrics",
    "compression_ratio",
    "TwoStageScreener",
    "ScreeningOutput",
]
