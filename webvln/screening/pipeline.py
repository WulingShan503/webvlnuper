"""4.6 两阶段筛选流水线。

对应论文式 (4.4.1)：

    C_final = LLM-Rank(I, RuleFilter(C_t))

先用零成本规则把候选从约 45 压到约 22，再由 LLM 排序取 Top-k。
两阶段串联的意义在于：规则过滤承担了大部分压缩（51.1%）却不消耗 API，
使 LLM 只需处理更短的清单，论文报告提示词 token 从约 2250 降至约 1100。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from webvln.screening.candidate import Candidate
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.metrics import ScreeningMetrics
from webvln.screening.rule_filter import RuleFilter


@dataclass
class ScreeningOutput:
    """一次筛选的完整输出。

    Attributes:
        candidates: 最终保留的候选，按相关性降序。
        kept_indices: 对应的原始下标，供决策层索引特征张量。
        n_original: 筛选前的候选数。
        n_after_rule: 规则过滤后的候选数。
        from_cache: LLM 结果是否来自缓存。
    """

    candidates: List[Candidate]
    kept_indices: List[int]
    n_original: int
    n_after_rule: int
    from_cache: bool = False


@dataclass
class TwoStageScreener:
    """两阶段候选筛选器。

    Attributes:
        rule_filter: 阶段一。为 None 时跳过（用于 5.5 节的单阶段对照实验）。
        ranker: 阶段二。为 None 时仅做规则过滤。
        metrics: CR / RR 累计器。
    """

    rule_filter: Optional[RuleFilter] = field(default_factory=RuleFilter)
    ranker: Optional[LLMRanker] = None
    metrics: ScreeningMetrics = field(default_factory=ScreeningMetrics)

    def screen(
        self,
        instruction: str,
        candidates: Sequence[Candidate],
        target_index: Optional[int] = None,
    ) -> ScreeningOutput:
        """筛选一步的候选动作。

        Args:
            instruction: 导航指令（问题 Q 与辅助描述 D 的拼接）。
            candidates: 模拟器给出的原始候选。
            target_index: 教师动作下标，仅用于统计 RR，不参与筛选决策——
                筛选在推理时同样运行，不能依赖标注。

        Returns:
            ScreeningOutput。
        """
        n_original = len(candidates)

        after_rule = (
            list(self.rule_filter(candidates))
            if self.rule_filter is not None
            else list(candidates)
        )
        n_after_rule = len(after_rule)

        from_cache = False
        if self.ranker is not None and after_rule:
            result = self.ranker.rank(instruction, after_rule)
            from_cache = result.from_cache
            by_index = {c.index: c for c in after_rule}
            final = [by_index[i] for i in result.indices if i in by_index]
        else:
            final = after_rule

        kept_indices = [c.index for c in final]
        self.metrics.update(
            n_original=n_original,
            n_kept=len(final),
            target_index=target_index,
            kept_indices=kept_indices,
        )

        return ScreeningOutput(
            candidates=final,
            kept_indices=kept_indices,
            n_original=n_original,
            n_after_rule=n_after_rule,
            from_cache=from_cache,
        )

    def stats(self) -> dict:
        """汇总筛选与 LLM 调用统计。"""
        out = {"screening": self.metrics.as_dict()}
        if self.ranker is not None:
            out["ranker"] = self.ranker.stats()
        if self.rule_filter is not None:
            out["rule_filter_last"] = self.rule_filter.last_stats.as_dict()
        return out
