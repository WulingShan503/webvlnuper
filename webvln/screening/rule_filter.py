"""4.4 阶段一：规则过滤。

直接把全部候选交给 LLM 排序在代价上不可接受：平均 45 个候选、每个约 50 token，
单步提示词约 2250 token，乘以 200,000 次迭代的 rollout 后 API 开销无法承受。
因此先用零成本的确定性规则剔除明显无关的候选，再把剩余部分交给 LLM。

论文报告该阶段把平均候选数从 45 压缩到 22（压缩 51.1%），
同时保持 98.2% 的目标召回率——即规则误删目标的比例低于 2%。

三条规则：
    1. 去重：文本与 href 均相同的候选只保留首个
    2. 区域剪枝：剔除页脚、侧边栏等低价值区域的元素
    3. 关键词黑名单：href 中含广告 / 弹窗 / 追踪特征的链接
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

from webvln.screening.candidate import Candidate, PageArea

#: 默认剪枝的页面区域。论文 4.4 节点名 footer 与 sidebar：
#: 这两处几乎全是跨页面复用的通用链接（隐私政策、订阅入口等），
#: 与具体商品问答指令的相关性极低。导航栏虽同属高频区域，但常包含
#: 品类入口（如 "Men's Shoes"），对导航仍有价值，故默认保留。
DEFAULT_PRUNED_AREAS: Tuple[PageArea, ...] = (PageArea.FOOTER, PageArea.SIDEBAR)

#: href 黑名单关键词。论文 4.4 节列出 advertisement / popup / tracking 三类特征。
DEFAULT_BLOCKED_KEYWORDS: Tuple[str, ...] = (
    "advertisement",
    "popup",
    "tracking",
    "doubleclick",
    "utm_",
    "/ads/",
    "javascript:void",
)


@dataclass
class FilterStats:
    """一次过滤的统计信息，用于 5.4 节的压缩率分析与调参。"""

    n_input: int = 0
    n_output: int = 0
    n_dropped_duplicate: int = 0
    n_dropped_area: int = 0
    n_dropped_keyword: int = 0
    n_dropped_empty: int = 0

    @property
    def compression_ratio(self) -> float:
        """本阶段的压缩率，即被剔除候选占输入的比例。"""
        if self.n_input == 0:
            return 0.0
        return 1.0 - self.n_output / self.n_input

    def as_dict(self) -> dict:
        d = {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_dropped_duplicate": self.n_dropped_duplicate,
            "n_dropped_area": self.n_dropped_area,
            "n_dropped_keyword": self.n_dropped_keyword,
            "n_dropped_empty": self.n_dropped_empty,
        }
        d["compression_ratio"] = round(self.compression_ratio, 4)
        return d


@dataclass
class RuleFilter:
    """基于规则的候选过滤器，对应论文式 (4.4.1) 中的 ``RuleFilter(·)``。

    Attributes:
        drop_duplicates: 是否启用去重。
        pruned_areas: 需剔除的页面区域。
        blocked_keywords: href 黑名单关键词（小写匹配）。
        drop_empty: 是否剔除无文本、无上下文的候选。
        min_keep: 输出下界。若规则过于激进导致候选被清空，决策层将无动作可选，
            此时按原顺序回填至该数量——保证流水线不会因过滤而失效。
    """

    drop_duplicates: bool = True
    pruned_areas: Sequence[PageArea] = DEFAULT_PRUNED_AREAS
    blocked_keywords: Sequence[str] = DEFAULT_BLOCKED_KEYWORDS
    drop_empty: bool = True
    min_keep: int = 1
    _last_stats: FilterStats = field(default_factory=FilterStats, repr=False)

    def __call__(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """过滤候选列表，返回保留的子集（保持原有相对顺序）。"""
        kept, stats = self.filter_with_stats(candidates)
        self._last_stats = stats
        return kept

    def filter_with_stats(
        self, candidates: Sequence[Candidate]
    ) -> Tuple[List[Candidate], FilterStats]:
        """过滤并同时返回统计信息。"""
        stats = FilterStats(n_input=len(candidates))
        pruned = set(self.pruned_areas)
        keywords = [k.lower() for k in self.blocked_keywords]
        seen: Set[str] = set()
        kept: List[Candidate] = []

        for cand in candidates:
            if self.drop_empty and cand.is_empty:
                stats.n_dropped_empty += 1
                continue
            if self.drop_duplicates:
                key = cand.dedup_key()
                if key in seen:
                    stats.n_dropped_duplicate += 1
                    continue
                seen.add(key)
            if cand.area in pruned:
                stats.n_dropped_area += 1
                continue
            if _matches_any(cand.href, keywords):
                stats.n_dropped_keyword += 1
                continue
            kept.append(cand)

        if len(kept) < self.min_keep:
            kept = _backfill(kept, candidates, self.min_keep)

        stats.n_output = len(kept)
        return kept, stats

    @property
    def last_stats(self) -> FilterStats:
        """最近一次调用的统计信息。"""
        return self._last_stats


def _matches_any(href: str, keywords: Sequence[str]) -> bool:
    """href 是否命中任一黑名单关键词。"""
    if not href:
        return False
    lowered = href.lower()
    return any(k in lowered for k in keywords)


def _backfill(
    kept: List[Candidate], original: Sequence[Candidate], target: int
) -> List[Candidate]:
    """候选被过滤至空时，按原顺序回填到 ``target`` 个。"""
    kept_ids = {id(c) for c in kept}
    result = list(kept)
    for cand in original:
        if len(result) >= target:
            break
        if id(cand) not in kept_ids:
            result.append(cand)
    # 回填打乱了顺序，按原始下标恢复，保证决策层看到的候选顺序稳定。
    result.sort(key=lambda c: c.index)
    return result
