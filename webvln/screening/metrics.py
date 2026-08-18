"""4.5 节：筛选机制的评价指标。

导航指标（SR / SPL / TL）衡量端到端性能，但无法说明筛选本身的行为，
因此论文引入两个直接刻画筛选质量的指标：

    CR (Compression Ratio)  压缩率，式 (4.5.1)：CR = 1 - k / n
    RR (Recall Retention)   召回保持率：目标动作仍留在筛选结果中的比例

两者构成权衡：CR 越高决策空间越小，但过度压缩会把目标动作一并删除，
使 RR 下降。论文 5.4 节报告 Top-5 在 CR 35.5% 时保持 RR 93.5%。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


def compression_ratio(n_original: int, n_kept: int) -> float:
    """式 (4.5.1)：CR = 1 - k / n。

    Args:
        n_original: 原始候选数 n。
        n_kept: 筛选后保留数 k。

    Returns:
        压缩率，取值 [0, 1]。n 为 0 时定义为 0（无候选可压缩）。
    """
    if n_original <= 0:
        return 0.0
    return 1.0 - n_kept / n_original


@dataclass
class ScreeningMetrics:
    """跨步累计的筛选指标。

    指标按「步」累计而非按 episode：筛选发生在每一步决策上，
    以步为单位统计才能反映 rollout 中每次压缩的真实平均水平。
    """

    n_steps: int = 0
    sum_original: int = 0
    sum_kept: int = 0
    n_steps_with_target: int = 0
    n_target_retained: int = 0

    def update(
        self,
        n_original: int,
        n_kept: int,
        target_index: Optional[int] = None,
        kept_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """记录一步的筛选结果。

        Args:
            n_original: 该步原始候选数。
            n_kept: 筛选后候选数。
            target_index: 教师动作（ground-truth）的候选下标。为 None 表示
                该步无标注目标（如已到达目标页需选 [EOA]），不计入 RR 分母。
            kept_indices: 保留的候选下标，用于判断目标是否被保留。
        """
        self.n_steps += 1
        self.sum_original += n_original
        self.sum_kept += n_kept

        if target_index is not None and kept_indices is not None:
            self.n_steps_with_target += 1
            if target_index in set(kept_indices):
                self.n_target_retained += 1

    @property
    def avg_original(self) -> float:
        """平均原始候选数（论文报告基线约 45，规则过滤后约 22）。"""
        return self.sum_original / self.n_steps if self.n_steps else 0.0

    @property
    def avg_kept(self) -> float:
        """平均保留候选数。"""
        return self.sum_kept / self.n_steps if self.n_steps else 0.0

    @property
    def cr(self) -> float:
        """整体压缩率。

        用候选总数之比而非各步 CR 的平均值：后者会让候选很少的步
        与候选很多的步获得相同权重，低估在长候选列表上的实际压缩效果。
        """
        return compression_ratio(self.sum_original, self.sum_kept)

    @property
    def rr(self) -> float:
        """召回保持率。无标注目标的步不参与计算。"""
        if self.n_steps_with_target == 0:
            return 1.0
        return self.n_target_retained / self.n_steps_with_target

    def as_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "avg_original": round(self.avg_original, 2),
            "avg_kept": round(self.avg_kept, 2),
            "CR": round(self.cr * 100, 2),
            "RR": round(self.rr * 100, 2),
            "n_steps_with_target": self.n_steps_with_target,
            "n_target_retained": self.n_target_retained,
        }
