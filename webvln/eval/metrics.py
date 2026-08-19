"""5.1 节：导航与问答的评测指标。

对应官方 ``r2r_src/eval.py`` 的 ``Evaluation``：

    SR    成功率，停在目标页即成功
    OSR   oracle 成功率，轨迹**经过**目标页即成功（不要求停在那）
    SPL   成功率按路径长度加权，式 (5.1.1)
    TL    轨迹长度（经过的页面数）
    WUPS  答案质量，仅在导航成功时计算

论文 3.9 / 5.1 节用 Best Score = SR + WUPS0.9 选模型：只看 SR 会选出
「找对页面但答不对」的模型，只看 WUPS 则无法反映导航能力。

注意 SPL 与 TL 的单位都是**页面数**而非物理距离——网页导航图里
相邻页面之间没有距离概念，官方直接用 ``len(path)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from webvln.eval.wups import wups_official


@dataclass
class Result:
    """单条 episode 的预测结果。

    Attributes:
        idx: 记录 ID，与 ``Episode.idx`` 对应。
        trajectory: 智能体实际走过的 urlID 序列，含起点。
        answer: 生成的答案。
    """

    idx: Any
    trajectory: List[str] = field(default_factory=list)
    answer: str = ""


@dataclass
class NavigationScores:
    """一个划分上的累计得分。

    各指标按 episode 累加后取平均，与官方 ``score_summary`` 一致。
    """

    n: int = 0
    success: float = 0.0
    oracle_success: float = 0.0
    spl: float = 0.0
    tl: float = 0.0
    wups_9: float = 0.0
    wups_0: float = 0.0

    def add(
        self,
        success: bool,
        oracle_success: bool,
        spl: float,
        tl: int,
        wups_9: float = 0.0,
        wups_0: float = 0.0,
    ) -> None:
        self.n += 1
        self.success += float(success)
        self.oracle_success += float(oracle_success)
        self.spl += spl
        self.tl += tl
        self.wups_9 += wups_9
        self.wups_0 += wups_0

    def as_dict(self, percent: bool = True) -> Dict[str, float]:
        """汇总为论文表 5.1 的字段。

        Args:
            percent: True 时把比率类指标换算成百分数（论文表格的形式）。
                TL 是长度不是比率，任何情况下都不乘 100。
        """
        if self.n == 0:
            return {k: 0.0 for k in ("SR", "OSR", "SPL", "TL", "WUPS0.9", "WUPS0.0")}
        scale = 100.0 if percent else 1.0
        return {
            "SR": round(self.success / self.n * scale, 2),
            "OSR": round(self.oracle_success / self.n * scale, 2),
            "SPL": round(self.spl / self.n * scale, 2),
            "TL": round(self.tl / self.n, 2),
            "WUPS0.9": round(self.wups_9 / self.n * scale, 2),
            "WUPS0.0": round(self.wups_0 / self.n * scale, 2),
        }

    def best_score(self) -> float:
        """Best Score = SR + WUPS0.9（论文 3.9 节的模型选择准则）。"""
        summary = self.as_dict()
        return round(summary["SR"] + summary["WUPS0.9"], 2)


def score_episode(
    gt_path: Sequence[str],
    trajectory: Sequence[str],
    gt_answer: str = "",
    answer: str = "",
    wups_fn: Optional[Callable[[str, str, float], float]] = None,
) -> Dict[str, float]:
    """给单条 episode 打分，对应官方 ``_score_item``。

    Args:
        gt_path: ground-truth 最短路径。
        trajectory: 智能体走过的页面序列，须以 ``gt_path[0]`` 开头。
        gt_answer: 参考答案。
        answer: 生成的答案。
        wups_fn: WUPS 实现，默认 ``wups_official``（复刻官方逐字符行为，
            论文表 5.1 的数字由它产出）。

    Returns:
        含 success / oracle_success / spl / tl / wups_0.9 / wups_0.0 的字典。

    Raises:
        ValueError: 轨迹起点与 ground-truth 不一致。官方在这里用 assert，
            这类不一致通常意味着结果文件与划分对不上，必须暴露出来。
    """
    if not trajectory:
        raise ValueError("轨迹为空，至少应包含起始页")
    if trajectory[0] != gt_path[0]:
        raise ValueError(
            f"轨迹起点 {trajectory[0]!r} 与 ground-truth 起点 {gt_path[0]!r} 不一致"
        )

    goal = gt_path[-1]
    tl = len(trajectory)
    success = trajectory[-1] == goal
    # OSR 只要求经过目标页：它衡量「找到了但没停下」，
    # 与 SR 的差值即停止时机造成的损失。
    oracle_success = goal in set(trajectory)

    scores = {
        "success": float(success),
        "oracle_success": float(oracle_success),
        "tl": float(tl),
        "spl": 0.0,
        "wups_0.9": 0.0,
        "wups_0.0": 0.0,
    }

    if success:
        # 式 (5.1.1)：SPL = S · L / max(L, P)。走得比最短路径长则打折，
        # 失败样本 SPL 记 0（官方同样处理）。
        scores["spl"] = len(gt_path) / max(len(gt_path), tl)
        # 答案只在导航成功时评分：没找到页面就无从取信息作答，
        # 此时的答案分数没有意义（官方失败样本 WUPS 记 0）。
        fn = wups_fn or wups_official
        scores["wups_0.9"] = fn(gt_answer, answer, 0.9)
        scores["wups_0.0"] = fn(gt_answer, answer, 0.0)

    return scores


def score_results(
    results: Sequence[Result],
    ground_truth: Mapping[Any, Any],
    wups_fn: Optional[Callable[[str, str, float], float]] = None,
) -> NavigationScores:
    """给一批结果打分。

    Args:
        results: 预测结果。
        ground_truth: ``{idx: Episode}``。键按 ``str(idx)`` 与原值同时查找——
            官方 ``Evaluation`` 用 ``str(item['idx'])`` 建键而结果里是原始类型，
            两边类型不一致时会静默漏掉整个划分。
        wups_fn: WUPS 实现。

    Returns:
        NavigationScores。重复的 idx 只计一次（官方靠 ``ids.remove`` 去重，
        因为 ``valid()`` 的循环会绕回开头重复评估部分样本）。
    """
    scores = NavigationScores()
    seen = set()

    for result in results:
        if result.idx in seen:
            continue
        gt = ground_truth.get(result.idx, ground_truth.get(str(result.idx)))
        if gt is None:
            continue
        seen.add(result.idx)

        item = score_episode(
            gt_path=gt.path,
            trajectory=result.trajectory,
            gt_answer=gt.answer,
            answer=result.answer,
            wups_fn=wups_fn,
        )
        scores.add(
            success=bool(item["success"]),
            oracle_success=bool(item["oracle_success"]),
            spl=item["spl"],
            tl=int(item["tl"]),
            wups_9=item["wups_0.9"],
            wups_0=item["wups_0.0"],
        )

    return scores


def best_score(summary: Mapping[str, float]) -> float:
    """由汇总字典算 Best Score = SR + WUPS0.9。"""
    return round(summary.get("SR", 0.0) + summary.get("WUPS0.9", 0.0), 2)
