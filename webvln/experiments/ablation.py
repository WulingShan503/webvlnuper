"""5.3 / 5.4 / 5.5 节实验的配置装配与结果比对。

各实验只在筛选配置上有差别，模型与训练流程完全相同：

    5.2  基线            screening.enabled = false
    5.3  Top-k 消融      llm_ranker.k ∈ {3, 5, 8}
    5.5  单阶段对照      rule_filter.enabled = false（45 个候选直接给 LLM）

因此这里只生成配置字典，交给 ``screening/config.py:build_screener`` 装配，
避免为每个实验各写一份训练脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from webvln.experiments.reference import (
    REFERENCE_ABLATION,
    REFERENCE_SCREENING,
    REFERENCE_TWO_STAGE,
)

#: 论文 5.3 节消融的 k 取值。k=5 为最优。
TOPK_VALUES = (3, 5, 8)


def screening_config_for_k(
    k: int, base: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """生成 Top-k 消融的配置。

    Args:
        k: 保留的候选数。
        base: 基准配置（通常是读入的 ``configs/screening.yaml``）。

    Returns:
        新的配置字典。不修改 ``base``——同一份基准要派生出多个实验配置，
        就地改会让后一个实验继承前一个的改动。
    """
    cfg = _deep_copy(base) if base else _default_config()
    cfg.setdefault("screening", {})["enabled"] = True
    cfg["screening"].setdefault("llm_ranker", {})["enabled"] = True
    cfg["screening"]["llm_ranker"]["k"] = k
    return cfg


def baseline_config(base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """5.2 节基线：完全关闭筛选。

    ``build_screener`` 会返回 None，训练走原始的全量候选路径。
    """
    cfg = _deep_copy(base) if base else _default_config()
    cfg.setdefault("screening", {})["enabled"] = False
    return cfg


def single_stage_config(
    k: int = 5, base: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """5.5 节单阶段对照：跳过规则过滤，把全量候选直接交给 LLM。

    这是论文用来说明两阶段必要性的对照组——单阶段 SR 39.71 与两阶段
    39.67 仅差 0.05，但 API 开销是后者的两倍多（表 5.4）。
    """
    cfg = screening_config_for_k(k, base)
    cfg["screening"].setdefault("rule_filter", {})["enabled"] = False
    return cfg


def rule_filter_only_config(base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """仅阶段一的对照：关闭 LLM 排序。

    用于分离两个阶段各自的贡献——规则过滤本身就把候选从 45 压到 22
    （CR 51.1%，RR 98.2%）。
    """
    cfg = _deep_copy(base) if base else _default_config()
    cfg.setdefault("screening", {})["enabled"] = True
    cfg["screening"].setdefault("rule_filter", {})["enabled"] = True
    cfg["screening"].setdefault("llm_ranker", {})["enabled"] = False
    return cfg


@dataclass
class ExperimentResult:
    """一组实验配置下的实测结果。

    Attributes:
        name: 实验名，与 ``reference.py`` 的键对应（如 ``llm_top5``）。
        val: 验证集指标（``NavigationScores.as_dict()`` 的输出）。
        test: 测试集指标。
        screening: 筛选统计（CR / RR / API 调用与缓存命中）。
    """

    name: str
    val: Dict[str, float] = field(default_factory=dict)
    test: Dict[str, float] = field(default_factory=dict)
    screening: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """摊平为表 5.2 的一行：Val SR / Test SR / SPL / TL。"""
        return {
            "name": self.name,
            "val_SR": self.val.get("SR"),
            "test_SR": self.test.get("SR"),
            "SPL": self.val.get("SPL"),
            "TL": self.val.get("TL"),
        }


def compare_with_reference(
    results: Sequence[ExperimentResult],
    reference: Optional[Mapping[str, Mapping[str, float]]] = None,
    tolerance: float = 0.5,
) -> List[Dict[str, Any]]:
    """把实测结果与论文数字逐项比对。

    Args:
        results: 实测结果。
        reference: 参照表，默认表 5.2（``REFERENCE_ABLATION``）。
        tolerance: 允许的绝对偏差（百分点）。默认 0.5——随机种子与
            GPT-3.5-Turbo 响应波动带来的差异通常在此量级内。

    Returns:
        每项含 metric / actual / expected / diff / within_tolerance。
        参照表里没有的实验名会被跳过（如自定义的 k 值）。
    """
    ref = reference if reference is not None else REFERENCE_ABLATION
    rows: List[Dict[str, Any]] = []

    for result in results:
        expected = ref.get(result.name)
        if expected is None:
            continue
        actual = result.as_row()
        for metric, want in expected.items():
            got = actual.get(metric)
            if got is None:
                continue
            diff = round(got - want, 3)
            rows.append(
                {
                    "name": result.name,
                    "metric": metric,
                    "actual": got,
                    "expected": want,
                    "diff": diff,
                    "within_tolerance": abs(diff) <= tolerance,
                }
            )
    return rows


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """把结果渲染成 Markdown 表格，便于粘进论文或日志。

    Args:
        rows: 数据行。
        columns: 列名与取值键。

    Returns:
        Markdown 表格字符串。缺失值渲染为 ``-`` 而非 ``None``，
        免得表格里出现 Python 字面量。
    """
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append("-" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def screening_summary_row(name: str, stats: Mapping[str, Any]) -> Dict[str, Any]:
    """把 ``TwoStageScreener.stats()`` 整理成表 5.3 的一行。

    ``stats()`` 把筛选指标放在 ``screening`` 键下，这里摊平取用。
    """
    screening = stats.get("screening", stats)
    return {
        "name": name,
        "avg_candidates": screening.get("avg_kept"),
        "CR": screening.get("CR"),
        "RR": screening.get("RR"),
    }


def _default_config() -> Dict[str, Any]:
    """无基准配置时的最小骨架。"""
    return {
        "screening": {
            "enabled": True,
            "rule_filter": {"enabled": True},
            "llm_ranker": {"enabled": True, "k": 5},
        }
    }


def _deep_copy(value: Any) -> Any:
    """递归拷贝嵌套的 dict / list 配置。"""
    if isinstance(value, Mapping):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value
