"""4.5 / 4.6 节：指标与两阶段流水线的单元测试。"""

import json

import pytest

from webvln.screening.candidate import Candidate, ElementType, PageArea
from webvln.screening.llm_backend import ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.metrics import ScreeningMetrics, compression_ratio
from webvln.screening.pipeline import TwoStageScreener
from webvln.screening.rule_filter import RuleFilter


def make(index, text=None, area=PageArea.MAIN, href=""):
    return Candidate(
        index=index,
        text=text if text is not None else f"Item {index}",
        elem_type=ElementType.LINK,
        area=area,
        href=href or f"/p/{index}",
    )


# --- 指标 -------------------------------------------------------------------


def test_compression_ratio_formula():
    # 式 (4.5.1)：CR = 1 - k / n
    assert compression_ratio(45, 5) == pytest.approx(1 - 5 / 45)
    assert compression_ratio(0, 0) == 0.0


def test_metrics_aggregate_by_total_counts_not_per_step_mean():
    m = ScreeningMetrics()
    m.update(n_original=100, n_kept=5)
    m.update(n_original=6, n_kept=5)
    # 按总量计算：1 - 10/106；若按各步 CR 平均则会得到明显更低的压缩率。
    assert m.cr == pytest.approx(1 - 10 / 106)


def test_recall_retention_counts_only_steps_with_target():
    m = ScreeningMetrics()
    m.update(10, 5, target_index=3, kept_indices=[1, 2, 3, 4, 5])  # 命中
    m.update(10, 5, target_index=9, kept_indices=[1, 2, 3, 4, 5])  # 漏掉
    m.update(10, 5, target_index=None, kept_indices=[1])  # 无目标，不计入
    assert m.n_steps == 3
    assert m.n_steps_with_target == 2
    assert m.rr == pytest.approx(0.5)


def test_rr_defaults_to_one_without_any_target():
    assert ScreeningMetrics().rr == 1.0


# --- 流水线 -----------------------------------------------------------------


def test_two_stage_order_rule_then_llm():
    # 页脚候选应在 LLM 之前被剔除，因此 LLM 看到的清单里不含它们。
    cands = [make(i) for i in range(6)] + [
        make(i, area=PageArea.FOOTER) for i in range(6, 12)
    ]
    backend = ScriptedBackend([json.dumps({"indices": [2, 0, 5]})])
    screener = TwoStageScreener(
        rule_filter=RuleFilter(),
        ranker=LLMRanker(backend=backend, k=3),
    )
    out = screener.screen("find the price", cands)

    assert out.n_original == 12
    assert out.n_after_rule == 6
    assert out.kept_indices == [2, 0, 5]
    prompt = backend.calls[0]
    assert "6. " not in prompt and "11. " not in prompt


def test_rule_only_mode_skips_llm():
    cands = [make(i) for i in range(4)] + [make(9, area=PageArea.FOOTER)]
    screener = TwoStageScreener(rule_filter=RuleFilter(), ranker=None)
    out = screener.screen("q", cands)
    assert out.kept_indices == [0, 1, 2, 3]


def test_llm_only_mode_skips_rule_filter():
    cands = [make(i, area=PageArea.FOOTER) for i in range(8)]
    backend = ScriptedBackend([json.dumps({"indices": [1, 3]})])
    screener = TwoStageScreener(
        rule_filter=None, ranker=LLMRanker(backend=backend, k=2)
    )
    out = screener.screen("q", cands)
    assert out.n_after_rule == 8
    assert out.kept_indices == [1, 3]


def test_target_index_does_not_influence_selection():
    # 筛选在推理时同样运行，不能依赖教师动作。
    cands = [make(i) for i in range(8)]
    backend = ScriptedBackend([json.dumps({"indices": [0, 1]})])
    screener = TwoStageScreener(ranker=LLMRanker(backend=backend, k=2))
    out = screener.screen("q", cands, target_index=7)
    assert out.kept_indices == [0, 1]
    assert screener.metrics.rr == 0.0


def test_metrics_track_compression_across_steps():
    screener = TwoStageScreener(
        ranker=LLMRanker(
            backend=ScriptedBackend([json.dumps({"indices": [0, 1, 2, 3, 4]})] * 2),
            k=5,
        )
    )
    for _ in range(2):
        screener.screen("q", [make(i) for i in range(20)], target_index=1)

    stats = screener.stats()["screening"]
    assert stats["n_steps"] == 2
    assert stats["avg_original"] == 20.0
    assert stats["avg_kept"] == 5.0
    assert stats["CR"] == pytest.approx(75.0)
    assert stats["RR"] == pytest.approx(100.0)


def test_empty_candidate_list_is_handled():
    screener = TwoStageScreener(
        ranker=LLMRanker(backend=ScriptedBackend([]), k=5)
    )
    out = screener.screen("q", [])
    assert out.kept_indices == []
    assert out.n_original == 0
