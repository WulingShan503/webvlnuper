"""实验编排的单元测试。

只覆盖配置枚举与报告渲染——``run_one`` 要跑训练，需 torch 与 API key。
"""

import pytest

from webvln.experiments.ablation import ExperimentResult
from webvln.experiments.runner import build_experiment_configs, report
from webvln.screening.config import build_screener


def base():
    return {
        "screening": {
            "enabled": True,
            "rule_filter": {"enabled": True},
            "llm_ranker": {"enabled": True, "k": 5},
        }
    }


def test_ablation_covers_baseline_and_three_k_values():
    configs = build_experiment_configs("ablation", base())
    # 名称须与 reference.py 的键一致，比对才能对上。
    assert set(configs) == {"baseline", "llm_top8", "llm_top5", "llm_top3"}
    assert configs["llm_top3"]["screening"]["llm_ranker"]["k"] == 3
    assert configs["baseline"]["screening"]["enabled"] is False


def test_two_stage_experiment_pairs_single_and_two():
    configs = build_experiment_configs("two_stage", base())
    assert set(configs) == {"single_stage", "two_stage"}
    # 单阶段跳过规则过滤，把全量候选直接交给 LLM。
    assert configs["single_stage"]["screening"]["rule_filter"]["enabled"] is False
    assert configs["two_stage"]["screening"]["rule_filter"]["enabled"] is True


def test_stage_one_experiment_disables_llm():
    configs = build_experiment_configs("stage_one", base())
    screener = build_screener(config=configs["rule_filter_only"])
    assert screener.ranker is None


def test_baseline_experiment_is_single_config():
    configs = build_experiment_configs("baseline", base())
    assert list(configs) == ["baseline"]


def test_unknown_experiment_rejected():
    with pytest.raises(ValueError, match="未知实验"):
        build_experiment_configs("nope", base())


def test_report_renders_all_sections():
    results = [
        ExperimentResult(
            name="llm_top5",
            val={"SR": 39.12, "SPL": 39.01, "TL": 3.72},
            test={"SR": 35.03},
            screening={"screening": {"avg_kept": 5.0, "CR": 59.7, "RR": 93.5}},
        )
    ]
    out = report(results, "ablation")
    assert "### 导航指标" in out
    assert "### 筛选指标" in out
    assert "### 与论文数字的差异" in out
    assert "llm_top5" in out


def test_report_omits_screening_section_for_baseline():
    results = [
        ExperimentResult(name="baseline", val={"SR": 38.35}, test={"SR": 34.21})
    ]
    out = report(results, "baseline")
    # 基线无筛选器，stats 为空，不该渲染空表。
    assert "### 筛选指标" not in out
    assert "### 导航指标" in out
