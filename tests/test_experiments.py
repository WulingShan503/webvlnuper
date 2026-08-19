"""实验配置装配与论文数字比对的单元测试。

同时把论文表 5.2 / 5.3 / 5.4 的数字与正文的算术关系钉住：
CR 与平均候选数的对应、API 开销与 token 量的对应。
这些断言在论文数字被误改时会立刻失败。
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
    REFERENCE_SCREENING,
    REFERENCE_STAGE_ONE,
    REFERENCE_TOKEN_COST,
    REFERENCE_TWO_STAGE,
)
from webvln.screening.config import build_screener


def base():
    return {
        "screening": {
            "enabled": True,
            "rule_filter": {"enabled": True, "pruned_areas": ["FOOTER"]},
            "llm_ranker": {"enabled": True, "k": 5},
        }
    }


# --- 配置装配 ---------------------------------------------------------------


def test_topk_config_sets_k():
    cfg = screening_config_for_k(3, base())
    assert cfg["screening"]["llm_ranker"]["k"] == 3
    # 其余设置须原样保留，否则消融就不只是在改 k。
    assert cfg["screening"]["rule_filter"]["pruned_areas"] == ["FOOTER"]


def test_config_derivation_does_not_mutate_base():
    b = base()
    screening_config_for_k(8, b)
    single_stage_config(3, b)
    # 同一份基准要派生多个实验配置，就地改会让后一个继承前一个的改动。
    assert b["screening"]["llm_ranker"]["k"] == 5
    assert b["screening"]["rule_filter"]["enabled"] is True


def test_baseline_config_disables_screening():
    cfg = baseline_config(base())
    assert cfg["screening"]["enabled"] is False
    # build_screener 返回 None 即走全量候选的基线路径。
    assert build_screener(config=cfg) is None


def test_single_stage_config_skips_rule_filter():
    cfg = single_stage_config(5, base())
    assert cfg["screening"]["rule_filter"]["enabled"] is False
    assert cfg["screening"]["llm_ranker"]["k"] == 5
    screener = build_screener(config=cfg, backend=object())
    assert screener is not None
    assert screener.rule_filter is None


def test_rule_filter_only_config_skips_llm():
    cfg = rule_filter_only_config(base())
    screener = build_screener(config=cfg)
    assert screener is not None
    assert screener.ranker is None
    assert screener.rule_filter is not None


def test_default_config_when_no_base_given():
    cfg = screening_config_for_k(8)
    assert cfg["screening"]["llm_ranker"]["k"] == 8
    assert cfg["screening"]["enabled"] is True


# --- 论文数字的内部一致性 ---------------------------------------------------


def test_topk_values_match_paper():
    assert TOPK_VALUES == (3, 5, 8)
    assert set(REFERENCE_ABLATION) == {"baseline", "llm_top8", "llm_top5", "llm_top3"}


def test_top5_is_the_best_configuration():
    # 论文结论：k=5 最优。SR / SPL 越大越好。
    ab = REFERENCE_ABLATION
    assert ab["llm_top5"]["val_SR"] == max(v["val_SR"] for v in ab.values())
    assert ab["llm_top5"]["test_SR"] == max(v["test_SR"] for v in ab.values())
    # TL 越小越好，但最小的是 top3——论文据此说明 k 过小会漏掉目标。
    assert ab["llm_top3"]["TL"] < ab["llm_top5"]["TL"]


def test_screening_cr_matches_avg_candidate_count():
    """表 5.3 的 CR 是相对基线 12.4 计算的，而非正文的 45。

    这条断言把口径钉死：算错基数会得到完全不同的 CR。
    """
    baseline_avg = REFERENCE_SCREENING["baseline"]["avg_candidates"]
    assert baseline_avg == 12.4
    for name in ("llm_top8", "llm_top5", "llm_top3"):
        row = REFERENCE_SCREENING[name]
        expected_cr = (1 - row["avg_candidates"] / baseline_avg) * 100
        assert abs(expected_cr - row["CR"]) < 0.1, name


def test_recall_retention_decreases_as_compression_increases():
    # CR 与 RR 的权衡：压得越狠，目标被删的概率越大。
    rows = [REFERENCE_SCREENING[n] for n in ("llm_top8", "llm_top5", "llm_top3")]
    crs = [r["CR"] for r in rows]
    rrs = [r["RR"] for r in rows]
    assert crs == sorted(crs)
    assert rrs == sorted(rrs, reverse=True)


def test_stage_one_compression_matches_45_to_22():
    # 4.4 节：45 → 22，压缩 51.1%，召回 98.2%。
    s = REFERENCE_STAGE_ONE
    expected = (1 - s["avg_after"] / s["avg_before"]) * 100
    assert abs(expected - s["CR"]) < 0.1
    assert s["RR"] == 98.2


def test_two_stage_halves_api_cost_for_negligible_sr_loss():
    single = REFERENCE_TWO_STAGE["single_stage"]
    two = REFERENCE_TWO_STAGE["two_stage"]
    # SR 只差 0.05，开销降到 0.48（即 52%）。
    assert abs(single["SR"] - two["SR"]) <= 0.05
    assert abs((1 - two["api_cost"]) * 100 - 52) < 1.0


def test_token_cost_matches_candidate_counts():
    t = REFERENCE_TOKEN_COST
    assert t["single_stage_tokens"] == 45 * t["tokens_per_candidate"]
    assert t["two_stage_tokens"] == 22 * t["tokens_per_candidate"]
    # token 量之比应与表 5.4 的 api_cost 大致吻合。
    ratio = t["two_stage_tokens"] / t["single_stage_tokens"]
    assert abs(ratio - REFERENCE_TWO_STAGE["two_stage"]["api_cost"]) < 0.05


def test_known_inconsistencies_are_documented():
    # 论文表格与正文有数处数字不一致，代码按表格为准，
    # 但必须记录下来——否则读者会以为实现有误。
    topics = {item["topic"] for item in KNOWN_INCONSISTENCIES}
    assert "Top-k 的 SR 数值" in topics
    assert all(item["adopted"] for item in KNOWN_INCONSISTENCIES)
    assert all(item["detail"] for item in KNOWN_INCONSISTENCIES)


# --- 结果比对 ---------------------------------------------------------------


def test_compare_flags_within_tolerance():
    result = ExperimentResult(
        name="llm_top5",
        val={"SR": 39.20, "SPL": 39.01, "TL": 3.72},
        test={"SR": 35.03},
    )
    rows = compare_with_reference([result])
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["val_SR"]["diff"] == 0.08
    assert by_metric["val_SR"]["within_tolerance"] is True
    assert by_metric["test_SR"]["diff"] == 0.0


def test_compare_flags_outside_tolerance():
    result = ExperimentResult(name="llm_top5", val={"SR": 30.0}, test={"SR": 35.03})
    rows = compare_with_reference([result], tolerance=0.5)
    val_sr = next(r for r in rows if r["metric"] == "val_SR")
    assert val_sr["within_tolerance"] is False
    assert val_sr["diff"] == -9.12


def test_compare_skips_unknown_experiment_names():
    rows = compare_with_reference([ExperimentResult(name="llm_top4", val={"SR": 39.0})])
    assert rows == []


def test_screening_summary_row_flattens_stats():
    stats = {"screening": {"avg_kept": 5.0, "CR": 59.7, "RR": 93.5}}
    row = screening_summary_row("llm_top5", stats)
    assert row == {"name": "llm_top5", "avg_candidates": 5.0, "CR": 59.7, "RR": 93.5}


def test_format_table_renders_markdown():
    rows = [{"name": "llm_top5", "val_SR": 39.12, "TL": None}]
    out = format_table(rows, ["name", "val_SR", "TL"])
    lines = out.splitlines()
    assert lines[0] == "| name | val_SR | TL |"
    assert lines[1] == "| --- | --- | --- |"
    # 缺失值渲染成 -，免得表格里出现 Python 的 None。
    assert lines[2] == "| llm_top5 | 39.12 | - |"
