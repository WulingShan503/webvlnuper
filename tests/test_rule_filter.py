"""4.4 节规则过滤的单元测试。"""

import pytest

from webvln.screening.candidate import Candidate, ElementType, PageArea
from webvln.screening.rule_filter import RuleFilter


def make(index, text="link", area=PageArea.MAIN, href="", context=""):
    return Candidate(
        index=index,
        text=text,
        elem_type=ElementType.LINK,
        area=area,
        href=href,
        context=context,
    )


def test_removes_duplicate_text_and_href():
    cands = [
        make(0, "Contact Us", href="/contact"),
        make(1, "Contact Us", href="/contact"),
        make(2, "Contact Us", href="/support"),  # href 不同，不算重复
    ]
    kept = RuleFilter()(cands)
    assert [c.index for c in kept] == [0, 2]


def test_dedup_is_case_insensitive():
    cands = [make(0, "Reviews", href="/r"), make(1, "reviews", href="/R")]
    assert len(RuleFilter()(cands)) == 1


def test_prunes_footer_and_sidebar_but_keeps_nav():
    cands = [
        make(0, "Product Reviews", area=PageArea.MAIN),
        make(1, "Privacy Policy", area=PageArea.FOOTER),
        make(2, "Newsletter", area=PageArea.SIDEBAR),
        make(3, "Men's Shoes", area=PageArea.NAV),
    ]
    kept = RuleFilter()(cands)
    assert [c.index for c in kept] == [0, 3]


def test_blocks_blacklisted_href_keywords():
    cands = [
        make(0, "Details", href="/product/12"),
        make(1, "Sale", href="/ads/banner?id=3"),
        make(2, "Click", href="https://x.com/?utm_source=news"),
        make(3, "Info", href="javascript:void(0)"),
    ]
    kept = RuleFilter()(cands)
    assert [c.index for c in kept] == [0]


def test_drops_candidates_without_any_text():
    cands = [make(0, "Buy Now"), make(1, "")]
    kept = RuleFilter()(cands)
    assert [c.index for c in kept] == [0]


def test_empty_candidate_with_context_is_kept():
    # 图标链接缺失 alt 但周围有标题文本时，仍可供 LLM 判断。
    cands = [make(0, "", context="Customer Feedback")]
    assert len(RuleFilter()(cands)) == 1


def test_backfill_when_all_filtered_out():
    # 全部候选位于页脚时不应返回空列表，否则决策层无动作可选。
    cands = [make(i, f"Link {i}", area=PageArea.FOOTER) for i in range(3)]
    kept = RuleFilter(min_keep=2)(cands)
    assert len(kept) == 2
    assert [c.index for c in kept] == [0, 1]


def test_stats_account_for_every_dropped_candidate():
    cands = [
        make(0, "Keep", href="/a"),
        make(1, "Keep", href="/a"),
        make(2, "Foot", area=PageArea.FOOTER),
        make(3, "Ad", href="/ads/x"),
        make(4, ""),
    ]
    kept, stats = RuleFilter().filter_with_stats(cands)
    assert stats.n_input == 5
    assert stats.n_output == len(kept) == 1
    assert stats.n_dropped_duplicate == 1
    assert stats.n_dropped_area == 1
    assert stats.n_dropped_keyword == 1
    assert stats.n_dropped_empty == 1


def test_compression_ratio_matches_paper_scale():
    # 论文报告 45 -> 22，压缩 51.1%。这里验证压缩率的计算方式一致。
    cands = [make(i, f"Item {i}") for i in range(45)]
    for c in cands[22:]:
        c.area = PageArea.FOOTER
    kept, stats = RuleFilter().filter_with_stats(cands)
    assert len(kept) == 22
    assert stats.compression_ratio == pytest.approx(0.511, abs=1e-3)


def test_disabling_rules_keeps_everything():
    cands = [
        make(0, "A", href="/a"),
        make(0, "A", href="/a"),
        make(1, "F", area=PageArea.FOOTER),
    ]
    f = RuleFilter(drop_duplicates=False, pruned_areas=(), blocked_keywords=())
    assert len(f(cands)) == 3
