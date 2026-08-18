"""模拟器对接层的单元测试。

候选记录格式取自 WebVLN 官方实现 ``r2r_src/env.py``：
``connectivity[websiteID][urlID]["data"]`` 中每条含
``clickable_id`` / ``next_url_id`` / ``text``(列表) / ``href_full`` / ``imgs``。
"""

import json

from webvln.screening.adapter import (
    EOA_KEY,
    apply_screening_to_state,
    build_instruction,
    candidate_from_record,
    candidates_from_state,
    instruction_from_obs,
)
from webvln.screening.candidate import ElementType, PageArea
from webvln.screening.llm_backend import ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.pipeline import TwoStageScreener
from webvln.screening.rule_filter import RuleFilter


def record(text, href, next_url_id="u1", imgs=None, cid="c1"):
    return {
        "clickable_id": cid,
        "next_url_id": next_url_id,
        "text": text if isinstance(text, list) else [text],
        "href_full": href,
        "imgs": imgs if imgs is not None else [],
    }


def state(*pairs):
    """构造 state['candidate']，键为 clickable_id。"""
    return {
        cid: record(text, href, next_url_id=f"u{i}", cid=cid)
        for i, (cid, text, href) in enumerate(pairs)
    }


# --- 单条记录转换 -----------------------------------------------------------


def test_maps_official_fields():
    r = record("Product Reviews", "https://shop.com/products/sock/reviews", imgs=["a.jpg"])
    c = candidate_from_record(4, r, clickable_id="btn_7")

    assert c.index == 4
    assert c.text == "Product Reviews"
    assert c.href == "https://shop.com/products/sock/reviews"
    assert c.has_image is True
    assert c.raw["clickable_id"] == "btn_7"
    assert c.raw["next_url_id"] == "u1"
    assert c.raw["imgs"] == ["a.jpg"]


def test_text_list_is_joined_not_truncated():
    # text 是列表，官方注释只取首项；这里保留全部片段以免丢失语义
    # （图片 alt 与相邻标题常分列两项）。
    r = record(["Blue Sock", "Material: cotton"], "/products/sock")
    assert candidate_from_record(0, r).text == "Blue Sock Material: cotton"


def test_empty_text_list_yields_empty_text():
    r = record([], "/products/sock")
    assert candidate_from_record(0, r).text == ""


def test_none_entries_in_text_list_are_skipped():
    r = {"text": [None, "Size Guide"], "href_full": "/x"}
    assert candidate_from_record(0, r).text == "Size Guide"


def test_defaults_to_link_type():
    # 官方数据不含标签名；候选均为可跳转元素，按 LINK 处理。
    assert candidate_from_record(0, record("A", "/a")).elem_type is ElementType.LINK


def test_explicit_tag_overrides_default():
    r = record("Add to cart", "/cart")
    r["tag"] = "button"
    assert candidate_from_record(0, r).elem_type is ElementType.BUTTON


def test_area_inferred_when_absent():
    c = candidate_from_record(0, record("Privacy", "/pages/privacy-policy"))
    assert c.area is PageArea.FOOTER


def test_explicit_area_string_is_respected():
    r = record("X", "/products/x")
    r["area"] = "FOOTER"
    assert candidate_from_record(0, r).area is PageArea.FOOTER
    r["area"] = "page footer"
    assert candidate_from_record(0, r).area is PageArea.FOOTER


def test_unrecognised_area_string_falls_back_to_unknown():
    r = record("X", "/x")
    r["area"] = "somewhere-odd"
    assert candidate_from_record(0, r).area is PageArea.UNKNOWN


def test_missing_href_field_does_not_crash():
    assert candidate_from_record(0, {"text": ["A"]}).href == ""


# --- 整步候选转换 -----------------------------------------------------------


def test_index_follows_dict_iteration_order():
    # index 必须与 make_candidate 的 enumerate 顺序一致，
    # 否则筛选结果会索引到错误的特征行。
    st = state(("a", "First", "/products/1"), ("b", "Second", "/products/2"))
    cands = candidates_from_state(st)
    assert [c.index for c in cands] == [0, 1]
    assert [c.raw["clickable_id"] for c in cands] == ["a", "b"]


def test_eoa_is_excluded_from_screening():
    st = state(("a", "First", "/products/1"))
    st[EOA_KEY] = record("", "")
    cands = candidates_from_state(st)
    assert [c.raw["clickable_id"] for c in cands] == ["a"]


def test_eoa_index_gap_is_preserved():
    # [EOA] 在字典中占据一个位置，跳过它不能让后续候选的下标前移，
    # 否则与 make_candidate 的行号错位。
    st = {}
    st["a"] = record("First", "/products/1", cid="a")
    st[EOA_KEY] = record("", "")
    st["b"] = record("Second", "/products/2", cid="b")
    cands = candidates_from_state(st)
    assert [c.index for c in cands] == [0, 2]


def test_non_mapping_entries_are_ignored():
    st = state(("a", "First", "/products/1"))
    st["broken"] = "not-a-dict"
    assert len(candidates_from_state(st)) == 1


# --- 指令拼接 ---------------------------------------------------------------


def test_instruction_joins_question_and_description():
    assert build_instruction("How much is it?", "A blue sock.") == (
        "How much is it? A blue sock."
    )


def test_empty_description_leaves_no_trailing_separator():
    # 论文脚注：问题足以定位目标页时 D 为空。
    assert build_instruction("How much is it?", "") == "How much is it?"


def test_instruction_from_obs_handles_string_dict_and_list():
    assert instruction_from_obs({"text": " How much? "}) == "How much?"
    assert instruction_from_obs({"text": {"Q": "How much?", "D": "Blue sock."}}) == (
        "How much? Blue sock."
    )
    assert instruction_from_obs({"text": ["How much?", "Blue sock."]}) == (
        "How much? Blue sock."
    )
    assert instruction_from_obs({}) == ""


# --- 回写候选字典 -----------------------------------------------------------


def test_rebuilds_state_dict_with_kept_candidates_only():
    st = state(
        ("a", "First", "/products/1"),
        ("b", "Second", "/products/2"),
        ("c", "Third", "/products/3"),
    )
    out = apply_screening_to_state(st, kept_indices=[0, 2])
    assert list(out.keys()) == ["a", "c"]


def test_eoa_survives_screening_even_if_not_selected():
    # 剔除 [EOA] 会让智能体永远无法停止并作答。
    st = state(("a", "First", "/products/1"), ("b", "Second", "/products/2"))
    st[EOA_KEY] = record("", "")
    out = apply_screening_to_state(st, kept_indices=[0])
    assert EOA_KEY in out
    assert list(out.keys()) == ["a", EOA_KEY]


def test_rebuilt_dict_keeps_original_page_order():
    # 回写按原始顺序而非 LLM 相关性顺序，保证同一页面的 logits 排布稳定。
    st = state(
        ("a", "First", "/products/1"),
        ("b", "Second", "/products/2"),
        ("c", "Third", "/products/3"),
    )
    out = apply_screening_to_state(st, kept_indices=[2, 0])
    assert list(out.keys()) == ["a", "c"]


def test_records_are_passed_through_unmodified():
    st = state(("a", "First", "/products/1"))
    out = apply_screening_to_state(st, kept_indices=[0])
    assert out["a"] is st["a"]


# --- 端到端 -----------------------------------------------------------------


def test_full_flow_from_simulator_state_to_filtered_state():
    st = {}
    for cid, text, href in [
        ("c0", "Blue Striped Sock", "/products/blue-striped-sock"),
        ("c1", "Product Reviews", "/products/blue-striped-sock/reviews"),
        ("c2", "Privacy Policy", "/pages/privacy-policy"),
        ("c3", "Newsletter", "/pages/newsletter"),
        ("c4", "Blue Striped Sock", "/products/blue-striped-sock"),  # 重复
        ("c5", "Sponsored", "/ads/banner"),
        ("c6", "Material Care", "/products/blue-striped-sock/description"),
    ]:
        st[cid] = record(text, href, cid=cid)
    st[EOA_KEY] = record("", "")

    cands = candidates_from_state(st)
    # 规则过滤应剔除页脚(c2)、侧边栏(c3)、重复(c4)、广告(c5)
    kept_after_rule = RuleFilter()(cands)
    assert [c.raw["clickable_id"] for c in kept_after_rule] == ["c0", "c1", "c6"]

    screener = TwoStageScreener(
        rule_filter=RuleFilter(),
        ranker=LLMRanker(backend=ScriptedBackend([json.dumps({"indices": [6, 1]})]), k=2),
    )
    out = screener.screen("What material is this sock made of?", cands, target_index=6)

    assert out.n_original == 7
    assert out.n_after_rule == 3
    assert out.kept_indices == [6, 1]

    new_state = apply_screening_to_state(st, out.kept_indices)
    assert list(new_state.keys()) == ["c1", "c6", EOA_KEY]
    assert screener.metrics.rr == 1.0
