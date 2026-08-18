"""4.2 节候选序列化的单元测试。"""

from webvln.screening.candidate import Candidate, ElementType, PageArea
from webvln.screening.serializer import (
    MAX_TEXT_LEN,
    build_candidate_block,
    serialize,
    truncate,
)


def test_matches_paper_template():
    cand = Candidate(
        index=3,
        text="Product Reviews",
        elem_type=ElementType.LINK,
        area=PageArea.MAIN,
        context="Customer Feedback",
    )
    assert serialize(cand) == (
        '[TYPE: LINK][TEXT: "Product Reviews"]'
        "located in the [AREA: main content section]"
        'near [CONTEXT: "Customer Feedback"]'
    )


def test_omits_unknown_area_and_missing_context():
    cand = Candidate(index=0, text="Buy", elem_type=ElementType.BUTTON)
    assert serialize(cand) == '[TYPE: BUTTON][TEXT: "Buy"]'


def test_truncates_long_inner_text():
    long_text = "x" * 250
    cand = Candidate(index=0, text=long_text, elem_type=ElementType.LINK)
    out = serialize(cand)
    assert "x" * MAX_TEXT_LEN in out
    assert out.count("x") == MAX_TEXT_LEN
    assert "..." in out


def test_truncate_keeps_short_text_untouched():
    assert truncate("short", 100) == "short"


def test_element_type_inferred_from_tag():
    assert ElementType.from_tag("a") is ElementType.LINK
    assert ElementType.from_tag("BUTTON") is ElementType.BUTTON
    assert ElementType.from_tag("select") is ElementType.INPUT
    assert ElementType.from_tag("div") is ElementType.UNKNOWN
    assert ElementType.from_tag(None) is ElementType.UNKNOWN


def test_candidate_normalises_whitespace():
    cand = Candidate(index=0, text="  Product\n   Reviews  ")
    assert cand.text == "Product Reviews"


def test_block_uses_candidate_index_not_row_number():
    # 规则过滤后下标不连续，提示词必须沿用原始 index，
    # 否则 LLM 返回的编号无法映射回特征张量的行。
    cands = [
        Candidate(index=7, text="A", elem_type=ElementType.LINK),
        Candidate(index=19, text="B", elem_type=ElementType.LINK),
    ]
    block = build_candidate_block(cands)
    assert block.startswith("7. ")
    assert "\n19. " in block
