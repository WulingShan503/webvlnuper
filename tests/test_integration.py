"""集成入口与配置装配的单元测试。"""

import json

import pytest

from webvln.screening.adapter import EOA_KEY
from webvln.screening.config import build_screener
from webvln.screening.integration import screen_candidates, screen_state
from webvln.screening.llm_backend import ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.pipeline import TwoStageScreener
from webvln.screening.rule_filter import RuleFilter


def rec(text, href, cid):
    return {
        "clickable_id": cid,
        "next_url_id": f"url_{cid}",
        "text": [text],
        "href_full": href,
        "imgs": [],
    }


def sim_state():
    """模拟 Simulator.getState() 的返回值。"""
    entries = [
        ("c0", "Blue Sock", "/products/blue-sock"),
        ("c1", "Material Care", "/products/blue-sock/material-care"),
        ("c2", "Privacy Policy", "/pages/privacy-policy"),
        ("c3", "Contact Us", "/pages/contact"),
        ("c4", "Newsletter", "/pages/newsletter"),
        ("c5", "Size Guide", "/products/blue-sock/size-guide"),
        ("c6", "Reviews", "/products/blue-sock/reviews"),
    ]
    cand = {cid: rec(t, h, cid) for cid, t, h in entries}
    cand[EOA_KEY] = {"clickable_id": EOA_KEY, "next_url_id": "", "text": [], "href_full": ""}
    return {"websiteID": "SA", "urlID": "u42", "candidate": cand}


def screener(indices, k=3):
    return TwoStageScreener(
        rule_filter=RuleFilter(),
        ranker=LLMRanker(backend=ScriptedBackend([json.dumps({"indices": indices})]), k=k),
    )


# --- screen_state -----------------------------------------------------------


def test_returns_state_with_same_shape():
    st = sim_state()
    out = screen_state(st, {"text": "What material?"}, screener([1, 5, 6]))
    assert set(out.keys()) == set(st.keys())
    assert out["websiteID"] == "SA" and out["urlID"] == "u42"


def test_prunes_footer_and_sidebar_then_keeps_top_k():
    st = sim_state()
    out = screen_state(st, {"text": "What material?"}, screener([1, 5, 6]))
    # c2/c3 页脚、c4 侧边栏先被规则剔除；LLM 再从余下选出 3 个。
    assert list(out["candidate"].keys()) == ["c1", "c5", "c6", EOA_KEY]


def test_eoa_always_survives():
    st = sim_state()
    out = screen_state(st, {"text": "q"}, screener([1]))
    assert EOA_KEY in out["candidate"]


def test_original_state_is_not_mutated():
    # 模拟器内部持有同一 candidate 对象，就地裁剪会污染后续步骤的候选图。
    st = sim_state()
    before = list(st["candidate"].keys())
    screen_state(st, {"text": "q"}, screener([1]))
    assert list(st["candidate"].keys()) == before


def test_none_screener_is_identity():
    st = sim_state()
    out = screen_state(st, {"text": "q"}, None)
    assert list(out["candidate"].keys()) == list(st["candidate"].keys())


def test_empty_candidate_dict_short_circuits():
    st = {"websiteID": "SA", "urlID": "u1", "candidate": {}}
    out = screen_state(st, {"text": "q"}, screener([0]))
    assert out["candidate"] == {}


def test_explicit_instruction_overrides_item():
    backend = ScriptedBackend([json.dumps({"indices": [0]})])
    sc = TwoStageScreener(ranker=LLMRanker(backend=backend, k=1))
    screen_state(sim_state(), {"text": "from item"}, sc, instruction="explicit one")
    assert "explicit one" in backend.calls[0]
    assert "from item" not in backend.calls[0]


def test_instruction_read_from_item_text():
    backend = ScriptedBackend([json.dumps({"indices": [0]})])
    sc = TwoStageScreener(ranker=LLMRanker(backend=backend, k=1))
    screen_state(sim_state(), {"text": "What material is it?"}, sc)
    assert "What material is it?" in backend.calls[0]


# --- 教师动作与 RR ----------------------------------------------------------


def test_target_clickable_id_counted_when_retained():
    sc = screener([1, 5, 6])
    screen_state(sim_state(), {"text": "q"}, sc, target_clickable_id="c1")
    assert sc.metrics.rr == 1.0
    assert sc.metrics.n_steps_with_target == 1


def test_target_missed_lowers_recall():
    sc = screener([1, 5, 6])
    screen_state(sim_state(), {"text": "q"}, sc, target_clickable_id="c0")
    assert sc.metrics.rr == 0.0


def test_eoa_target_excluded_from_recall_denominator():
    # 已到达目标页时教师动作是 [EOA]，不是页面元素，不应算作召回失败。
    sc = screener([1, 5, 6])
    screen_state(sim_state(), {"text": "q"}, sc, target_clickable_id=EOA_KEY)
    assert sc.metrics.n_steps_with_target == 0
    assert sc.metrics.rr == 1.0


def test_unknown_target_id_excluded_from_denominator():
    sc = screener([1, 5, 6])
    screen_state(sim_state(), {"text": "q"}, sc, target_clickable_id="does-not-exist")
    assert sc.metrics.n_steps_with_target == 0


def test_screen_candidates_returns_metrics_payload():
    out = screen_candidates(
        sim_state()["candidate"], "What material?", screener([1, 5, 6]), "c1"
    )
    assert out.n_original == 7  # [EOA] 不计入
    assert out.n_after_rule == 4  # c2/c3/c4 被剪枝
    assert out.kept_indices == [1, 5, 6]


# --- 配置装配 ---------------------------------------------------------------


def test_build_screener_from_dict():
    cfg = {
        "screening": {
            "enabled": True,
            "rule_filter": {"enabled": True, "pruned_areas": ["FOOTER"]},
            "llm_ranker": {"enabled": True, "k": 4},
            "cache": {"enabled": False},
        }
    }
    sc = build_screener(cfg, backend=ScriptedBackend([]))
    assert sc.ranker.k == 4
    assert sc.ranker.cache is None
    assert sc.rule_filter.pruned_areas[0].name == "FOOTER"


def test_disabled_screening_returns_none():
    # 5.2 节的基线复现走不筛选路径。
    assert build_screener({"screening": {"enabled": False}}) is None


def test_rule_only_and_llm_only_configs():
    llm_only = build_screener(
        {"screening": {"rule_filter": {"enabled": False}, "cache": {"enabled": False}}},
        backend=ScriptedBackend([]),
    )
    assert llm_only.rule_filter is None and llm_only.ranker is not None

    rule_only = build_screener(
        {"screening": {"llm_ranker": {"enabled": False}}}, backend=ScriptedBackend([])
    )
    assert rule_only.ranker is None and rule_only.rule_filter is not None


def test_area_values_accepted_in_both_spellings():
    sc = build_screener(
        {"screening": {"rule_filter": {"pruned_areas": ["sidebar", "page footer"]}}},
        backend=ScriptedBackend([]),
    )
    names = {a.name for a in sc.rule_filter.pruned_areas}
    assert names == {"SIDEBAR", "FOOTER"}


def test_invalid_area_raises():
    with pytest.raises(ValueError, match="未知的页面区域配置"):
        build_screener(
            {"screening": {"rule_filter": {"pruned_areas": ["nowhere"]}}},
            backend=ScriptedBackend([]),
        )


def test_shipped_yaml_config_loads():
    # 仓库内的默认配置须可直接装配，避免文档与代码脱节。
    sc = build_screener(path="configs/screening.yaml", backend=ScriptedBackend([]))
    assert sc.ranker.k == 5  # 5.3 节最优
    assert sc.ranker.model_name == "gpt-3.5-turbo"
