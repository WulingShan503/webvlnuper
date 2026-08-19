"""导航环境与观测构造的单元测试。

重点是教师动作在**筛选前后**的对齐：筛选改变候选集合，
教师动作须先按 clickable_id 定位、再映射到筛选后的下标，
否则会指向错误的候选（本项目最易出错的一处）。
"""

from webvln.data.episode import Episode
from webvln.data.features import FeatureStore
from webvln.data.graph import NavigationGraph
from webvln.screening.candidate import PageArea
from webvln.screening.llm_backend import ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.pipeline import TwoStageScreener
from webvln.screening.rule_filter import RuleFilter
from webvln.train.env import WebVLNEnv


def record(cid, next_url, text="link", href=""):
    return {
        "clickable_id": cid,
        "next_url_id": next_url,
        "text": [text],
        "href_full": href or f"https://site/{cid}",
        "imgs": [],
    }


def build_graph():
    connectivity = {
        "S": {
            "S_0": {
                "data": {
                    # href 带 /collections 前缀，会被 area.py 判为导航区
                    "nav": record("nav", "S_9", "Shop All",
                                  "https://site/collections/all"),
                    "prod": record("prod", "S_1", "Grey Socks",
                                   "https://site/products/socks"),
                }
            },
            "S_1": {"data": {"detail": record("detail", "S_2", "Details")}},
            "S_2": {"data": {}},
            "S_9": {"data": {"back": record("back", "S_0", "Back")}},
        }
    }
    shortest_paths = {
        "S": {
            "S_0": {"S_2": ["S_0", "S_1", "S_2"], "S_1": ["S_0", "S_1"]},
            "S_1": {"S_2": ["S_1", "S_2"]},
            "S_2": {"S_2": ["S_2"]},
            "S_9": {"S_2": ["S_9", "S_0", "S_1", "S_2"]},
        }
    }
    return NavigationGraph(connectivity, shortest_paths)


def episode():
    return Episode(
        idx=1,
        path=["S_0", "S_1", "S_2"],
        question="what material are the socks?",
        answer="cotton",
        text="Target: Grey Socks, what material are the socks?",
        text_enc=[101, 5, 102],
        answer_enc=[1, 7],
        answer_enc_w_eos=[7, 2],
    )


def features():
    return FeatureStore(
        text_feats={"nav": [1], "prod": [2], "detail": [3]},
        screenshot_feats={"nav": [4], "prod": [5], "detail": [6]},
        feature_size=1,
    )


def env(screener=None):
    return WebVLNEnv(build_graph(), features(), screener=screener)


def test_reset_places_batch_at_start_page():
    e = env()
    obs = e.reset([episode()])
    assert obs[0].url_id == "S_0"
    assert obs[0].candidate_ids == ["nav", "prod"]
    assert obs[0].n_actions == 3  # 2 候选 + [EOA]
    assert obs[0].eoa_index == 2


def test_observation_carries_features_in_official_order():
    obs = env().reset([episode()])[0]
    # [文本, 按钮图, 截图]；无图元素按钮图为零向量。
    assert obs.candidate_feats[1] == [[2], [0.0], [5]]


def test_teacher_points_at_shortest_path_next_hop():
    obs = env().reset([episode()])[0]
    # 最短路径 S_0 -> S_1 -> S_2，下一跳是 S_1，对应候选 prod（下标 1）。
    assert obs.teacher == 1
    assert obs.distance == 3


def test_teacher_recomputed_after_going_off_path():
    e = env()
    e.reset([episode()])
    # 点 nav 走到 S_9，偏离 ground-truth 路径。
    obs = e.step([0])[0]
    assert obs.url_id == "S_9"
    # 教师按最短路径重算：S_9 -> S_0，对应候选 back（下标 0），
    # 而非按 gt_path 固定下标取一个当前页面没有链接的页面。
    assert obs.teacher == 0


def test_teacher_is_eoa_at_target_page():
    e = env()
    e.reset([episode()])
    e.step([1])          # S_0 -> S_1
    obs = e.step([0])[0]  # S_1 -> S_2 (目标页)
    assert obs.url_id == "S_2"
    assert obs.candidate_ids == []
    # 已在目标页：教师动作是停止，落在 [EOA] 位。
    assert obs.teacher == obs.eoa_index == 0


def test_stop_action_keeps_page_unchanged():
    e = env()
    e.reset([episode()])
    obs = e.step([-1])[0]
    assert obs.url_id == "S_0"


def test_screening_reindexes_teacher_action():
    """筛选剔除候选后，教师动作须跟着改下标。

    这里规则过滤会剪掉 nav（被判为导航区），prod 从下标 1 变成 0。
    """
    screener = TwoStageScreener(
        rule_filter=RuleFilter(pruned_areas=(PageArea.NAV, PageArea.FOOTER)),
        ranker=None,
    )
    obs = env(screener).reset([episode()])[0]
    assert obs.candidate_ids == ["prod"]
    assert obs.teacher == 0
    assert obs.n_actions == 2


def test_screening_that_drops_target_falls_back_to_eoa():
    """目标候选被 LLM 筛掉时教师动作落到 [EOA]。

    让排序器只保留 nav，prod（教师目标）被剔除。
    """
    backend = ScriptedBackend(responses=['{"indices": [0]}'])
    screener = TwoStageScreener(
        rule_filter=None, ranker=LLMRanker(backend=backend, k=1)
    )
    obs = env(screener).reset([episode()])[0]
    assert "prod" not in obs.candidate_ids
    # 没有可达目标的候选可选时，学会停止好过学会点错误链接。
    assert obs.teacher == obs.eoa_index


def test_screening_stats_available_and_empty_without_screener():
    assert env().screening_stats() == {}
    screener = TwoStageScreener(rule_filter=RuleFilter(), ranker=None)
    e = env(screener)
    e.reset([episode()])
    assert "CR" in e.screening_stats()["screening"]


def test_url_for_action_returns_none_for_eoa():
    obs = env().reset([episode()])[0]
    assert obs.url_for_action(1) == "S_1"
    assert obs.url_for_action(obs.eoa_index) is None
    assert obs.url_for_action(99) is None
