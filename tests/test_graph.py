"""导航图与最短路径表的单元测试。

结构取自官方 ``map.json`` / ``shortest_paths.json``：
``connectivity[websiteID][urlID]["data"][clickable_id]`` 与
``shortest_paths[websiteID][起点][终点] -> [起点, ..., 终点]``。
"""

import json
import os

from webvln.data.graph import MAP_FILE, SHORTEST_PATHS_FILE, NavigationGraph


def record(cid, next_url):
    return {
        "clickable_id": cid,
        "next_url_id": next_url,
        "text": [f"link {cid}"],
        "href_full": f"https://site/{cid}",
        "imgs": [],
    }


def graph():
    connectivity = {
        "SAHB": {
            "SAHB_0": {"data": {"c1": record("c1", "SAHB_1"), "c2": record("c2", "SAHB_2")}},
            "SAHB_1": {"data": {"c3": record("c3", "SAHB_2")}},
            "SAHB_2": {"data": {}},  # 叶子页：只被指向，自身无出边
        }
    }
    shortest_paths = {
        "SAHB": {
            "SAHB_0": {"SAHB_2": ["SAHB_0", "SAHB_2"], "SAHB_1": ["SAHB_0", "SAHB_1"]},
            "SAHB_1": {"SAHB_2": ["SAHB_1", "SAHB_2"]},
            "SAHB_2": {"SAHB_2": ["SAHB_2"]},
        }
    }
    return NavigationGraph(connectivity, shortest_paths)


def test_candidates_and_transition():
    g = graph()
    assert set(g.candidates("SAHB", "SAHB_0")) == {"c1", "c2"}
    assert g.next_url_id("SAHB", "SAHB_0", "c2") == "SAHB_2"


def test_missing_page_returns_empty_instead_of_raising():
    g = graph()
    # 官方在无出边或未知页面上取候选会 KeyError。
    assert g.candidates("SAHB", "SAHB_2") == {}
    assert g.candidates("SAHB", "nope") == {}
    assert g.candidates("nope", "SAHB_0") == {}
    assert g.next_url_id("SAHB", "SAHB_0", "unknown") is None


def test_shortest_path_and_distance():
    g = graph()
    assert g.shortest_path("SAHB", "SAHB_0", "SAHB_2") == ["SAHB_0", "SAHB_2"]
    assert g.distance("SAHB", "SAHB_1", "SAHB_2") == 2
    # 不可达不抛错：官方三重下标索引会中断整个 batch。
    assert g.shortest_path("SAHB", "SAHB_2", "SAHB_0") == []
    assert g.distance("SAHB", "SAHB_2", "SAHB_0") == 0


def test_teacher_clickable_id_is_stable_key():
    g = graph()
    assert g.teacher_clickable_id("SAHB", "SAHB_0", "SAHB_1") == "c1"
    assert g.teacher_clickable_id("SAHB", "SAHB_0", "SAHB_9") is None


def test_teacher_action_index_uses_candidate_order():
    g = graph()
    candidates = {"c1": {"urlID": "SAHB_1"}, "c2": {"urlID": "SAHB_2"}}
    assert g.teacher_action_index(candidates, "SAHB_1") == 0
    assert g.teacher_action_index(candidates, "SAHB_2") == 1


def test_teacher_action_index_returns_eoa_slot():
    g = graph()
    candidates = {"c1": {"urlID": "SAHB_1"}, "c2": {"urlID": "SAHB_2"}}
    # 已在目标页：教师动作是 [EOA]，位于候选之后的最后一位。
    assert g.teacher_action_index(candidates, None) == 2
    # 目标候选被筛掉时同样落到 [EOA]，而非误指某个真实候选。
    assert g.teacher_action_index({"c1": {"urlID": "SAHB_1"}}, "SAHB_2") == 1


def test_teacher_action_index_accepts_next_url_id_key():
    # 筛选前的原始候选用 next_url_id，筛选后经 make_candidate 变成 urlID。
    g = graph()
    assert g.teacher_action_index({"c1": {"next_url_id": "SAHB_1"}}, "SAHB_1") == 0


def test_candidate_count_stats():
    g = graph()
    stats = g.candidate_count_stats()
    assert stats["n_pages"] == 3
    assert stats["max"] == 2.0
    assert stats["min"] == 0.0
    assert NavigationGraph({}, {}).candidate_count_stats()["n_pages"] == 0


def test_websites_and_iter_pages():
    g = graph()
    assert g.websites() == ["SAHB"]
    assert sorted(g.iter_pages()) == ["SAHB/SAHB_0", "SAHB/SAHB_1", "SAHB/SAHB_2"]


def test_from_dir_loads_both_files(tmp_path):
    g = graph()
    (tmp_path / MAP_FILE).write_text(json.dumps(g.connectivity), encoding="utf-8")
    (tmp_path / SHORTEST_PATHS_FILE).write_text(
        json.dumps(g.shortest_paths), encoding="utf-8"
    )

    loaded = NavigationGraph.from_dir(str(tmp_path))
    assert loaded.next_url_id("SAHB", "SAHB_0", "c1") == "SAHB_1"
    assert loaded.distance("SAHB", "SAHB_0", "SAHB_2") == 2


def test_from_dir_tolerates_missing_files(tmp_path):
    loaded = NavigationGraph.from_dir(str(tmp_path))
    assert loaded.websites() == []
    assert loaded.candidates("SAHB", "SAHB_0") == {}
