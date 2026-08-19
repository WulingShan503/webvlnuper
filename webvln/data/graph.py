"""网站导航图与最短路径表。

官方把每个网站的页面连接关系存在 ``map.json``：

    connectivity[websiteID][urlID]["data"][clickable_id] -> 候选记录

最短路径存在 ``shortest_paths.json``：

    shortest_paths[websiteID][起点 urlID][终点 urlID] -> [起点, ..., 终点]

这两张表撑起三件事：模拟器的状态转移（点击哪个候选去哪个页面）、
教师动作（下一步该点哪个候选）、以及 SPL 与 TL 的分母（ground-truth 路径长度）。

官方把它们直接塞进 ``R2RBatch`` 与 ``Evaluation`` 两处各读一遍
（``eval.py`` 里又 ``json.load`` 了一次 shortest_paths）。这里抽成独立对象，
训练与评测共享同一份，省下一次数 GB 的重复解析。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

MAP_FILE = "map.json"
SHORTEST_PATHS_FILE = "shortest_paths.json"


class NavigationGraph:
    """一个数据集的全部网站导航图。

    Attributes:
        connectivity: ``map.json`` 的内容。
        shortest_paths: ``shortest_paths.json`` 的内容。
    """

    def __init__(
        self,
        connectivity: Mapping[str, Mapping[str, Mapping[str, Any]]],
        shortest_paths: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]],
    ) -> None:
        self.connectivity = connectivity
        self.shortest_paths = shortest_paths

    @classmethod
    def from_dir(cls, data_dir: str) -> "NavigationGraph":
        """从数据目录加载两张表。"""
        return cls(
            connectivity=_load_json(os.path.join(data_dir, MAP_FILE)),
            shortest_paths=_load_json(os.path.join(data_dir, SHORTEST_PATHS_FILE)),
        )

    # --- 候选与状态转移 -----------------------------------------------------

    def candidates(self, website_id: str, url_id: str) -> Dict[str, Any]:
        """取某页面的候选字典，对应 ``Simulator.load_candidate``。

        缺失页面返回空字典而非抛错：图中存在只被指向、自身无出边的叶子页
        （如售罄商品页），官方在这类页面上取候选会 KeyError。
        """
        site = self.connectivity.get(website_id) or {}
        page = site.get(url_id) or {}
        return dict(page.get("data") or {})

    def next_url_id(
        self, website_id: str, url_id: str, clickable_id: str
    ) -> Optional[str]:
        """点击某候选后到达的页面。为未知候选时返回 None。"""
        record = self.candidates(website_id, url_id).get(clickable_id)
        if not record:
            return None
        nxt = record.get("next_url_id")
        return str(nxt) if nxt is not None else None

    def websites(self) -> List[str]:
        """全部网站 ID。论文 2.3 节：三个购物网站，各划分均覆盖。"""
        return sorted(self.connectivity.keys())

    # --- 最短路径 -----------------------------------------------------------

    def shortest_path(
        self, website_id: str, from_url: str, to_url: str
    ) -> List[str]:
        """两页面间的最短路径（含起点与终点）。

        不可达时返回空列表。官方 ``_get_obs`` 直接三重下标索引，
        遇到不可达的组合会 KeyError 中断整个 batch。
        """
        site = self.shortest_paths.get(website_id) or {}
        starts = site.get(from_url) or {}
        return [str(p) for p in (starts.get(to_url) or [])]

    def distance(self, website_id: str, from_url: str, to_url: str) -> int:
        """到目标页的剩余距离，即最短路径的页面数。

        与官方 ``obs['distance'] = len(shortest_paths[...])`` 一致。
        不可达返回 0——官方在这里没有兜底，这个值只用于日志与 A2C 奖励，
        返回 0 不会污染 SR / SPL。
        """
        return len(self.shortest_path(website_id, from_url, to_url))

    def teacher_clickable_id(
        self, website_id: str, url_id: str, next_url_id: str
    ) -> Optional[str]:
        """当前页面上通往 ``next_url_id`` 的候选键。

        对应官方 ``_teacher_action`` 里遍历候选比对 ``urlID`` 的那段。
        返回候选键而非下标，因为筛选会改变候选集合、下标随之变化，
        而键在整个流程中稳定（见 ``screening/adapter.py`` 的说明）。
        """
        for clickable_id, record in self.candidates(website_id, url_id).items():
            if str(record.get("next_url_id") or "") == str(next_url_id):
                return clickable_id
        return None

    def teacher_action_index(
        self,
        candidates: Mapping[str, Mapping[str, Any]],
        next_url_id: Optional[str],
    ) -> int:
        """教师动作在给定候选序列中的下标。

        Args:
            candidates: 当前步的候选字典（**筛选之后**的那份，
                否则下标与模型 logits 对不上）。
            next_url_id: 下一步应到达的页面。为 None 表示已在目标页，
                教师动作是 [EOA]。

        Returns:
            候选下标；[EOA] 时返回 ``len(candidates)``——官方把停止动作
            放在候选之后的最后一位（``a[i] = len(ob['candidate'])``）。
            目标候选被筛掉时同样返回该值：没有可达目标的候选可选时，
            让智能体学会停止比学会点一个错误链接更好。
        """
        if next_url_id is None:
            return len(candidates)
        for k, record in enumerate(candidates.values()):
            url = record.get("urlID", record.get("next_url_id"))
            if str(url or "") == str(next_url_id):
                return k
        return len(candidates)

    # --- 自检 ---------------------------------------------------------------

    def iter_pages(self) -> Iterator[str]:
        """遍历全部 ``websiteID/urlID``，用于统计候选规模。"""
        for website_id, site in self.connectivity.items():
            for url_id in site:
                yield f"{website_id}/{url_id}"

    def candidate_count_stats(self) -> Dict[str, float]:
        """全图的候选数分布。

        论文 4.1 / 3.3 节称平均约 45、最多约 100 个候选，
        这是两阶段筛选的动机数据；加载后跑一次可核对数据版本是否一致。
        """
        counts = [
            len(page.get("data") or {})
            for site in self.connectivity.values()
            for page in site.values()
        ]
        if not counts:
            return {"n_pages": 0, "avg": 0.0, "max": 0.0, "min": 0.0}
        return {
            "n_pages": len(counts),
            "avg": round(sum(counts) / len(counts), 2),
            "max": float(max(counts)),
            "min": float(min(counts)),
        }


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh) or {}
