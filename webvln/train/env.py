"""模拟器与观测构造。

对应官方 ``r2r_src/env.py`` 的 ``Simulator`` / ``EnvBatch`` / ``R2RBatch``，
但只保留数据流：一批 episode 各自处在某个页面，每步给出观测
（候选特征 + 教师动作 + 到目标的距离），接受动作后跳转页面。

第四章的候选筛选插在 ``make_candidate`` **之前**——这里体现为
``Observation`` 构造时先对原始候选调 ``screener``，再查特征表。
筛选后候选变少，特征查表与后续前向计算相应减少，这正是论文
5.5 节所述 API 与算力开销的来源。

不依赖 torch：特征以 list 形式传出，由 ``trainer.py`` 负责堆成张量。
这样 rollout 的下标对齐、停止判定与轨迹记录都能在本机测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from webvln.data.episode import Episode
from webvln.data.features import FeatureStore
from webvln.data.graph import NavigationGraph
from webvln.screening.integration import screen_state
from webvln.screening.pipeline import TwoStageScreener

#: 停止动作的键，与 ``screening/adapter.py`` 保持一致。
EOA_KEY = "[EOA]"


@dataclass
class Observation:
    """一步的观测，对应官方 ``_get_obs()`` 的一个元素。

    Attributes:
        idx: episode ID。
        website_id: 网站 ID。
        url_id: 当前页面。
        candidate_ids: 候选键，顺序与 ``candidate_feats`` 的行一致。
        candidate_feats: 每个候选的三段特征 ``[文本, 按钮图, 截图]``。
        candidate_url_ids: 各候选点击后到达的页面。
        teacher: 教师动作下标。``len(candidate_ids)`` 表示 [EOA]。
        distance: 到目标页的剩余距离，仅用于日志。
        gt_path: ground-truth 路径。
        text_enc: 指令 token id。
        answer_enc: 答案解码器输入。
        answer_enc_w_eos: 答案预测目标。
    """

    idx: Any
    website_id: str
    url_id: str
    candidate_ids: List[str] = field(default_factory=list)
    candidate_feats: List[List[Any]] = field(default_factory=list)
    candidate_url_ids: List[str] = field(default_factory=list)
    teacher: int = 0
    distance: int = 0
    gt_path: List[str] = field(default_factory=list)
    text_enc: List[int] = field(default_factory=list)
    answer_enc: List[int] = field(default_factory=list)
    answer_enc_w_eos: List[int] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def n_actions(self) -> int:
        """动作空间大小：候选数 + 1（[EOA]）。

        对应官方 ``candidate_leng = len(ob['candidate']) + 1``。
        """
        return len(self.candidate_ids) + 1

    @property
    def eoa_index(self) -> int:
        """[EOA] 在动作空间中的下标，即最后一位。"""
        return len(self.candidate_ids)

    def url_for_action(self, action: int) -> Optional[str]:
        """动作下标对应的目标页面。[EOA] 或越界返回 None。"""
        if 0 <= action < len(self.candidate_url_ids):
            return self.candidate_url_ids[action]
        return None


class WebVLNEnv:
    """一批 episode 的导航环境。

    Attributes:
        graph: 导航图。
        features: 特征表。
        screener: 候选筛选器。为 None 时走基线路径（不筛选）。
    """

    def __init__(
        self,
        graph: NavigationGraph,
        features: FeatureStore,
        screener: Optional[TwoStageScreener] = None,
    ) -> None:
        self.graph = graph
        self.features = features
        self.screener = screener
        self.batch: List[Episode] = []
        self.url_ids: List[str] = []

    def reset(self, batch: Sequence[Episode]) -> List[Observation]:
        """开始新一批 episode，各自置于起始页。"""
        self.batch = list(batch)
        self.url_ids = [ep.start_url_id for ep in self.batch]
        return self.observe()

    def step(self, actions: Sequence[int]) -> List[Observation]:
        """执行一批动作并返回新观测。

        动作为 -1（停止 / 已结束 / 忽略）时页面不变——官方
        ``make_equiv_action`` 中 ``action != -1`` 的判断即此意。
        """
        for i, action in enumerate(actions):
            if action == -1:
                continue
            obs_url = self._observe_one(i).url_for_action(action)
            if obs_url is not None:
                self.url_ids[i] = obs_url
        return self.observe()

    def observe(self) -> List[Observation]:
        """构造整批观测。"""
        return [self._observe_one(i) for i in range(len(self.batch))]

    def _observe_one(self, i: int) -> Observation:
        ep = self.batch[i]
        website_id = ep.website_id or ""
        url_id = self.url_ids[i]

        raw = self.graph.candidates(website_id, url_id)

        # 教师动作先按**原始**候选定位到 clickable_id：筛选会改变候选集合，
        # 用键定位再映射到筛选后的下标，才不会指向错误的候选。
        teacher_url = self._teacher_url(ep, url_id)
        teacher_key = None
        if teacher_url is not None:
            teacher_key = self.graph.teacher_clickable_id(
                website_id, url_id, teacher_url
            )

        if self.screener is not None and raw:
            state = screen_state(
                {"websiteID": website_id, "urlID": url_id, "candidate": raw},
                item={"text": ep.text},
                screener=self.screener,
                instruction=ep.text,
                target_clickable_id=teacher_key,
            )
            raw = state.get("candidate") or {}

        candidate_ids = list(raw.keys())
        candidate_url_ids = [
            str(raw[cid].get("next_url_id") or "") for cid in candidate_ids
        ]
        candidate_feats = [
            self.features.candidate_features(cid, raw[cid].get("imgs") or [])
            for cid in candidate_ids
        ]

        # 教师动作：目标候选在筛选后的序列里找不到（被筛掉，或本就已在目标页）
        # 时落到 [EOA]。让智能体学会停止，好过学会点一个错误链接。
        if teacher_key is not None and teacher_key in candidate_ids:
            teacher = candidate_ids.index(teacher_key)
        else:
            teacher = len(candidate_ids)

        return Observation(
            idx=ep.idx,
            website_id=website_id,
            url_id=url_id,
            candidate_ids=candidate_ids,
            candidate_feats=candidate_feats,
            candidate_url_ids=candidate_url_ids,
            teacher=teacher,
            distance=self.graph.distance(website_id, url_id, ep.target_url_id),
            gt_path=list(ep.path),
            text_enc=list(ep.text_enc),
            answer_enc=list(ep.answer_enc),
            answer_enc_w_eos=list(ep.answer_enc_w_eos),
        )

    def _teacher_url(self, ep: Episode, url_id: str) -> Optional[str]:
        """当前页面上教师应跳转到的页面。

        用**最短路径**而非 ground-truth 路径的下一跳：智能体偏离
        ground-truth 后，按 gt_path 的固定下标取下一跳会指向一个当前页面
        根本没有链接的页面，教师动作随即退化为 [EOA]。
        官方 ``_argmax_action`` 用 ``ob['teacher'][1]``（即最短路径的第二个页面）
        正是这个道理。
        """
        path = self.graph.shortest_path(ep.website_id or "", url_id, ep.target_url_id)
        if len(path) <= 1:
            # 已在目标页（或不可达）：教师动作是停止。
            return None
        return path[1]

    def screening_stats(self) -> Dict[str, Any]:
        """筛选统计（CR / RR / API 调用与缓存命中）。

        无筛选器时返回空字典，便于 5.2 节基线与 5.3 节消融共用同一份日志代码。
        """
        if self.screener is None:
            return {}
        return self.screener.stats()
