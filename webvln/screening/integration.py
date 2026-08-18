"""与 WebVLN 官方实现的集成入口。

官方 ``r2r_src/env.py`` 中，一步观测的构造流程为：

    _get_obs()  ->  make_candidate(feature, state)  ->  obs['candidate']

筛选须插在 ``make_candidate`` **之前**：该函数会为每个候选查三张特征表
（text_feats / img_feats / screenshot_crop_feats）并拼成 4864 维向量，
先筛选才能省下被剔除候选的查表与后续前向计算。

集成方式是在 ``R2RBatch._get_obs`` 中改一行：

    for i, (feature, state) in enumerate(self.env.getStates()):
        item = self.batch[i]
    +   state = screen_state(state, item, screener)      # 新增
        candidate_feature = self.make_candidate(feature, state)

``screen_state`` 返回结构相同、仅候选变少的 state，
因此 ``make_candidate`` 及其下游无需任何改动。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from webvln.screening.adapter import (
    EOA_KEY,
    apply_screening_to_state,
    candidates_from_state,
    instruction_from_obs,
)
from webvln.screening.pipeline import ScreeningOutput, TwoStageScreener


def screen_state(
    state: Mapping[str, Any],
    item: Optional[Mapping[str, Any]] = None,
    screener: Optional[TwoStageScreener] = None,
    instruction: Optional[str] = None,
    target_clickable_id: Optional[str] = None,
) -> Dict[str, Any]:
    """对模拟器 state 中的候选做两阶段筛选。

    Args:
        state: ``Simulator.getState()`` 的返回值，含 ``candidate`` 字典。
        item: ``self.batch[i]``，用于取出指令文本。
        screener: 筛选器。为 None 时原样返回，等价于基线。
        instruction: 显式指令，优先于从 ``item`` 推断。
        target_clickable_id: 教师动作对应的 ``clickable_id``，仅用于统计 RR。

    Returns:
        新的 state 字典（浅拷贝，仅替换 ``candidate``）。
        原 state 不被修改——模拟器内部持有同一对象，
        就地修改会让后续步骤看到被裁剪的候选图。
    """
    if screener is None:
        return dict(state)

    raw = state.get("candidate") or {}
    if not raw:
        return dict(state)

    if instruction is None:
        instruction = instruction_from_obs(item or {})

    candidates = candidates_from_state(raw)
    target_index = _resolve_target_index(candidates, target_clickable_id)

    out = screener.screen(instruction, candidates, target_index=target_index)

    new_state = dict(state)
    new_state["candidate"] = apply_screening_to_state(raw, out.kept_indices)
    return new_state


def screen_candidates(
    raw_candidates: Mapping[str, Mapping[str, Any]],
    instruction: str,
    screener: TwoStageScreener,
    target_clickable_id: Optional[str] = None,
) -> ScreeningOutput:
    """直接筛选候选字典并返回筛选结果对象。

    需要读取 CR / RR 或调试筛选行为时使用；仅需接入训练流程用
    ``screen_state`` 即可。
    """
    candidates = candidates_from_state(raw_candidates)
    target_index = _resolve_target_index(candidates, target_clickable_id)
    return screener.screen(instruction, candidates, target_index=target_index)


def _resolve_target_index(candidates, target_clickable_id: Optional[str]) -> Optional[int]:
    """把教师动作的 clickable_id 映射为候选下标。

    教师动作可能是 [EOA]（已到达目标页），或指向已被上一步跳转移除的元素，
    此时返回 None——该步不计入 RR 分母，否则会把「本就无候选目标」
    的步算作召回失败，低估筛选质量。
    """
    if not target_clickable_id or target_clickable_id == EOA_KEY:
        return None
    for cand in candidates:
        if cand.raw.get("clickable_id") == target_clickable_id:
            return cand.index
    return None
