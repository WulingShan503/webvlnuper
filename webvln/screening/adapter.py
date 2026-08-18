"""模拟器候选格式的对接层。

WebVLN 官方模拟器（``r2r_src/env.py``）中，一步的候选来自
``connectivity[websiteID][urlID]["data"]``，形如：

    {
      "<clickable_id>": {
        "clickable_id": "...",
        "next_url_id": "...",
        "text": ["Product Reviews", ...],   # 注意是列表
        "href_full": "https://site/collections/x/products/y",
        "imgs": ["img_0012.jpg", ...]
      },
      ...
    }

``make_candidate`` 随后把每个条目转成三段特征
（文本 / 按钮图 / 截图裁剪），并以 ``clickable_id`` 为键返回有序字典。
筛选必须发生在这一步**之前**，才能省下被剔除候选的特征查表与前向计算。

本模块负责把上述原始字典转为 ``Candidate`` 列表，并保证：

1. ``index`` 与 ``make_candidate`` 遍历顺序严格一致——决策层的 logits
   按该顺序排布，下标错位会让模型选中完全无关的链接；
2. ``clickable_id`` / ``next_url_id`` 被保留，使筛选结果能映射回
   特征字典与模拟器的跳转动作。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from webvln.screening.area import infer_area
from webvln.screening.candidate import Candidate, ElementType, PageArea

#: [EOA] 停止动作的候选键。基线模型把它与可点击候选并列送入 softmax，
#: 但它不是页面元素，不参与筛选——一旦被规则或 LLM 剔除，
#: 智能体将永远无法停止并作答。
EOA_KEY = "[EOA]"


def candidate_from_record(
    index: int,
    record: Mapping[str, Any],
    clickable_id: Optional[str] = None,
) -> Candidate:
    """把一条原始候选记录转为 ``Candidate``。

    Args:
        index: 该候选在本步候选序列中的位置，须与 ``make_candidate``
            的遍历顺序一致。
        record: ``map.json`` 中的单条候选记录。
        clickable_id: 候选键。记录内通常自带 ``clickable_id``，
            但字典键才是权威来源，故允许显式传入覆盖。

    Returns:
        Candidate。
    """
    text = _first_text(record.get("text"))
    href = str(record.get("href_full") or record.get("href") or "")
    imgs = record.get("imgs") or []
    dom_path = str(record.get("dom_path") or record.get("xpath") or "")

    cid = clickable_id or record.get("clickable_id") or ""

    # 官方数据未标注元素类型。候选全部来自 href 可跳转的可点击元素，
    # 按 LINK 处理；若记录显式给出标签则据其判定。
    tag = record.get("tag") or record.get("tag_name")
    elem_type = ElementType.from_tag(tag) if tag else ElementType.LINK

    area = record.get("area")
    if area is None:
        area = infer_area(href=href, text=text, dom_path=dom_path)
    elif isinstance(area, str):
        area = _coerce_area(area)

    return Candidate(
        index=index,
        text=text,
        elem_type=elem_type,
        area=area,
        context=_first_text(record.get("context")),
        href=href,
        dom_path=dom_path,
        has_image=bool(imgs),
        raw={
            "clickable_id": str(cid),
            "next_url_id": str(record.get("next_url_id") or ""),
            "imgs": list(imgs),
        },
    )


def candidates_from_state(
    state_candidates: Mapping[str, Mapping[str, Any]],
    skip_keys: Sequence[str] = (EOA_KEY,),
) -> List[Candidate]:
    """把模拟器一步的候选字典转为 ``Candidate`` 列表。

    Python 3.7+ 的 dict 保序，``make_candidate`` 中
    ``enumerate(state['candidate'])`` 的顺序即插入顺序，
    这里的遍历顺序与之相同，因此 ``index`` 可直接对应特征张量的行。

    Args:
        state_candidates: ``state['candidate']``。
        skip_keys: 不参与筛选的键，默认排除 [EOA]。

    Returns:
        Candidate 列表，``index`` 从 0 连续编号。
    """
    skip = set(skip_keys)
    out: List[Candidate] = []
    for index, (key, record) in enumerate(state_candidates.items()):
        if key in skip:
            continue
        if not isinstance(record, Mapping):
            continue
        out.append(candidate_from_record(index, record, clickable_id=key))
    return out


def build_instruction(
    question: str, description: str = "", separator: str = " "
) -> str:
    """拼接导航指令。

    WebVLN 的指令由问题 Q 与辅助描述 D 组成（论文 3.2 节以
    ``[CLS] Q [SEP] D [SEP]`` 输入 BERT）。论文脚注指出，
    当问题本身足以定位目标页时 D 为空，此处相应地跳过空串，
    避免在提示词中留下多余分隔符。
    """
    parts = [p.strip() for p in (question, description) if p and p.strip()]
    return separator.join(parts)


def instruction_from_obs(obs: Mapping[str, Any]) -> str:
    """从 ``_get_obs()`` 的观测项中取出指令文本。

    官方观测把指令放在 ``text`` 字段（``item['text']``）。
    若该字段是 Q / D 两段的列表或字典，一并拼接。
    """
    text = obs.get("text")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, Mapping):
        return build_instruction(
            str(text.get("Q") or text.get("question") or ""),
            str(text.get("D") or text.get("description") or ""),
        )
    if isinstance(text, Sequence):
        return build_instruction(*(str(t) for t in list(text)[:2]))
    return ""


def apply_screening_to_state(
    state_candidates: Mapping[str, Mapping[str, Any]],
    kept_indices: Iterable[int],
    skip_keys: Sequence[str] = (EOA_KEY,),
) -> Dict[str, Mapping[str, Any]]:
    """按筛选结果重建候选字典，供 ``make_candidate`` 继续处理。

    [EOA] 等 ``skip_keys`` 中的键无论是否被选中都会保留：它们不是页面元素，
    剔除会让智能体失去停止动作。

    Args:
        state_candidates: 原始候选字典。
        kept_indices: 筛选保留的下标（``Candidate.index``）。
        skip_keys: 始终保留的键。

    Returns:
        新的候选字典。顺序按原始下标升序，而非 LLM 的相关性顺序——
        决策层的候选顺序须在筛选前后保持一致的语义，
        否则同一页面在不同步会得到不同的 logits 排布。
    """
    keep = set(kept_indices)
    skip = set(skip_keys)
    result: Dict[str, Mapping[str, Any]] = {}
    for index, (key, record) in enumerate(state_candidates.items()):
        if key in skip or index in keep:
            result[key] = record
    return result


def _first_text(value: Any) -> str:
    """取候选文本。

    官方 ``text`` 字段是列表（同一元素可能含多段文本，如图片 alt
    与相邻标题）。``make_candidate`` 注释中的用法为 ``cur_cc['text'][0]``，
    这里沿用首项为主文本，其余合并为补充，避免丢失可用语义。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return _first_text(list(value.values()))
    if isinstance(value, Sequence):
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return " ".join(parts)
    return str(value).strip()


def _coerce_area(value: str) -> PageArea:
    """把字符串转为 PageArea，兼容枚举名与枚举值两种写法。"""
    try:
        return PageArea(value)
    except ValueError:
        pass
    try:
        return PageArea[value.strip().upper()]
    except KeyError:
        return PageArea.UNKNOWN
