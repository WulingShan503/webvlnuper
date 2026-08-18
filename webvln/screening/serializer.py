"""4.2 候选特征的语义文本重构。

基线模型中的候选是结构化多模态特征（4864 维拼接向量），LLM 无法直接读取。
本模块把候选还原为自然语言描述，使其可以进入 LLM 的上下文窗口。

论文给出的模板为：

    [TYPE: LINK][TEXT: "Product Reviews"]located in the [AREA: main content
    section]near [CONTEXT: "Customer Feedback"]

保留类型、文本、区域、上下文四个字段，是因为它们分别对应「元素能做什么」、
「元素说了什么」、「元素在页面何处」、「元素周围在讲什么」，
四者共同决定了元素与导航指令的语义相关性。
"""

from __future__ import annotations

from typing import Iterable, List

from webvln.screening.candidate import Candidate

#: innerText 截断长度。论文 4.2 节规定超过 100 字符的文本予以截断，
#: 以控制送入 LLM 的 token 数——过长的正文段落对判断可点击意图并无增益。
MAX_TEXT_LEN = 100

#: 上下文文本相对更次要，采用更短的预算。
MAX_CONTEXT_LEN = 60


def truncate(text: str, limit: int) -> str:
    """按字符数截断文本，超长时补省略号。"""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def serialize(candidate: Candidate) -> str:
    """把单个候选转为语义文本描述。

    区域与上下文字段可能缺失（模拟器未提供 DOM 位置，或元素周围无文本），
    此时省略对应片段，而不是输出 "unspecified area" 这类噪声词——
    空信息不应占用 LLM 的注意力。
    """
    text = truncate(candidate.text, MAX_TEXT_LEN) if candidate.text else ""
    parts = [f"[TYPE: {candidate.elem_type.value}]", f'[TEXT: "{text}"]']

    if candidate.area.name != "UNKNOWN":
        parts.append(f"located in the [AREA: {candidate.area.value}]")

    if candidate.context:
        ctx = truncate(candidate.context, MAX_CONTEXT_LEN)
        parts.append(f'near [CONTEXT: "{ctx}"]')

    return "".join(parts)


def serialize_all(candidates: Iterable[Candidate]) -> List[str]:
    """批量序列化。"""
    return [serialize(c) for c in candidates]


def build_candidate_block(candidates: Iterable[Candidate]) -> str:
    """构造送入 LLM 提示词的候选清单。

    每行前缀为候选的 ``index``，而非其在清单中的位置。规则过滤会在
    LLM 排序之前剔除部分候选，若改用行号，LLM 返回的编号将无法映射回
    原始特征张量的行。以 ``index`` 为准可让两阶段解耦。
    """
    lines = [f"{c.index}. {serialize(c)}" for c in candidates]
    return "\n".join(lines)
