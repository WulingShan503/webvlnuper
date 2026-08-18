"""4.1 候选动作的数据表示。

WebVLN 模拟器在每一步给出当前页面的可点击元素集合。基线模型直接把这些元素的
多模态特征（文本 768 + 截图 2048 + 按钮图 2048）拼接后送入决策层，
不区分元素的语义价值。筛选机制需要在特征编码之前介入，因此这里保留
元素的原始结构化信息（文本、类型、DOM 位置、上下文），供后续序列化与排序使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ElementType(str, Enum):
    """可点击元素的类型。

    论文 4.2 节将 DOM 标签归并为三类，作为序列化模板中的 [TYPE:] 字段。
    """

    LINK = "LINK"
    BUTTON = "BUTTON"
    INPUT = "INPUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_tag(cls, tag: Optional[str]) -> "ElementType":
        """由 HTML 标签名推断元素类型。"""
        if not tag:
            return cls.UNKNOWN
        tag = tag.strip().lower()
        if tag == "a":
            return cls.LINK
        if tag in ("button", "summary"):
            return cls.BUTTON
        if tag in ("input", "select", "textarea"):
            return cls.INPUT
        return cls.UNKNOWN


class PageArea(str, Enum):
    """元素在页面中所处的区域。

    区域是规则过滤（4.4 节）的主要依据：导航栏、页脚、侧边栏中的链接
    多为高频通用链接，与具体指令的相关性通常低于主内容区。
    """

    MAIN = "main content section"
    NAV = "navigation bar"
    HEADER = "page header"
    FOOTER = "page footer"
    SIDEBAR = "sidebar"
    UNKNOWN = "unspecified area"


@dataclass
class Candidate:
    """一个候选点击动作。

    Attributes:
        index: 该候选在当前步原始候选列表中的下标。LLM 排序返回的是下标，
            需靠它映射回原始特征张量的行，因此在筛选全程保持不变。
        text: 元素的可见文本；对图片链接则取 ``alt`` 属性。
        elem_type: 元素类型。
        area: 所处页面区域。
        context: 邻近文本（如所属区块标题），为 LLM 提供判断依据。
        href: 链接目标 URL，规则过滤据此识别广告 / 追踪链接。
        dom_path: DOM 路径或 XPath，用于区域推断与调试。
        has_image: 是否携带按钮图像特征。
        raw: 模拟器给出的原始字段，便于回溯，不参与筛选逻辑。
    """

    index: int
    text: str = ""
    elem_type: ElementType = ElementType.UNKNOWN
    area: PageArea = PageArea.UNKNOWN
    context: str = ""
    href: str = ""
    dom_path: str = ""
    has_image: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 模拟器字段可能为 None 或带首尾空白，统一归一化，
        # 避免后续序列化与去重时出现 "None" 字面量或空白差异。
        self.text = _clean(self.text)
        self.context = _clean(self.context)
        self.href = (self.href or "").strip()
        self.dom_path = (self.dom_path or "").strip()
        if isinstance(self.elem_type, str):
            self.elem_type = ElementType(self.elem_type)
        if isinstance(self.area, str):
            self.area = PageArea(self.area)

    @property
    def is_empty(self) -> bool:
        """是否无任何可供语义判断的文本。

        这类候选（如纯图标链接且缺失 alt）无法被 LLM 有效排序，
        规则过滤阶段会将其剔除。
        """
        return not self.text and not self.context

    def dedup_key(self) -> str:
        """去重键。

        论文 4.4 节指出页面常存在指向同一目标的重复链接（如导航栏与页脚
        并列出现的同名入口）。文本与 href 同时相同即视为重复。
        """
        return f"{self.text.lower()}|{self.href.lower()}"


def _clean(value: Optional[str]) -> str:
    """折叠空白并去除首尾空格。"""
    if not value:
        return ""
    return " ".join(str(value).split())
