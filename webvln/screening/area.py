"""页面区域推断。

论文 4.4 节的区域剪枝规则依赖元素所处的页面区域，但 WebVLN 官方模拟器的
候选记录（``map.json`` 中 ``connectivity[websiteID][urlID]["data"]`` 的条目）
只包含 ``clickable_id`` / ``next_url_id`` / ``text`` / ``href_full`` / ``imgs``
五个字段，**不含 DOM 路径或区域标注**。因此区域必须从 href 与文本反推。

推断依据是购物网站的通用结构约定：页脚集中放置法务与客服链接
（Privacy Policy、Terms、Contact Us），导航栏放置品类入口（Shop、Collections），
侧边栏放置筛选与订阅控件。这些链接的 URL 路径与锚文本高度模式化，
足以支撑区域级别的粗分类。

推断是启发式的，会有误判。规则过滤因此保留 ``min_keep`` 回填与
可关闭区域剪枝的开关：当误判把目标链接判入页脚时，
整体召回率仍由论文报告的 98.2% 兜底，而非退化为不可用。
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

from webvln.screening.candidate import PageArea

#: 页脚特征词。法务、客服、公司信息类链接在购物站几乎只出现在页脚。
FOOTER_PATTERNS: Tuple[str, ...] = (
    "privacy",
    "terms",
    "conditions",
    "cookie",
    "disclaimer",
    "copyright",
    "accessibility",
    "sitemap",
    "contact",
    "about-us",
    "aboutus",
    "careers",
    "press",
    "affiliate",
    "wholesale",
    "shipping-policy",
    "return-policy",
    "refund",
    "faq",
    "help",
    "customer-service",
    "track-order",
    "gift-card",
)

#: 侧边栏特征词。订阅、筛选、排序类控件。
SIDEBAR_PATTERNS: Tuple[str, ...] = (
    "newsletter",
    "subscribe",
    "filter",
    "sort-by",
    "sortby",
    "refine",
    "facet",
    "price-range",
    "recently-viewed",
    "related-product",
    "you-may-also-like",
)

#: 导航栏特征词。品类与集合入口。
NAV_PATTERNS: Tuple[str, ...] = (
    "/collections",
    "/collection/",
    "/category",
    "/categories",
    "/shop",
    "/catalog",
    "/products?",
    "/all-products",
    "/new-arrivals",
    "/best-sellers",
    "/sale",
    "/brands",
    "/menu",
)

#: 页头特征词。账户与购物车入口。
HEADER_PATTERNS: Tuple[str, ...] = (
    "/cart",
    "/checkout",
    "/login",
    "/signin",
    "/sign-in",
    "/register",
    "/signup",
    "/sign-up",
    "/account",
    "/wishlist",
    "/search",
    "/logo",
)

#: 主内容区特征词。具体商品与其详情页。
MAIN_PATTERNS: Tuple[str, ...] = (
    "/products/",
    "/product/",
    "/item/",
    "/p/",
    "/dp/",
    "/reviews",
    "/description",
    "/specification",
)

#: DOM 路径中的语义标签。若数据集扩展后提供了 DOM 路径，
#: 它比 URL 猜测可靠得多，故优先采用。
_DOM_HINTS: Tuple[Tuple[str, PageArea], ...] = (
    ("footer", PageArea.FOOTER),
    ("aside", PageArea.SIDEBAR),
    ("sidebar", PageArea.SIDEBAR),
    ("nav", PageArea.NAV),
    ("header", PageArea.HEADER),
    ("banner", PageArea.HEADER),
    ("main", PageArea.MAIN),
    ("article", PageArea.MAIN),
    ("[role=contentinfo]", PageArea.FOOTER),
    ("[role=navigation]", PageArea.NAV),
    ("[role=main]", PageArea.MAIN),
)

_NON_WORD = re.compile(r"[^a-z0-9]+")


def infer_area(
    href: str = "",
    text: str = "",
    dom_path: str = "",
) -> PageArea:
    """推断元素所处的页面区域。

    优先级：DOM 路径 > href 路径 > 锚文本。DOM 路径若存在即为直接证据；
    href 次之，因为 URL 结构由站点模板决定，比人工撰写的锚文本更规整；
    锚文本最后，仅在前两者均无线索时使用。

    Args:
        href: 链接目标（官方字段为 ``href_full``）。
        text: 元素可见文本或 ``alt``。
        dom_path: DOM 路径 / XPath，官方数据集未提供，留作扩展。

    Returns:
        推断出的 PageArea；无任何线索时为 ``PageArea.UNKNOWN``。
        返回 UNKNOWN 而非默认 MAIN，是为了让区域剪枝对无证据的候选保持中立——
        默认成 MAIN 会掩盖数据缺失，默认成页脚则会误删正常候选。
    """
    if dom_path:
        area = _match_dom(dom_path)
        if area is not None:
            return area

    if href:
        area = _match_href(href)
        if area is not None:
            return area

    if text:
        area = _match_text(text)
        if area is not None:
            return area

    return PageArea.UNKNOWN


def _match_dom(dom_path: str) -> Optional[PageArea]:
    """按 DOM 路径中最靠后（最深）的语义标签判定。

    取最深匹配是因为 DOM 路径自外向内书写，越靠后的标签越贴近元素本身：
    ``body > footer > nav > a`` 中的元素属于页脚里的导航块，
    按最深匹配得到 NAV，按最浅则会误判为 FOOTER。
    """
    lowered = dom_path.lower()
    best: Optional[PageArea] = None
    best_pos = -1
    for hint, area in _DOM_HINTS:
        pos = lowered.rfind(hint)
        if pos > best_pos:
            best_pos = pos
            best = area
    return best


def _match_href(href: str) -> Optional[PageArea]:
    """按 href 路径关键词判定。

    只检查路径与查询部分，剔除协议与域名：域名中出现的
    "shop"（如 ``shop.example.com``）不应把站内全部链接判为导航栏。
    """
    path = _href_path(href.lower())
    if not path:
        return None
    # MAIN 先于其他类别判定：商品详情链接可能同时含有 /collections/
    # 这类导航特征（如 /collections/socks/products/blue-sock），
    # 而它实际是主内容区的商品入口。
    for patterns, area in (
        (MAIN_PATTERNS, PageArea.MAIN),
        (FOOTER_PATTERNS, PageArea.FOOTER),
        (SIDEBAR_PATTERNS, PageArea.SIDEBAR),
        (HEADER_PATTERNS, PageArea.HEADER),
        (NAV_PATTERNS, PageArea.NAV),
    ):
        if any(p in path for p in patterns):
            return area
    return None


def _match_text(text: str) -> Optional[PageArea]:
    """按锚文本判定。

    文本以空白与标点归一为连字符形式后匹配，使 "Privacy Policy"
    能命中 ``privacy``、"Contact Us" 能命中 ``contact``。
    """
    slug = _NON_WORD.sub("-", text.lower()).strip("-")
    if not slug:
        return None
    for patterns, area in (
        (FOOTER_PATTERNS, PageArea.FOOTER),
        (SIDEBAR_PATTERNS, PageArea.SIDEBAR),
    ):
        if any(_text_hit(slug, p) for p in patterns):
            return area
    return None


def _text_hit(slug: str, pattern: str) -> bool:
    """锚文本匹配。

    去掉模式中的斜杠后按子串比较——URL 模式（``/cart``）与文本
    模式（``privacy``）共用同一词表，斜杠在文本里不会出现。
    """
    p = pattern.strip("/?")
    return bool(p) and p in slug


def _href_path(href: str) -> str:
    """取出 href 的路径与查询部分。"""
    # 去掉协议与域名。相对路径（/products/x）不含 "//"，直接返回。
    if "//" in href:
        rest = href.split("//", 1)[1]
        slash = rest.find("/")
        return rest[slash:] if slash >= 0 else ""
    return href


def infer_area_batch(
    hrefs: Sequence[str], texts: Sequence[str]
) -> list:
    """批量推断，便于对整页候选一次处理。"""
    return [infer_area(href=h, text=t) for h, t in zip(hrefs, texts)]
