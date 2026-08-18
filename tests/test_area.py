"""页面区域推断的单元测试。

官方数据集不提供区域标注，区域由 href 与锚文本反推，
因此这里重点验证购物站常见链接模式的分类是否符合论文的剪枝意图。
"""

from webvln.screening.area import infer_area
from webvln.screening.candidate import PageArea


# --- href 推断 --------------------------------------------------------------


def test_legal_and_service_links_are_footer():
    assert infer_area(href="https://shop.com/pages/privacy-policy") is PageArea.FOOTER
    assert infer_area(href="/pages/terms-and-conditions") is PageArea.FOOTER
    assert infer_area(href="/pages/contact") is PageArea.FOOTER
    assert infer_area(href="/pages/refund-policy") is PageArea.FOOTER


def test_subscription_and_filter_links_are_sidebar():
    assert infer_area(href="/pages/newsletter") is PageArea.SIDEBAR
    assert infer_area(href="/collections/all?filter=color") is PageArea.SIDEBAR


def test_category_entries_are_nav():
    assert infer_area(href="https://shop.com/collections/socks") is PageArea.NAV
    assert infer_area(href="/new-arrivals") is PageArea.NAV


def test_account_and_cart_links_are_header():
    assert infer_area(href="/cart") is PageArea.HEADER
    assert infer_area(href="/account/login") is PageArea.HEADER


def test_product_links_are_main():
    assert infer_area(href="/products/blue-striped-sock") is PageArea.MAIN
    assert infer_area(href="/products/x/reviews") is PageArea.MAIN


def test_product_under_collection_is_main_not_nav():
    # 商品详情链接常嵌在集合路径下，这类链接属于主内容区，
    # 若误判为导航栏并不会被默认剪枝，但会歪曲 5.4 节的区域统计。
    href = "https://shop.com/collections/socks/products/blue-sock"
    assert infer_area(href=href) is PageArea.MAIN


def test_domain_keywords_do_not_leak_into_path_matching():
    # 域名中的 shop 不应让站内所有链接都变成导航栏。
    assert infer_area(href="https://shop.example.com/") is PageArea.UNKNOWN


def test_unknown_when_no_evidence():
    # 无证据时保持中立，既不默认 MAIN 掩盖缺失，也不误判为页脚。
    assert infer_area(href="/x/y/z", text="Click") is PageArea.UNKNOWN
    assert infer_area() is PageArea.UNKNOWN


# --- 锚文本推断 -------------------------------------------------------------


def test_anchor_text_used_when_href_has_no_signal():
    assert infer_area(href="/a/b", text="Privacy Policy") is PageArea.FOOTER
    assert infer_area(href="/a/b", text="Contact Us") is PageArea.FOOTER
    assert infer_area(href="/a/b", text="Subscribe") is PageArea.SIDEBAR


def test_href_takes_priority_over_text():
    # URL 结构由模板决定，比人工锚文本更规整。
    assert infer_area(href="/products/sock", text="Privacy Policy") is PageArea.MAIN


# --- DOM 路径推断 -----------------------------------------------------------


def test_dom_path_overrides_href():
    area = infer_area(href="/products/sock", dom_path="html > body > footer > a")
    assert area is PageArea.FOOTER


def test_deepest_dom_tag_wins():
    # footer 内的 nav 块属于导航，取最深匹配而非最浅。
    area = infer_area(dom_path="body > footer > nav > ul > li > a")
    assert area is PageArea.NAV


def test_dom_role_attributes_recognised():
    assert infer_area(dom_path="div[role=contentinfo] > a") is PageArea.FOOTER
    assert infer_area(dom_path="div[role=main] > a") is PageArea.MAIN
