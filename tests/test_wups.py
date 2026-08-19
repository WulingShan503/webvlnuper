"""WUPS 的单元测试。

WordNet 需要 nltk 与语料下载，本机不具备，因此全部用注入的
``similarity_fn`` 替身：测试关心的是集合匹配与降权逻辑，不是 WordNet 本身。
"""

from webvln.eval.wups import (
    INTERPRETATION_PENALTY,
    items2list,
    wup_measure,
    wups,
    wups_official,
)


def sim(a, b):
    """相似度替身：sock/socks 近义，shirt/pants 远亲，其余无关。"""
    pairs = {
        ("sock", "socks"): 0.98,
        ("shirt", "pants"): 0.6,
    }
    return pairs.get((a, b), pairs.get((b, a), 0.0))


def test_items2list_splits_on_comma():
    assert items2list("book, chair ,table") == ["book", "chair", "table"]


def test_identical_terms_score_one_without_wordnet():
    # 价格、型号等非词典词靠精确匹配得分，不查 WordNet。
    assert wup_measure("$16.10", "$16.10", similarity_fn=sim) == 1.0


def test_near_synonym_keeps_full_weight():
    score = wup_measure("sock", "socks", similarity_threshold=0.925, similarity_fn=sim)
    assert score == 0.98


def test_distant_relation_is_downweighted():
    # 靠语义场沾亲带故匹配上的词不应与同义词同权。
    score = wup_measure("shirt", "pants", similarity_threshold=0.925, similarity_fn=sim)
    assert abs(score - 0.6 * INTERPRETATION_PENALTY) < 1e-9


def test_unrelated_terms_score_zero():
    assert wup_measure("sock", "laptop", similarity_fn=sim) == 0.0
    assert wup_measure("", "sock", similarity_fn=sim) == 0.0


def test_wups_exact_match_is_one():
    assert wups("black socks", "black socks", 0.9, sim) == 1.0


def test_wups_is_symmetric_and_penalises_extra_terms():
    # 多预测一个无关词项会拉低乘积，防止「万能词」刷分。
    full = wups("sock", "sock", 0.9, sim)
    padded = wups("sock", "sock,laptop", 0.9, sim)
    assert full == 1.0
    assert padded == 0.0


def test_wups_set_semantics_ignore_order():
    a = wups("book,chair", "chair,book", 0.9, sim)
    assert a == 1.0


def test_dirac_measure_when_thresh_is_minus_one():
    # thresh=-1 即官方所说的 standard accuracy。
    assert wups("sock", "sock", -1, sim) == 1.0
    assert wups("sock", "socks", -1, sim) == 0.0


def test_official_variant_compares_character_by_character():
    """官方 calculate_wups 收到字符串后 zip 出的是字符对。

    论文表 5.1 的 WUPS 数字由此产生，所以复现基线必须用这个版本。
    """
    # 完全相同的字符串逐字符也全对。
    assert wups_official("abc", "abc", 0.9, sim) == 1.0
    # 首字符相同、其余不同：3 对中 1 对得分。
    assert abs(wups_official("abc", "axy", 0.9, sim) - 1 / 3) < 1e-9


def test_official_variant_truncates_to_shorter_string():
    # zip 以较短的串为准，长出的部分不参与评分——这也是官方的行为。
    assert wups_official("ab", "abcdef", 0.9, sim) == 1.0


def test_official_variant_handles_empty_input():
    # 官方在这里会 ZeroDivisionError。
    assert wups_official("", "", 0.9, sim) == 0.0
    assert wups_official("abc", "", 0.9, sim) == 0.0


def test_official_variant_gives_full_score_to_a_prefix():
    """官方版本对「预测是真值前缀」的情况给满分。

    逐字符配对 + 按较短串截断，两个效果叠加的结果：
    ``"The price is 16.1"`` 是 ``"The price is 16.10"`` 的前缀，
    每一对字符都相同，于是拿到 1.0。集合版本则按词项判断，
    ``16.1`` 与 ``16.10`` 不是同一个词项。这就是两种算法不可混用的原因。
    """
    gt = "The price is 16.10"
    assert wups_official(gt, "The price is 16.1", 0.9, sim) == 1.0
    assert wups(gt, "The price is 16.1", 0.9, sim) < 1.0


def test_set_variant_scores_word_level_difference():
    gt = "black socks"
    # 集合版本按逗号切词项，整句不同即 0；换成近义词项才有部分分。
    assert wups(gt, "black sock", 0.9, sim) == 0.0
    assert wups("sock", "socks", 0.9, sim) == 0.98
