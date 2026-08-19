"""WUPS：基于 WordNet 的自由形式答案评分（论文 5.1 节指标 4）。

WUPS (Wu-Palmer Set) 由 Malinowski & Fritz (2014) 提出，用于开放式 VQA：
答案视为词项集合，两集合的匹配度由 Wu-Palmer 词义相似度加权。
阈值 0.9 严格（近义词才算命中），0.0 宽松。

    score(A, T) = min( prod_{a in A} m(a, T), prod_{t in T} m(t, A) )
    m(x, S)     = max_{s in S} wup(x, s)

**与官方实现的一处差异**：官方 ``calculate_wups(gt, pred, thresh)`` 被
``eval.py`` 以两个**字符串**调用，而函数内部 ``zip(input_gt, input_pred)``
是按元素配对——字符串的元素是字符，于是逐字符算了 WUPS，
分数还被截断到较短那个字符串的长度。论文表 5.1 的 WUPS0.9 24.26 /
WUPS0.0 31.87 就是这么算出来的。

因此这里提供两个函数：

    ``wups``           按词项集合计算，指标的本来定义
    ``wups_official``  复刻官方逐字符行为，用于对齐已发表的数字

复现基线时用 ``wups_official``，否则会得到与论文不同（且更高）的分数。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

#: 官方 ``wup_measure`` 的默认阈值。低于此值的相似度被降权到 0.1。
DEFAULT_SIMILARITY_THRESHOLD = 0.925
#: 降权系数：靠语义场匹配上（而非同义）时的折扣。
INTERPRETATION_PENALTY = 0.1


def items2list(text: str) -> List[str]:
    """把逗号分隔的答案切成词项列表，与官方 ``items2list`` 一致。"""
    return [item.strip() for item in text.split(",")]


def _wordnet_similarity(a: str, b: str) -> float:
    """两个词的 Wu-Palmer 相似度，取所有名词词义中的最大值。

    延迟导入 nltk：本机无 nltk 时其余评测代码（SR / SPL / TL）仍可运行，
    调用方可注入自定义相似度函数做测试。
    """
    from nltk.corpus import wordnet as wn

    syn_a = wn.synsets(a, pos=wn.NOUN)
    syn_b = wn.synsets(b, pos=wn.NOUN)
    if not syn_a or not syn_b:
        return 0.0

    best = 0.0
    for x in syn_a:
        for y in syn_b:
            score = x.wup_similarity(y)
            if score and score > best:
                best = score
    return best


def wup_measure(
    a: str,
    b: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """单个词项对的 WUP 相似度。

    Args:
        a: 词项一。
        b: 词项二。
        similarity_threshold: 低于该相似度时乘 0.1 降权——靠语义场沾亲带故
            匹配上的词不应与同义词同权。
        similarity_fn: 相似度函数，默认走 WordNet。测试与离线复现可注入替身。

    Returns:
        [0, 1] 区间的分数。完全相同返回 1.0（不查 WordNet，
        因此数字、价格等非词典词也能精确匹配）。
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    fn = similarity_fn or _wordnet_similarity
    best = fn(a, b)
    if best <= 0.0:
        return 0.0

    weight = 1.0 if best >= similarity_threshold else INTERPRETATION_PENALTY
    return best * weight


def _set_membership(
    x: str, group: Sequence[str], element_measure: Callable[[str, str], float]
) -> float:
    """模糊集合成员度 m(x ∈ A) = max_{a ∈ A} m(x, a)。"""
    if not group:
        return 0.0
    return max(element_measure(x, a) for a in group)


def _score_pair(
    truth: Sequence[str],
    pred: Sequence[str],
    element_measure: Callable[[str, str], float],
) -> float:
    """两个词项集合的对称匹配分。

    取两个方向乘积的较小值：只看一个方向的话，预测出一个万能词
    就能在「每个真值都被覆盖」上得满分。
    """
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0

    left = 1.0
    for t in truth:
        left *= _set_membership(t, pred, element_measure)
    right = 1.0
    for p in pred:
        right *= _set_membership(p, truth, element_measure)
    return min(left, right)


def _element_measure(
    thresh: float, similarity_fn: Optional[Callable[[str, str], float]]
) -> Callable[[str, str], float]:
    """按阈值选择词项间的度量。

    ``thresh == -1`` 时退化为 Dirac 测度（完全相等才得 1），
    即官方所说的 standard accuracy。
    """
    if thresh == -1:
        return lambda a, b: float(bool(a) and a == b)
    return lambda a, b: wup_measure(a, b, thresh, similarity_fn)


def wups(
    truth: str,
    prediction: str,
    thresh: float = 0.9,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """按词项集合计算 WUPS —— 指标的本来定义。

    Args:
        truth: 参考答案。
        prediction: 模型生成的答案。
        thresh: 相似度阈值，0.9 / 0.0 对应论文的两档；-1 为精确匹配。
        similarity_fn: 注入的词相似度函数。

    Returns:
        [0, 1] 区间的分数。
    """
    return _score_pair(
        items2list(truth), items2list(prediction), _element_measure(thresh, similarity_fn)
    )


def wups_official(
    truth: str,
    prediction: str,
    thresh: float = 0.9,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """复刻官方 ``calculate_wups`` 的逐字符行为。

    官方以字符串调用，函数内 ``zip(input_gt, input_pred)`` 于是按字符配对，
    对每一对字符各算一次集合分再取平均，长度以较短的字符串为准。
    论文表 5.1 的 WUPS 数字由此产生，复现基线必须用这个版本。

    Returns:
        [0, 1] 区间的分数。两串均为空时返回 0.0——官方在这里会
        ``ZeroDivisionError``（``len(score_list)`` 为 0）。
    """
    measure = _element_measure(thresh, similarity_fn)
    pairs = list(zip(truth, prediction))
    if not pairs:
        return 0.0
    scores = [
        _score_pair(items2list(t), items2list(p), measure) for t, p in pairs
    ]
    return sum(scores) / len(scores)
