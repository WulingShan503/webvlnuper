"""批次长度计算的单元测试。

张量构造需要 torch，本机不具备，因此只覆盖两个纯 Python 的长度函数——
它们正是最易错的地方：token 序列长 ``3n+1``、动作空间长 ``n+1``，
两者不同，混用会让 argmax 落在错误的候选上。
"""

from webvln.train.batching import candidate_lengths, token_lengths
from webvln.train.env import Observation


def obs(n):
    return Observation(
        idx=1,
        website_id="S",
        url_id="S_0",
        candidate_ids=[f"c{i}" for i in range(n)],
        candidate_feats=[[[0.0]] * 3 for _ in range(n)],
        candidate_url_ids=[f"S_{i}" for i in range(n)],
    )


def test_action_space_is_n_plus_one():
    # 官方 candidate_leng = len(ob['candidate']) + 1
    assert candidate_lengths([obs(2), obs(5)]) == [3, 6]


def test_token_length_is_three_n_plus_one():
    # 官方 len(ob['candidate'])*3 + 1，末位留给 [EOA]
    assert token_lengths([obs(2), obs(5)]) == [7, 16]


def test_two_lengths_differ_and_must_not_be_mixed():
    o = [obs(45)]  # 论文报告的平均候选数
    assert token_lengths(o) == [136]
    assert candidate_lengths(o) == [46]


def test_empty_candidate_page_still_has_stop_action():
    # 目标页无出链，动作空间只剩 [EOA]。
    assert candidate_lengths([obs(0)]) == [1]
    assert token_lengths([obs(0)]) == [1]
