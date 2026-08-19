"""导航与问答指标的单元测试。

对齐官方 ``eval.py``：失败样本 SPL 与 WUPS 记 0，
OSR 只要求轨迹经过目标页，TL 与 SPL 的单位都是页面数。
WUPS 用替身注入，避免依赖 nltk。
"""

import pytest

from webvln.data.episode import Episode
from webvln.eval.metrics import (
    NavigationScores,
    Result,
    best_score,
    score_episode,
    score_results,
)


def fake_wups(gt, pred, thresh):
    """完全一致给 1，阈值 0.0 时给部分分——用于区分两档 WUPS。"""
    if gt == pred:
        return 1.0
    return 0.5 if thresh == 0.0 else 0.0


GT_PATH = ["S_0", "S_1", "S_2"]


def test_successful_episode_scores_full():
    s = score_episode(GT_PATH, ["S_0", "S_1", "S_2"], "ans", "ans", fake_wups)
    assert s["success"] == 1.0
    assert s["oracle_success"] == 1.0
    assert s["tl"] == 3.0
    assert s["spl"] == 1.0
    assert s["wups_0.9"] == 1.0


def test_detour_reduces_spl_but_not_sr():
    # 式 (5.1.1)：走得比最短路径长则按 L / max(L, P) 打折。
    s = score_episode(GT_PATH, ["S_0", "S_9", "S_1", "S_2"], "a", "a", fake_wups)
    assert s["success"] == 1.0
    assert s["tl"] == 4.0
    assert s["spl"] == 3 / 4


def test_passing_through_goal_without_stopping_counts_only_osr():
    # OSR 与 SR 的差值即停止时机造成的损失。
    s = score_episode(GT_PATH, ["S_0", "S_2", "S_5"], "a", "a", fake_wups)
    assert s["success"] == 0.0
    assert s["oracle_success"] == 1.0
    assert s["spl"] == 0.0


def test_failed_episode_gets_zero_wups():
    # 没找到目标页就无从取信息作答，答案分数记 0（官方同样处理）。
    s = score_episode(GT_PATH, ["S_0", "S_7"], "ans", "ans", fake_wups)
    assert s["wups_0.9"] == 0.0
    assert s["wups_0.0"] == 0.0


def test_two_wups_thresholds_are_reported_separately():
    s = score_episode(GT_PATH, GT_PATH, "black socks", "socks", fake_wups)
    assert s["wups_0.9"] == 0.0
    assert s["wups_0.0"] == 0.5


def test_single_page_episode():
    # 起点即目标页：TL 为 1，SPL 满分。
    s = score_episode(["S_0"], ["S_0"], "a", "a", fake_wups)
    assert s["success"] == 1.0
    assert s["tl"] == 1.0
    assert s["spl"] == 1.0


def test_mismatched_start_is_rejected():
    # 结果文件与划分对不上时必须报错，而非静默算出错误的 SR。
    with pytest.raises(ValueError, match="起点"):
        score_episode(GT_PATH, ["OTHER_0", "S_1"], "a", "a", fake_wups)
    with pytest.raises(ValueError):
        score_episode(GT_PATH, [], "a", "a", fake_wups)


def episodes():
    return {
        1: Episode(idx=1, path=["S_0", "S_1"], answer="ans one"),
        2: Episode(idx=2, path=["S_0", "S_3"], answer="ans two"),
    }


def test_score_results_aggregates():
    results = [
        Result(idx=1, trajectory=["S_0", "S_1"], answer="ans one"),  # 成功
        Result(idx=2, trajectory=["S_0", "S_9"], answer="wrong"),    # 失败
    ]
    scores = score_results(results, episodes(), wups_fn=fake_wups)
    summary = scores.as_dict()
    assert scores.n == 2
    assert summary["SR"] == 50.0
    assert summary["SPL"] == 50.0
    assert summary["TL"] == 2.0
    assert summary["WUPS0.9"] == 50.0


def test_score_results_deduplicates_repeated_idx():
    # 官方 valid() 的循环会绕回开头重复评估部分样本，需按 idx 去重。
    results = [
        Result(idx=1, trajectory=["S_0", "S_1"], answer="ans one"),
        Result(idx=1, trajectory=["S_0", "S_9"], answer="x"),
    ]
    scores = score_results(results, episodes(), wups_fn=fake_wups)
    assert scores.n == 1
    assert scores.as_dict()["SR"] == 100.0


def test_score_results_tolerates_str_keyed_ground_truth():
    # 官方 Evaluation 用 str(idx) 建键，结果里却是原始类型。
    gt = {"1": Episode(idx=1, path=["S_0", "S_1"], answer="a")}
    scores = score_results(
        [Result(idx=1, trajectory=["S_0", "S_1"], answer="a")], gt, wups_fn=fake_wups
    )
    assert scores.n == 1


def test_score_results_skips_unknown_idx():
    scores = score_results(
        [Result(idx=99, trajectory=["S_0"], answer="a")], episodes(), wups_fn=fake_wups
    )
    assert scores.n == 0
    assert scores.as_dict()["SR"] == 0.0


def test_best_score_is_sr_plus_wups9():
    # 论文 3.9 节的模型选择准则。
    scores = NavigationScores()
    scores.add(success=True, oracle_success=True, spl=1.0, tl=3, wups_9=0.4, wups_0=0.6)
    summary = scores.as_dict()
    assert summary["SR"] == 100.0
    assert summary["WUPS0.9"] == 40.0
    assert scores.best_score() == 140.0
    assert best_score(summary) == 140.0


def test_empty_scores_do_not_divide_by_zero():
    summary = NavigationScores().as_dict()
    assert summary == {
        "SR": 0.0, "OSR": 0.0, "SPL": 0.0, "TL": 0.0,
        "WUPS0.9": 0.0, "WUPS0.0": 0.0,
    }


def test_tl_is_never_scaled_to_percent():
    scores = NavigationScores()
    scores.add(success=True, oracle_success=True, spl=1.0, tl=4, wups_9=1.0, wups_0=1.0)
    # TL 是长度不是比率，percent 与否都不乘 100。
    assert scores.as_dict(percent=True)["TL"] == 4.0
    assert scores.as_dict(percent=False)["TL"] == 4.0
    assert scores.as_dict(percent=False)["SR"] == 1.0
