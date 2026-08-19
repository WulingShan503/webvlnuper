"""动作解析、停止判定与轨迹记录的单元测试。

对齐官方 ``agent.py:rollout``：动作等于 [EOA] 位、等于 ignoreid、
或样本已结束时，环境动作置 -1 并标记结束。
"""

from webvln.train.env import Observation
from webvln.train.rollout import (
    FEEDBACK_MIX,
    MIX_PASSES,
    RolloutRecorder,
    resolve_action,
    teacher_action,
)


def obs(url="S_0", n=2, teacher=1, idx=1):
    return Observation(
        idx=idx,
        website_id="S",
        url_id=url,
        candidate_ids=[f"c{i}" for i in range(n)],
        candidate_feats=[[[0.0]] * 3 for _ in range(n)],
        candidate_url_ids=[f"S_{i + 1}" for i in range(n)],
        teacher=teacher,
    )


def test_action_space_includes_stop_slot():
    o = obs(n=2)
    assert o.n_candidates == 2
    assert o.n_actions == 3
    assert o.eoa_index == 2


def test_teacher_action_ignores_ended_samples():
    o = obs(teacher=1)
    assert teacher_action(o, ended=False) == 1
    # 已结束的样本返回 ignoreid，不产生损失。
    assert teacher_action(o, ended=True) == -100


def test_resolve_normal_action():
    d = resolve_action(0, obs(), ended=False)
    assert d.env_action == 0
    assert d.stops is False


def test_resolve_stop_action():
    o = obs(n=2)
    d = resolve_action(o.eoa_index, o, ended=False)
    assert d.env_action == -1
    assert d.stops is True


def test_resolve_ignore_id_stops():
    d = resolve_action(-100, obs(), ended=False)
    assert d.env_action == -1
    assert d.stops is True


def test_ended_sample_cannot_revive():
    # 已结束的样本无论给什么动作都不再移动。
    d = resolve_action(0, obs(), ended=True)
    assert d.env_action == -1
    assert d.stops is True


def test_out_of_range_action_treated_as_stop():
    # 筛选后候选变少时旧下标可能越界，按停止处理而非跳到无关页面。
    o = obs(n=2)
    assert resolve_action(5, o, ended=False).env_action == -1
    assert resolve_action(-3, o, ended=False).env_action == -1


def test_recorder_tracks_trajectory():
    o = [obs(url="S_0", n=2)]
    rec = RolloutRecorder(o)
    assert rec.trajectories == [["S_0"]]

    env_actions = rec.record(0, [resolve_action(0, o[0], False)], o)
    assert env_actions == [0]
    assert rec.trajectories == [["S_0", "S_1"]]
    assert rec.ended == [False]


def test_recorder_marks_answer_step_at_first_stop():
    o = [obs(n=2)]
    rec = RolloutRecorder(o)
    rec.record(0, [resolve_action(0, o[0], False)], o)
    rec.record(1, [resolve_action(o[0].eoa_index, o[0], False)], o)
    # 回答状态取自停止那一步。
    assert rec.answer_step == [1]
    assert rec.ended == [True]


def test_answer_step_not_overwritten_after_ending():
    o = [obs(n=2)]
    rec = RolloutRecorder(o)
    rec.record(0, [resolve_action(o[0].eoa_index, o[0], False)], o)
    rec.record(1, [resolve_action(o[0].eoa_index, o[0], True)], o)
    assert rec.answer_step == [0]


def test_stopped_sample_does_not_move():
    o = [obs(n=2)]
    rec = RolloutRecorder(o)
    rec.record(0, [resolve_action(o[0].eoa_index, o[0], False)], o)
    assert rec.trajectories == [["S_0"]]


def test_all_ended_and_finalize():
    o = [obs(n=2, idx=1), obs(n=2, idx=2)]
    rec = RolloutRecorder(o)
    rec.record(0, [resolve_action(2, o[0], False), resolve_action(0, o[1], False)], o)
    assert rec.all_ended() is False

    # 跑满步数仍未停止的样本，回答状态取最后一步而非起始页。
    rec.finalize(max_step=9)
    assert rec.all_ended() is True
    assert rec.answer_step == [0, 9]


def test_as_results_matches_eval_format():
    o = [obs(n=2)]
    rec = RolloutRecorder(o)
    rec.record(0, [resolve_action(0, o[0], False)], o)
    rec.answers[0] = "cotton"
    results = rec.as_results()
    assert results == [{"idx": 1, "trajectory": ["S_0", "S_1"], "answer": "cotton"}]


def test_mix_expands_into_two_passes():
    """官方的 mix 是「先 sample 跑一遍、再 teacher 跑一遍」，

    两次损失累进同一个 loss，不是按概率二选一。
    """
    assert FEEDBACK_MIX == "mix"
    assert [name for name, _ in MIX_PASSES] == ["sample", "teacher"]
    assert all(w == 1.0 for _, w in MIX_PASSES)
