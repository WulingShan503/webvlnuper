"""一个 episode 的推进：动作选择、停止判定与轨迹记录。

官方 ``agent.py:rollout`` 的三种 feedback：

    teacher  用教师动作前进（imitation learning）
    argmax   用模型 argmax 前进（student forcing），评测也用它
    sample   按 softmax 采样前进，增加探索

``run/train.bash`` 用 ``mix``——官方在 ``train()`` 里把它展开成
先 sample 跑一遍、再 teacher 跑一遍，两次损失都累进同一个 ``self.loss``，
即一次迭代做两次前向。这不是「按概率二选一」，容易看错。

停止判定与官方一致：动作等于 ``candidate_leng - 1``（即 [EOA] 位）、
等于 ``ignoreid``、或该样本已结束时，环境动作置 -1 且标记为结束。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from webvln.train.env import Observation

FEEDBACK_TEACHER = "teacher"
FEEDBACK_ARGMAX = "argmax"
FEEDBACK_SAMPLE = "sample"
#: 官方 ``run/train.bash`` 的设置：一次迭代先 sample 再 teacher，各跑一遍。
FEEDBACK_MIX = "mix"

#: ``mix`` 展开后的两趟顺序与各趟的模仿学习权重。
#: 官方 sample 趟用 ``train_ml=1.0``，teacher 趟用 ``--teacherWeight 1``。
MIX_PASSES = ((FEEDBACK_SAMPLE, 1.0), (FEEDBACK_TEACHER, 1.0))


def teacher_action(obs: Observation, ended: bool, ignore_id: int = -100) -> int:
    """教师动作，对应官方 ``_argmax_action``。

    Args:
        obs: 当前观测。
        ended: 该样本是否已结束。
        ignore_id: 交叉熵忽略值。

    Returns:
        动作下标；已结束的样本返回 ``ignore_id``，使其不产生损失。
        已在目标页时返回 [EOA] 位。
    """
    if ended:
        return ignore_id
    return obs.teacher


@dataclass
class StepDecision:
    """一步的动作解析结果。

    Attributes:
        env_action: 交给环境的动作。-1 表示不移动（停止 / 已结束 / 忽略）。
        stops: 该步之后样本是否结束。
    """

    env_action: int
    stops: bool


def resolve_action(
    action: int, obs: Observation, ended: bool, ignore_id: int = -100
) -> StepDecision:
    """把模型输出的动作下标转成环境动作并判定是否结束。

    对应官方那段：

        if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:
            cpu_a_t[i] = -1

    Args:
        action: 模型选出的动作下标（0..n 为候选，n 为 [EOA]）。
        obs: 当前观测。
        ended: 该样本此前是否已结束。
        ignore_id: 忽略值。

    Returns:
        StepDecision。已结束的样本恒为 (-1, True)，不会因为新动作复活。
    """
    if ended or action == ignore_id or action == obs.eoa_index:
        return StepDecision(env_action=-1, stops=True)
    # 越界动作（候选被筛掉后仍指向旧下标）按停止处理，
    # 而非静默跳到某个无关页面。
    if action < 0 or action >= obs.n_candidates:
        return StepDecision(env_action=-1, stops=True)
    return StepDecision(env_action=action, stops=False)


class RolloutRecorder:
    """记录一批 episode 的轨迹与结束状态。

    轨迹用于评测（SR / OSR / SPL / TL），格式与 ``eval.Result.trajectory``
    一致：以起始页开头，每次成功跳转追加一页。

    Attributes:
        answer_step: 各样本在哪一步进入回答阶段。
    """

    def __init__(self, obs: Sequence[Observation]) -> None:
        self.trajectories: List[List[str]] = [[ob.url_id] for ob in obs]
        self.idxs: List[Any] = [ob.idx for ob in obs]
        self.ended: List[bool] = [False] * len(obs)
        # 回答用的状态取自「停止那一步」的隐状态。官方用
        # ``len(traj_i["path"]) - 2`` 反推该步下标，等价于记录停止时的步号；
        # 这里直接记下来，避免依赖轨迹长度的间接推算。
        self.answer_step: List[int] = [0] * len(obs)
        self.answers: List[str] = [""] * len(obs)

    def record(
        self, step: int, decisions: Sequence[StepDecision], obs: Sequence[Observation]
    ) -> List[int]:
        """记录一步的动作结果。

        Args:
            step: 当前步号（从 0 开始）。
            decisions: 各样本的动作解析结果。
            obs: 该步的观测（用于把动作映射成目标页面）。

        Returns:
            交给环境的动作列表。
        """
        env_actions: List[int] = []
        for i, decision in enumerate(decisions):
            if not self.ended[i] and decision.stops:
                # 首次停止的那一步即回答阶段的状态来源。
                self.answer_step[i] = step
            if decision.env_action != -1:
                url = obs[i].url_for_action(decision.env_action)
                if url:
                    self.trajectories[i].append(url)
            self.ended[i] = self.ended[i] or decision.stops
            env_actions.append(decision.env_action)
        return env_actions

    def all_ended(self) -> bool:
        """整批是否都已结束，用于提前跳出循环。"""
        return all(self.ended)

    def finalize(self, max_step: int) -> None:
        """把跑满步数仍未停止的样本收尾。

        达到 ``maxAction``（官方 10 步）被强制中断的样本，其回答状态取最后一步。
        不收尾会让这些样本的 answer_step 停在 0，用起始页的状态去作答。
        """
        for i, ended in enumerate(self.ended):
            if not ended:
                self.answer_step[i] = max_step
                self.ended[i] = True

    def as_results(self) -> List[Dict[str, Any]]:
        """导出为评测所需的结果列表。"""
        return [
            {
                "idx": self.idxs[i],
                "trajectory": list(self.trajectories[i]),
                "answer": self.answers[i],
            }
            for i in range(len(self.idxs))
        ]
