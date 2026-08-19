"""训练与评测循环（论文 3.9 / 5.1 节）。

拆成三层，torch 依赖只落在最内层：

    ``env.py``       模拟器与观测构造（纯 Python，可离线测试）
    ``rollout.py``   一个 episode 的推进：动作选择、停止判定、轨迹记录
    ``trainer.py``   优化循环：损失反传、验证调度、模型选择

论文超参：AdamW lr 1e-5（官方）/ 1e-4（论文），weight decay 1e-2，
batch 4（官方）/ 8（论文），200,000 迭代，140,000 后每 1,000 步验证，
按 Best Score = SR + WUPS0.9 选模型。
"""

from webvln.train.batching import (
    build_candidate_tensor,
    candidate_lengths,
    token_lengths,
)
from webvln.train.env import Observation, WebVLNEnv
from webvln.train.rollout import (
    FEEDBACK_ARGMAX,
    FEEDBACK_MIX,
    FEEDBACK_SAMPLE,
    FEEDBACK_TEACHER,
    RolloutRecorder,
    StepDecision,
    resolve_action,
    teacher_action,
)

__all__ = [
    "FEEDBACK_ARGMAX",
    "FEEDBACK_MIX",
    "FEEDBACK_SAMPLE",
    "FEEDBACK_TEACHER",
    "Observation",
    "RolloutRecorder",
    "StepDecision",
    "WebVLNEnv",
    "build_candidate_tensor",
    "candidate_lengths",
    "resolve_action",
    "teacher_action",
    "token_lengths",
]

# Trainer 需要 torch，故不在包导入时拉起——本机无 torch 时
# ``from webvln.train import WebVLNEnv`` 仍应可用。
# 使用方式：``from webvln.train.trainer import Trainer``。
