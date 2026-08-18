"""3.6 动作预测。

论文式 (3.6.1) / (3.6.2)：对 M 个候选（含 [EOA]）做 softmax，取 argmax 为动作。

    p_i = exp(logit_i) / Σ_j exp(logit_j)        (3.6.1)
    a_t = argmax_i p_i                           (3.6.2)

实现上有一处必须处理的错位：候选特征以**每候选 3 个 token**的形式送入
跨模态 Transformer（见 ``candidate_encoder.py``），因此注意力分数的长度是
``n_candidates * 3 + 1``，而动作空间的大小是 ``n_candidates + 1``。
两者不能直接对齐——若把 3n+1 维的 logits 当作动作分布，
argmax 得到的下标会落在错误的候选上。

官方 ``agent.py`` 中掩码用的是 ``candidate_leng = len(ob['candidate']) + 1``，
即按动作维度而非 token 维度构造，注释里保留的 ``a_t = logit[:,::3]``
也说明了需要按步长 3 取样。本模块把这一归约显式化为 ``pool_token_logits``。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from webvln.models.config import WebVLNConfig


def pool_token_logits(
    token_logits: torch.Tensor,
    tokens_per_candidate: int = 3,
    reduction: str = "first",
) -> torch.Tensor:
    """把 token 级 logits 归约为候选级 logits。

    批内的候选数差异由 ``mask_invalid_actions`` 处理，此处只做形状归约，
    因此不需要各样本的候选数。

    Args:
        token_logits: [batch, n_tokens]，n_tokens = max_cand * k + 1。
        tokens_per_candidate: 每候选的 token 数 k。
        reduction: 归约方式。
            ``"first"`` 取每候选的首个 token（文本段），等价于官方注释中的
            ``logit[:,::3]``——文本 token 承载语义主体，与指令的相关性最强；
            ``"mean"`` 取三段平均；``"max"`` 取三段最大。

    Returns:
        [batch, max_cand + 1]，末位为 [EOA]。
    """
    batch, n_tokens = token_logits.shape
    k = tokens_per_candidate
    max_cand = (n_tokens - 1) // k

    body = token_logits[:, : max_cand * k].view(batch, max_cand, k)
    if reduction == "first":
        cand_logits = body[:, :, 0]
    elif reduction == "mean":
        cand_logits = body.mean(dim=2)
    elif reduction == "max":
        cand_logits = body.max(dim=2).values
    else:
        raise ValueError(f"未知的归约方式：{reduction!r}")

    # [EOA] 位于 token 序列的最后一位（其特征为全零，见 candidate_encoder）。
    eoa_logit = token_logits[:, max_cand * k : max_cand * k + 1]
    return torch.cat((cand_logits, eoa_logit), dim=1)


def mask_invalid_actions(
    logits: torch.Tensor, action_lengths: Sequence[int]
) -> torch.Tensor:
    """屏蔽超出候选数的动作位。

    批内各样本候选数不同，短样本的尾部是零填充。若不屏蔽，
    模型可能选中一个不存在的候选，模拟器无法执行该跳转。

    Args:
        logits: [batch, max_actions]。
        action_lengths: 各样本的有效动作数（候选数 + 1，含 [EOA]）。

    Returns:
        屏蔽后的 logits，无效位为 -inf。
    """
    batch, max_actions = logits.shape
    idx = torch.arange(max_actions, device=logits.device).unsqueeze(0)
    lens = torch.tensor(list(action_lengths), device=logits.device).unsqueeze(1)
    invalid = idx >= lens
    return logits.masked_fill(invalid, float("-inf"))


class ActionPredictor(nn.Module):
    """动作预测头。

    本身不含可学习参数——动作 logits 直接来自跨模态注意力分数
    （见 ``cross_modal.py``）。本类只负责 token→候选归约、掩码与采样，
    单独成类是为了让 rollout 的动作选择逻辑集中在一处。
    """

    def __init__(self, config: WebVLNConfig, reduction: str = "first") -> None:
        super().__init__()
        self.config = config
        self.reduction = reduction

    def forward(
        self,
        token_logits: torch.Tensor,
        action_lengths: Sequence[int],
    ) -> torch.Tensor:
        """由 token 级分数得到掩码后的候选级 logits。"""
        logits = pool_token_logits(
            token_logits,
            tokens_per_candidate=self.config.tokens_per_candidate,
            reduction=self.reduction,
        )
        return mask_invalid_actions(logits, action_lengths)

    @staticmethod
    def select(logits: torch.Tensor, feedback: str = "argmax") -> torch.Tensor:
        """按反馈策略选择动作。

        Args:
            logits: [batch, max_actions] 已掩码的 logits。
            feedback: ``"argmax"`` 为学生强制（式 3.6.2）；
                ``"sample"`` 从 softmax 分布采样，官方训练用 ``mix``
                即两者混合，以缓解暴露偏差。

        Returns:
            [batch] 动作下标。
        """
        if feedback == "argmax":
            return logits.argmax(dim=1)
        if feedback == "sample":
            probs = F.softmax(logits, dim=1)
            return torch.distributions.Categorical(probs).sample()
        raise ValueError(f"未知的反馈策略：{feedback!r}")


def teacher_action(
    candidate_url_ids: Sequence[Sequence[str]],
    gt_paths: Sequence[Sequence[str]],
    step: int,
    ended: Sequence[bool],
    ignore_id: int = -100,
) -> torch.Tensor:
    """构造教师动作，对应官方 ``agent.py:_teacher_action``。

    Args:
        candidate_url_ids: 各样本候选的 ``next_url_id`` 列表，顺序与 logits 一致。
        gt_paths: 各样本的真实路径（url id 序列）。
        step: 当前步 t。
        ended: 各样本是否已结束。
        ignore_id: 交叉熵忽略值。

    Returns:
        [batch] 教师动作下标。已结束或越界的样本置 ``ignore_id``，
        使其不产生梯度——否则已停止的样本会持续贡献损失，
        等价于对短轨迹样本重复加权。
    """
    actions = []
    for i, urls in enumerate(candidate_url_ids):
        gt = gt_paths[i]
        if ended[i] or step >= len(gt):
            actions.append(ignore_id)
        elif step == len(gt) - 1:
            # 已到达目标页，正确动作是 [EOA]，位于候选之后的末位。
            actions.append(len(urls))
        else:
            target_url = gt[step + 1]
            match = next((k for k, u in enumerate(urls) if u == target_url), None)
            # 目标不在候选中（可能被筛选误删，或数据图不连通）时忽略该步，
            # 而非强行指向某个候选——错误的监督信号比缺失更有害。
            actions.append(match if match is not None else ignore_id)
    return torch.tensor(actions, dtype=torch.long)
