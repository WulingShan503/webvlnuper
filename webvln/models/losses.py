"""3.8 损失函数。

论文式 (3.8.1)：

    L_total = L_nav + λ · L_ans

导航损失是候选上的交叉熵（模仿学习 / teacher forcing），
回答损失是式 (3.8.2) 的自回归负对数似然：

    L_ans = - Σ_l log P(w_l | w_<l, s_[EOA])       (3.8.2)

官方实现的两处细节值得注意：
- 导航损失用 ``CrossEntropyLoss(ignore_index=ignoreid, size_average=False)``，
  即**按和而非按均值**累计，再在 rollout 结束后除以 batch。若逐步取均值，
  长轨迹样本每步的权重会被摊薄，短轨迹样本反而被放大。
- 回答损失用 label smoothing 0.1。答案是自由形式句子，
  存在多种合法表述，硬标签会过度惩罚同义表达。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from webvln.models.config import WebVLNConfig


class NavigationLoss(nn.Module):
    """导航损失，对应式 (3.8.1) 中的 L_nav。

    Attributes:
        ignore_index: 已结束或无有效教师动作的样本标签值。
        reduction: 官方用 ``"sum"``，跨步累计后再除以 batch。
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "sum") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """单步导航损失。

        Args:
            logits: [batch, n_actions] 已掩码的候选 logits。
            target: [batch] 教师动作下标。

        Returns:
            标量损失。
        """
        # 被掩码的无效动作位是 -inf，与 ignore_index 的样本相遇时
        # 交叉熵会得到 nan。先把整行无效的样本替换为 ignore_index 已由
        # teacher_action 保证，这里只需确保 -inf 不参与 log_softmax 的分子。
        return F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class LabelSmoothingLoss(nn.Module):
    """带标签平滑的交叉熵，对应官方 ``QADecoder.LabelSmoothingLoss``。

    Attributes:
        num_classes: 词表大小。
        smoothing: 平滑系数，官方 0.1。
        ignore_index: PAD 位不计入损失。
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        ignore_index: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing 应在 [0, 1) 内，实际为 {smoothing}")
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算平滑交叉熵。

        Args:
            pred: [N, C] logits。
            target: [N] 标签。

        Returns:
            标量损失，对非 PAD 位取均值。
        """
        log_probs = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs, self.smoothing / (self.num_classes - 1)
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * log_probs, dim=-1)
        valid = target != self.ignore_index
        n_valid = valid.sum()
        if n_valid == 0:
            return loss.new_zeros(())
        return (loss * valid).sum() / n_valid


class AnsweringLoss(nn.Module):
    """回答损失，对应式 (3.8.2)。

    输入是解码器对整个答案序列的 logits，内部做 teacher forcing 的错位对齐：
    位置 l 的预测应匹配 l+1 位的真实 token。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.criterion = LabelSmoothingLoss(
            num_classes=config.vocab_size,
            smoothing=config.label_smoothing,
            ignore_index=config.pad_token_id,
        )

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """计算答案生成损失。

        Args:
            logits: [batch, seq_len, vocab_size]。
            target_ids: [batch, seq_len] 带终止符的目标序列
                （官方 ``answer_enc_w_eos``）。

        Returns:
            标量损失。
        """
        if logits.size(1) != target_ids.size(1):
            raise ValueError(
                f"logits 与目标序列长度不符：{logits.size(1)} vs {target_ids.size(1)}"
            )
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_target = target_ids.reshape(-1)
        return self.criterion(flat_logits, flat_target)


class WebVLNLoss(nn.Module):
    """总损失，实现式 (3.8.1)。

    Attributes:
        qa_weight: 式中的 λ，默认 1.0。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.nav_loss = NavigationLoss(ignore_index=config.ignore_id, reduction="sum")
        self.ans_loss = AnsweringLoss(config)
        self.qa_weight = config.qa_loss_weight

    def forward(
        self,
        nav_logits_per_step: Optional[list] = None,
        nav_targets_per_step: Optional[list] = None,
        answer_logits: Optional[torch.Tensor] = None,
        answer_targets: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> dict:
        """汇总导航与回答损失。

        Args:
            nav_logits_per_step: 各步的候选 logits 列表。
            nav_targets_per_step: 各步的教师动作列表。
            answer_logits: 回答头输出。
            answer_targets: 目标答案序列。
            batch_size: 用于把按和累计的导航损失还原为每样本量级。

        Returns:
            含 ``total`` / ``nav`` / ``ans`` 三项的字典，便于分别记录曲线。
        """
        device = None
        nav_total = None

        if nav_logits_per_step:
            for logits, target in zip(nav_logits_per_step, nav_targets_per_step or []):
                step_loss = self.nav_loss(logits, target)
                nav_total = step_loss if nav_total is None else nav_total + step_loss
                device = logits.device
            # 按 batch 归一：与官方 ml_loss / batch_size 的处理一致。
            nav_total = nav_total / max(batch_size, 1)

        ans_total = None
        if answer_logits is not None and answer_targets is not None:
            ans_total = self.ans_loss(answer_logits, answer_targets)
            device = answer_logits.device

        zero = torch.zeros((), device=device) if device is not None else torch.zeros(())
        nav_val = nav_total if nav_total is not None else zero
        ans_val = ans_total if ans_total is not None else zero

        return {
            "total": nav_val + self.qa_weight * ans_val,
            "nav": nav_val,
            "ans": ans_val,
        }
