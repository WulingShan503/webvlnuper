"""3.4 / 3.5 状态 token 递归与跨模态 Transformer。

对应官方 ``vlnbert_PREVALENT.py`` 的 ``LXRTXLayer``。单层内的数据流是：

    1. 状态 token 与候选 token 拼成视觉序列
       visn = [state] + candidates
    2. 该序列对语言 token 做交叉注意力（语言只作 key/value，不更新）
    3. 拼接后的序列做自注意力，此步的注意力分数即动作 logits 的来源
    4. FFN
    5. 拆回：状态 token 写回语言序列的第 0 位，候选 token 继续向下传

第 2 步中语言序列传入的是 ``lang_feats[:, 1:, :]``——**跳过第 0 位**，
因为第 0 位已被状态 token 占据（论文 3.4 节：``[CLS]`` 位在导航过程中
被状态 token 替换）。若不跳过，状态 token 会对自己做交叉注意力。

动作 logits 取自第 3 步自注意力分数的 ``[:, :, 0, 1:]``：
第 0 个 query（状态 token）对第 1 位起（候选 token）的注意力，
再对注意力头取平均。这解释了为什么模型没有独立的动作分类头——
「选哪个候选」本身就被表示为「状态最关注哪个候选」。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from webvln.models.attention import (
    CrossAttentionBlock,
    FeedForward,
    SelfAttentionBlock,
)
from webvln.models.config import WebVLNConfig


class CrossModalLayer(nn.Module):
    """跨模态层。

    Attributes:
        cross_attention: 视觉序列（状态 + 候选）对语言的交叉注意力。
        self_attention: 视觉序列内部的自注意力，其分数产出动作 logits。
        ffn: 前馈层。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_attention = CrossAttentionBlock(config)
        self.self_attention = SelfAttentionBlock(config)
        self.ffn = FeedForward(config)

    def forward(
        self,
        lang_feats: torch.Tensor,
        lang_mask: torch.Tensor,
        visn_feats: torch.Tensor,
        visn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单层前向。

        Args:
            lang_feats: [batch, lang_len, hidden]，第 0 位为状态 token。
            lang_mask: 可加偏置掩码 [batch, 1, 1, lang_len]。
            visn_feats: [batch, n_tokens, hidden] 候选 token。
            visn_mask: 可加偏置掩码 [batch, 1, 1, n_tokens]。

        Returns:
            (lang_out, visn_out, lang_scores, visn_scores)
            ``visn_scores`` 形状 [batch, heads, n_tokens]，
            即状态 token 对各候选 token 的注意力分数。
        """
        # 状态 token 与候选拼成一个序列，使二者能在自注意力中直接交互。
        state_visn = torch.cat((lang_feats[:, 0:1, :], visn_feats), dim=1)
        state_visn_mask = torch.cat((lang_mask[:, :, :, 0:1], visn_mask), dim=-1)

        # 语言侧跳过第 0 位：该位已是状态 token，不应作为语言上下文。
        state_visn, cross_scores = self.cross_attention(
            state_visn, lang_feats[:, 1:, :], lang_mask[:, :, :, 1:]
        )
        lang_scores = cross_scores[:, :, 0, :]

        state_visn, self_scores = self.self_attention(state_visn, state_visn_mask)
        state_visn = self.ffn(state_visn)

        # 动作 logits 的来源：状态 token（query 0）对候选 token（key 1 起）的分数。
        visn_scores = self_scores[:, :, 0, 1:]

        visn_out = state_visn[:, 1:, :]
        # 更新后的状态 token 写回语言序列第 0 位，语言 token 本身保持不变——
        # 论文指出初始化得到的语言表示已足够好，导航中无需重新编码。
        lang_out = torch.cat((state_visn[:, 0:1, :], lang_feats[:, 1:, :]), dim=1)

        return lang_out, visn_out, lang_scores, visn_scores


class CrossModalEncoder(nn.Module):
    """跨模态编码器，堆叠 ``vl_layers`` 层。

    官方 ``vl_layers=2``；论文 3.5 节写 4 层（``PAPER_CONFIG``）。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [CrossModalLayer(config) for _ in range(config.vl_layers)]
        )
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(
        self,
        lang_feats: torch.Tensor,
        lang_mask: torch.Tensor,
        visn_feats: torch.Tensor,
        visn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """跨模态前向。

        Returns:
            (state, action_scores, attended_visual)
            ``state`` 形状 [batch, hidden]，即更新后的状态 token s_t；
            ``action_scores`` 形状 [batch, n_tokens]，对注意力头取平均后的
            动作 logits（尚未按候选长度掩码）；
            ``attended_visual`` 形状 [batch, hidden]，按注意力加权的候选表示，
            供回答头使用。
        """
        lang_out, visn_out = lang_feats, visn_feats
        visn_scores = None

        for layer in self.layers:
            lang_out, visn_out, _, visn_scores = layer(
                lang_out, lang_mask, visn_out, visn_mask
            )

        # 对头取平均：官方 visual_action_scores = visual_attention_scores.mean(dim=1)
        action_scores = visn_scores.mean(dim=1)

        # 状态 token 经 pooler 得到 h_t，作为下一步的输入与回答头的条件。
        state = self.pooler_activation(self.pooler_dense(lang_out[:, 0, :]))

        visual_probs = nn.Softmax(dim=-1)(action_scores.clone()).unsqueeze(-1)
        attended_visual = (visual_probs * visn_feats).sum(1)

        return state, action_scores, attended_visual


class StateActionProjection(nn.Module):
    """3.4 状态 token 的动作融合。

    对应官方 ``model_OSCAR.py`` 的 ``action_state_project``：
    把上一步的状态与所选动作的特征拼接后投影回 hidden_size，
    使状态 token 携带「我上一步点了什么」的信息——
    这是模型具备历史记忆的唯一途径（没有额外的 RNN）。
    """

    def __init__(self, config: WebVLNConfig, action_feat_size: Optional[int] = None) -> None:
        super().__init__()
        action_feat_size = (
            action_feat_size if action_feat_size is not None else config.hidden_size
        )
        self.dense = nn.Linear(config.hidden_size + action_feat_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, state: torch.Tensor, action_feat: torch.Tensor) -> torch.Tensor:
        """融合状态与上一步动作。

        Args:
            state: [batch, hidden] 上一步的状态 token。
            action_feat: [batch, action_feat_size] 上一步所选候选的特征。

        Returns:
            [batch, hidden]。
        """
        combined = torch.cat((state, action_feat), dim=1)
        return self.layer_norm(self.activation(self.dense(combined)))
