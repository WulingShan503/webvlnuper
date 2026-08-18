"""3.5 跨模态 Transformer 的注意力构件。

对应官方 ``vlnbert_PREVALENT.py`` 中的 ``BertSelfAttention`` /
``BertOutAttention`` / ``LXRTXLayer``。与标准 BERT 层的差别在于
**注意力分数需要返回给外部**：动作 logits 就是状态 token 对候选 token 的
注意力分数（见 ``navigator.py``），而非独立的分类头，
所以这里的 forward 一律同时返回 context 与 scores。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from webvln.models.config import WebVLNConfig


class MultiHeadAttention(nn.Module):
    """多头注意力，同时返回注意力分数。

    ``ctx_dim`` 不为 None 时为交叉注意力：query 来自 ``hidden_states``，
    key / value 来自 ``context``。
    """

    def __init__(self, config: WebVLNConfig, ctx_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.attention_head_size
        self.all_head_size = self.num_heads * self.head_size

        ctx_dim = ctx_dim if ctx_dim is not None else config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] -> [batch, heads, seq, head_size]。"""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算注意力。

        Args:
            hidden_states: [batch, q_len, hidden]，query 来源。
            context: [batch, kv_len, ctx_dim]。为 None 时退化为自注意力。
            attention_mask: 可加的偏置掩码 [batch, 1, 1, kv_len]，
                PAD 位为 -10000。

        Returns:
            (context_layer, attention_scores)
            ``context_layer`` 形状 [batch, q_len, hidden]；
            ``attention_scores`` 形状 [batch, heads, q_len, kv_len]，
            **未经 softmax**——外部取它做动作 logits 时需要原始分数，
            softmax 会在掩码与跨步合并之后才施加。
        """
        kv = context if context is not None else hidden_states

        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(kv))
        v = self.transpose_for_scores(self.value(kv))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = self.dropout(nn.Softmax(dim=-1)(scores))
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.size()[:-2], self.all_head_size)
        return ctx, scores


class AttentionOutput(nn.Module):
    """注意力输出的残差 + LayerNorm。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.dense(hidden_states))
        return self.layer_norm(h + input_tensor)


class SelfAttentionBlock(nn.Module):
    """自注意力子层（注意力 + 残差 LayerNorm）。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.output = AttentionOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, scores = self.attention(hidden_states, attention_mask=attention_mask)
        return self.output(ctx, hidden_states), scores


class CrossAttentionBlock(nn.Module):
    """交叉注意力子层。"""

    def __init__(self, config: WebVLNConfig, ctx_dim: Optional[int] = None) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config, ctx_dim=ctx_dim)
        self.output = AttentionOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx, scores = self.attention(hidden_states, context, attention_mask)
        return self.output(ctx, hidden_states), scores


class FeedForward(nn.Module):
    """FFN 子层。论文 3.5 节记中间层维度为 3072。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense_out = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.dense_in(hidden_states))
        h = self.dropout(self.dense_out(h))
        return self.layer_norm(h + hidden_states)


class TransformerLayer(nn.Module):
    """标准 Transformer 编码层，用于语言分支的 ``la_layers``。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.attention = SelfAttentionBlock(config)
        self.ffn = FeedForward(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h, _ = self.attention(hidden_states, attention_mask)
        return self.ffn(h)
