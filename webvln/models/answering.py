"""3.7 回答头。

论文式 (3.7.1)：以最终状态 s_[EOA] 为条件，自回归生成答案。

    P(w_l | w_<l, s_[EOA]) = softmax(W_o h_l + b_o)

结构为带交叉注意力的 Transformer 解码器（官方 ``QADecoder``，2 层；
论文 3.7 节写 4 层）。训练用 teacher forcing，
损失为 label smoothing 后的交叉熵（官方 ``LabelSmoothingLoss(smoothing=0.1)``）。

因果掩码用位置比较矩阵构造并注册为 buffer——每步重新生成掩码会在
200,000 迭代中累积可观开销，且掩码只依赖序列长度，可复用。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from webvln.models.attention import (
    CrossAttentionBlock,
    FeedForward,
    SelfAttentionBlock,
)
from webvln.models.config import WebVLNConfig


class AnswerEmbeddings(nn.Module):
    """答案 token 的词嵌入与位置嵌入。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos = self.position_ids[:, :seq_len]
        embeds = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        return self.dropout(self.layer_norm(embeds))


class AnswerDecoderLayer(nn.Module):
    """解码层：因果自注意力 + 对状态的交叉注意力 + FFN。"""

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.self_attention = SelfAttentionBlock(config)
        self.cross_attention = CrossAttentionBlock(config)
        self.ffn = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        h, _ = self.self_attention(hidden_states, causal_mask)
        h, _ = self.cross_attention(h, encoder_hidden_states)
        return self.ffn(h)


class AnsweringHead(nn.Module):
    """自回归回答头，对应式 (3.7.1)。

    Attributes:
        config: 模型配置。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = AnswerEmbeddings(config)
        self.layers = nn.ModuleList(
            [AnswerDecoderLayer(config) for _ in range(config.qa_layers)]
        )
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_activation = nn.GELU()
        self.transform_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        max_pos = config.max_position_embeddings
        pos = torch.arange(max_pos)
        # 下三角为 True：位置 i 只能看到 j <= i。
        causal = pos[None, :] <= pos[:, None]
        self.register_buffer("causal", causal, persistent=False)

    def forward(
        self,
        state: torch.Tensor,
        answer_ids: torch.Tensor,
        last_step_only: bool = False,
    ) -> torch.Tensor:
        """生成答案 token 的 logits。

        Args:
            state: [batch, hidden] 或 [batch, n, hidden]，最终状态 s_[EOA]。
                为 2 维时自动补出长度 1 的序列维——交叉注意力要求 key/value
                带序列维度。
            answer_ids: [batch, seq_len] 答案 token id（teacher forcing 输入）。
            last_step_only: 推理时只需最后一步的 logits，可省去无用计算。

        Returns:
            [batch, seq_len, vocab_size]；``last_step_only`` 为 True 时
            形状为 [batch, 1, vocab_size]。
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)

        h = self.embeddings(answer_ids)
        seq_len = answer_ids.size(1)
        mask = self.causal[:seq_len, :seq_len]
        # 转成可加偏置：允许位加 0，禁止位加 -10000。
        causal_mask = (1.0 - mask.to(dtype=h.dtype)) * -10000.0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, state, causal_mask)

        if last_step_only:
            h = h[:, -1:, :]

        h = self.transform_norm(self.transform_activation(self.transform(h)))
        return self.decoder(h)

    @torch.no_grad()
    def generate(
        self,
        state: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """贪心解码，用于评测时生成自由形式答案。

        Args:
            state: [batch, hidden] 最终状态。
            bos_token_id: 起始 token。
            eos_token_id: 终止 token。
            max_len: 最大长度，默认取 ``config.max_answer_len``。

        Returns:
            [batch, gen_len] 生成的 token id（不含起始 token）。
        """
        max_len = max_len or self.config.max_answer_len
        batch = state.size(0)
        device = state.device

        tokens = torch.full((batch, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.forward(state, tokens, last_step_only=True)
            next_token = logits[:, -1, :].argmax(dim=-1)
            # 已生成 EOS 的样本后续一律填 PAD，避免继续产出无意义 token
            # 干扰 WUPS 计算。
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)
            finished = finished | (next_token == eos_token_id)
            if bool(finished.all()):
                break

        return tokens[:, 1:]
