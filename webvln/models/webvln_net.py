"""WebVLN-Net 主模型，串联第三章各组件。

论文 3.1 节的三阶段结构：

    初始化  encode_language()   BERT 编码 Q&D，[CLS] 作为初始状态 s_0
    导航    navigate_step()     逐步更新状态并预测动作，直到选中 [EOA]
    回答    answer()            以 s_[EOA] 为条件生成答案 R

模型不持有 rollout 循环——循环涉及模拟器交互与教师强制策略，
属于训练逻辑（见 `webvln/train/`）。这里只暴露单步前向，
使模型本身可独立测试，也让候选筛选（第四章）能在调用 ``navigate_step``
之前介入而无需改动模型。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from webvln.models.action import ActionPredictor
from webvln.models.answering import AnsweringHead
from webvln.models.attention import TransformerLayer
from webvln.models.candidate_encoder import CandidateEncoder
from webvln.models.config import WebVLNConfig
from webvln.models.cross_modal import CrossModalEncoder, StateActionProjection
from webvln.models.language import extend_attention_mask, length2mask


class WebVLNNet(nn.Module):
    """Website-aware VLN Network。

    Attributes:
        config: 模型配置。
        bert: 注入的 BERT（语言分支）。为 None 时使用内置的
            ``la_layers`` 层 Transformer，需自行训练词嵌入。
    """

    def __init__(self, config: WebVLNConfig, bert: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        self.bert = bert

        if bert is None:
            # 无预训练 BERT 时的退化路径：自建词嵌入与语言层。
            # 论文 3.2 节用 BERT-base 初始化，随机初始化的语言分支
            # 性能会显著下降（基线论文表 2 中 VLN-BERT 随机初始化 vs
            # LXMERT 初始化的差距即说明这一点）。
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.lang_layer_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
            self.lang_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.lang_layers = nn.ModuleList(
                [TransformerLayer(config) for _ in range(config.la_layers)]
            )

        self.candidate_encoder = CandidateEncoder(config)
        self.state_action_project = StateActionProjection(config)
        self.cross_modal = CrossModalEncoder(config)
        self.action_predictor = ActionPredictor(config)
        self.answering_head = AnsweringHead(config)

    # --- 初始化阶段 ---------------------------------------------------------

    def encode_language(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码指令，仅在 episode 开始时调用一次。

        Returns:
            (lang_feats, state)
            ``lang_feats`` 形状 [batch, seq_len, hidden]，第 0 位为状态 token；
            ``state`` 形状 [batch, hidden]，即 s_0。
        """
        if self.bert is not None:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            lang_feats = (
                outputs[0]
                if isinstance(outputs, (tuple, list))
                else outputs.last_hidden_state
            )
        else:
            seq_len = input_ids.size(1)
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            h = self.word_embeddings(input_ids) + self.position_embeddings(pos)
            h = self.lang_dropout(self.lang_layer_norm(h))
            mask = extend_attention_mask(attention_mask, h.dtype)
            for layer in self.lang_layers:
                h = layer(h, mask)
            lang_feats = h

        return lang_feats, lang_feats[:, 0, :]

    # --- 导航阶段 ----------------------------------------------------------

    def navigate_step(
        self,
        lang_feats: torch.Tensor,
        lang_attention_mask: torch.Tensor,
        candidate_feats: torch.Tensor,
        action_lengths: Sequence[int],
        state: Optional[torch.Tensor] = None,
        prev_action_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步导航前向。

        Args:
            lang_feats: [batch, seq_len, hidden] 语言特征。
            lang_attention_mask: [batch, seq_len] 0/1 掩码。
            candidate_feats: [batch, n_tokens, feature_size] 候选特征。
            action_lengths: 各样本的有效动作数（候选数 + 1）。
            state: [batch, hidden] 上一步状态。t=0 时为 None，
                此时用 ``lang_feats`` 的第 0 位。
            prev_action_feat: [batch, hidden] 上一步所选候选的特征。
                提供时与状态融合，使状态携带历史动作信息。

        Returns:
            (state, action_logits, attended_visual)
            ``action_logits`` 形状 [batch, max_actions]，已按候选数掩码，
            可直接送入交叉熵或 argmax。
        """
        if state is not None:
            if prev_action_feat is not None:
                state = self.state_action_project(state, prev_action_feat)
            # 状态写回语言序列第 0 位——论文 3.4 节的递归状态更新。
            lang_feats = torch.cat(
                (state.unsqueeze(1), lang_feats[:, 1:, :]), dim=1
            )

        visn_feats = self.candidate_encoder(candidate_feats)

        # 候选掩码按 token 维度构造：length2mask 返回 True 表示 PAD，
        # 这里取反得到「有效位为 1」的形式再扩展为可加偏置。
        n_tokens = candidate_feats.size(1)
        token_lengths = [
            (n - 1) * self.config.tokens_per_candidate + 1 for n in action_lengths
        ]
        visn_valid = (~length2mask(token_lengths, n_tokens)).long().to(
            candidate_feats.device
        )

        lang_mask = extend_attention_mask(lang_attention_mask, visn_feats.dtype)
        visn_mask = extend_attention_mask(visn_valid, visn_feats.dtype)

        new_state, token_logits, attended_visual = self.cross_modal(
            lang_feats, lang_mask, visn_feats, visn_mask
        )
        action_logits = self.action_predictor(token_logits, action_lengths)
        return new_state, action_logits, attended_visual

    # --- 回答阶段 ----------------------------------------------------------

    def answer(
        self, state: torch.Tensor, answer_ids: torch.Tensor
    ) -> torch.Tensor:
        """生成答案 logits（teacher forcing）。"""
        return self.answering_head(state, answer_ids)

    def generate_answer(
        self,
        state: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """贪心解码生成答案，用于评测。"""
        return self.answering_head.generate(state, bos_token_id, eos_token_id, max_len)
