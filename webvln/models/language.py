"""3.2 语言编码器与指令编码。

论文 3.2 节：问题 Q 与辅助描述 D 拼成 ``[CLS] Q [SEP] D [SEP]``，
经 WordPiece 切分后送入 BERT-base（12 层，768 维），
取输出的 token 序列作为语言特征，其中 ``[CLS]`` 位作为智能体的初始状态 token。

关键点在于**语言编码只在初始化时执行一次**。论文指出，初始化得到的语言 token
已经是 Q&D 的良好表示，导航过程中无需重新编码——这既省算力，
也让语言表示在整个 episode 中保持一致。因此导航步里语言 token 只作为
Transformer 的 key / value，不再更新（见 ``navigator.py``）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from webvln.models.config import WebVLNConfig


def build_instruction_tokens(
    question: str,
    description: str = "",
    tokenizer=None,
    max_len: int = 50,
) -> Tuple[List[int], List[int]]:
    """把 Q 与 D 编码为 ``[CLS] Q [SEP] D [SEP]`` 的 token id 序列。

    Args:
        question: 问题 Q。
        description: 辅助描述 D。论文脚注指出，问题本身足以定位目标页时 D 为空，
            此时退化为 ``[CLS] Q [SEP]``，不补空的第二段。
        tokenizer: HuggingFace BertTokenizer。
        max_len: 截断长度。

    Returns:
        (token_ids, attention_mask)，长度均为 ``max_len``（右侧 PAD 补齐）。
    """
    if tokenizer is None:
        raise ValueError("需要传入 BertTokenizer")

    q_tokens = tokenizer.tokenize(question or "")
    d_tokens = tokenizer.tokenize(description or "")

    tokens = ["[CLS]"] + q_tokens + ["[SEP]"]
    if d_tokens:
        tokens += d_tokens + ["[SEP]"]

    # 截断时保留末尾的 [SEP]：BERT 依赖它标记序列边界，
    # 直接截断会让最后一段缺少结束符。
    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(ids)

    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids += [0] * pad_len
        mask += [0] * pad_len

    return ids, mask


class LanguageEncoder(nn.Module):
    """BERT 语言分支。

    对应官方 ``vlnbert_PREVALENT.py`` 中 ``mode == 'language'`` 的分支：
    embedding 后过 ``la_layers`` 层自注意力，返回 token 序列与 pooled 输出。

    Attributes:
        config: 模型配置。
    """

    def __init__(self, config: WebVLNConfig, bert: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        # bert 由外部注入，便于单元测试用轻量替身，也便于加载官方权重。
        self.bert = bert

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码指令。

        Args:
            input_ids: [batch, seq_len] token id。
            attention_mask: [batch, seq_len]，1 为真实 token，0 为 PAD。
            token_type_ids: [batch, seq_len] 段标记。

        Returns:
            (sequence_output, state_token)
            ``sequence_output`` 形状 [batch, seq_len, hidden]，
            ``state_token`` 为其 ``[CLS]`` 位，形状 [batch, hidden]，
            即导航起始的状态 token s_0。
        """
        if self.bert is None:
            raise RuntimeError("LanguageEncoder 未注入 BERT 模型")

        if token_type_ids is None:
            token_type_ids = build_token_type_ids(
                input_ids, sep_token_id=self.config.sep_token_id
            )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = (
            outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        )
        return sequence_output, sequence_output[:, 0, :]


def build_token_type_ids(
    input_ids: torch.Tensor, sep_token_id: int = 102
) -> torch.Tensor:
    """由 token id 推断段标记。

    ``[CLS] Q [SEP]`` 记为段 0，其后的 ``D [SEP]`` 记为段 1。
    官方实现直接传全 0（``torch.zeros_like(mask)``），
    因为 Q 与 D 的边界已由 ``[SEP]`` 标出、且两段同为指令语义；
    这里提供标准的两段划分，供需要区分 Q / D 的消融实验使用。
    """
    is_sep = input_ids == sep_token_id
    # 首个 [SEP] 之后（不含该 [SEP] 本身）为段 1。
    # cumsum 在 [SEP] 位就已置 1，故右移一位。
    seg = torch.cumsum(is_sep.long(), dim=1)
    seg = torch.nn.functional.pad(seg[:, :-1], (1, 0), value=0)
    return seg.clamp(max=1)


def extend_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """把 [batch, seq] 的 0/1 掩码扩展为可加到注意力分数上的偏置。

    真实 token 加 0，PAD 位加 -10000——softmax 后其概率趋近 0。
    与官方 ``vlnbert_PREVALENT.py`` 的处理一致。
    """
    extended = mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype)
    return (1.0 - extended) * -10000.0


def length2mask(lengths: Sequence[int], max_len: Optional[int] = None) -> torch.Tensor:
    """由长度列表构造 PAD 掩码。

    返回值中 **True 表示 PAD 位**（需被屏蔽），与官方 ``utils.length2mask``
    的语义一致——官方随后用 ``logit.masked_fill_(candidate_mask, -inf)``，
    若语义反过来会把所有真实候选屏蔽掉。
    """
    size = max_len or max(lengths)
    idx = torch.arange(size).unsqueeze(0)
    lens = torch.tensor(list(lengths)).unsqueeze(1)
    return idx >= lens
