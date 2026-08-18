"""3.3 候选特征编码。

论文 3.3 节把每个候选的三段特征描述为拼接后投影：

    768 (文本 / alt 经 BERT) + 2048 (按钮图 ResNet152) + 2048 (截图裁剪 ResNet152)
    = 4864                                                        式 (3.3.1)

**官方实现与此不同**：``agent.py:_candidate_variable`` 把三段特征放成
**三个独立 token**（``candidate_feat[i, j*3+k, :] = feat``），每段 512 维，
序列长度为 ``len(candidate) * 3 + 1``。这样跨模态 Transformer 可以对
「文本」「按钮图」「截图」三种模态分别做注意力，而非在投影前就压成一个向量。

本模块实现官方形式，并保留 ``ConcatCandidateEncoder`` 对应论文写法，
供 5.x 节对照。两者的 ``forward`` 签名一致，可直接替换。

末位 token 是 [EOA]。官方注释明确 "The candidate_feat at len(ob['candidate'])
is the feature for the END which is zero in my implementation"——
即 [EOA] 的特征为全零，其被选中的 logit 完全由跨模态注意力从状态 token 学得。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from webvln.models.config import WebVLNConfig


def build_candidate_features(
    per_candidate_feats: Sequence[Sequence[np.ndarray]],
    feature_size: int = 512,
    tokens_per_candidate: int = 3,
) -> Tuple[np.ndarray, int]:
    """把一个样本的候选特征铺成 token 序列。

    Args:
        per_candidate_feats: 每个候选的特征段列表，形如
            ``[[text_feat, button_img_feat, screenshot_feat], ...]``，
            对应官方 ``ob['candidate'][cc]["feature"]``。
        feature_size: 单段特征维度。
        tokens_per_candidate: 每候选的 token 数。

    Returns:
        (feats, n_tokens)
        ``feats`` 形状 [n_candidates * tokens_per_candidate + 1, feature_size]，
        末位为 [EOA]（全零）。``n_tokens`` 为**候选数 + 1**，
        不是 token 数——官方 ``candidate_leng`` 在函数末尾被重新赋值为
        ``len(ob['candidate']) + 1``，用于构造动作维度上的掩码。
    """
    n_cand = len(per_candidate_feats)
    total_tokens = n_cand * tokens_per_candidate + 1
    feats = np.zeros((total_tokens, feature_size), dtype=np.float32)

    for j, segs in enumerate(per_candidate_feats):
        for k, seg in enumerate(segs):
            if k >= tokens_per_candidate:
                break
            vec = np.asarray(seg, dtype=np.float32).reshape(-1)
            # 特征段长度可能与 feature_size 不符（不同骨干网络输出维度不同），
            # 截断或零填充而非抛错，避免单条脏数据中断 200,000 迭代的训练。
            n = min(vec.shape[0], feature_size)
            feats[j * tokens_per_candidate + k, :n] = vec[:n]

    return feats, n_cand + 1


def batch_candidate_features(
    batch_feats: Sequence[Sequence[Sequence[np.ndarray]]],
    feature_size: int = 512,
    tokens_per_candidate: int = 3,
) -> Tuple[torch.Tensor, List[int]]:
    """按批构造候选特征张量。

    批内各样本的候选数不同，按最长者右侧零填充。

    Returns:
        (tensor, candidate_lengths)
        ``tensor`` 形状 [batch, max_tokens, feature_size]；
        ``candidate_lengths`` 为各样本的「候选数 + 1」，供 ``length2mask`` 使用。
    """
    per_sample = [
        build_candidate_features(feats, feature_size, tokens_per_candidate)
        for feats in batch_feats
    ]
    max_tokens = max(f.shape[0] for f, _ in per_sample)

    out = np.zeros((len(per_sample), max_tokens, feature_size), dtype=np.float32)
    lengths: List[int] = []
    for i, (feats, n_len) in enumerate(per_sample):
        out[i, : feats.shape[0], :] = feats
        lengths.append(n_len)

    return torch.from_numpy(out), lengths


class CandidateEncoder(nn.Module):
    """候选特征投影（官方形式：三段各占一个 token）。

    把 [batch, n_tokens, feature_size] 投影到 hidden_size 并 LayerNorm，
    对应官方 ``model_OSCAR.py`` 中的 ``img_projection`` + ``cand_LayerNorm``，
    以及 ``vlnbert_PREVALENT.py`` 的 ``VisionEncoder``。
    """

    def __init__(self, config: WebVLNConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.feature_size, config.hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 官方对视觉特征额外施加 featdropout（0.4），强度高于常规 dropout：
        # 候选特征来自固定的预训练骨干，不参与训练，容易被模型过度依赖。
        self.feat_dropout = nn.Dropout(p=config.feat_dropout)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, candidate_feats: torch.Tensor) -> torch.Tensor:
        """投影候选特征。

        Args:
            candidate_feats: [batch, n_tokens, feature_size]。

        Returns:
            [batch, n_tokens, hidden_size]。
        """
        x = self.feat_dropout(candidate_feats)
        x = self.projection(x)
        x = self.layer_norm(x)
        return self.dropout(x)


class ConcatCandidateEncoder(nn.Module):
    """候选特征投影（论文形式：三段拼接为 4864 维）。

    实现论文式 (3.3.1)。与 ``CandidateEncoder`` 的区别在于输入形状：
    此处每个候选只占一个 token，维度为三段之和。
    官方权重与该形式不兼容，仅供对照实验从头训练。

    Attributes:
        text_dim / button_dim / screenshot_dim: 三段特征维度，
            默认 768 / 2048 / 2048，和为 4864。
    """

    def __init__(
        self,
        config: WebVLNConfig,
        text_dim: int = 768,
        button_dim: int = 2048,
        screenshot_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.config = config
        self.text_dim = text_dim
        self.button_dim = button_dim
        self.screenshot_dim = screenshot_dim
        self.input_dim = text_dim + button_dim + screenshot_dim  # 4864

        self.projection = nn.Linear(self.input_dim, config.hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feat_dropout = nn.Dropout(p=config.feat_dropout)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, candidate_feats: torch.Tensor) -> torch.Tensor:
        """投影拼接后的候选特征。

        Args:
            candidate_feats: [batch, n_candidates, 4864]。

        Returns:
            [batch, n_candidates, hidden_size]，即式 (3.3.2) 的 H_cand。
        """
        if candidate_feats.size(-1) != self.input_dim:
            raise ValueError(
                f"候选特征维度应为 {self.input_dim}（式 3.3.1），"
                f"实际为 {candidate_feats.size(-1)}"
            )
        x = self.feat_dropout(candidate_feats)
        x = self.projection(x)
        x = self.layer_norm(x)
        return self.dropout(x)


def concat_candidate_features(
    text_feats: np.ndarray,
    button_feats: np.ndarray,
    screenshot_feats: np.ndarray,
) -> np.ndarray:
    """按式 (3.3.1) 拼接三段特征。

    缺失的按钮图（无图标的纯文本链接）以全零填充，
    保持维度一致——论文 3.3 节指出无图候选的图像特征置零。
    """
    return np.concatenate([text_feats, button_feats, screenshot_feats], axis=-1)
