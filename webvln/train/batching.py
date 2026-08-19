"""观测批次到张量的转换。

把 ``Observation`` 列表堆成模型输入。单独成模块的原因是这里集中了
两处与官方对齐的下标约定，它们出错时不报异常、只是分数上不去：

1. 候选特征按 ``j*3+k`` 铺排（候选 j 的第 k 段特征），
   序列长度 ``max(n_candidates)*3 + 1``，末位留给 [EOA] 的全零特征；
2. 动作空间长度 ``n_candidates + 1``，与特征序列长度**不同**——
   注意力分数需按步长 3 归约才能对上（见 ``models/action.py``）。

torch 在函数内部导入，使 ``env.py`` / ``rollout.py`` 的测试不受影响。
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from webvln.train.env import Observation


def candidate_lengths(obs: Sequence[Observation]) -> List[int]:
    """各样本的动作空间大小（候选数 + 1）。

    对应官方 ``candidate_leng = [len(ob['candidate']) + 1 for ob in obs]``。
    """
    return [ob.n_actions for ob in obs]


def token_lengths(obs: Sequence[Observation], tokens_per_candidate: int = 3) -> List[int]:
    """各样本的候选 token 数。

    对应官方 ``len(ob['candidate'])*3 + 1``。
    """
    return [ob.n_candidates * tokens_per_candidate + 1 for ob in obs]


def build_candidate_tensor(
    obs: Sequence[Observation],
    feature_size: int = 512,
    tokens_per_candidate: int = 3,
) -> Any:
    """把候选特征堆成 [batch, max_tokens, feature_size] 张量。

    复刻官方 ``_candidate_variable``：候选 j 的第 k 段特征写在
    ``j*tokens_per_candidate + k`` 行，[EOA] 位保持全零。

    [EOA] 的特征全零并非疏漏——它不是页面元素，没有可提取的特征，
    其 logit 完全由跨模态注意力从状态 token 学得。

    Returns:
        float32 张量。
    """
    import torch

    lengths = token_lengths(obs, tokens_per_candidate)
    max_len = max(lengths) if lengths else 1
    feats = torch.zeros((len(obs), max_len, feature_size), dtype=torch.float32)

    for i, ob in enumerate(obs):
        for j, segments in enumerate(ob.candidate_feats):
            for k, segment in enumerate(segments[:tokens_per_candidate]):
                row = j * tokens_per_candidate + k
                feats[i, row, :] = _as_row(torch, segment, feature_size)

    return feats


def _as_row(torch_mod: Any, segment: Any, feature_size: int) -> Any:
    """把一段特征规整成长度 ``feature_size`` 的一维张量。

    特征表里的条目形状不统一：官方 pkl 中有的是 ``(1, 512)``、
    缺失时补的是 ``np.zeros((1,512))``，重抽的是一维向量。
    统一 flatten 后按长度裁剪或补零，避免形状不匹配在 200,000 迭代的
    某一批才暴露出来。
    """
    tensor = torch_mod.as_tensor(segment, dtype=torch_mod.float32).reshape(-1)
    if tensor.numel() == feature_size:
        return tensor
    if tensor.numel() > feature_size:
        return tensor[:feature_size]
    out = torch_mod.zeros(feature_size, dtype=torch_mod.float32)
    out[: tensor.numel()] = tensor
    return out


def build_language_tensors(obs: Sequence[Observation], pad_token_id: int = 0) -> Tuple[Any, Any]:
    """把指令 token 堆成 (input_ids, attention_mask)。

    掩码按「非 PAD」判定而非记录长度：``text_enc`` 已是定长补齐的，
    有效长度信息只剩 PAD 值本身。
    """
    import torch

    ids = torch.tensor([ob.text_enc for ob in obs], dtype=torch.long)
    mask = (ids != pad_token_id).long()
    return ids, mask


def build_answer_tensors(obs: Sequence[Observation]) -> Tuple[Any, Any]:
    """把答案序列堆成 (decoder_input, target)。

    分别对应官方 ``answer_enc``（以 [unused0] 起始）与
    ``answer_enc_w_eos``（以 [unused1] 结尾）。
    """
    import torch

    inp = torch.tensor([ob.answer_enc for ob in obs], dtype=torch.long)
    tgt = torch.tensor([ob.answer_enc_w_eos for ob in obs], dtype=torch.long)
    return inp, tgt


def build_teacher_tensor(
    obs: Sequence[Observation], ended: Sequence[bool], ignore_id: int = -100
) -> Any:
    """把教师动作堆成 [batch] 长整型张量。

    已结束的样本填 ``ignore_id``，交叉熵会跳过它们。
    """
    import torch

    from webvln.train.rollout import teacher_action

    targets = [
        teacher_action(ob, ended=bool(e), ignore_id=ignore_id)
        for ob, e in zip(obs, ended)
    ]
    return torch.tensor(targets, dtype=torch.long)
