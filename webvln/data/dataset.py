"""WebVLN-v1 数据集的加载与批次取样。

官方目录结构（``args.data_dir`` / ``args.setting``）：

    Downloads/Data/
      shortest_paths.json          导航图的最短路径表
      map.json                     各网站各页面的候选元素
      img_feats.pkl                按钮图特征
      text_feats.pkl               候选文本特征
      screenshot_crop_feats.pkl    截图裁剪特征
      seen/                        setting 目录
        train.json / train_enc.json
        val.json   / val_enc.json
        test.json  / test_enc.json

``*_enc.json`` 是带 token 编码的缓存，首次运行由 ``prepare_dataset`` 生成。
本模块保持同样的约定：优先读 ``_enc.json``，缺失时读原始 json 并现场编码，
使复现流程不必重跑编码，也不必手工准备缓存。

论文 2.3 节的划分规模：训练 8,960 / 验证 1,262 / 测试 4,603。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

from webvln.data.episode import Episode
from webvln.data.text import (
    DEFAULT_MAX_ANSWER_LEN,
    DEFAULT_MAX_INSTR_LEN,
    build_instruction_text,
    encode_answer,
    encode_instruction,
)

#: 论文 2.3 节报告的划分规模，用于加载后自检。
SPLIT_SIZES = {"train": 8960, "val": 1262, "test": 4603}


def split_path(data_dir: str, setting: str, split: str, encoded: bool = False) -> str:
    """拼出某个划分的 json 路径。"""
    name = f"{split}_enc.json" if encoded else f"{split}.json"
    return os.path.join(data_dir, setting, name)


def load_split(
    data_dir: str, setting: str = "seen", split: str = "train"
) -> List[Dict[str, Any]]:
    """读取一个划分的原始记录。

    优先 ``{split}_enc.json``，不存在时回退 ``{split}.json``——与官方
    ``load_datasets`` 的 try/except 行为一致，但这里用 ``os.path.exists``
    显式判断：官方那个裸 except 会把 json 语法错误也当成「文件不存在」，
    转而去读另一个文件，报错信息会指向错误的路径。
    """
    encoded = split_path(data_dir, setting, split, encoded=True)
    raw = split_path(data_dir, setting, split, encoded=False)
    path = encoded if os.path.exists(encoded) else raw
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def prepare_episodes(
    records: Sequence[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    max_instr_len: int = DEFAULT_MAX_INSTR_LEN,
    max_answer_len: int = DEFAULT_MAX_ANSWER_LEN,
) -> List[Episode]:
    """把原始记录转成 ``Episode``，缺失的编码就地补齐。

    Args:
        records: ``load_split`` 的返回值。
        tokenizer: BertTokenizer。记录已含 ``text_enc`` 时可为 None。
        max_instr_len: 指令截断长度。
        max_answer_len: 答案截断长度。

    Returns:
        Episode 列表，顺序与输入一致。

    Raises:
        ValueError: 记录未编码且未提供 tokenizer。
    """
    episodes: List[Episode] = []
    for record in records:
        ep = Episode.from_dict(record)
        if not ep.text_enc:
            if tokenizer is None:
                raise ValueError(
                    f"记录 {ep.idx} 缺少 text_enc，需传入 tokenizer 现场编码"
                )
            _encode_episode(ep, tokenizer, max_instr_len, max_answer_len)
        episodes.append(ep)
    return episodes


def _encode_episode(
    ep: Episode, tokenizer: Any, max_instr_len: int, max_answer_len: int
) -> None:
    """就地补齐一条 episode 的 token 编码。"""
    ep.text = ep.text or build_instruction_text(ep.target, ep.question)
    ep.text_enc, ep.text_words = encode_instruction(ep.text, tokenizer, max_instr_len)
    ep.answer_enc, ep.answer_enc_w_eos, ep.answer_words = encode_answer(
        ep.answer, tokenizer, max_answer_len
    )


def save_encoded(
    episodes: Sequence[Episode], data_dir: str, setting: str, split: str
) -> str:
    """把编码结果写回 ``{split}_enc.json``，供后续运行直接复用。

    对应官方 ``prepare_dataset`` 中 ``save_flag`` 的分支。
    编码 8,960 条训练样本要跑一遍 WordPiece，缓存后再训练可省下这段时间。

    Returns:
        写入的文件路径。
    """
    out_dir = os.path.join(data_dir, setting)
    os.makedirs(out_dir, exist_ok=True)
    path = split_path(data_dir, setting, split, encoded=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([ep.to_official_dict() for ep in episodes], fh, indent=4)
    return path


class WebVLNDataset:
    """一个划分的 episode 集合与批次取样器。

    对应官方 ``R2RBatch`` 中与数据有关的部分（``_next_minibatch`` /
    ``reset_epoch``）。取样逻辑单独拆出来，是为了让它能脱离模拟器与特征表
    被测试——官方那份代码要构造完整环境才能验证一个下标是否回绕正确。

    Attributes:
        episodes: 该划分的全部 episode。
        split: 划分名。
        batch_size: 批大小。官方 ``run/train.bash`` 用 4，论文 5.1 节写 8。
        seed: 随机种子。官方在 ``env.py`` 顶部 ``random.seed(0)``。
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        split: str = "train",
        batch_size: int = 4,
        seed: int = 0,
        shuffle: Optional[bool] = None,
    ) -> None:
        self.episodes = list(episodes)
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self._rng = random.Random(seed)
        # 训练划分打乱、验证与测试保持原序：评测要求结果可复现，
        # 且 ``Evaluation`` 按 idx 索引，顺序不影响分数但影响日志可比性。
        self.shuffle = split == "train" if shuffle is None else shuffle
        if self.shuffle:
            self._rng.shuffle(self.episodes)
        self.ix = 0
        self.batch: List[Episode] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def size(self) -> int:
        """与官方 ``R2RBatch.size()`` 同名，便于替换。"""
        return len(self.episodes)

    def next_minibatch(
        self, batch_size: Optional[int] = None, tile_one: bool = False
    ) -> List[Episode]:
        """取下一批 episode，末尾不足时从头回绕。

        Args:
            batch_size: 覆盖默认批大小。
            tile_one: 把同一条 episode 复制成整批，官方用于调试。

        Returns:
            长度为 ``batch_size`` 的列表。

        Note:
            回绕而非丢弃尾批，是因为训练按**迭代数**而非 epoch 计数
            （200,000 次迭代），每次迭代都必须拿到满批，
            否则批内长度对齐与损失归一化会随批大小抖动。
        """
        size = batch_size or self.batch_size
        if tile_one:
            batch = [self.episodes[self.ix]] * size
            self.ix += 1
            if self.ix >= len(self.episodes):
                self._reshuffle()
                self.ix = 0
            self.batch = batch
            return batch

        batch = self.episodes[self.ix : self.ix + size]
        if len(batch) < size:
            self._reshuffle()
            self.ix = size - len(batch)
            batch = batch + self.episodes[: self.ix]
        else:
            self.ix += size
        self.batch = batch
        return batch

    def reset_epoch(self, shuffle: bool = False) -> None:
        """把取样下标复位到开头。评测前调用，保证遍历完整覆盖。"""
        if shuffle:
            self._rng.shuffle(self.episodes)
        self.ix = 0

    def iter_batches(self, batch_size: Optional[int] = None) -> Iterator[List[Episode]]:
        """按顺序遍历一遍全部 episode，最后一批可能不足。

        评测用：官方 ``valid()`` 靠 ``looped`` 标志判断是否绕回，
        会把开头的样本重复评一遍再靠 idx 去重。这里直接给出不重复的遍历，
        使 SR / SPL 的分母就是该划分的样本数。
        """
        size = batch_size or self.batch_size
        for start in range(0, len(self.episodes), size):
            yield self.episodes[start : start + size]

    def _reshuffle(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.episodes)

    @classmethod
    def from_dir(
        cls,
        data_dir: str,
        setting: str = "seen",
        split: str = "train",
        tokenizer: Optional[Any] = None,
        batch_size: int = 4,
        seed: int = 0,
        max_instr_len: int = DEFAULT_MAX_INSTR_LEN,
        max_answer_len: int = DEFAULT_MAX_ANSWER_LEN,
    ) -> "WebVLNDataset":
        """从数据目录直接构造。"""
        records = load_split(data_dir, setting, split)
        episodes = prepare_episodes(
            records, tokenizer, max_instr_len=max_instr_len, max_answer_len=max_answer_len
        )
        return cls(episodes, split=split, batch_size=batch_size, seed=seed)

    def check_size(self, strict: bool = False) -> Optional[str]:
        """核对样本数是否与论文 2.3 节一致。

        WebVLN-v1 的公开数据可能随版本微调，规模不符不一定是错误，
        因此默认只返回提示字符串由调用方决定如何处理；
        复现基线时可设 ``strict=True`` 强制一致，避免用错划分却浑然不觉。

        Returns:
            规模一致时返回 None，否则返回描述差异的字符串。

        Raises:
            ValueError: ``strict`` 为 True 且规模不符。
        """
        expected = SPLIT_SIZES.get(self.split)
        if expected is None or len(self.episodes) == expected:
            return None
        msg = (
            f"划分 {self.split} 实际 {len(self.episodes)} 条，"
            f"论文 2.3 节为 {expected} 条"
        )
        if strict:
            raise ValueError(msg)
        return msg
