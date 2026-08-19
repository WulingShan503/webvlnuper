"""WebVLN-v1 数据集加载与文本编码。

论文 2.3 节的数据划分：8,960 / 1,262 / 4,603（训练 / 验证 / 测试），
对应基线论文 60% / 10% / 30% 的比例，三个划分覆盖全部三个网站
但记录与路径互不重叠。

本子包只负责**离线数据**：episode 记录、指令与答案的 token 编码、
批次取样。导航图（``map.json`` / ``shortest_paths.json``）与特征表
属于运行时环境，见 ``webvln/data/graph.py`` 与 ``features.py``。
"""

from webvln.data.dataset import (
    SPLIT_SIZES,
    WebVLNDataset,
    load_split,
    prepare_episodes,
    save_encoded,
    split_path,
)
from webvln.data.episode import Episode, infer_website_id
from webvln.data.text import (
    ANSWER_BOS_TOKEN,
    ANSWER_EOS_TOKEN,
    DEFAULT_MAX_ANSWER_LEN,
    DEFAULT_MAX_INSTR_LEN,
    attention_mask_from_length,
    build_instruction_text,
    encode_answer,
    encode_instruction,
    load_bert_tokenizer,
    pad_answer_tokens,
    pad_instr_tokens,
)

__all__ = [
    "ANSWER_BOS_TOKEN",
    "ANSWER_EOS_TOKEN",
    "DEFAULT_MAX_ANSWER_LEN",
    "DEFAULT_MAX_INSTR_LEN",
    "Episode",
    "SPLIT_SIZES",
    "WebVLNDataset",
    "attention_mask_from_length",
    "build_instruction_text",
    "encode_answer",
    "encode_instruction",
    "infer_website_id",
    "load_bert_tokenizer",
    "load_split",
    "pad_answer_tokens",
    "pad_instr_tokens",
    "prepare_episodes",
    "save_encoded",
    "split_path",
]
