"""一条 WebVLN episode 的数据表示。

官方 ``r2r_src/utils.py:prepare_dataset`` 把 ``{split}.json`` 的原始记录
转成带 token 编码的字典，字段为：

    idx, target, path, Q, A, text, text_enc, text_words,
    answer_enc, answer_words, answer_enc_w_eos

其中 ``path`` 是 ``[起始 urlID, ..., 目标 urlID]``，元素形如
``"<websiteID>_<页面序号>"``——官方从 ``path[0].split("_")[0]`` 取网站 ID，
所以下划线前缀即网站标识。

这里用 dataclass 承载同样的字段，理由是训练循环要频繁访问
``gt_path[-1]`` 与 ``answer_enc``，字典的字符串键在 200,000 迭代中
既慢也容易写错；``to_official_dict`` 保证仍能喂给官方 ``R2RBatch``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class Episode:
    """一条导航 + 问答记录。

    Attributes:
        idx: 记录 ID。官方 ``Evaluation`` 以 ``str(idx)`` 为键索引 ground truth，
            因此这里保留原始类型不做转换。
        path: 最短路径上的 urlID 序列，``path[0]`` 为首页、``path[-1]`` 为目标页。
        question: 问题 Q。
        answer: 参考答案 A（自由形式句子，WUPS 的比对对象）。
        target: 目标商品/页面的标题，与 Q 拼成指令文本。
        text: 送入 BERT 的指令字符串。
        text_enc: 指令 token id，长度 ``max_instr_len``。
        text_words: 指令的有效 token 数（含 [CLS] 与 [SEP]）。
        answer_enc: 解码器输入，以 ``[unused0]`` 起始。
        answer_enc_w_eos: 解码目标，以 ``[unused1]`` 结尾。
        answer_words: 答案的有效 token 数。
        website_id: 网站 ID。为 None 时由 ``path[0]`` 推断。
    """

    idx: Any
    path: List[str]
    question: str = ""
    answer: str = ""
    target: str = ""
    text: str = ""
    text_enc: List[int] = field(default_factory=list)
    text_words: int = 0
    answer_enc: List[int] = field(default_factory=list)
    answer_enc_w_eos: List[int] = field(default_factory=list)
    answer_words: int = 0
    website_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.path = [str(p) for p in self.path]
        if self.website_id is None:
            self.website_id = infer_website_id(self.path)

    @property
    def start_url_id(self) -> str:
        """起始页（首页）的 urlID。"""
        return self.path[0]

    @property
    def target_url_id(self) -> str:
        """目标页的 urlID。"""
        return self.path[-1]

    @property
    def path_length(self) -> int:
        """ground-truth 路径长度（页面数）。

        论文 5.1 节的 TL 与 SPL 都以「经过的页面数」为单位而非物理距离，
        与官方 ``eval.py`` 中 ``len(path)`` 的算法一致。
        数据集平均路径长度 3.32（基线论文 Automatic Path Generation 一节）。
        """
        return len(self.path)

    def teacher_url_id(self, step: int) -> Optional[str]:
        """第 ``step`` 步应跳转到的 urlID。

        越界返回 None：此时智能体已走到路径末端，教师动作是 [EOA]
        而非任何候选（官方 ``_teacher_action`` 中
        ``step == len(gt_path) - 1`` 的分支）。
        """
        nxt = step + 1
        if nxt < len(self.path):
            return self.path[nxt]
        return None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Episode":
        """从官方 ``{split}_enc.json`` 的一条记录构造。

        兼容两种形态：已编码的 ``_enc.json``（含 ``text`` / ``text_enc``），
        以及未编码的原始 ``{split}.json``（问答放在 ``QA`` 二元组里）。
        未编码时文本字段留空，由 ``prepare_episodes`` 补齐编码，
        避免在数据结构层引入 tokenizer 依赖。
        """
        qa = record.get("QA") or []
        question = record.get("Q") or (qa[0] if len(qa) > 0 else "")
        answer = record.get("A") or (qa[1] if len(qa) > 1 else "")

        return cls(
            idx=record.get("idx"),
            path=list(record.get("path") or []),
            question=str(question or ""),
            answer=str(answer or ""),
            target=str(record.get("target") or ""),
            text=str(record.get("text") or ""),
            text_enc=list(record.get("text_enc") or []),
            text_words=int(record.get("text_words") or 0),
            answer_enc=list(record.get("answer_enc") or []),
            answer_enc_w_eos=list(record.get("answer_enc_w_eos") or []),
            answer_words=int(record.get("answer_words") or 0),
            website_id=record.get("websiteID"),
        )

    def to_official_dict(self) -> Dict[str, Any]:
        """转回官方 ``R2RBatch`` 期望的字典。

        字段名与 ``prepare_dataset`` 的输出严格一致，
        使本模块的数据可以直接喂给未改动的官方 env / agent。
        """
        return {
            "idx": self.idx,
            "target": self.target,
            "path": list(self.path),
            "Q": self.question,
            "A": self.answer,
            "text": self.text,
            "text_enc": list(self.text_enc),
            "text_words": self.text_words,
            "answer_enc": list(self.answer_enc),
            "answer_words": self.answer_words,
            "answer_enc_w_eos": list(self.answer_enc_w_eos),
        }


def infer_website_id(path: Sequence[str]) -> str:
    """从 urlID 前缀取网站 ID。

    官方 ``R2RBatch.reset`` 用 ``item['path'][0].split("_")[0]`` 作为
    ``scanId`` 传给模拟器，即网站标识就在 urlID 的下划线之前。
    """
    if not path:
        return ""
    return str(path[0]).split("_")[0]
