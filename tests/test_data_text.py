"""指令与答案编码的单元测试。

对齐官方 ``r2r_src/utils.py`` 的 ``pad_instr_tokens`` / ``pad_answer_tokens``：
指令定长 50（``--maxInput 50``），答案定长 40，截断按 ``max_length - 3``。
"""

import pytest

from webvln.data.text import (
    ANSWER_BOS_TOKEN,
    ANSWER_EOS_TOKEN,
    attention_mask_from_length,
    build_instruction_text,
    encode_answer,
    encode_instruction,
    pad_answer_tokens,
    pad_instr_tokens,
)


class FakeTokenizer:
    """按空格切词的 tokenizer 替身。

    真实的 BertTokenizer 需要下载词表，单元测试只关心补齐与截断逻辑，
    因此这里用可预测的 id 映射：特殊 token 取固定值，普通词取长度。
    """

    SPECIAL = {"[PAD]": 0, "[CLS]": 101, "[SEP]": 102, "[unused0]": 1, "[unused1]": 2}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.SPECIAL.get(t, 1000 + len(t)) for t in tokens]


def test_instruction_text_puts_target_before_question():
    # 官方 prepare_dataset 的拼法：目标标题充当辅助描述 D。
    text = build_instruction_text("Grey Socks", "how much does it cost?")
    assert text == "Target: Grey Socks, how much does it cost?"


def test_pad_instr_tokens_wraps_and_pads():
    tokens, num_words = pad_instr_tokens(["a", "b", "c"], max_length=8)
    assert tokens[0] == "[CLS]"
    assert tokens[4] == "[SEP]"
    assert num_words == 5  # [CLS] a b c [SEP]
    assert tokens[5:] == ["[PAD]"] * 3
    assert len(tokens) == 8


def test_pad_instr_tokens_truncates_but_keeps_sep():
    tokens, num_words = pad_instr_tokens([f"w{i}" for i in range(20)], max_length=6)
    assert len(tokens) == 6
    assert tokens[0] == "[CLS]"
    # 截断后结尾仍须是 [SEP]，否则 BERT 缺少序列边界标记。
    assert tokens[-1] == "[SEP]"
    assert num_words == 6
    assert "[PAD]" not in tokens


def test_pad_instr_tokens_rejects_too_short():
    # 官方返回 None 后在 convert_tokens_to_ids 处崩溃；这里提前报错。
    with pytest.raises(ValueError):
        pad_instr_tokens(["a", "b"], max_length=10)


def test_answer_input_and_target_are_shifted():
    inp, n_in = pad_answer_tokens(["x", "y"], max_length=6, eos_flag=False)
    tgt, _ = pad_answer_tokens(["x", "y"], max_length=6, eos_flag=True)
    assert inp[:3] == [ANSWER_BOS_TOKEN, "x", "y"]
    assert tgt[:3] == ["x", "y", ANSWER_EOS_TOKEN]
    assert n_in == 3
    assert len(inp) == len(tgt) == 6


def test_answer_truncates_by_max_minus_three():
    # 官方按 max_length - 3 截断；改成 -1 会让序列比官方长两位。
    tokens, _ = pad_answer_tokens([f"w{i}" for i in range(20)], max_length=10)
    assert len(tokens) == 10
    assert tokens.count("[PAD]") == 2
    assert tokens[0] == ANSWER_BOS_TOKEN


def test_encode_instruction_returns_ids():
    ids, num_words = encode_instruction("Target: Socks, cost?", FakeTokenizer(), 8)
    assert ids[0] == 101
    assert ids[num_words - 1] == 102
    assert ids[num_words:] == [0] * (8 - num_words)


def test_encode_answer_returns_two_sequences():
    inp, tgt, num_words = encode_answer("it costs ten dollars", FakeTokenizer(), 10)
    assert inp[0] == 1  # [unused0]
    assert tgt[num_words - 1] == 2  # [unused1] 落在有效末位
    assert len(inp) == len(tgt) == 10
    assert num_words == 5


def test_attention_mask_from_length():
    assert attention_mask_from_length(3, 6) == [1, 1, 1, 0, 0, 0]
    # 越界长度不应产出超长掩码。
    assert attention_mask_from_length(9, 4) == [1, 1, 1, 1]
