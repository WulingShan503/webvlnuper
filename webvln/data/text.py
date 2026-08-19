"""指令与答案的 WordPiece 编码。

论文 3.2 节把 Q 与 D 拼成 ``[CLS] Q [SEP] D [SEP]``；官方
``prepare_dataset`` 实际拼的是 ``f"Target: {target}, {Q}"`` 后走
``pad_instr_tokens``（单段，只有一个 [SEP]）——即用商品标题充当辅助描述 D。
两种写法都在这里提供：``build_instruction_text`` 复刻官方，
``webvln/models/language.py:build_instruction_tokens`` 是论文的两段式。

答案的编码有两份，对应 teacher forcing 的输入与目标：

    answer_enc        [unused0] w_1 ... w_L [PAD]...   解码器输入（右移一位）
    answer_enc_w_eos  w_1 ... w_L [unused1] [PAD]...   预测目标

BERT 词表里 ``[unused0]`` / ``[unused1]`` 是未使用槽位，官方借用它们当
BOS / EOS。不能改用 ``[CLS]`` / ``[SEP]``：那两个 token 在指令编码中
已有各自含义，共享会让回答头与语言分支争夺同一嵌入。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

#: 答案序列的起止 token。官方 ``pad_answer_tokens`` 用 BERT 的未使用槽位。
ANSWER_BOS_TOKEN = "[unused0]"
ANSWER_EOS_TOKEN = "[unused1]"

#: 官方 ``--maxInput 50``（``param.py``）。论文 3.2 节未给具体截断长度。
DEFAULT_MAX_INSTR_LEN = 50
#: 官方 ``prepare_dataset`` 传入 ``pad_answer_tokens(..., 40, ...)``。
DEFAULT_MAX_ANSWER_LEN = 40


def load_bert_tokenizer(model_name: str = "bert-base-uncased") -> Any:
    """加载 BertTokenizer。

    延迟导入 transformers：数据结构与编码逻辑的单元测试用轻量替身即可，
    不应为了跑测试而拉起 400MB 的依赖。
    """
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(model_name)


def build_instruction_text(target: str, question: str) -> str:
    """拼接指令文本，复刻官方 ``prepare_dataset``。

    官方形如 ``"Target: Grey Striped Socks, how much does it cost?"``。
    目标标题放在问题之前，是因为它承担论文所述辅助描述 D 的作用——
    问题本身常常不足以定位目标页（基线论文脚注 1）。
    """
    return f"Target: {target}, {question}"


def pad_instr_tokens(
    instr_tokens: Sequence[str], max_length: int = DEFAULT_MAX_INSTR_LEN
) -> Tuple[List[str], int]:
    """给指令 token 加 ``[CLS]`` / ``[SEP]`` 并补齐到定长。

    Args:
        instr_tokens: WordPiece 切分后的 token（不含特殊 token）。
        max_length: 目标长度。

    Returns:
        (tokens, num_words)。``num_words`` 含 ``[CLS]`` 与 ``[SEP]``，
        供构造 attention mask 使用。

    Raises:
        ValueError: token 数不足 3 个。官方 ``pad_instr_tokens`` 在这种情况下
            返回 None，调用方随后会在 ``convert_tokens_to_ids`` 处崩溃；
            这里直接报错，把问题定位在数据而非编码环节。
    """
    if len(instr_tokens) <= 2:
        raise ValueError(f"指令过短，无法编码：{list(instr_tokens)!r}")

    tokens = list(instr_tokens)
    # -2 给 [CLS] 与 [SEP] 留位；先截断再加特殊 token，
    # 否则加完再截会把结尾的 [SEP] 切掉。
    if len(tokens) > max_length - 2:
        tokens = tokens[: max_length - 2]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    num_words = len(tokens)
    tokens += ["[PAD]"] * (max_length - len(tokens))
    return tokens, num_words


def pad_answer_tokens(
    answer_tokens: Sequence[str],
    max_length: int = DEFAULT_MAX_ANSWER_LEN,
    eos_flag: bool = False,
) -> Tuple[List[str], int]:
    """给答案 token 加 BOS 或 EOS 并补齐到定长。

    Args:
        answer_tokens: WordPiece 切分后的答案 token。
        max_length: 目标长度。
        eos_flag: True 时产出目标序列（结尾加 ``[unused1]``），
            False 时产出解码器输入（开头加 ``[unused0]``）。

    Returns:
        (tokens, num_words)。

    Note:
        官方按 ``max_length - 3`` 截断，注释写「-2 for [BOS] and [SEP] and [EOS]」
        （数目对不上，实际只加一个特殊 token）。这里沿用 -3：改成 -1 会让
        序列比官方长两位，与预训练权重的位置嵌入分布不一致。
    """
    tokens = list(answer_tokens)
    if len(tokens) > max_length - 3:
        tokens = tokens[: max_length - 3]

    if eos_flag:
        tokens = tokens + [ANSWER_EOS_TOKEN]
    else:
        tokens = [ANSWER_BOS_TOKEN] + tokens

    num_words = len(tokens)
    tokens += ["[PAD]"] * (max_length - len(tokens))
    return tokens, num_words


def encode_instruction(
    text: str, tokenizer: Any, max_length: int = DEFAULT_MAX_INSTR_LEN
) -> Tuple[List[int], int]:
    """把指令文本编码为定长 token id 序列。

    Returns:
        (token_ids, num_words)。
    """
    tokens, num_words = pad_instr_tokens(tokenizer.tokenize(text), max_length)
    return tokenizer.convert_tokens_to_ids(tokens), num_words


def encode_answer(
    answer: str, tokenizer: Any, max_length: int = DEFAULT_MAX_ANSWER_LEN
) -> Tuple[List[int], List[int], int]:
    """把答案编码为解码器输入与目标两份序列。

    Returns:
        (answer_enc, answer_enc_w_eos, num_words)。
        ``num_words`` 取输入序列的有效长度，与官方 ``answer_words`` 一致。
    """
    tokens = tokenizer.tokenize(answer)
    inp, num_words = pad_answer_tokens(tokens, max_length, eos_flag=False)
    tgt, _ = pad_answer_tokens(tokens, max_length, eos_flag=True)
    return (
        tokenizer.convert_tokens_to_ids(inp),
        tokenizer.convert_tokens_to_ids(tgt),
        num_words,
    )


def attention_mask_from_length(num_words: int, max_length: int) -> List[int]:
    """由有效 token 数构造 0/1 注意力掩码。

    官方在 agent 中用 ``text_enc != 0`` 现算掩码，效果相同；
    显式用长度更稳妥——若答案中出现 id 为 0 的 token（``[PAD]`` 之外的
    词表首项），按值比较会误判为 PAD。
    """
    valid = max(0, min(num_words, max_length))
    return [1] * valid + [0] * (max_length - valid)
