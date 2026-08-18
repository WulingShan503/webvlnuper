"""4.3 节使用的提示词模板。

单独成模块，便于在 5.3 / 5.5 节的消融实验中替换措辞而不触碰排序逻辑。
"""

SYSTEM_PROMPT = (
    "You are a web navigation assistant. You judge which clickable elements on "
    "a webpage are most likely to lead toward the information requested by a "
    "user instruction. Always answer with valid JSON and nothing else."
)

#: 论文 4.3 节给出的排序提示词。要求返回 JSON 格式的下标列表，
#: 是为了让输出可被稳定解析——自由文本回复在 200,000 次迭代的规模下
#: 解析失败率不可控。
RANK_PROMPT = """You are a web navigation assistant. Given the instruction:
'{instruction}', rank the following candidate actions by their relevance to \
completing the task. Return the indices of the top-{k} most relevant actions \
in JSON format.

Candidate actions:
{candidates}

Respond with only a JSON object of the form {{"indices": [...]}}, ordered from \
most to least relevant."""


def build_rank_prompt(instruction: str, candidate_block: str, k: int) -> str:
    """填充排序提示词。"""
    return RANK_PROMPT.format(
        instruction=instruction.strip(),
        candidates=candidate_block,
        k=k,
    )
