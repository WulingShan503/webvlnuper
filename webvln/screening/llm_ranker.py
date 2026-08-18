"""4.3 阶段二：LLM 语义排序与 Top-k 选择。

对应论文式 (4.3.2)：

    C_t^{top-k} = Top-k(LLM-Rank(I, C_t))

流程为：候选序列化 → 构造提示词 → 调用 LLM → 解析 JSON 下标 → 取前 k 个。

k 的取值影响精度与压缩的权衡，论文 5.3 节的消融显示 k=5 最优
（Val SR 39.12%）：k=3 压缩过度会漏掉目标链接，k=8 保留了过多噪声。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from webvln.screening.cache import RankCache, make_key
from webvln.screening.candidate import Candidate
from webvln.screening.llm_backend import LLMBackend
from webvln.screening.prompts import SYSTEM_PROMPT, build_rank_prompt
from webvln.screening.serializer import build_candidate_block

#: 从自由文本中兜底提取 JSON 对象。GPT-3.5-Turbo 偶尔会在 JSON 前后
#: 附加解释性文字或 Markdown 代码块围栏，直接 json.loads 会失败。
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")


@dataclass
class RankResult:
    """一次排序的结果。

    Attributes:
        indices: LLM 给出的候选下标，按相关性降序，长度不超过 k。
        from_cache: 是否来自缓存。
        parse_failed: JSON 解析是否失败并触发回退。
    """

    indices: List[int]
    from_cache: bool = False
    parse_failed: bool = False


def parse_indices(raw: str, valid: Sequence[int]) -> Optional[List[int]]:
    """从 LLM 输出解析候选下标。

    解析分三层，逐层放宽：标准 JSON → 文本中嵌入的 JSON 对象 → 裸整数序列。
    三层全部失败时返回 None，由调用方决定回退策略。

    Args:
        raw: LLM 原始输出。
        valid: 合法下标集合。LLM 可能返回不存在的下标（幻觉），
            或复述提示词中的示例编号，必须据此过滤，
            否则错误下标会索引到无关的特征行。

    Returns:
        去重后的合法下标列表；无法解析时为 None。
    """
    valid_set = set(valid)
    payload = _try_json(raw)

    if payload is not None:
        indices = _extract_from_payload(payload)
        if indices is not None:
            return _sanitize(indices, valid_set)

    # 末层回退：直接抓取文本中的整数。仅在前两层失败时启用，
    # 因为它无法区分下标与其他数字（如 k 值本身）。
    nums = [int(m) for m in _INT_RE.findall(raw)]
    filtered = _sanitize(nums, valid_set)
    return filtered or None


def _try_json(raw: str) -> Optional[object]:
    """尝试把输出解析为 JSON，容忍代码块围栏与前后缀文字。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _extract_from_payload(payload: object) -> Optional[List[int]]:
    """从已解析的 JSON 中取出下标列表。

    提示词要求 ``{"indices": [...]}``，但模型也可能直接返回数组，
    或换用 ranking / top_k 等键名，因此这里一并接受。
    """
    if isinstance(payload, list):
        return _coerce_int_list(payload)
    if isinstance(payload, dict):
        for key in ("indices", "index", "ranking", "top_k", "actions", "results"):
            if key in payload:
                value = payload[key]
                if isinstance(value, list):
                    return _coerce_int_list(value)
                if isinstance(value, int):
                    return [value]
        # 键名未知但仅有一个数组值时，按该数组处理。
        arrays = [v for v in payload.values() if isinstance(v, list)]
        if len(arrays) == 1:
            return _coerce_int_list(arrays[0])
    return None


def _coerce_int_list(values: Sequence[object]) -> List[int]:
    """把混合类型的列表转为整数列表。

    模型可能返回字符串下标（"3"）或包裹成对象（{"index": 3}）。
    """
    out: List[int] = []
    for v in values:
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str):
            m = _INT_RE.search(v)
            if m:
                out.append(int(m.group(0)))
        elif isinstance(v, dict):
            for key in ("index", "idx", "id"):
                if isinstance(v.get(key), int):
                    out.append(v[key])
                    break
    return out


def _sanitize(indices: Sequence[int], valid: set) -> List[int]:
    """过滤非法下标并去重，保持原有顺序。"""
    seen = set()
    out = []
    for i in indices:
        if i in valid and i not in seen:
            seen.add(i)
            out.append(i)
    return out


@dataclass
class LLMRanker:
    """基于 LLM 的候选语义排序器。

    Attributes:
        backend: LLM 调用后端。
        k: Top-k 的 k 值，论文 5.3 节最优取 5。
        cache: 可选的响应缓存。
        model_name: 参与缓存键构造，避免跨模型错误命中。
    """

    backend: LLMBackend
    k: int = 5
    cache: Optional[RankCache] = None
    model_name: str = "gpt-3.5-turbo"
    n_parse_failures: int = field(default=0, init=False)
    n_llm_calls: int = field(default=0, init=False)

    def rank(self, instruction: str, candidates: Sequence[Candidate]) -> RankResult:
        """对候选排序并返回 Top-k 下标。

        候选数不超过 k 时跳过 LLM 调用——此时排序不会改变被选集合，
        调用只是徒增开销。
        """
        valid = [c.index for c in candidates]
        if len(candidates) <= self.k:
            return RankResult(indices=list(valid))

        block = build_candidate_block(candidates)
        key = make_key(instruction, block, self.k, self.model_name)

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return RankResult(indices=cached, from_cache=True)

        prompt = build_rank_prompt(instruction, block, self.k)
        raw = self.backend.complete(SYSTEM_PROMPT, prompt)
        self.n_llm_calls += 1

        indices = parse_indices(raw, valid)
        parse_failed = indices is None
        if parse_failed:
            # 解析失败时退回原始顺序的前 k 个，而非丢弃该步：
            # 训练不能因单次 API 异常中断，退化为「不筛选」是安全的下界。
            self.n_parse_failures += 1
            indices = list(valid)

        indices = indices[: self.k]

        if self.cache is not None and not parse_failed:
            self.cache.put(key, indices)

        return RankResult(indices=indices, parse_failed=parse_failed)

    def select(
        self, instruction: str, candidates: Sequence[Candidate]
    ) -> List[Candidate]:
        """返回排序后保留的候选对象。

        输出按 LLM 给出的相关性顺序排列，不还原为原始页面顺序——
        下游可据此做加权，且候选自带 ``index`` 可随时映射回特征张量。
        """
        result = self.rank(instruction, candidates)
        by_index = {c.index: c for c in candidates}
        return [by_index[i] for i in result.indices if i in by_index]

    def stats(self) -> dict:
        d = {"n_llm_calls": self.n_llm_calls, "n_parse_failures": self.n_parse_failures}
        if self.cache is not None:
            d["cache"] = self.cache.stats()
        return d
