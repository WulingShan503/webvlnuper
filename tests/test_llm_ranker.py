"""4.3 节 LLM 语义排序的单元测试。"""

import json

from webvln.screening.cache import RankCache
from webvln.screening.candidate import Candidate, ElementType
from webvln.screening.llm_backend import ScriptedBackend
from webvln.screening.llm_ranker import LLMRanker, parse_indices


def make_candidates(n, start=0):
    return [
        Candidate(index=i, text=f"Item {i}", elem_type=ElementType.LINK)
        for i in range(start, start + n)
    ]


def ranker(responses, k=5, cache=None):
    return LLMRanker(backend=ScriptedBackend(responses), k=k, cache=cache)


# --- 输出解析 ---------------------------------------------------------------


def test_parses_documented_json_format():
    raw = '{"indices": [3, 1, 7]}'
    assert parse_indices(raw, valid=range(10)) == [3, 1, 7]


def test_parses_bare_array():
    assert parse_indices("[2, 4]", valid=range(10)) == [2, 4]


def test_parses_json_inside_markdown_fence():
    raw = '```json\n{"indices": [5, 2]}\n```'
    assert parse_indices(raw, valid=range(10)) == [5, 2]


def test_parses_json_with_surrounding_prose():
    raw = 'Sure! Here are the results: {"indices": [1, 0]} Hope this helps.'
    assert parse_indices(raw, valid=range(10)) == [1, 0]


def test_parses_alternative_key_names():
    assert parse_indices('{"ranking": [8, 3]}', valid=range(10)) == [8, 3]
    assert parse_indices('{"top_k": [4]}', valid=range(10)) == [4]


def test_parses_string_and_object_elements():
    assert parse_indices('{"indices": ["3", "1"]}', valid=range(10)) == [3, 1]
    raw = '{"indices": [{"index": 6}, {"index": 2}]}'
    assert parse_indices(raw, valid=range(10)) == [6, 2]


def test_falls_back_to_bare_integers():
    assert parse_indices("The best are 4 then 9.", valid=range(10)) == [4, 9]


def test_drops_hallucinated_out_of_range_indices():
    # 规则过滤后下标不连续，LLM 可能返回已被剔除或根本不存在的编号。
    assert parse_indices('{"indices": [99, 3, -1]}', valid=[3, 7]) == [3]


def test_deduplicates_repeated_indices():
    assert parse_indices('{"indices": [2, 2, 5]}', valid=range(10)) == [2, 5]


def test_returns_none_when_nothing_usable():
    assert parse_indices("I cannot determine the answer.", valid=range(10)) is None


def test_ignores_booleans_in_array():
    assert parse_indices('{"indices": [true, 3]}', valid=range(10)) == [3]


# --- 排序行为 ---------------------------------------------------------------


def test_returns_top_k_in_llm_order():
    r = ranker([json.dumps({"indices": [7, 2, 9, 0, 4, 1]})], k=5)
    res = r.rank("find the price", make_candidates(10))
    assert res.indices == [7, 2, 9, 0, 4]
    assert not res.parse_failed


def test_skips_llm_when_candidates_fit_within_k():
    # 候选数 <= k 时排序不改变被选集合，不应浪费一次 API 调用。
    backend = ScriptedBackend([])
    r = LLMRanker(backend=backend, k=5)
    res = r.rank("q", make_candidates(3))
    assert res.indices == [0, 1, 2]
    assert backend.n_calls == 0
    assert r.n_llm_calls == 0


def test_parse_failure_falls_back_to_original_order():
    r = ranker(["sorry, I can't help"], k=3)
    res = r.rank("q", make_candidates(10))
    assert res.parse_failed
    assert res.indices == [0, 1, 2]
    assert r.n_parse_failures == 1


def test_select_returns_candidate_objects_in_ranked_order():
    r = ranker([json.dumps({"indices": [4, 1]})], k=2)
    cands = make_candidates(6)
    picked = r.select("q", cands)
    assert [c.index for c in picked] == [4, 1]
    assert [c.text for c in picked] == ["Item 4", "Item 1"]


def test_works_with_non_contiguous_indices_after_rule_filter():
    cands = [
        Candidate(index=i, text=f"Item {i}", elem_type=ElementType.LINK)
        for i in (3, 11, 19, 25, 31, 40)
    ]
    r = ranker([json.dumps({"indices": [19, 40]})], k=2)
    assert r.rank("q", cands).indices == [19, 40]


# --- 缓存 -------------------------------------------------------------------


def test_second_identical_request_hits_cache():
    cache = RankCache()
    r = ranker([json.dumps({"indices": [1, 2]})], k=2, cache=cache)
    cands = make_candidates(8)

    first = r.rank("same question", cands)
    second = r.rank("same question", cands)

    assert first.indices == second.indices == [1, 2]
    assert not first.from_cache and second.from_cache
    # 只发生了一次真实调用，脚本后端仅预设了一条响应。
    assert r.n_llm_calls == 1
    assert cache.hits == 1


def test_cache_key_separates_different_k():
    cache = RankCache()
    cands = make_candidates(8)
    r5 = ranker([json.dumps({"indices": [1, 2, 3, 4, 5]})], k=5, cache=cache)
    r5.rank("q", cands)
    r3 = ranker([json.dumps({"indices": [7, 6, 0]})], k=3, cache=cache)
    res = r3.rank("q", cands)
    assert res.indices == [7, 6, 0]
    assert not res.from_cache


def test_failed_parse_is_not_cached():
    # 缓存解析失败的回退结果会让该页面永久退化为「不筛选」。
    cache = RankCache()
    r = ranker(["garbage", json.dumps({"indices": [2, 3]})], k=2, cache=cache)
    cands = make_candidates(8)
    assert r.rank("q", cands).parse_failed
    assert r.rank("q", cands).indices == [2, 3]


def test_cache_persists_across_instances(tmp_path):
    path = str(tmp_path / "llm_cache.jsonl")
    cands = make_candidates(8)

    r1 = ranker([json.dumps({"indices": [5, 6]})], k=2, cache=RankCache(path))
    r1.rank("q", cands)

    r2 = LLMRanker(backend=ScriptedBackend([]), k=2, cache=RankCache(path))
    res = r2.rank("q", cands)
    assert res.indices == [5, 6]
    assert res.from_cache
    assert r2.n_llm_calls == 0
