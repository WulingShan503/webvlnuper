"""从 YAML 配置构建筛选器。

把 ``configs/screening.yaml`` 的内容装配为 ``TwoStageScreener``，
使 5.3 节的 Top-k 消融与 5.5 节的单阶段对照实验只需改配置、不改代码。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from webvln.screening.cache import RankCache
from webvln.screening.candidate import PageArea
from webvln.screening.llm_backend import LLMBackend, OpenAIBackend
from webvln.screening.llm_ranker import LLMRanker
from webvln.screening.pipeline import TwoStageScreener
from webvln.screening.rule_filter import (
    DEFAULT_BLOCKED_KEYWORDS,
    DEFAULT_PRUNED_AREAS,
    RuleFilter,
)

DEFAULT_CONFIG_PATH = os.path.join("configs", "screening.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    import yaml  # 延迟导入：仅在使用配置文件时才需要该依赖

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_rule_filter(cfg: Mapping[str, Any]) -> Optional[RuleFilter]:
    """按配置构建阶段一。``enabled: false`` 时返回 None。"""
    if not cfg.get("enabled", True):
        return None

    areas = cfg.get("pruned_areas")
    pruned = (
        tuple(_coerce_area(a) for a in areas)
        if areas is not None
        else DEFAULT_PRUNED_AREAS
    )
    keywords = cfg.get("blocked_keywords")
    return RuleFilter(
        drop_duplicates=bool(cfg.get("drop_duplicates", True)),
        pruned_areas=pruned,
        blocked_keywords=(
            tuple(str(k) for k in keywords)
            if keywords is not None
            else DEFAULT_BLOCKED_KEYWORDS
        ),
        drop_empty=bool(cfg.get("drop_empty", True)),
        min_keep=int(cfg.get("min_keep", 1)),
    )


def build_cache(cfg: Mapping[str, Any]) -> Optional[RankCache]:
    """按配置构建缓存。"""
    if not cfg.get("enabled", True):
        return None
    return RankCache(path=cfg.get("path") or None)


def build_ranker(
    cfg: Mapping[str, Any],
    cache: Optional[RankCache] = None,
    backend: Optional[LLMBackend] = None,
) -> Optional[LLMRanker]:
    """按配置构建阶段二。

    Args:
        cfg: ``screening.llm_ranker`` 段。
        cache: 已构建的缓存。
        backend: 显式注入的后端。离线复现与测试传入替身即可，
            无需 API key——默认的 OpenAI 后端会在首次调用时才导入 openai。
    """
    if not cfg.get("enabled", True):
        return None

    model = str(cfg.get("model", "gpt-3.5-turbo"))
    if backend is None:
        backend = OpenAIBackend(
            model=model,
            temperature=float(cfg.get("temperature", 0.0)),
            max_retries=int(cfg.get("max_retries", 3)),
            timeout=float(cfg.get("timeout", 30.0)),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    return LLMRanker(
        backend=backend,
        k=int(cfg.get("k", 5)),
        cache=cache,
        model_name=model,
    )


def build_screener(
    config: Optional[Mapping[str, Any]] = None,
    path: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
) -> Optional[TwoStageScreener]:
    """构建完整的两阶段筛选器。

    Args:
        config: 已解析的配置字典。与 ``path`` 二选一。
        path: 配置文件路径，默认 ``configs/screening.yaml``。
        backend: 注入的 LLM 后端。

    Returns:
        TwoStageScreener；``screening.enabled: false`` 时返回 None，
        调用方据此走基线（不筛选）路径，用于 5.2 节的复现对照。
    """
    if config is None:
        config = load_yaml(path or DEFAULT_CONFIG_PATH)

    cfg = dict(config.get("screening", config))
    if not cfg.get("enabled", True):
        return None

    cache = build_cache(dict(cfg.get("cache", {})))
    return TwoStageScreener(
        rule_filter=build_rule_filter(dict(cfg.get("rule_filter", {}))),
        ranker=build_ranker(dict(cfg.get("llm_ranker", {})), cache=cache, backend=backend),
    )


def _coerce_area(value: Any) -> PageArea:
    """把配置中的区域写法转为 PageArea，兼容枚举名与枚举值。"""
    if isinstance(value, PageArea):
        return value
    text = str(value).strip()
    try:
        return PageArea[text.upper()]
    except KeyError:
        pass
    try:
        return PageArea(text)
    except ValueError:
        raise ValueError(f"未知的页面区域配置：{value!r}") from None
