"""LLM 响应缓存。

论文 5.5 节指出：训练与评测中同一页面会被反复访问（rollout 采样、
多个 episode 共享同一站点结构），相同的「指令 + 候选集」组合大量重复。
引入缓存后 API 调用次数下降 52%，而 SR 无变化——缓存是纯粹的开销优化，
命中时返回的结果与重新调用完全一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Dict, List, Optional


def make_key(instruction: str, candidate_block: str, k: int, model: str) -> str:
    """构造缓存键。

    键必须覆盖全部影响输出的输入：指令、候选清单、k、模型名。
    漏掉任一项都会导致跨配置的错误命中，使消融实验结果失真。
    """
    payload = json.dumps(
        {"i": instruction, "c": candidate_block, "k": k, "m": model},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RankCache:
    """排序结果缓存，可选持久化到 JSONL 文件。

    以 JSONL 而非单个 JSON 保存，是为了让训练中途崩溃时已有的缓存条目
    依然可读——追加写入不会破坏此前的内容。

    Attributes:
        path: 持久化路径。为 None 时仅驻留内存。
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._store: Dict[str, List[int]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            self._load(path)

    def _load(self, path: str) -> None:
        """从 JSONL 载入已有缓存，跳过损坏行。"""
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._store[record["key"]] = list(record["indices"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    # 上次运行被中断时最后一行可能不完整，忽略即可。
                    continue

    def get(self, key: str) -> Optional[List[int]]:
        """查询缓存，同时累计命中 / 未命中计数。"""
        with self._lock:
            if key in self._store:
                self.hits += 1
                return list(self._store[key])
            self.misses += 1
            return None

    def put(self, key: str, indices: List[int]) -> None:
        """写入缓存并追加持久化。"""
        with self._lock:
            self._store[key] = list(indices)
            if self.path:
                self._append(key, indices)

    def _append(self, key: str, indices: List[int]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"key": key, "indices": indices}, ensure_ascii=False) + "\n"
            )

    @property
    def hit_rate(self) -> float:
        """命中率，对应论文报告的 API 调用节省比例。"""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }
