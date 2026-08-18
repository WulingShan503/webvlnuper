"""LLM 调用后端。

抽象出 ``LLMBackend`` 协议，使排序逻辑不依赖具体服务商：
论文实验使用 OpenAI GPT-3.5-Turbo，而单元测试与离线复现需要可控的替身。
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol


class LLMBackend(Protocol):
    """一次对话补全调用。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """返回模型输出的原始文本。"""
        ...


class OpenAIBackend:
    """OpenAI Chat Completions 后端。

    论文 5.1 节配置：GPT-3.5-Turbo，温度设为 0 以保证排序结果可复现——
    排序是筛选步骤而非生成任务，采样随机性只会引入不必要的方差。

    Attributes:
        model: 模型名。
        temperature: 采样温度，默认 0。
        max_retries: 失败重试次数。
        timeout: 单次请求超时（秒）。
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import openai  # 延迟导入：离线复现与单元测试无需安装该依赖

        if self._api_key:
            openai.api_key = self._api_key

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    temperature=self.temperature,
                    request_timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp["choices"][0]["message"]["content"]
            except Exception as err:  # 网络 / 限流 / 服务端错误
                last_err = err
                # 指数退避：训练中的高频调用容易触发限流，
                # 立即重试只会加剧拥塞。
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM 调用在 {self.max_retries} 次重试后仍失败") from last_err


class ScriptedBackend:
    """按预设脚本返回响应的测试替身。"""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        if not self._responses:
            raise RuntimeError("ScriptedBackend 预设响应已耗尽")
        return self._responses.pop(0)

    @property
    def n_calls(self) -> int:
        return len(self.calls)
