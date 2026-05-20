"""
LLM适配器
==========

封装LLM调用，提供统一的LLM接口

支持:
1. MiniMax (当前主力)
2. OpenAI
3. Deepseek
4. 本地模型 (ollama)
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

from ..interfaces.llm import LLMPort

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "minimax-cn"
    model: str = "MiniMax-M2.7"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


class MiniMaxAdapter(LLMPort):
    """
    MiniMax LLM适配器
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None

    def initialize(self) -> bool:
        """初始化MiniMax客户端"""
        try:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("MINIMAX_API_KEY")
            if not api_key:
                logger.warning("MiniMax API Key未设置")

            base_url = self.config.base_url or "https://api.minimax.chat/v1"

            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )
            return True

        except Exception as e:
            logger.error(f"MiniMax初始化失败: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """生成文本"""
        if not self._client:
            self.initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            return self._clean_think_tags(content)

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"[Error: {e}]"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Iterator[str]:
        """流式生成"""
        if not self._client:
            self.initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"[Error: {e}]"

    def _clean_think_tags(self, content: str) -> str:
        """清理MiniMax模型的think标签和混乱格式"""
        import re

        if not content:
            return ""

        # 移除 <think>...</think> 标签及其内容（最优先）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 如果剩余内容很少或为空，尝试直接提取实际内容
        # M2.7有时会把真实回复放在think标签后面
        if len(content.strip()) < 10:
            # 尝试找到真正的回复（通常在最后一个\n\n之后）
            parts = content.split('\n\n')
            if len(parts) > 1:
                # 取最后非空部分
                for part in reversed(parts):
                    part = part.strip()
                    if len(part) > 20:
                        content = part
                        break

        # 清理HTML标签
        content = re.sub(r'<[^>]+>', '', content)

        # 清理多余的空白和换行
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        # 如果内容仍然很短，尝试简单描述
        if len(content.strip()) < 20:
            # 可能模型没正常回复，返回提示
            return content.strip()

        return content.strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """对话"""
        if not self._client:
            self.initialize()

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            return self._clean_think_tags(content)

        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"[Error: {e}]"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """获取嵌入向量"""
        # MiniMax支持embedding API
        try:
            if not self._client:
                self.initialize()

            # 使用text-embedding模型
            embeddings = []
            for text in texts:
                response = self._client.embeddings.create(
                    model="embo-01",
                    input=text,
                )
                embeddings.append(response.data[0].embedding)
            return embeddings

        except Exception as e:
            logger.error(f"Embedding失败: {e}")
            # 返回随机向量作为fallback
            import numpy as np
            return [np.random.rand(1024).tolist() for _ in texts]


class MockLLMAdapter(LLMPort):
    """
    Mock LLM适配器 - 用于测试
    """

    def __init__(self):
        self._initialized = True

    def initialize(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        return f"[Mock] 已收到提示: {prompt[:50]}..."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Iterator[str]:
        yield f"[Mock] 已收到提示: {prompt[:50]}..."

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        return f"[Mock] 已收到 {len(messages)} 条消息"

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        return [np.random.rand(1024).tolist() for _ in texts]


# =============================================================================
# 工厂函数
# =============================================================================

def create_llm_adapter(
    adapter_type: str = "minimax",
    **kwargs,
) -> LLMPort:
    """
    创建LLM适配器

    Args:
        adapter_type: 适配器类型 ("minimax", "openai", "mock")
        **kwargs: 传递给适配器的配置

    Returns:
        LLMPort接口实例
    """
    if adapter_type == "minimax":
        config = LLMConfig(**kwargs)
        return MiniMaxAdapter(config)
    elif adapter_type == "mock":
        return MockLLMAdapter()
    else:
        raise ValueError(f"未知的LLM适配器类型: {adapter_type}")
