"""
LLM接口定义
"""
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """
    LLM调用端口接口

    支持:
    - OpenAI
    - MiniMax
    - Claude
    - Deepseek
    - 本地模型
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的文本
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """
        流式生成文本

        Yields:
            文本片段
        """
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """
        对话模式

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            助手回复
        """
        ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        ...
