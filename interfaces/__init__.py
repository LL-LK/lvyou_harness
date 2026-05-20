"""
LvyouHarness 接口抽象层
========================

定义各组件的接口协议(Protocol)，实现:
1. 清晰的接口边界
2. 模块间低耦合
3. 易于扩展和Mock测试

接口设计原则:
- 接口尽量小而专注
- 使用Protocol而不是ABC(duck typing)
- 依赖注入而非硬编码
"""
from .rag import RAGPort
from .llm import LLMPort
from .crawler import CrawlerPort
from .storage import StoragePort

__all__ = [
    "RAGPort",
    "LLMPort",
    "CrawlerPort",
    "StoragePort",
]
