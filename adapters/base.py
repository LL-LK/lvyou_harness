"""
适配器基类
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type

T = TypeVar("T")


class BaseAdapter(ABC):
    """
    适配器基类

    设计原则:
    1. 适配器封装具体实现，对外暴露统一接口
    2. 支持热插拔 - 可以随时替换底层实现
    3. 延迟初始化 - 避免不必要的资源占用
    """

    _instance: Optional[BaseAdapter] = None

    def __init__(self):
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """初始化适配器"""
        pass

    def ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            self.initialize()

    @classmethod
    def get_instance(cls) -> "BaseAdapter":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例(用于测试)"""
        cls._instance = None


from typing import Optional
