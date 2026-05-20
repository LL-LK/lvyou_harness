"""
存储接口定义
"""
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class StoragePort(Protocol):
    """
    存储端口接口

    支持:
    - 本地文件
    - S3/MinIO
    - 数据库
    """

    def save(self, key: str, data: Any) -> bool:
        """保存数据"""
        ...

    def load(self, key: str) -> Optional[Any]:
        """加载数据"""
        ...

    def delete(self, key: str) -> bool:
        """删除数据"""
        ...

    def exists(self, key: str) -> bool:
        """检查是否存在"""
        ...

    def list_keys(self, prefix: str = "") -> List[str]:
        """列出所有键"""
        ...
