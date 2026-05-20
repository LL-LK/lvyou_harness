"""
RAG接口定义
"""
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class RAGPort(Protocol):
    """
    RAG检索端口接口

    所有RAG实现必须实现此接口:
    - Milvus向量库
    - Elasticsearch
    - Chroma
    - 内存存储(测试用)
    """

    def initialize(self) -> bool:
        """初始化连接"""
        ...

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索相似文档

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            文档列表，每项包含 content, metadata, score
        """
        ...

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """
        添加文档到向量库

        Args:
            documents: 文档列表，每项包含 content, metadata
            batch_size: 批处理大小

        Returns:
            成功添加的数量
        """
        ...

    def count(self) -> int:
        """返回向量库中的文档数量"""
        ...

    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        ...

    def list_collections(self) -> List[str]:
        """列出所有集合"""
        ...
