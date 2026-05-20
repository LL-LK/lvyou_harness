"""
RAG适配器
=========

封装向量库实现，提供统一的RAG接口

支持多种实现:
1. MilvusAdapter (pymilvus)
2. SimpleVectorAdapter (内存/测试用)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..interfaces.rag import RAGPort

logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    """Milvus配置"""
    uri: str = "./milvus_rag.db"
    collection_name: str = "lvyou_guilin"
    embedding_model: str = "BAAI/bge-m3"
    dim: int = 1024
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"


class MilvusAdapter(RAGPort):
    """
    Milvus向量库适配器

    使用pymilvus连接Milvus Lite
    """

    def __init__(self, config: Optional[MilvusConfig] = None):
        self.config = config or MilvusConfig()
        self._client = None
        self._collection = None

    def initialize(self) -> bool:
        """初始化Milvus连接"""
        try:
            from pymilvus import connections, Collection

            connections.connect(uri=self.config.uri)
            logger.info(f"Milvus连接成功: {self.config.uri}")

            if self.config.collection_name not in self._list_collections():
                logger.warning(f"Collection {self.config.collection_name} 不存在")

            self._collection = Collection(self.config.collection_name)
            self._collection.load()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Milvus初始化失败: {e}")
            return False

    def _list_collections(self) -> List[str]:
        """列出所有collection"""
        try:
            from pymilvus import utility
            return utility.list_collections()
        except:
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索相似文档"""
        if not self._initialized:
            self.initialize()

        try:
            # 简化实现 - 实际应该用embedding搜索
            # 这里返回空列表作为占位
            logger.warning("Milvus检索需要embedding模型支持")
            return []

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """添加文档"""
        if not self._initialized:
            self.initialize()

        if not documents:
            return 0

        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            import numpy as np

            # 创建collection
            if not self._collection_exists():
                self._create_collection()

            texts = []
            embeddings = []
            sources = []
            data_types = []
            cities = []
            titles = []

            for doc in documents:
                texts.append(doc.get("content", "")[:1000])
                sources.append(doc.get("source", ""))
                data_types.append(doc.get("data_type", ""))
                cities.append(doc.get("city", "桂林"))
                titles.append(doc.get("title", "")[:100])
                # 占位embedding
                embeddings.append(np.random.rand(self.config.dim).tolist())

            # 插入
            for i in range(0, len(texts), batch_size):
                batch = [
                    texts[i:i+batch_size],
                    embeddings[i:i+batch_size],
                    sources[i:i+batch_size],
                    data_types[i:i+batch_size],
                    cities[i:i+batch_size],
                    titles[i:i+batch_size],
                ]
                self._collection.insert(batch)

            return len(documents)

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return 0

    def _collection_exists(self) -> bool:
        """检查collection是否存在"""
        return self.config.collection_name in self._list_collections()

    def _create_collection(self):
        """创建collection"""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields=fields, description=f"桂林旅游数据")
        self._collection = Collection(name=self.config.collection_name, schema=schema)
        logger.info(f"创建Collection: {self.config.collection_name}")

    def count(self) -> int:
        """返回文档数量"""
        if not self._initialized:
            self.initialize()
        if self._collection:
            return self._collection.num_entities
        return 0

    def delete_collection(self, name: str) -> bool:
        """删除collection"""
        try:
            from pymilvus import Collection
            if name in self._list_collections():
                coll = Collection(name)
                coll.drop()
                logger.info(f"删除Collection: {name}")
                return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
        return False

    def list_collections(self) -> List[str]:
        """列出所有collection"""
        return self._list_collections()

    def close(self):
        """关闭连接"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
            self._initialized = False
        except:
            pass


class SimpleVectorAdapter(RAGPort):
    """
    简单内存向量适配器

    用于测试或数据量小的场景
    """

    def __init__(self):
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        for doc in documents:
            self._documents.append(doc)
            # 简化: 使用随机embedding
            import numpy as np
            self._embeddings.append(np.random.rand(1024).tolist())
        return len(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # 简化: 直接返回所有文档
        results = self._documents[:top_k]
        return [
            {**doc, "score": 0.9}
            for doc in results
        ]

    def count(self) -> int:
        return len(self._documents)

    def delete_collection(self, name: str) -> bool:
        self._documents.clear()
        self._embeddings.clear()
        return True

    def list_collections(self) -> List[str]:
        return ["simple"]


# =============================================================================
# 工厂函数
# =============================================================================

def create_rag_adapter(
    adapter_type: str = "milvus",
    **kwargs,
) -> RAGPort:
    """
    创建RAG适配器

    Args:
        adapter_type: 适配器类型 ("milvus", "simple")
        **kwargs: 传递给适配器的配置

    Returns:
        RAGPort接口实例
    """
    if adapter_type == "milvus":
        config = MilvusConfig(**kwargs)
        return MilvusAdapter(config)
    elif adapter_type == "simple":
        return SimpleVectorAdapter()
    else:
        raise ValueError(f"未知的RAG适配器类型: {adapter_type}")
