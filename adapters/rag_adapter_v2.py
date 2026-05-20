"""
RAG适配器 V2
============

集成BGE-m3真实嵌入向量的RAG适配器

支持:
1. Milvus向量库 + BGE嵌入
2. 本地向量存储
3. 简易内存模式
4. BM25+向量混合检索
5. BGE-reranker重排
"""
from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..interfaces.rag import RAGPort
from ..embedding import create_embedder, BGEEmbedder

logger = logging.getLogger(__name__)

# BM25和重排依赖（可选）
try:
    from rank_bm25 import BM25Plus
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25未安装，BM25检索将不可用")

try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False
    logger.warning("sentence_transformers未安装，重排功能将不可用")


@dataclass
class MilvusRAGConfig:
    """Milvus RAG配置"""
    uri: str = os.environ.get("LVYOU_DB", "/home/l2140/milvus_rag.db")
    collection_name: str = "lvyou_guilin"
    embedder_path: str = "/mnt/f/LLM/models/bge-small-zh-v1.5"
    embedding_dim: int = 512
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    nlist: int = 128
    # BM25配置
    use_bm25: bool = True
    bm25_top_k: int = 20
    # 重排配置
    use_rerank: bool = True
    rerank_model_path: str = os.environ.get("RERANK_MODEL_PATH", "/mnt/f/LLM/models/bge-reranker-large")
    rerank_top_k: int = 3


class MilvusRAGAdapter(RAGPort):
    """
    Milvus RAG适配器 V2

    集成BGE嵌入和Milvus向量库
    """

    def __init__(self, config: Optional[MilvusRAGConfig] = None):
        self.config = config or MilvusRAGConfig()
        self._client = None
        self._collection = None
        self._embedder: Optional[BGEEmbedder] = None
        self._initialized = False
        # BM25相关
        self._bm25: Optional["BM25Plus"] = None
        self._bm25_texts: List[str] = []
        self._bm25_ids: List[int] = []
        # 重排相关
        self._reranker: Optional["CrossEncoder"] = None
        self._rerank_initialized = False

    def initialize(self) -> bool:
        """初始化Milvus和Embedding模型"""
        if self._initialized:
            return True

        try:
            # 1. 初始化embedding模型
            logger.info("初始化BGE嵌入模型...")
            self._embedder = create_embedder(
                model_path=self.config.embedder_path,
                dimension=self.config.embedding_dim,
            )
            if not self._embedder.initialize():
                logger.error("BGE模型初始化失败")
                return False

            # 2. 连接Milvus
            logger.info(f"连接Milvus: {self.config.uri}")
            from pymilvus import connections, Collection

            connections.connect(uri=self.config.uri, timeout=30)
            logger.info("Milvus连接成功")

            # 3. 加载或创建collection
            self._load_or_create_collection()

            # 4. 初始化BM25索引
            self._init_bm25()

            # 5. 初始化重排模型
            self._init_reranker()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_bm25(self):
        """初始化BM25索引"""
        if not self.config.use_bm25 or not BM25_AVAILABLE:
            logger.info("BM25检索已禁用")
            return

        try:
            if not self._collection or self._collection.num_entities == 0:
                logger.info("Collection为空，跳过BM25初始化")
                return

            logger.info("初始化BM25索引...")
            # 从Milvus获取所有文本用于BM25
            from pymilvus import Collection
            results = self._collection.query(
                expr="id >= 0",
                output_fields=["id", "text"],
                limit=10000,
            )

            if not results:
                logger.info("无文本数据用于BM25索引")
                return

            # 构建BM25索引
            self._bm25_texts = [r["text"][:2000] for r in results]  # 限制长度
            self._bm25_ids = [r["id"] for r in results]

            # 使用jieba分词（如果可用）或简单空格分词
            try:
                import jieba
                tokenized_corpus = [list(jieba.cut(text)) for text in self._bm25_texts]
            except ImportError:
                logger.warning("jieba未安装，使用空格分词")
                tokenized_corpus = [text.split() for text in self._bm25_texts]

            self._bm25 = BM25Plus(tokenized_corpus)
            logger.info(f"BM25索引构建完成，{len(self._bm25_texts)}条文档")

        except Exception as e:
            logger.warning(f"BM25初始化失败: {e}")
            self._bm25 = None

    def _init_reranker(self):
        """初始化重排模型"""
        if not self.config.use_rerank or not RERANK_AVAILABLE:
            logger.info("重排功能已禁用")
            return

        try:
            import os
            model_path = self.config.rerank_model_path
            if not os.path.exists(model_path):
                logger.warning(f"重排模型不存在: {model_path}，跳过初始化")
                return

            logger.info(f"初始化重排模型: {model_path}")
            self._reranker = CrossEncoder(model_path)
            self._rerank_initialized = True
            logger.info("重排模型初始化完成")

        except Exception as e:
            logger.warning(f"重排模型初始化失败: {e}")
            self._reranker = None

    def _load_or_create_collection(self):
        """加载或创建collection"""
        from pymilvus import Collection, utility

        if utility.has_collection(self.config.collection_name):
            logger.info(f"加载已有Collection: {self.config.collection_name}")
            self._collection = Collection(self.config.collection_name)
            self._collection.load()
        else:
            logger.info(f"创建新Collection: {self.config.collection_name}")
            self._create_collection()

    def _create_collection(self):
        """创建collection"""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields=fields, description="桂林旅游数据")
        self._collection = Collection(name=self.config.collection_name, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {"nlist": self.config.nlist},
        }
        self._collection.create_index(field_name="embedding", index_params=index_params)
        self._collection.load()
        logger.info("Collection创建并索引完成")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """添加文档（带真实embedding）"""
        if not self._initialized:
            self.initialize()

        if not documents:
            return 0

        try:
            from pymilvus import Collection

            texts = []
            embeddings = []
            sources = []
            data_types = []
            cities = []
            titles = []

            # 提取文本
            for doc in documents:
                texts.append(doc.get("content", "")[:5000])  # 限制长度
                sources.append(doc.get("source", "unknown"))
                data_types.append(doc.get("data_type", "attraction"))
                cities.append(doc.get("city", "桂林"))
                titles.append(doc.get("title", "")[:200])

            # 生成embedding
            logger.info(f"生成{len(texts)}个文本的embedding...")
            embeddings = self._embedder.embed(texts, normalize=True)
            logger.info("Embedding生成完成")

            # 分批插入
            total_inserted = 0
            for i in range(0, len(texts), batch_size):
                batch_data = [
                    texts[i:i+batch_size],
                    embeddings[i:i+batch_size],
                    sources[i:i+batch_size],
                    data_types[i:i+batch_size],
                    cities[i:i+batch_size],
                    titles[i:i+batch_size],
                ]
                self._collection.insert(batch_data)
                total_inserted += len(texts[i:i+batch_size])

            # 刷新
            self._collection.flush()
            logger.info(f"成功插入{total_inserted}条文档")

            return total_inserted

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """混合检索：BM25 + 向量 + 重排"""
        if not self._initialized:
            self.initialize()

        try:
            from pymilvus import Collection

            # ========== 阶段1: BM25初筛 ==========
            bm25_results = []
            if self._bm25 and BM25_AVAILABLE:
                try:
                    import jieba
                    query_tokens = list(jieba.cut(query))
                except ImportError:
                    query_tokens = query.split()

                bm25_scores = self._bm25.get_scores(query_tokens)
                bm25_top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:self.config.bm25_top_k]

                for idx in bm25_top_indices:
                    bm25_results.append({
                        "id": self._bm25_ids[idx],
                        "text": self._bm25_texts[idx],
                        "bm25_score": bm25_scores[idx]
                    })
                logger.info(f"BM25初筛: {len(bm25_results)}条")

            # ========== 阶段2: 向量检索 ==========
            query_embedding = self._embedder.embed_single(query, normalize=True)
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"nprobe": 10},
            }

            # 扩大检索范围以合并BM25结果
            vector_limit = max(top_k * 4, self.config.bm25_top_k)
            vector_results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=vector_limit,
                output_fields=["id", "text", "source", "data_type", "city", "title"],
            )

            vector_docs = []
            for hits in vector_results:
                for hit in hits:
                    vector_docs.append({
                        "id": hit.id,
                        "text": hit.entity.get("text", ""),
                        "source": hit.entity.get("source", ""),
                        "data_type": hit.entity.get("data_type", ""),
                        "city": hit.entity.get("city", ""),
                        "title": hit.entity.get("title", ""),
                        "vector_score": hit.distance
                    })
            logger.info(f"向量检索: {len(vector_docs)}条")

            # ========== 阶段3: 合并去重 ==========
            merged_results = {}
            for doc in vector_docs:
                doc_id = doc["id"]
                if doc_id not in merged_results:
                    merged_results[doc_id] = doc

            for doc in bm25_results:
                doc_id = doc["id"]
                if doc_id in merged_results:
                    merged_results[doc_id]["bm25_score"] = doc["bm25_score"]
                else:
                    doc["vector_score"] = 0
                    merged_results[doc_id] = doc

            candidates = list(merged_results.values())
            logger.info(f"合并去重后: {len(candidates)}条候选")

            # ========== 阶段4: 重排 (如果可用) ==========
            if self._reranker and RERANK_AVAILABLE and len(candidates) > top_k:
                logger.info("使用重排模型重排...")
                # 准备重排输入
                pairs = [[query, doc["text"]] for doc in candidates]
                rerank_scores = self._reranker.predict(pairs)

                # 添加重排分数并排序
                for i, doc in enumerate(candidates):
                    doc["rerank_score"] = float(rerank_scores[i])

                # 按重排分数排序
                candidates.sort(key=lambda d: d["rerank_score"], reverse=True)
                final_results = candidates[:top_k]

                logger.info(f"重排后: {len(final_results)}条")
            else:
                # 无重排时，使用向量+BM25综合分数
                for doc in candidates:
                    # 归一化分数
                    vector_s = doc.get("vector_score", 0)
                    bm25_s = doc.get("bm25_score", 0)
                    # 综合分数 = 0.7*向量 + 0.3*BM25
                    doc["score"] = 0.7 * vector_s + 0.3 * bm25_s if bm25_s > 0 else vector_s

                candidates.sort(key=lambda d: d.get("score", 0), reverse=True)
                final_results = candidates[:top_k]

            # 格式化输出
            docs = []
            for doc in final_results:
                docs.append({
                    "id": doc.get("id"),
                    "content": doc.get("text", ""),
                    "source": doc.get("source", ""),
                    "data_type": doc.get("data_type", ""),
                    "city": doc.get("city", ""),
                    "title": doc.get("title", ""),
                    "score": doc.get("score", doc.get("vector_score", 0)),
                    "metadata": {
                        "source": doc.get("source", ""),
                        "data_type": doc.get("data_type", ""),
                        "city": doc.get("city", ""),
                    },
                })

            logger.info(f"最终返回{len(docs)}条相关文档")
            return docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """单独调用重排接口"""
        if not self._reranker or not RERANK_AVAILABLE:
            return documents[:top_k]

        try:
            pairs = [[query, doc.get("content", doc.get("text", ""))] for doc in documents]
            scores = self._reranker.predict(pairs)

            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])

            documents.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
            return documents[:top_k]

        except Exception as e:
            logger.error(f"重排失败: {e}")
            return documents[:top_k]

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
            from pymilvus import Collection, utility
            if utility.has_collection(name):
                coll = Collection(name)
                coll.drop()
                logger.info(f"删除Collection: {name}")
            # 重置状态，强制重新初始化
            self._collection = None
            self._initialized = False
            return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
        return False

    def list_collections(self) -> List[str]:
        """列出所有collection"""
        try:
            from pymilvus import utility
            return utility.list_collections()
        except:
            return []

    def close(self):
        """关闭连接"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
            if self._embedder:
                self._embedder.close()
            self._initialized = False
        except:
            pass


class SimpleRAGAdapter(RAGPort):
    """
    简单RAG适配器（内存模式）

    用于测试或小数据量场景
    """

    def __init__(self, dimension: int = 512):
        self._dimension = dimension
        self._documents: List[Dict[str, Any]] = []
        self._embedder = create_embedder(dimension=dimension)
        self._embeddings: List[List[float]] = []
        self._embedder.initialize()

    def initialize(self) -> bool:
        return True

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        for doc in documents:
            self._documents.append(doc)
            text = doc.get("content", "")[:5000]
            emb = self._embedder.embed_single(text, normalize=True)
            self._embeddings.append(emb)
        return len(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._documents:
            return []

        # Query embedding
        query_emb = self._embedder.embed_single(query, normalize=True)

        # 计算相似度
        scores = []
        for emb in self._embeddings:
            score = sum(q * e for q, e in zip(query_emb, emb))
            scores.append(score)

        # 排序取top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {**self._documents[i], "score": scores[i]}
            for i in top_indices
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
        config = MilvusRAGConfig(**kwargs)
        return MilvusRAGAdapter(config)
    elif adapter_type == "simple":
        dimension = kwargs.get("dimension", 512)
        return SimpleRAGAdapter(dimension=dimension)
    else:
        raise ValueError(f"未知的RAG适配器类型: {adapter_type}")
