"""
LvyouRAGAdapter - 旅游领域RAG适配器
=======================================

封装RAG-Harness，提供旅游领域专用的:
- 景点知识检索
- 问答生成
- 知识库管理
- BM25 + 向量 + 重排 混合检索
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# RAG客户端封装
# =============================================================================

class LvyouRAGAdapter:
    """
    旅游领域RAG适配器

    封装RAG-Harness的UnifiedHarnessRAGAdapter，提供:
    - 景点信息检索
    - 问答生成
    - 历史记录管理
    - BM25 + 向量 + 重排 混合检索
    """

    def __init__(
        self,
        collection_name: str = "lvyou_guilin",
        top_k: int = 5,
        rerank: bool = True,
        use_hybrid: bool = True,
    ):
        self.collection_name = collection_name
        self.top_k = top_k
        self.rerank = rerank
        self.use_hybrid = use_hybrid
        self._client = None
        self._initialized = False
        self._hybrid_retriever = None

    def initialize(self) -> bool:
        """
        初始化RAG客户端

        Returns:
            是否初始化成功
        """
        if self._initialized:
            return True

        try:
            import sys
            sys.path.insert(0, "/home/l2140/RAG-Harness")

            from rag_harness.unified_adapter import UnifiedHarnessRAGAdapter, HarnessConfig

            # 根据collection选择配置
            if "guilin" in self.collection_name.lower():
                cfg = HarnessConfig.for_lvyou() if hasattr(HarnessConfig, "for_lvyou") else None
            else:
                cfg = None

            if cfg is None:
                # 使用默认配置
                cfg = self._create_default_config()

            self._client = UnifiedHarnessRAGAdapter(cfg)
            self._initialized = True
            logger.info(f"RAG初始化成功: {self.collection_name}")
            return True

        except ImportError as e:
            logger.warning(f"RAG-Harness未安装: {e}")
            return False
        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            return False

    def _create_default_config(self):
        """创建默认配置"""
        try:
            from rag_harness.unified_adapter import HarnessConfig
            return HarnessConfig()
        except ImportError:
            return None

    def _init_hybrid_retriever(self) -> bool:
        """
        初始化混合检索器

        Returns:
            是否初始化成功
        """
        if not self.use_hybrid:
            return False

        if self._hybrid_retriever is not None:
            return True

        try:
            from .hybrid_retriever import HybridRetriever, HybridRetrievalConfig
            from rag_harness.embedding import BGEEmbedder
            from rag_harness.storage import MilvusVectorStore

            # 创建配置
            config = HybridRetrievalConfig(
                reranker_model="/mnt/f/LLM/models/bge-reranker-large",
                reranker_device="cpu",
                embedder_model=self.config.bge_model if hasattr(self, 'config') else "/mnt/f/LLM/models/bge-small-zh-v1.5",
                vector_top_k=20,
                bm25_top_k=20,
                rerank_top_k=10,
                final_top_k=self.top_k,
            )

            # 获取 Milvus 配置
            if hasattr(self, 'config') and self.config:
                milvus_uri = self.config.milvus_uri
                collection = self.config.milvus_collection
                bge_model = self.config.bge_model
                device = self.config.bge_device
            else:
                milvus_uri = str(Path("/mnt/g/embeddings/lvyou/milvus") / "lvyou.db")
                collection = "lvyou_travel_kb"
                bge_model = "/mnt/f/LLM/models/bge-small-zh-v1.5"
                device = "cpu"

            # 初始化嵌入器
            embedder = BGEEmbedder(
                model_name=bge_model,
                device=device,
                batch_size=32,
            )

            # 初始化向量存储
            vector_store = MilvusVectorStore(
                uri=milvus_uri,
                collection_name=collection,
                dimension=embedder.get_dimension(),
                metric_type="COSINE",
            )

            # 获取所有文档用于 BM25 索引
            try:
                # 尝试使用 query 方法获取所有文档
                if hasattr(vector_store, 'query'):
                    # Milvus 的 query 方法可以用空 filter 获取所有
                    all_docs = vector_store.query(expr="", limit=10000)
                    bm25_docs = [doc.get("text", "") for doc in all_docs]
                elif hasattr(vector_store, 'get_all_documents'):
                    all_docs = vector_store.get_all_documents()
                    bm25_docs = [doc.get("text", "") for doc in all_docs]
                else:
                    bm25_docs = []
            except Exception as e:
                logger.warning(f"获取文档用于 BM25 索引失败: {e}")
                bm25_docs = []

            # 创建混合检索器
            self._hybrid_retriever = HybridRetriever(
                vector_store=vector_store,
                embedder=embedder,
                config=config,
                bm25_documents=bm25_docs,
            )

            logger.info(f"混合检索器初始化成功，BM25 文档数: {len(bm25_docs)}")
            return True

        except ImportError as e:
            logger.warning(f"混合检索器依赖未安装: {e}")
            return False
        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}")
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, Any] = None,
        use_hybrid: bool = None,
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据筛选条件
            use_hybrid: 是否使用混合检索 (True=BM25+向量+重排, False=仅向量)

        Returns:
            [{id, content, title, score, metadata}, ...]
        """
        if not self._initialized:
            self.initialize()

        k = top_k or self.top_k

        # 决定使用混合检索还是普通检索
        _use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        # 尝试使用混合检索
        if _use_hybrid and self._hybrid_retriever is None:
            self._init_hybrid_retriever()

        if _use_hybrid and self._hybrid_retriever is not None:
            try:
                results = self._hybrid_retriever.retrieve(query, top_k=k)

                # 应用过滤
                if filters:
                    results = self._apply_filters(results, filters)

                # 格式化结果
                formatted = []
                for r in results:
                    formatted.append({
                        "id": r.get("id", ""),
                        "content": r.get("text", ""),
                        "text": r.get("text", ""),
                        "title": r.get("metadata", {}).get("source", "unknown"),
                        "score": r.get("rerank_score", r.get("score", 0)),
                        "metadata": r.get("metadata", {}),
                    })

                return formatted

            except Exception as e:
                logger.error(f"混合检索失败，回退到普通检索: {e}")

        # 普通检索 (回退方案)
        if not self._client:
            return []

        try:
            results = self._client.retrieve(query, top_k=k)

            # 应用过滤
            if filters:
                results = self._apply_filters(results, filters)

            return results

        except Exception as e:
            logger.error(f"RAG检索失败: {e}")
            return []

    def ask(
        self,
        question: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        问答

        Args:
            question: 问题
            context: 额外上下文

        Returns:
            {answer, sources, confidence}
        """
        if not self._initialized:
            self.initialize()

        if not self._client:
            return {
                "answer": "RAG系统未初始化，无法回答。",
                "sources": [],
                "confidence": 0.0,
            }

        try:
            # 检索相关文档
            docs = self.retrieve(question, top_k=5)

            if not docs:
                return {
                    "answer": "未找到相关信息，请换个问题或扩展搜索范围。",
                    "sources": [],
                    "confidence": 0.0,
                }

            # 构建上下文
            context_text = "\n".join(
                f"[{i + 1}] {d.get('content', d.get('text', ''))[:300]}"
                for i, d in enumerate(docs[:3])
            )

            # 生成回答 (如果client支持直接QA)
            if hasattr(self._client, "ask"):
                result = self._client.ask(question)
                return result

            # 否则手动构造
            answer = self._generate_answer(question, docs)
            return answer

        except Exception as e:
            logger.error(f"RAG问答失败: {e}")
            return {
                "answer": f"处理问题时出错: {str(e)}",
                "sources": [],
                "confidence": 0.0,
            }

    def _generate_answer(
        self,
        question: str,
        docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """生成回答"""
        # 简单实现：直接使用检索到的文档内容
        top_doc = docs[0] if docs else {}

        answer_text = top_doc.get("content", top_doc.get("text", ""))

        return {
            "answer": answer_text,
            "sources": [
                {
                    "title": d.get("title", "unknown"),
                    "content": d.get("content", "")[:100],
                    "score": d.get("score", 0),
                }
                for d in docs[:3]
            ],
            "confidence": top_doc.get("score", 0),
        }

    def _apply_filters(
        self,
        results: List[Dict],
        filters: Dict[str, Any],
    ) -> List[Dict]:
        """应用元数据过滤"""
        filtered = []
        for doc in results:
            metadata = doc.get("metadata", {})
            match = all(
                metadata.get(k) == v for k, v in filters.items()
            )
            if match:
                filtered.append(doc)
        return filtered

    # -------------------------------------------------------------------------
    # 知识库管理
    # -------------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """列出可用知识库"""
        if not self._client:
            return []

        try:
            if hasattr(self._client, "list_collections"):
                return self._client.list_collections()
            return [self.collection_name]
        except Exception as e:
            logger.error(f"列出知识库失败: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        if not self._client:
            return {}

        try:
            if hasattr(self._client, "get_stats"):
                return self._client.get_stats(self.collection_name)
            return {"collection": self.collection_name, "status": "unknown"}
        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            return {}


# =============================================================================
# 单例实例
# =============================================================================

_global_adapter: Optional[LvyouRAGAdapter] = None


def get_global_adapter() -> LvyouRAGAdapter:
    """获取全局RAG适配器实例"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = LvyouRAGAdapter()
    return _global_adapter


def initialize_adapter(
    collection: str = "lvyou_guilin",
    top_k: int = 5,
) -> bool:
    """初始化全局RAG适配器"""
    adapter = get_global_adapter()
    adapter.collection_name = collection
    adapter.top_k = top_k
    return adapter.initialize()


# =============================================================================
# 便捷函数
# =============================================================================

def retrieve_scenics(
    query: str,
    top_k: int = 5,
    region: str = None,
) -> List[Dict[str, Any]]:
    """
    检索景点信息

    Args:
        query: 查询文本
        top_k: 返回数量
        region: 地区筛选

    Returns:
        景点信息列表
    """
    adapter = get_global_adapter()
    filters = {"region": region} if region else None
    return adapter.retrieve(query, top_k=top_k, filters=filters)


def ask_scenic(
    question: str,
    scenic_name: str = None,
) -> str:
    """
    询问景点相关问题

    Args:
        question: 问题
        scenic_name: 指定景点 (可选)

    Returns:
        回答文本
    """
    adapter = get_global_adapter()

    if scenic_name:
        question = f"关于{scenic_name}: {question}"

    result = adapter.ask(question)
    return result.get("answer", "")


def search_food(
    query: str,
    region: str = None,
) -> List[Dict[str, Any]]:
    """
    检索美食推荐

    Args:
        query: 查询文本
        region: 地区

    Returns:
        美食信息列表
    """
    adapter = get_global_adapter()
    full_query = f"{region or ''} {query} 美食推荐"
    return adapter.retrieve(full_query, top_k=5)


def search_accommodation(
    query: str,
    region: str = None,
    budget: str = None,
) -> List[Dict[str, Any]]:
    """
    检索住宿推荐

    Args:
        query: 查询文本
        region: 地区
        budget: 预算级别

    Returns:
        住宿信息列表
    """
    adapter = get_global_adapter()
    parts = []
    if region:
        parts.append(region)
    if budget:
        parts.append(f"{budget}价位")
    parts.append(query)
    full_query = " ".join(parts)
    return adapter.retrieve(full_query, top_k=5)
