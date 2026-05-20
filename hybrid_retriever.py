"""
HybridRetriever - BM25 + 向量 + 重排 混合检索器
================================================

实现旅游领域RAG的混合检索链路:
1. BM25 稀疏检索 (关键词匹配)
2. 向量检索 (语义相似度)
3. RRF 融合 (倒数排序融合)
4. BGE-reranker 重排 (精排)

参考 RAG-Harness 的 AdvancedRetrieval 实现
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BM25 实现 (使用 rank_bm25)
# =============================================================================

try:
    from rank_bm25 import BM25Okapi
    from rank_bm25 import BM25Plus
    from rank_bm25 import BM25L
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    logger.warning("rank-bm25 未安装，将使用内置 BM25 实现")


# =============================================================================
# 配置
# =============================================================================

@dataclass
class HybridRetrievalConfig:
    """混合检索配置"""
    # BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_algorithm: str = "bm25"  # bm25 / bm25l / bm25plus
    
    # 检索数量
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 5
    
    # RRF 融合参数
    rrf_k: int = 60
    
    # 重排模型
    reranker_model: str = "/mnt/f/LLM/models/bge-reranker-large"
    reranker_device: str = "cpu"
    reranker_use_fp16: bool = False
    
    # 向量嵌入模型
    embedder_model: str = "/mnt/f/LLM/models/bge-small-zh-v1.5"
    embedder_dimension: int = 512
    embedder_device: str = "cpu"


class BM25Indexer:
    """
    BM25 索引器
    
    使用 rank_bm25 库或内置实现建立 BM25 索引
    """
    
    def __init__(
        self,
        documents: List[str],
        config: Optional[HybridRetrievalConfig] = None,
    ):
        """
        初始化 BM25 索引器
        
        Args:
            documents: 文档列表
            config: 配置
        """
        self.config = config or HybridRetrievalConfig()
        self.documents = documents
        self.doc_ids: List[str] = [f"doc_{i}" for i in range(len(documents))]
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """简单中英文分词"""
        import re
        # 转小写
        text = text.lower()
        # 去除标点，保留中文、英文、数字
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    def _build_index(self) -> None:
        """构建 BM25 索引"""
        if not self.documents:
            logger.warning("BM25 索引为空文档列表")
            return
        
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        if RANK_BM25_AVAILABLE:
            algo = self.config.bm25_algorithm.lower()
            if algo == "bm25l":
                self.bm25 = BM25L(tokenized_docs)
            elif algo == "bm25plus":
                self.bm25 = BM25Plus(tokenized_docs)
            else:
                self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"使用 rank_bm25 构建 {self.config.bm25_algorithm} 索引: {len(self.documents)} 个文档")
        else:
            # Fallback: 使用内置简单实现
            self.bm25 = SimpleBM25(tokenized_docs, self.config)
            logger.info(f"使用内置 BM25 索引: {len(self.documents)} 个文档")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        if not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        # 计算分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 按分数排序
        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_idx, score) in enumerate(doc_scores[:top_k], 1):
            results.append({
                "id": self.doc_ids[doc_idx],
                "doc_idx": doc_idx,
                "text": self.documents[doc_idx],
                "score": float(score),
                "rank": rank,
                "source": "bm25",
            })
        
        return results


class SimpleBM25:
    """内置简单 BM25 实现 (当 rank_bm25 不可用时使用)"""
    
    def __init__(self, tokenized_docs: List[List[str]], config: HybridRetrievalConfig):
        self.config = config
        self.tokenized_docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(doc) for doc in tokenized_docs) / self.N if self.N > 0 else 0
        
        # 计算文档频率
        from collections import Counter
        self.doc_freq = Counter()
        for doc in tokenized_docs:
            self.doc_freq.update(set(doc))
        
        # 计算 IDF
        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """计算每个文档的 BM25 分数"""
        scores = np.zeros(self.N)
        k1 = self.config.bm25_k1
        b = self.config.bm25_b
        
        for i, doc in enumerate(self.tokenized_docs):
            doc_len = len(doc)
            doc_tf = Counter(doc)
            
            score = 0.0
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1))
                score += idf * numerator / denominator
            
            scores[i] = score
        
        return scores


# =============================================================================
# 重排器
# =============================================================================

class Reranker:
    """
    BGE-reranker 重排器
    
    使用交叉编码器对检索结果进行重排序
    """
    
    def __init__(
        self,
        model_name: str = "/mnt/f/LLM/models/bge-reranker-large",
        device: str = "cpu",
        use_fp16: bool = False,
    ):
        """
        初始化重排器
        
        Args:
            model_name: 模型路径或 HuggingFace 模型名
            device: 设备 (cpu/cuda)
            use_fp16: 是否使用 FP16
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._is_loaded = False
    
    def _load_model(self) -> bool:
        """加载模型"""
        if self._is_loaded:
            return True
        
        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"加载重排模型: {self.model_name}, 设备: {self.device}")
            self._model = FlagReranker(
                self.model_name,
                device=self.device,
                use_fp16=self.use_fp16,
                cache_dir="./models",
            )
            self._is_loaded = True
            logger.info("重排模型加载成功")
            return True
        except Exception as e:
            logger.error(f"重排模型加载失败: {e}")
            self._is_loaded = False
            return False
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_n: 返回前 N 个结果
            
        Returns:
            重排后的结果列表
        """
        if not results:
            return []
        
        # 延迟加载模型
        if not self._is_loaded:
            if not self._load_model():
                logger.warning("重排模型加载失败，返回原始结果")
                return results[:top_n]
        
        # 准备 query-document 对
        pairs = []
        for r in results:
            doc_text = r.get("text", "")
            pairs.append((query, doc_text))
        
        try:
            # 计算相关性分数
            scores = self._model.compute_score(pairs)
            
            # 添加分数到结果
            for r, score in zip(results, scores):
                r["rerank_score"] = float(score)
            
            # 按分数排序
            reranked = sorted(
                results,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )
            
            # 重新设置 rank
            for i, r in enumerate(reranked):
                r["rerank_rank"] = i + 1
            
            logger.debug(f"重排完成: {len(results)} 个结果")
            return reranked[:top_n]
            
        except Exception as e:
            logger.error(f"重排失败: {e}")
            return results[:top_n]
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded


# =============================================================================
# RRF 融合
# =============================================================================

def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    倒数排序融合 (RRF)
    
    将多个检索结果列表融合为一个统一的排序列表
    
    RRF 公式: score(d) = sum(1 / (k + rank(d)))
    
    Args:
        result_lists: 多个检索结果列表
        k: RRF 参数 (典型值 60)
        
    Returns:
        融合后的结果列表
    """
    fused_scores: Dict[str, Dict[str, Any]] = {}
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, 1):
            doc_id = result.get("id")
            if doc_id is None:
                continue
            
            # RRF 分数
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    **result,
                    "rrf_score": rrf_score,
                    "sources": [result.get("source", "unknown")],
                }
            else:
                fused_scores[doc_id]["rrf_score"] += rrf_score
                # 合并 sources
                sources = set(fused_scores[doc_id]["sources"])
                sources.add(result.get("source", "unknown"))
                fused_scores[doc_id]["sources"] = list(sources)
    
    # 按 RRF 分数排序
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x.get("rrf_score", 0),
        reverse=True,
    )
    
    # 重新设置 rank
    for i, result in enumerate(sorted_results):
        result["final_rank"] = i + 1
    
    return sorted_results


# =============================================================================
# 混合检索器
# =============================================================================

class HybridRetriever:
    """
    BM25 + 向量 + 重排 混合检索器
    
    检索流程:
    1. BM25 检索 (稀疏检索，关键词匹配)
    2. 向量检索 (密向检索，语义相似度)
    3. RRF 融合 (结合两种检索结果)
    4. BGE-reranker 重排 (精排)
    
    使用方式:
        retriever = HybridRetriever(
            vector_store=milvus_store,
            embedder=bge_embedder,
            config=config,
        )
        results = retriever.retrieve("桂林景点推荐", top_k=5)
    """
    
    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        config: Optional[HybridRetrievalConfig] = None,
        bm25_documents: Optional[List[str]] = None,
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储 (MilvusVectorStore)
            embedder: 嵌入器 (BGEEmbedder)
            config: 配置
            bm25_documents: BM25 索引的文档列表
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or HybridRetrievalConfig()
        
        # 初始化 BM25 索引器
        self.bm25_indexer: Optional[BM25Indexer] = None
        if bm25_documents:
            self.set_bm25_documents(bm25_documents)
        
        # 初始化重排器
        self.reranker = Reranker(
            model_name=self.config.reranker_model,
            device=self.config.reranker_device,
            use_fp16=self.config.reranker_use_fp16,
        )
        
        logger.info(
            f"HybridRetriever 初始化完成: "
            f"vector_top_k={self.config.vector_top_k}, "
            f"bm25_top_k={self.config.bm25_top_k}, "
            f"rerank_top_k={self.config.rerank_top_k}"
        )
    
    def set_bm25_documents(self, documents: List[str]) -> None:
        """
        设置 BM25 索引的文档
        
        Args:
            documents: 文档列表 (需要与向量存储中的顺序一致)
        """
        self.bm25_indexer = BM25Indexer(documents, self.config)
        logger.info(f"BM25 文档已设置: {len(documents)} 个文档")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            
        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.final_top_k
        
        # Step 1: 向量检索
        vector_results = self._vector_search(query, self.config.vector_top_k)
        
        # Step 2: BM25 检索
        bm25_results = self._bm25_search(query, self.config.bm25_top_k)
        
        # Step 3: RRF 融合
        if vector_results and bm25_results:
            fused_results = reciprocal_rank_fusion(
                [vector_results, bm25_results],
                k=self.config.rrf_k,
            )
        elif vector_results:
            fused_results = vector_results
        elif bm25_results:
            fused_results = bm25_results
        else:
            logger.warning("向量检索和 BM25 检索均无结果")
            return []
        
        # Step 4: 重排
        if fused_results and self.reranker.is_loaded():
            reranked_results = self.reranker.rerank(
                query,
                fused_results,
                top_n=self.config.rerank_top_k,
            )
        else:
            # 无重排时直接截取
            reranked_results = fused_results[:self.config.rerank_top_k]
        
        return reranked_results[:top_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            query: 查询文本
            top_k: 检索数量
            
        Returns:
            向量检索结果
        """
        try:
            # 生成查询向量
            query_embedding = self.embedder.embed_single(query)
            
            # 搜索
            search_results = self.vector_store.search(
                query_vectors=[query_embedding],
                top_k=top_k,
            )
            
            results = []
            for rank, hit in enumerate(search_results[0].results, 1):
                results.append({
                    "id": hit.get("id", f"vec_{rank}"),
                    "text": hit.get("text", ""),
                    "score": hit.get("score", 0.0),
                    "rank": rank,
                    "source": "vector",
                    "metadata": hit.get("metadata", {}),
                })
            
            logger.debug(f"向量检索完成: {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 检索数量
            
        Returns:
            BM25 检索结果
        """
        if self.bm25_indexer is None:
            logger.warning("BM25 索引未初始化，跳过 BM25 检索")
            return []
        
        try:
            results = self.bm25_indexer.search(query, top_k=top_k)
            logger.debug(f"BM25 检索完成: {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 最终返回数量
            
        Returns:
            批量检索结果
        """
        return [self.retrieve(q, top_k) for q in queries]


# =============================================================================
# 工具函数
# =============================================================================

def create_hybrid_retriever(
    milvus_uri: str,
    collection_name: str,
    embedder_model: str = "/mnt/f/LLM/models/bge-small-zh-v1.5",
    reranker_model: str = "/mnt/f/LLM/models/bge-reranker-large",
    reranker_device: str = "cpu",
    **kwargs,
) -> Tuple[HybridRetriever, Any, Any]:
    """
    创建混合检索器 (便捷函数)
    
    Args:
        milvus_uri: Milvus 数据库路径
        collection_name: Collection 名称
        embedder_model: 嵌入模型路径
        reranker_model: 重排模型路径
        reranker_device: 重排设备
        **kwargs: 其他配置参数
        
    Returns:
        (HybridRetriever, vector_store, embedder)
    """
    from rag_harness.embedding import BGEEmbedder
    from rag_harness.storage import MilvusVectorStore
    
    # 初始化嵌入器
    embedder = BGEEmbedder(
        model_name=embedder_model,
        device=kwargs.get("embedder_device", "cpu"),
        batch_size=kwargs.get("batch_size", 32),
    )
    
    # 初始化向量存储
    vector_store = MilvusVectorStore(
        uri=milvus_uri,
        collection_name=collection_name,
        dimension=embedder.get_dimension(),
        metric_type=kwargs.get("metric_type", "COSINE"),
    )
    
    # 创建配置
    config = HybridRetrievalConfig(
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        **kwargs,
    )
    
    # 创建混合检索器
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        config=config,
    )
    
    return retriever, vector_store, embedder
