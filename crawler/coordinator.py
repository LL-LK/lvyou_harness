"""
多爬虫编排器
=============

并行/顺序执行多个爬虫，收集数据后存入向量库

核心流程:
    1. CrawlerCoordinator.run_parallel() - 多平台并行
    2. 统一产出 Document 列表
    3. save_to_vectorstore() - 存入Milvus
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseCrawler, CrawlerTask, CrawlerResult, Document, Platform, DataType
from .config import CrawlerConfig, GuilinScenicConfig
from .ctrip import CtripCrawler
from .meituan import MeituanCrawler
from .firecrawl_adapter import FirecrawlAdapter

logger = logging.getLogger(__name__)


class CrawlerCoordinator:
    """
    多爬虫编排器

    协调多个爬虫并行/顺序执行，统一输出格式

    使用方式:
        coordinator = CrawlerCoordinator(config)
        
        # 并行执行所有爬虫
        results = await coordinator.run_all()
        
        # 或指定平台
        results = await coordinator.run_platforms(["ctrip", "firecrawl"])
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        scenic_config: Optional[GuilinScenicConfig] = None,
    ):
        self.config = config or CrawlerConfig()
        self.scenic_config = scenic_config or GuilinScenicConfig()
        self._crawlers: Dict[str, BaseCrawler] = {}
        self._results: List[CrawlerResult] = []

    def _get_crawler(self, platform: str) -> BaseCrawler:
        """获取或创建爬虫实例"""
        if platform in self._crawlers:
            return self._crawlers[platform]

        if platform == "ctrip":
            from .ctrip import CtripCrawler
            crawler = CtripCrawler(
                city=self.config.city,
                max_concurrent=self.config.max_concurrent,
            )
        elif platform == "meituan":
            from .meituan import MeituanCrawler
            crawler = MeituanCrawler(
                city=self.config.city,
                max_concurrent=self.config.max_concurrent,
            )
        elif platform == "firecrawl":
            from .firecrawl_adapter import FirecrawlAdapter
            crawler = FirecrawlAdapter(
                max_concurrent=self.config.max_concurrent,
            )
        else:
            raise ValueError(f"未知平台: {platform}")

        self._crawlers[platform] = crawler
        return crawler

    async def run_all(self) -> List[Document]:
        """
        并行执行所有已启用的爬虫

        Returns:
            所有爬虫产出的文档列表
        """
        platforms = []
        if self.config.enable_ctrip:
            platforms.append("ctrip")
        if self.config.enable_meituan:
            platforms.append("meituan")
        if self.config.enable_firecrawl:
            platforms.append("firecrawl")

        return await self.run_platforms(platforms)

    async def run_platforms(
        self,
        platforms: List[str],
    ) -> List[Document]:
        """
        执行指定平台的爬虫

        Args:
            platforms: 平台列表 ["ctrip", "meituan", "firecrawl"]

        Returns:
            所有文档列表
        """
        start = time.time()
        logger.info(f"开始执行爬虫: {platforms}")

        # 并行执行所有平台
        tasks = []
        for p in platforms:
            crawler = self._get_crawler(p)
            tasks.append(crawler.crawl_all())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集所有文档
        all_docs = []
        for i, r in enumerate(results):
            platform = platforms[i]
            if isinstance(r, Exception):
                logger.error(f"{platform} 爬虫异常: {r}")
                continue
            all_docs.extend(r)
            logger.info(f"{platform} 完成: {len(r)} 文档")

        duration = (time.time() - start) * 1000
        logger.info(f"全部完成: {len(all_docs)} 文档, 耗时 {duration:.0f}ms")

        return all_docs

    async def run_ctrip(self) -> List[Document]:
        """仅执行携程爬虫"""
        return await self.run_platforms(["ctrip"])

    async def run_meituan(self) -> List[Document]:
        """仅执行美团爬虫"""
        return await self.run_platforms(["meituan"])

    def get_stats(self) -> Dict[str, Any]:
        """获取爬虫统计"""
        stats = {
            "total_documents": 0,
            "by_platform": {},
            "by_data_type": {},
        }
        for result in self._results:
            p = result.platform.value
            dt = result.data_type.value
            stats["by_platform"][p] = stats["by_platform"].get(p, 0) + len(result.documents)
            stats["by_data_type"][dt] = stats["by_data_type"].get(dt, 0) + len(result.documents)
            stats["total_documents"] += len(result.documents)
        return stats


# =============================================================================
# 向量库写入
# =============================================================================

def save_to_vectorstore(
    documents: List[Document],
    collection: str = "lvyou_guilin",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    将爬取的文档存入Milvus向量库

    Args:
        documents: 文档列表
        collection: 集合名称
        batch_size: 批处理大小

    Returns:
        写入统计
    """
    if not documents:
        return {"status": "skipped", "reason": "no documents"}

    logger.info(f"开始存入向量库: {len(documents)} 文档 → {collection}")

    try:
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        import numpy as np

        # 连接Milvus
        connections.connect(uri="./milvus_rag.db")

        # 创建或获取collection
        dim = 1024  # BGE-m3维度
        if utility.has_collection(collection):
            coll = Collection(collection)
            coll.drop()  # 删除旧collection重新创建
            logger.info(f"删除旧collection: {collection}")

        # 创建schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields=fields, description=f"桂林旅游数据 - {collection}")

        # 创建collection
        coll = Collection(name=collection, schema=schema)
        logger.info(f"创建collection: {collection}")

        # 插入数据
        texts = []
        embeddings = []
        sources = []
        data_types = []
        cities = []
        titles = []

        for doc in documents:
            texts.append(doc.to_markdown()[:1000])  # 截断
            sources.append(doc.source)
            data_types.append(doc.data_type.value)
            cities.append(doc.metadata.get("city", "桂林"))
            titles.append(doc.title[:100])
            # 随机embedding作为占位(实际应该用BGE编码)
            embeddings.append(np.random.rand(dim).tolist())

        # 批量插入
        for i in range(0, len(texts), batch_size):
            batch = [
                texts[i:i+batch_size],
                embeddings[i:i+batch_size],
                sources[i:i+batch_size],
                data_types[i:i+batch_size],
                cities[i:i+batch_size],
                titles[i:i+batch_size],
            ]
            coll.insert(batch)
            logger.info(f"  写入进度: {min(i+batch_size, len(texts))}/{len(texts)}")

        # 创建索引
        coll.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "COSINE"}
        )
        coll.load()

        count = coll.num_entities
        connections.disconnect("default")

        logger.info(f"向量库写入完成: {count} 条数据")

        return {
            "status": "success",
            "collection": collection,
            "total": len(documents),
            "inserted": count,
        }

    except ImportError as e:
        logger.warning(f"pymilvus未安装: {e}")
        return {"status": "import_error", "error": str(e)}
    except Exception as e:
        logger.error(f"写入向量库失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def save_to_json(
    documents: List[Document],
    output_path: str,
) -> Dict[str, Any]:
    """保存到JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [doc.to_dict() for doc in documents]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "path": str(output_path),
        "count": len(documents),
    }
