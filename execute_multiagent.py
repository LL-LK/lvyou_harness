"""
多智能体多任务旅游数据采集执行器
=====================================
并行爬取多个平台 → BGE向量嵌入 → Milvus存储 → MCP服务

执行方式:
    cd /home/l2140/lvyou_harness
    python execute_multiagent.py
"""
import asyncio
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("multiagent")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/l2140")

# 导入BGE嵌入器
try:
    from embedding.bge_embedder import BGEEmbedder
    HAS_BGE = True
except ImportError as e:
    logger.warning(f"BGE导入失败: {e}")
    HAS_BGE = False

# 导入爬虫
from crawler.coordinator import CrawlerCoordinator, save_to_vectorstore, save_to_json
from crawler.config import CrawlerConfig, GuilinScenicConfig
from crawler.base import Document, DataType, Platform

# =============================================================================
# 多智能体并行爬取任务
# =============================================================================

class TravelDataCollector:
    """旅游数据采集器 - 协调多个平台并行爬取"""
    
    def __init__(self):
        self.config = CrawlerConfig(city="桂林")
        self.scenic_config = GuilinScenicConfig()
        self.coordinator = CrawlerCoordinator(self.config, self.scenic_config)
        self.bge = None
        if HAS_BGE:
            model_path = "/mnt/f/LLM/models/bge-small-zh-v1.5"
            from embedding.bge_embedder import EmbedderConfig, BGEEmbedder as _BGE
            config = EmbedderConfig(model_path=model_path)
            self.bge = _BGE(config)
            logger.info(f"BGE嵌入器初始化完成: {model_path}")
    
    async def crawl_all_platforms(self) -> List[Document]:
        """并行爬取所有平台"""
        logger.info("=" * 60)
        logger.info("开始多平台并行爬取...")
        logger.info("=" * 60)
        
        start = time.time()
        
        # 并行执行所有已启用的爬虫
        platforms = []
        if self.config.enable_ctrip:
            platforms.append("ctrip")
        if self.config.enable_meituan:
            platforms.append("meituan")
        if self.config.enable_firecrawl:
            platforms.append("firecrawl")
        
        logger.info(f"将执行以下平台: {platforms}")
        
        docs = await self.coordinator.run_platforms(platforms)
        
        elapsed = time.time() - start
        logger.info(f"爬取完成: {len(docs)} 文档, 耗时 {elapsed:.1f}s")
        
        return docs
    
    def embed_documents(self, documents: List[Document]) -> List[Dict]:
        """使用BGE对文档进行向量嵌入"""
        if not self.bge:
            logger.warning("BGE不可用，使用随机嵌入")
            return [self._random_embed(doc) for doc in documents]
        
        logger.info(f"开始BGE嵌入: {len(documents)} 文档")
        
        texts = [doc.to_markdown()[:500] for doc in documents]  # 截断
        
        # 批量嵌入
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self.bge.embed(batch)
            embeddings.extend(batch_emb)
            logger.info(f"  嵌入进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        # 组合结果
        results = []
        for doc, emb in zip(documents, embeddings):
            results.append({
                "document": doc,
                "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            })
        
        return results
    
    def _random_embed(self, doc: Document) -> Dict:
        """生成随机嵌入作为占位"""
        import numpy as np
        return {
            "document": doc,
            "embedding": np.random.rand(512).tolist()  # BGE-small是512维
        }
    
    def save_to_milvus(self, embedded_docs: List[Dict], collection: str = "lvyou_guilin") -> Dict:
        """保存到Milvus向量库"""
        if not embedded_docs:
            return {"status": "skipped", "reason": "no documents"}
        
        logger.info(f"开始存入Milvus: {len(embedded_docs)} 文档 → {collection}")
        
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            import numpy as np
            
            # 连接
            db_path = "/home/l2140/milvus_rag.db"
            connections.connect(uri=db_path)
            
            dim = len(embedded_docs[0]["embedding"])
            logger.info(f"向量维度: {dim}")
            
            # 删除旧collection
            if utility.has_collection(collection):
                coll = Collection(collection)
                coll.drop()
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
            
            # 准备数据
            texts, embeddings, sources, data_types, cities, titles = [], [], [], [], [], []
            
            for item in embedded_docs:
                doc = item["document"]
                texts.append(doc.to_markdown()[:2000])  # 截断到合理长度
                embeddings.append(item["embedding"])
                sources.append(doc.source)
                data_types.append(doc.data_type.value)
                cities.append(doc.metadata.get("city", "桂林"))
                titles.append(doc.title[:100] if doc.title else "")
            
            # 批量插入
            batch_size = 50
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
            
            logger.info(f"Milvus写入完成: {count} 条数据")
            
            return {
                "status": "success",
                "collection": collection,
                "dimension": dim,
                "total": len(embedded_docs),
                "inserted": count,
            }
            
        except ImportError as e:
            logger.error(f"pymilvus未安装: {e}")
            return {"status": "import_error", "error": str(e)}
        except Exception as e:
            logger.error(f"写入Milvus失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def save_to_json(self, documents: List[Document], output_path: str) -> Dict:
        """保存到JSON文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for doc in documents:
            d = asdict(doc) if hasattr(doc, '__dataclass_fields__') else doc.__dict__
            data.append(d)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "path": str(output_path),
            "count": len(documents),
        }


# =============================================================================
# 爬虫平台配置
# =============================================================================

async def run_ctrip_crawler() -> Dict:
    """携程爬虫任务"""
    logger.info("=" * 60)
    logger.info("任务1: 携程爬虫")
    logger.info("=" * 60)
    
    from crawler.ctrip import CtripCrawler
    from crawler.config import CrawlerConfig
    
    config = CrawlerConfig(city="桂林")
    crawler = CtripCrawler(city=config.city, max_concurrent=config.max_concurrent)
    
    start = time.time()
    docs = await crawler.crawl_all()
    elapsed = time.time() - start
    
    return {
        "platform": "ctrip",
        "count": len(docs),
        "elapsed": elapsed,
        "documents": docs
    }


async def run_meituan_crawler() -> Dict:
    """美团爬虫任务"""
    logger.info("=" * 60)
    logger.info("任务2: 美团爬虫")
    logger.info("=" * 60)
    
    from crawler.meituan import MeituanCrawler
    from crawler.config import CrawlerConfig
    
    config = CrawlerConfig(city="桂林")
    crawler = MeituanCrawler(city=config.city, max_concurrent=config.max_concurrent)
    
    start = time.time()
    docs = await crawler.crawl_all()
    elapsed = time.time() - start
    
    return {
        "platform": "meituan", 
        "count": len(docs),
        "elapsed": elapsed,
        "documents": docs
    }


async def run_firecrawl_crawler() -> Dict:
    """Firecrawl爬虫任务"""
    logger.info("=" * 60)
    logger.info("任务3: Firecrawl爬虫")
    logger.info("=" * 60)
    
    from crawler.firecrawl_adapter import FirecrawlAdapter
    
    crawler = FirecrawlAdapter(max_concurrent=3)
    
    start = time.time()
    docs = await crawler.crawl_all()
    elapsed = time.time() - start
    
    return {
        "platform": "firecrawl",
        "count": len(docs),
        "elapsed": elapsed,
        "documents": docs
    }


# =============================================================================
# 主执行流程
# =============================================================================

async def main():
    """主执行流程"""
    logger.info("=" * 70)
    logger.info("多智能体多任务旅游数据采集系统")
    logger.info("=" * 70)
    
    # Step 1: 初始化采集器
    collector = TravelDataCollector()
    
    # Step 2: 并行爬取所有平台
    logger.info("\n[Step 1] 并行爬取多平台...")
    docs = await collector.crawl_all_platforms()
    logger.info(f"爬取结果: {len(docs)} 文档")
    
    # Step 3: BGE向量嵌入
    logger.info("\n[Step 2] BGE向量嵌入...")
    embedded = collector.embed_documents(docs)
    logger.info(f"嵌入完成: {len(embedded)} 文档")
    
    # Step 4: 保存到Milvus
    logger.info("\n[Step 3] 保存到Milvus...")
    result = collector.save_to_milvus(embedded, "lvyou_guilin")
    logger.info(f"Milvus保存结果: {result}")
    
    # Step 5: 保存到JSON备份
    logger.info("\n[Step 4] 保存到JSON备份...")
    json_result = collector.save_to_json(docs, "/home/l2140/lvyou_harness/data/multiagent_crawl.json")
    logger.info(f"JSON保存结果: {json_result}")
    
    # 最终统计
    logger.info("\n" + "=" * 70)
    logger.info("执行完成!")
    logger.info(f"总文档数: {len(docs)}")
    logger.info(f"Milvus状态: {result}")
    logger.info("=" * 70)
    
    return {
        "total_documents": len(docs),
        "milvus_result": result,
        "json_result": json_result,
    }


async def parallel_crawl():
    """并行执行多个爬虫任务"""
    logger.info("=" * 70)
    logger.info("多智能体并行爬取模式")
    logger.info("=" * 70)
    
    start = time.time()
    
    # 并行执行三个爬虫任务
    results = await asyncio.gather(
        run_ctrip_crawler(),
        run_meituan_crawler(),
        run_firecrawl_crawler(),
        return_exceptions=True
    )
    
    elapsed = time.time() - start
    
    # 统计结果
    total_docs = 0
    for i, r in enumerate(results):
        platform_names = ["ctrip", "meituan", "firecrawl"]
        if isinstance(r, Exception):
            logger.error(f"{platform_names[i]} 爬虫异常: {r}")
        else:
            logger.info(f"{r['platform']}: {r['count']} 文档, 耗时 {r['elapsed']:.1f}s")
            total_docs += r["count"]
    
    logger.info(f"\n并行爬取完成: {total_docs} 文档, 总耗时 {elapsed:.1f}s")
    
    # 收集所有文档
    all_docs = []
    for r in results:
        if not isinstance(r, Exception) and "documents" in r:
            all_docs.extend(r["documents"])
    
    return {
        "total_documents": total_docs,
        "elapsed": elapsed,
        "documents": all_docs,
        "platforms": {
            "ctrip": results[0]["count"] if not isinstance(results[0], Exception) else 0,
            "meituan": results[1]["count"] if not isinstance(results[1], Exception) else 0,
            "firecrawl": results[2]["count"] if not isinstance(results[2], Exception) else 0,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="多智能体旅游数据采集")
    parser.add_argument("--mode", choices=["all", "parallel", "sequential"], default="parallel",
                       help="执行模式: all=完整流程, parallel=并行爬取, sequential=顺序爬取")
    args = parser.parse_args()
    
    if args.mode == "parallel":
        result = asyncio.run(parallel_crawl())
    else:
        result = asyncio.run(main())
    
    print("\n最终结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
