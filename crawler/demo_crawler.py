"""
桂林旅游数据爬虫演示
====================

演示如何使用爬虫模块:
1. 爬取桂林旅游数据
2. 保存到Milvus向量库
3. 验证数据

使用方式:
    python -m lvyou_harness.crawler.demo_crawler
"""
import asyncio
import logging
import sys

# 添加路径
sys.path.insert(0, "/home/l2140")

from lvyou_harness.crawler import (
    GuilinCrawler,
    CtripCrawler,
    MeituanCrawler,
    CrawlerConfig,
    save_to_vectorstore,
    save_to_json,
    Document,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_guilin_crawler():
    """演示桂林爬虫"""
    print("\n" + "=" * 60)
    print("演示1: 桂林专属爬虫(整合所有平台)")
    print("=" * 60)

    config = CrawlerConfig(city="桂林")
    crawler = GuilinCrawler(config)

    # 爬取所有数据
    docs = await crawler.crawl_all()

    print(f"\n爬取完成: {len(docs)} 文档")

    # 按类型统计
    by_type = {}
    for doc in docs:
        dt = doc.data_type.value
        by_type[dt] = by_type.get(dt, 0) + 1

    print("\n按类型统计:")
    for dtype, count in by_type.items():
        print(f"  {dtype}: {count}")

    # 按来源统计
    by_source = {}
    for doc in docs:
        src = doc.source
        by_source[src] = by_source.get(src, 0) + 1

    print("\n按来源统计:")
    for source, count in by_source.items():
        print(f"  {source}: {count}")

    # 保存到JSON
    json_path = "/home/l2140/lvyou_harness/data/guilin_data.json"
    result = save_to_json(docs, json_path)
    print(f"\n保存JSON: {result}")

    # 保存到向量库
    vector_result = save_to_vectorstore(docs, collection="lvyou_guilin")
    print(f"保存向量库: {vector_result}")

    return docs


async def demo_ctrip_crawler():
    """演示携程爬虫"""
    print("\n" + "=" * 60)
    print("演示2: 携程爬虫")
    print("=" * 60)

    crawler = CtripCrawler(city="桂林")
    docs = await crawler.crawl_all()

    print(f"爬取完成: {len(docs)} 文档")
    for doc in docs[:3]:
        print(f"\n- {doc.title}")
        print(f"  来源: {doc.source}")
        print(f"  类型: {doc.data_type.value}")
        print(f"  摘要: {doc.content[:100]}...")

    return docs


async def demo_meituan_crawler():
    """演示美团爬虫"""
    print("\n" + "=" * 60)
    print("演示3: 美团爬虫")
    print("=" * 60)

    crawler = MeituanCrawler(city="桂林")
    docs = await crawler.crawl_all()

    print(f"爬取完成: {len(docs)} 文档")
    for doc in docs[:3]:
        print(f"\n- {doc.title}")
        print(f"  来源: {doc.source}")
        print(f"  类型: {doc.data_type.value}")

    return docs


async def demo_vectorstore():
    """演示向量库操作"""
    print("\n" + "=" * 60)
    print("演示4: 向量库验证")
    print("=" * 60)

    try:
        from pymilvus import connections, Collection
        from pymilvus import utility

        connections.connect(uri="./milvus_rag.db")

        # 列出所有collection
        cols = utility.list_collections()
        print(f"Milvus Collections: {cols}")

        # 检查lvyou_guilin
        if "lvyou_guilin" in cols:
            collection = Collection("lvyou_guilin")
            collection.load()
            count = collection.num_entities
            print(f"lvyou_guilin: {count} 条数据")
        else:
            print("lvyou_guilin 集合不存在")

        connections.disconnect("default")

    except Exception as e:
        print(f"Milvus连接失败: {e}")


async def main():
    """主函数"""
    print("桂林旅游数据爬虫演示")
    print("=" * 60)

    # 演示1: 桂林专属爬虫
    await demo_guilin_crawler()

    # 演示2: 携程爬虫
    await demo_ctrip_crawler()

    # 演示3: 美团爬虫
    await demo_meituan_crawler()

    # 演示4: 向量库验证
    await demo_vectorstore()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
