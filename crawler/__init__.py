"""
LvyouHarness Crawler 模块
=========================

桂林旅游数据爬虫 + 多爬虫编排器

主要组件:
- CrawlerCoordinator: 多爬虫编排器(asyncio并行)
- CtripCrawler: 携程桂林数据爬虫
- MeituanCrawler: 美团桂林数据爬虫  
- FirecrawlAdapter: Firecrawl通用网页爬取适配器
- GuilinCrawler: 桂林专属爬虫(整合上述所有)

使用方式:
    from lvyou_harness.crawler import GuilinCrawler, CrawlerConfig

    config = CrawlerConfig(city="桂林")
    crawler = GuilinCrawler(config)
    
    # 爬取所有数据
    documents = await crawler.crawl_all()
    
    # 存入向量库
    from lvyou_harness.crawler import save_to_vectorstore
    save_to_vectorstore(documents, collection="lvyou_guilin")
"""
from .base import (
    BaseCrawler,
    CrawlerTask,
    CrawlerResult,
    Document,
    Platform,
)
from .config import CrawlerConfig, GuilinScenicConfig
from .ctrip import CtripCrawler
from .meituan import MeituanCrawler
from .firecrawl_adapter import FirecrawlAdapter
from .coordinator import CrawlerCoordinator, save_to_vectorstore, save_to_json
from .guilin import GuilinCrawler

__all__ = [
    # 基础
    "BaseCrawler",
    "CrawlerTask", 
    "CrawlerResult",
    "Document",
    "Platform",
    "DataType",
    # 配置
    "CrawlerConfig",
    "GuilinScenicConfig",
    "CtripCrawlerConfig",
    "MeituanCrawlerConfig",
    "FirecrawlConfig",
    # 爬虫
    "CtripCrawler",
    "MeituanCrawler",
    "FirecrawlAdapter",
    "CrawlerCoordinator",
    "GuilinCrawler",
    # 工具
    "save_to_vectorstore",
    "save_to_json",
]
