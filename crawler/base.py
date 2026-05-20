"""
爬虫基类和基础数据结构
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """支持的爬虫平台"""
    CTRIP = "ctrip"        # 携程
    MEITUAN = "meituan"    # 美团
    FLIGGY = "fliggy"      # 飞猪
    FIRECRAWL = "firecrawl" # Firecrawl通用
    QIONGYOU = "qiongyou"  # 穷游
    BAIDU = "baidu"        # 百度


class DataType(str, Enum):
    """数据类型"""
    ATTRACTION = "attraction"      # 景区
    HOTEL = "hotel"                # 酒店
    RESTAURANT = "restaurant"       # 美食
    REVIEW = "review"              # 评论
    GUIDE = "guide"                # 攻略


@dataclass
class Document:
    """
    统一文档格式
    爬虫产出 → RAG Pipeline 输入
    """
    content: str           # 文本内容(Markdown)
    title: str            # 标题
    source: str           # 来源平台
    url: str              # 原始URL
    data_type: DataType   # 数据类型
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata包含: city, district, price, rating, tags等

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "data_type": self.data_type.value,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """转为Markdown格式"""
        meta = self.metadata
        lines = [
            f"# {self.title}",
            f"",
            f"**来源**: {self.source}",
            f"**类型**: {self.data_type.value}",
            f"**URL**: {self.url}",
        ]
        if meta.get("city"):
            lines.append(f"**城市**: {meta.get('city')}")
        if meta.get("district"):
            lines.append(f"**区域**: {meta.get('district')}")
        if meta.get("price"):
            lines.append(f"**价格**: {meta.get('price')}元")
        if meta.get("rating"):
            lines.append(f"**评分**: {meta.get('rating')}")
        if meta.get("tags"):
            lines.append(f"**标签**: {', '.join(meta.get('tags', []))}")
        lines.extend(["", self.content])
        return "\n".join(lines)


@dataclass
class CrawlerTask:
    """爬虫任务"""
    id: str
    platform: Platform
    data_type: DataType
    url: str
    params: Dict[str, Any] = field(default_factory=dict)  # 平台特定参数
    city: str = "桂林"
    max_items: int = 100  # 最大抓取数量

    def __str__(self):
        return f"CrawlerTask({self.platform.value}/{self.data_type.value}: {self.url})"


@dataclass
class CrawlerResult:
    """爬虫执行结果"""
    task_id: str
    platform: Platform
    data_type: DataType
    success: bool
    documents: List[Document] = field(default_factory=list)
    error: Optional[str] = None
    items_scraped: int = 0
    duration_ms: float = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "platform": self.platform.value,
            "data_type": self.data_type.value,
            "success": self.success,
            "documents": len(self.documents),
            "items_scraped": self.items_scraped,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class BaseCrawler(ABC):
    """
    爬虫基类

    所有平台爬虫需实现:
    - get_tasks(): 返回该平台的任务列表
    - crawl_task(task): 执行单个任务
    """

    def __init__(
        self,
        platform: Platform,
        city: str = "桂林",
        max_concurrent: int = 3,
    ):
        self.platform = platform
        self.city = city
        self.max_concurrent = max_concurrent
        self._session = None

    @abstractmethod
    def get_tasks(self) -> List[CrawlerTask]:
        """返回该平台的爬虫任务列表"""
        pass

    @abstractmethod
    async def crawl_task(self, task: CrawlerTask) -> CrawlerResult:
        """执行单个爬虫任务"""
        pass

    async def crawl_all(self) -> List[Document]:
        """并行爬取所有任务"""
        import time
        start = time.time()

        tasks = self.get_tasks()
        logger.info(f"[{self.platform.value}] 开始爬取 {len(tasks)} 个任务")

        # 并发控制
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def crawl_with_semaphore(task: CrawlerTask) -> CrawlerResult:
            async with semaphore:
                return await self.crawl_task(task)

        # 并行执行
        results = await asyncio.gather(
            *[crawl_with_semaphore(t) for t in tasks],
            return_exceptions=True,
        )

        # 收集文档
        all_docs = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"任务异常: {r}")
                continue
            if r.success and r.documents:
                all_docs.extend(r.documents)
                logger.info(f"  ✓ {r.task_id}: {len(r.documents)} 文档")

        duration = (time.time() - start) * 1000
        logger.info(f"[{self.platform.value}] 完成: {len(all_docs)} 文档, 耗时 {duration:.0f}ms")

        return all_docs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """关闭资源"""
        if self._session:
            await self._session.close()
