"""
Firecrawl适配器
================

封装Firecrawl MCP工具，提供通用网页爬取能力

Firecrawl优势:
- JS渲染页面支持
- 爬取+结构化提取
- 搜索+爬取组合
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Any, Optional

from .base import BaseCrawler, CrawlerTask, CrawlerResult, Document, Platform, DataType
from .config import CrawlerConfig, FirecrawlConfig

logger = logging.getLogger(__name__)


class FirecrawlAdapter(BaseCrawler):
    """
    Firecrawl网页爬取适配器

    使用Firecrawl API爬取网页内容
    支持:
    - scrape: 爬取单个URL
    - crawl: 递归爬取整个网站
    - map: 发现网站所有URL
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        config: Optional[FirecrawlConfig] = None,
    ):
        super().__init__(Platform.FIRECRAWL, city="通用", max_concurrent=max_concurrent)
        self.cfg = config or FirecrawlConfig()

    def get_tasks(self) -> List[CrawlerTask]:
        """返回Firecrawl任务(外部调用时构造)"""
        return []

    async def crawl_task(self, task: CrawlerTask) -> CrawlerResult:
        """执行Firecrawl爬取"""
        start = asyncio.get_event_loop().time()

        result = CrawlerResult(
            task_id=task.id,
            platform=task.platform,
            data_type=task.data_type,
            success=False,
        )

        try:
            if self.cfg.mode == "scrape":
                docs = await self.scrape_urls([task.url])
            elif self.cfg.mode == "crawl":
                docs = await self.crawl(task.url, task.max_items)
            else:
                docs = []

            result.success = True
            result.documents = docs

        except Exception as e:
            logger.error(f"Firecrawl失败 {task.id}: {e}")
            result.error = str(e)

        result.duration_ms = (asyncio.get_event_loop().time() - start) * 1000
        return result

    async def scrape_urls(self, urls: List[str]) -> List[Document]:
        """
        爬取URL列表

        使用Firecrawl MCP工具

        Args:
            urls: URL列表

        Returns:
            Document列表
        """
        docs = []

        for url in urls:
            try:
                doc = await self.scrape_single(url)
                if doc:
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"Firecrawl爬取失败 {url}: {e}")
                continue

        return docs

    async def scrape_single(self, url: str) -> Optional[Document]:
        """
        爬取单个URL

        Returns:
            Document或None
        """
        try:
            # 调用Firecrawl MCP
            from mcp_tools import firecrawl_scrape

            response = await firecrawl_scrape(url=url)

            if not response:
                return None

            # 解析响应
            content = response.get("content", "")
            title = response.get("title", url)

            return Document(
                content=content,
                title=title,
                source="Firecrawl",
                url=url,
                data_type=DataType.ATTRACTION,  # 默认类型
                metadata={
                    "scraped_at": response.get("scraped_at"),
                    "provider": "firecrawl",
                },
            )

        except ImportError:
            # MCP工具不可用时使用HTTP客户端
            return await self._scrape_http(url)
        except Exception as e:
            logger.warning(f"Firecrawl scrape失败: {e}")
            return None

    async def _scrape_http(self, url: str) -> Optional[Document]:
        """使用HTTP客户端爬取(备用方案)"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                # 简单解析标题
                title = url.split("/")[-1]
                if "<title>" in response.text:
                    start = response.text.find("<title>") + 7
                    end = response.text.find("</title>")
                    if end > start:
                        title = response.text[start:end]

                return Document(
                    content=response.text[:5000],  # 限制长度
                    title=title,
                    source="HTTP",
                    url=url,
                    data_type=DataType.ATTRACTION,
                    metadata={"scraped_via": "httpx"},
                )

        except Exception as e:
            logger.warning(f"HTTP爬取失败 {url}: {e}")
            return None

    async def crawl(
        self,
        url: str,
        max_pages: int = 10,
    ) -> List[Document]:
        """
        递归爬取网站

        Args:
            url: 起始URL
            max_pages: 最大页面数

        Returns:
            Document列表
        """
        try:
            # 使用Firecrawl MCP crawl
            from mcp_tools import firecrawl_crawl

            response = await firecrawl_crawl(url=url, max_pages=max_pages)

            docs = []
            for page in response.get("pages", []):
                doc = Document(
                    content=page.get("content", ""),
                    title=page.get("title", url),
                    source="Firecrawl",
                    url=page.get("url", url),
                    data_type=DataType.ATTRACTION,
                    metadata={"scraped_at": page.get("scraped_at")},
                )
                docs.append(doc)

            return docs

        except ImportError:
            logger.warning("Firecrawl MCP不可用")
            return []
        except Exception as e:
            logger.warning(f"Firecrawl crawl失败: {e}")
            return []

    async def map_site(self, url: str) -> List[str]:
        """
        发现网站所有URL

        Returns:
            URL列表
        """
        try:
            from mcp_tools import firecrawl_map

            response = await firecrawl_map(url=url)
            return response.get("urls", [])

        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"Firecrawl map失败: {e}")
            return []


# =============================================================================
# 便捷函数
# =============================================================================

async def scrape_url(url: str) -> Optional[Document]:
    """快速爬取单个URL"""
    adapter = FirecrawlAdapter()
    return await adapter.scrape_single(url)


async def scrape_multiple(urls: List[str]) -> List[Document]:
    """快速爬取多个URL"""
    adapter = FirecrawlAdapter()
    return await adapter.scrape_urls(urls)
