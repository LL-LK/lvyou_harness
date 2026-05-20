"""
爬虫接口定义
"""
from typing import Protocol, List, Dict, Any, runtime_checkable


@runtime_checkable
class CrawlerPort(Protocol):
    """
    爬虫端口接口

    支持:
    - 携程
    - 美团
    - 飞猪
    - Firecrawl通用
    - 任何自定义爬虫
    """

    @property
    def platform(self) -> str:
        """平台名称"""
        ...

    def get_tasks(self) -> List[Dict[str, Any]]:
        """返回爬虫任务列表"""
        ...

    async def crawl_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个爬虫任务

        Args:
            task: 任务配置，包含 url, data_type 等

        Returns:
            爬取结果，包含 documents 列表
        """
        ...

    async def crawl_all(self) -> List[Dict[str, Any]]:
        """
        执行所有爬虫任务

        Returns:
            所有文档列表
        """
        ...
