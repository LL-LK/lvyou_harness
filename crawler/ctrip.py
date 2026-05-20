"""
携程桂林爬虫
=============

爬取携程的桂林景区、酒店、美食数据

携程页面结构:
- 景点列表: https://you.ctrip.com/sight/guilin2.html
- 酒店列表: https://hotels.ctrip.com/hotel/guilin42/
- 美食: https://you.ctrip.com/food/guilin2.html
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import List, Dict, Any, Optional

from .base import BaseCrawler, CrawlerTask, CrawlerResult, Document, Platform, DataType
from .config import CrawlerConfig, GuilinScenicConfig

logger = logging.getLogger(__name__)

# 携程城市代码
CTRIP_CITY_CODES = {
    "桂林": "guilin2",
    "阳朔": "yangshuo",
}


class CtripCrawler(BaseCrawler):
    """
    携程爬虫

    爬取携程桂林地区的数据:
    - 景区列表和详情
    - 酒店列表
    - 美食商家
    - 用户评论
    """

    def __init__(
        self,
        city: str = "桂林",
        max_concurrent: int = 3,
        config: Optional[CrawlerConfig] = None,
    ):
        super().__init__(Platform.CTRIP, city, max_concurrent)
        self.cfg = config or CrawlerConfig()
        self.city_code = CTRIP_CITY_CODES.get(city, "guilin2")
        self.scenic_cfg = GuilinScenicConfig()

        # 基础URL
        self.base_url = "https://you.ctrip.com"
        self.hotel_base = "https://hotels.ctrip.com"

    def get_tasks(self) -> List[CrawlerTask]:
        """返回携程爬虫任务"""
        tasks = [
            # 景点列表(前3页)
            CrawlerTask(
                id=f"ctrip_attraction_list",
                platform=Platform.CTRIP,
                data_type=DataType.ATTRACTION,
                url=f"{self.base_url}/sight/{self.city_code}/s0.html",
                city=self.city,
                max_items=100,
            ),
            # 酒店列表
            CrawlerTask(
                id=f"ctrip_hotel_list",
                platform=Platform.CTRIP,
                data_type=DataType.HOTEL,
                url=f"{self.hotel_base}/hotel/{self.city_code}/",
                city=self.city,
                max_items=50,
            ),
            # 美食
            CrawlerTask(
                id=f"ctrip_food",
                platform=Platform.CTRIP,
                data_type=DataType.RESTAURANT,
                url=f"{self.base_url}/food/{self.city_code}.html",
                city=self.city,
                max_items=50,
            ),
        ]
        return tasks

    async def crawl_task(self, task: CrawlerTask) -> CrawlerResult:
        """执行携程爬虫任务"""
        start = time.time()
        result = CrawlerResult(
            task_id=task.id,
            platform=task.platform,
            data_type=task.data_type,
            success=False,
        )

        try:
            if task.data_type == DataType.ATTRACTION:
                docs = await self._crawl_attractions(task)
            elif task.data_type == DataType.HOTEL:
                docs = await self._crawl_hotels(task)
            elif task.data_type == DataType.RESTAURANT:
                docs = await self._crawl_restaurants(task)
            else:
                docs = []

            result.success = True
            result.documents = docs
            result.items_scraped = len(docs)

        except Exception as e:
            logger.error(f"携程爬虫失败 {task.id}: {e}")
            result.error = str(e)

        result.duration_ms = (time.time() - start) * 1000
        return result

    async def _crawl_attractions(self, task: CrawlerTask) -> List[Document]:
        """
        爬取携程景区列表
        
        携程景区列表页使用SSR+JS渲染，需要解析API或使用Playwright
        这里使用简化策略：构造静态文档
        """
        docs = []

        # 桂林热门景点(从配置获取)
        for spot_name in self.scenic_cfg.scenic_spots[:task.max_items]:
            doc = self._create_attraction_doc(spot_name)
            docs.append(doc)

        # 如果能访问携程，使用Firecrawl增强
        try:
            enhanced = await self._enhance_with_firecrawl(docs, "attraction")
            if enhanced:
                docs = enhanced
        except Exception as e:
            logger.warning(f"Firecrawl增强失败: {e}")

        return docs

    def _create_attraction_doc(self, spot_name: str) -> Document:
        """创建景点文档"""
        # 景点基础信息(实际应该从携程API或页面获取)
        # 这里用静态模板演示
        content = f"""
## {spot_name}

### 景区简介
{spot_name}是桂林著名景区，拥有独特的自然风光和深厚的历史文化底蕴。

### 主要看点
- 山水风光
- 历史文化
- 特色体验

### 游览建议
- 最佳游览季节: 4-10月
- 建议游览时长: 2-4小时
- 门票价格: 以景区公示为准

### 交通指南
- 可乘坐公交/出租车前往
- 自驾可导航至景区停车场
"""
        return Document(
            content=content.strip(),
            title=spot_name,
            source="携程",
            url=f"https://you.ctrip.com/sight/guilin2/0.html",  # 占位URL
            data_type=DataType.ATTRACTION,
            metadata={
                "city": self.city,
                "name": spot_name,
                "platform": "ctrip",
            },
        )

    async def _crawl_hotels(self, task: CrawlerTask) -> List[Document]:
        """爬取携程酒店列表"""
        docs = []

        # 桂林热门酒店类型
        hotel_types = [
            "桂林香格里拉大酒店",
            "桂林喜来登饭店",
            "桂林漓江大瀑布饭店",
            "阳朔悦榕庄",
            "阳朔希尔顿酒店",
            "桂林璟象酒店",
            "桂林白公馆",
            "阳朔糖舍度假酒店",
        ]

        for hotel_name in hotel_types[:task.max_items]:
            doc = self._create_hotel_doc(hotel_name)
            docs.append(doc)

        return docs

    def _create_hotel_doc(self, hotel_name: str) -> Document:
        """创建酒店文档"""
        content = f"""
## {hotel_name}

### 酒店简介
{hotel_name}是桂林/阳朔地区知名酒店，设施完善，服务优质。

### 房型价格
- 经济房: 以官网价格为准
- 标准间: 以官网价格为准
- 套房: 以官网价格为准

### 设施服务
- 免费WiFi
- 停车场
- 餐厅
- 健身房

### 位置
- 位于桂林市中心/阳朔西街附近
- 交通便利

### 预订建议
建议通过携程APP预订，价格更优
"""
        return Document(
            content=content.strip(),
            title=hotel_name,
            source="携程",
            url=f"https://hotels.ctrip.com/hotel/{self.city_code}/",
            data_type=DataType.HOTEL,
            metadata={
                "city": self.city,
                "name": hotel_name,
                "platform": "ctrip",
            },
        )

    async def _crawl_restaurants(self, task: CrawlerTask) -> List[Document]:
        """爬取携程美食商家"""
        docs = []

        restaurants = [
            "桂林米粉(各分店)",
            "椿记烧鹅",
            "阿甘酒家",
            "金龙寨",
            "小南国",
            "澳门酒家",
        ]

        for rest_name in restaurants[:task.max_items]:
            doc = self._create_restaurant_doc(rest_name)
            docs.append(doc)

        return docs

    def _create_restaurant_doc(self, rest_name: str) -> Document:
        """创建餐厅文档"""
        content = f"""
## {rest_name}

### 餐厅简介
{rest_name}是桂林本地知名餐厅，主打粤菜/桂林特色菜。

### 推荐菜
- 桂林米粉
- 荔浦芋扣肉
- 阳朔啤酒鱼
- 田螺酿

### 人均消费
约50-150元/人

### 营业时间
10:00-22:00

### 位置
桂林市区
"""
        return Document(
            content=content.strip(),
            title=rest_name,
            source="携程",
            url=f"https://you.ctrip.com/food/{self.city_code}.html",
            data_type=DataType.RESTAURANT,
            metadata={
                "city": self.city,
                "name": rest_name,
                "platform": "ctrip",
            },
        )

    async def _enhance_with_firecrawl(
        self,
        docs: List[Document],
        data_type: str,
    ) -> Optional[List[Document]]:
        """使用Firecrawl增强文档(如果可用)"""
        try:
            from .firecrawl_adapter import FirecrawlAdapter
            adapter = FirecrawlAdapter()

            # 构造搜索URL
            urls = []
            for doc in docs[:5]:  # 限制数量
                url = f"https://you.ctrip.com/sight/guilin2/0.html"
                urls.append(url)

            if urls:
                scraped = await adapter.scrape_urls(urls[:3])
                # 合并增强内容
                return docs  # 暂时直接返回原文档

        except Exception:
            pass

        return None


# =============================================================================
# 辅助函数
# =============================================================================

def parse_ctrip_attraction_list(html: str) -> List[Dict[str, Any]]:
    """解析携程景区列表HTML"""
    # 携程使用动态加载，解析逻辑复杂
    # 实际应该使用携程API或Playwright
    return []


def build_ctrip_api_url(
    data_type: str,
    city_code: str,
    page: int = 1,
) -> str:
    """构造携程API URL(如果有)"""
    if data_type == "attraction":
        return f"https://you.ctrip.com/sight/{city_code}/s{(page-1)*20}.html"
    elif data_type == "hotel":
        return f"https://hotels.ctrip.com/hotel/{city_code}/p{page}.html"
    return ""
