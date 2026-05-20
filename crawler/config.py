"""
爬虫配置
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class CrawlerConfig:
    """爬虫统一配置"""
    city: str = "桂林"
    city_code: str = "guilin2"  # 携程城市代码

    # 爬虫开关
    enable_ctrip: bool = True
    enable_meituan: bool = True
    enable_firecrawl: bool = True

    # 并发控制
    max_concurrent: int = 3
    max_per_platform: int = 50  # 每个平台最多条目

    # 输出
    output_dir: Path = Path("/home/l2140/lvyou_harness/data")
    save_json: bool = True
    save_markdown: bool = False

    # Milvus向量库
    milvus_uri: str = "./milvus_rag.db"
    collection_name: str = "lvyou_guilin"
    embedding_model: str = "BAAI/bge-m3"

    # 重试
    max_retries: int = 3
    retry_delay: float = 1.0

    # 请求间隔(秒)
    request_delay: float = 0.5

    # User-Agent
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class GuilinScenicConfig:
    """桂林景点爬虫配置"""
    # 携程桂林景点URL
    ctrip_attraction_url: str = "https://you.ctrip.com/sight/guilin2.html"
    ctrip_hotel_url: str = "https://hotels.ctrip.com/hotel/guilin42/"
    ctrip_food_url: str = "https://you.ctrip.com/food/guilin2.html"
    ctrip_attraction_list_url: str = "https://you.ctrip.com/sight/guilin2/s0.html"

    # 美团桂林URL
    meituan_hotel_url: str = "https://hotel.meituan.com/guilin/"
    meituan_attraction_url: str = "https://www.meituan.com/guilin/changdi/"

    # 需要爬取的景区列表
    scenic_spots: List[str] = field(default_factory=lambda: [
        "漓江",
        "象山景区",
        "两江四湖",
        "独秀峰·王城",
        "靖江王府",
        "七星公园",
        "芦笛岩",
        "叠彩山",
        "伏波山",
        "阳朔西街",
        "遇龙河",
        "十里画廊",
        "兴坪古镇",
        "九马画山",
        "龙脊梯田",
        "猫儿山",
        "冠岩",
        "古东瀑布",
        "银子岩",
        "世外桃源",
    ])

    # 热门景区(优先爬取)
    top_spots: List[str] = field(default_factory=lambda: [
        "漓江",
        "象山景区", 
        "阳朔西街",
        "遇龙河",
        "十里画廊",
        "龙脊梯田",
    ])

    # 评分阈值(只保留>=这个分数的)
    min_rating: float = 4.0

    # 评论数量阈值
    min_reviews: int = 10


@dataclass
class CtripCrawlerConfig(CrawlerConfig):
    """携程爬虫配置"""
    city_code: str = "guilin2"
    base_url: str = "https://you.ctrip.com"

    # 携程API(备用)
    api_base: str = "https://m.ctrip.com"
    use_api: bool = False  # 优先用API(如果有)

    # 景区列表页
    attraction_list_pages: int = 3  # 爬取前N页


@dataclass
class MeituanCrawlerConfig(CrawlerConfig):
    """美团爬虫配置"""
    city_code: str = "guilin"
    base_url: str = "https://www.meituan.com"

    # 酒店列表页
    hotel_list_pages: int = 5
    # 景点列表页
    attraction_list_pages: int = 3


@dataclass
class FirecrawlConfig:
    """Firecrawl配置"""
    # 是否使用Firecrawl(需要API key)
    enabled: bool = True
    api_key: Optional[str] = None
    max_pages: int = 10

    # 默认爬取模式
    mode: str = "scrape"  # scrape | crawl | map
