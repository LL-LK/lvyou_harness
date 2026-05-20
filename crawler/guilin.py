"""
桂林专属爬虫
============

桂林旅游数据爬虫 - 整合所有平台数据

桂林重点数据:
1. 景区(20+): 漓江, 象山, 两江四湖, 阳朔西街, 遇龙河, 龙脊梯田等
2. 酒店: 桂林市区+阳朔
3. 美食: 桂林米粉, 啤酒鱼, 荔浦芋扣肉等
4. 交通: 两江机场, 桂林站, 阳朔站
5. 购物: 土特产, 桂林三宝

使用方式:
    from lvyou_harness.crawler import GuilinCrawler

    crawler = GuilinCrawler()
    
    # 爬取所有数据
    docs = await crawler.crawl_all()
    
    # 保存到向量库
    from lvyou_harness.crawler import save_to_vectorstore
    result = save_to_vectorstore(docs, collection="lvyou_guilin")
"""
from __future__ import annotations

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import Document, DataType
from .config import CrawlerConfig, GuilinScenicConfig
from .coordinator import CrawlerCoordinator, save_to_vectorstore, save_to_json
from .ctrip import CtripCrawler
from .meituan import MeituanCrawler
from .firecrawl_adapter import FirecrawlAdapter

logger = logging.getLogger(__name__)


class GuilinCrawler:
    """
    桂林旅游数据爬虫

    整合携程、美团等平台，爬取桂林旅游全量数据
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        scenic_config: Optional[GuilinScenicConfig] = None,
    ):
        self.config = config or CrawlerConfig()
        self.config.city = "桂林"  # 强制桂林
        self.scenic_config = scenic_config or GuilinScenicConfig()
        self.coordinator = CrawlerCoordinator(self.config, self.scenic_config)

    async def crawl_all(self) -> List[Document]:
        """
        爬取桂林所有旅游数据

        Returns:
            所有Document列表
        """
        logger.info("开始爬取桂林旅游数据...")

        # 并行执行所有爬虫
        docs = await self.coordinator.run_all()

        # 添加桂林专属补充数据
        supplementary = self._get_supplementary_data()
        docs.extend(supplementary)

        logger.info(f"桂林数据爬取完成: {len(docs)} 文档")
        return docs

    async def crawl_attractions(self) -> List[Document]:
        """仅爬取景区数据"""
        ctrip = CtripCrawler(city="桂林")
        return await ctrip.crawl_all()

    async def crawl_hotels(self) -> List[Document]:
        """仅爬取酒店数据"""
        ctrip = CtripCrawler(city="桂林")
        meituan = MeituanCrawler(city="桂林")

        ctrip_docs = await ctrip.crawl_all()
        meituan_docs = await meituan.crawl_all()

        return ctrip_docs + meituan_docs

    async def crawl_restaurants(self) -> List[Document]:
        """仅爬取美食数据"""
        ctrip = CtripCrawler(city="桂林")
        meituan = MeituanCrawler(city="桂林")

        ctrip_docs = await ctrip.crawl_all()
        meituan_docs = await meituan.crawl_all()

        return ctrip_docs + meituan_docs

    def _get_supplementary_data(self) -> List[Document]:
        """
        获取桂林专属补充数据

        这些是标准化、结构化的桂林特色信息
        """
        docs = []

        # 桂林米粉专题
        docs.append(Document(
            content="""
## 桂林米粉

### 简介
桂林米粉是桂林最著名的特色小吃，分为卤菜粉和汤粉两种。

### 推荐店铺
- 石记米粉(桂林市): 30年历史老店
- 崇善米粉(桂林市): 连锁品牌
- 日头火米粉(桂林市): 24小时营业
- 老东江米粉(桂林市): 当地人常去

### 价格
- 2两: 4-5元
- 3两: 5-6元

### 特色吃法
1. 先干拌卤菜粉
2. 喝完汤
3. 加酸豆角、辣椒
""".strip(),
            title="桂林米粉",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "桂林", "tags": ["美食", "小吃", "特色"]},
        ))

        # 漓江漂流
        docs.append(Document(
            content="""
## 漓江漂流

### 简介
漓江漂流是桂林旅游必玩项目，分为杨堤-九马画山、兴坪-九马画山等航段。

### 漂流航段
1. 杨堤-九马画山: 全程约4小时
2. 兴坪-九马画山: 精华段，约1.5小时
3. 冠岩-兴坪: 全程约6小时

### 价格
- 杨堤-兴坪: 约150-200元/人(漓江漂流公司官方价)
- 兴坪游船: 约80-120元/人

### 游览建议
- 建议早上去，光线好
- 丰水期(4-10月)体验最佳
- 记得带防晒和雨具

### 交通
- 从桂林市区乘船或旅游巴士
- 阳朔出发更方便
""".strip(),
            title="漓江漂流",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "桂林", "tags": ["景点", "漂流", "漓江"]},
        ))

        # 阳朔西街
        docs.append(Document(
            content="""
## 阳朔西街

### 简介
阳朔西街是桂林最繁华的步行街，全长约800米，保存完好的明清风格建筑。

### 特色
- 洋人街: 外国游客众多
- 酒吧一条街
- 小吃一条街
- 手工艺品店

### 必吃美食
-啤酒鱼
- 田螺酿
- 漓江虾
- 桂花糕

### 购物
- 桂林三花酒
- 桂林辣椒酱
- 荔浦芋头
- 阳朔绣球

### 住宿推荐
- 西街周边民宿
- 阳朔悦榕庄(高端)
- 阳朔希尔顿(五星)

### 最佳游览时间
- 傍晚-晚上
- 避开十一黄金周
""".strip(),
            title="阳朔西街",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "阳朔", "tags": ["景点", "购物", "美食", "夜生活"]},
        ))

        # 龙脊梯田
        docs.append(Document(
            content="""
## 龙脊梯田

### 简介
龙脊梯田位于龙胜各族自治县，距桂林市区约80公里，是壮族、瑶族先民用智慧创造的奇迹。

### 三大寨
1. 平安寨: 开发最早，设施完善
2. 金坑大寨: 规模最大，景色壮观
3. 古壮寨: 原始风貌，游客较少

### 最佳观赏期
- 5月灌水期: 波光粼粼
- 6月底-7月中: 绿油油的稻田
- 9月底-10月中: 金黄色稻谷

### 门票
- 平安寨: 80元/人
- 金坑大寨: 80元/人
- 套票(含多个寨子): 100-120元/人

### 游览建议
- 建议住一晚，看日出
- 带足现金(山里信号差)
- 穿舒适的运动鞋
- 防晒防雨

### 交通
- 桂林市区乘坐班车到龙胜
- 或参加一日游/包车
""".strip(),
            title="龙脊梯田",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "龙胜", "tags": ["景点", "梯田", "民俗"]},
        ))

        # 遇龙河漂流
        docs.append(Document(
            content="""
## 遇龙河漂流

### 简介
遇龙河是漓江支流，漂流项目被誉为"小漓江"，水质清澈，两岸青山如黛。

### 漂流航段
1. 金龙桥-旧县: 经典段，约2小时
2. 遇龙桥-大榕树: 约1.5小时
3. 全程: 约4小时

### 价格
- 金龙桥-旧县: 约120-150元/筏(双人)
- 淡旺季有浮动

### 特色
- 人工竹筏，更亲近自然
- 沿途经过富里桥(明代古桥)
- 堤坝较小，不刺激但惬意

### 建议
- 早上漂流，光线好，人少
- 建议漂流后骑行十里画廊
- 带防晒用品
""".strip(),
            title="遇龙河漂流",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "阳朔", "tags": ["景点", "漂流", "遇龙河"]},
        ))

        # 两江四湖
        docs.append(Document(
            content="""
## 两江四湖

### 简介
两江四湖是桂林市区的水系，包括漓江、桃花江、杉湖、榕湖、桂湖、木龙湖。

### 游览方式
1. 日游(免费): 步行/骑车环湖
2. 夜游(付费): 乘船游览，夜景璀璨

### 两江四湖夜游
- 乘船时间: 约70分钟
- 价格: 约80-120元/人
- 推荐指数: ★★★★★

### 日月双塔
- 位于杉湖中央
- 桂林城市地标
- 夜景最佳拍摄点

### 游览建议
- 建议傍晚开始，日夜通吃
- 从象山景区附近上船
- 步行环湖约2-3小时
""".strip(),
            title="两江四湖",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "桂林", "tags": ["景点", "夜景", "船游"]},
        ))

        # 桂林交通指南
        docs.append(Document(
            content="""
## 桂林交通指南

### 外部交通

#### 航空
- 两江国际机场: 距市区约28公里
  - 北京/上海/广州等直飞
  - 机场大巴: 20元/人，约1小时到市区

#### 铁路
- 桂林站(市中心): 主要高铁站
- 桂林北站: 往北方向车次多
- 桂林西站: 较少使用
- 阳朔站: 离阳朔最近的高铁站

#### 长途汽车
- 桂林汽车站: 各省际班车
- 桂林琴潭汽车站: 往阳朔、机场方向

### 内部交通

#### 景区间交通
- 桂林-阳朔: 大巴约1.5小时，或坐船(4-5小时)
- 桂林-龙脊梯田: 大巴约2.5小时

#### 阳朔内部
- 租电动车: 约30-50元/天
- 摩的: 议价
- 包车: 约300-500元/天

### 建议
- 桂林适合作为中转站
- 阳朔可住2-3晚
- 龙脊梯田建议跟团或包车
""".strip(),
            title="桂林交通指南",
            source="桂林百科",
            url="",
            data_type=DataType.ATTRACTION,
            metadata={"city": "桂林", "tags": ["交通", "攻略"]},
        ))

        return docs

    async def run_and_save(
        self,
        collection: str = "lvyou_guilin",
        output_json: bool = False,
    ) -> Dict[str, Any]:
        """
        爬取并保存数据

        Args:
            collection: Milvus集合名
            output_json: 是否同时保存JSON

        Returns:
            执行结果统计
        """
        self.config.ensure_output_dir()

        # 1. 爬取
        docs = await self.crawl_all()

        # 2. 保存JSON
        json_result = {}
        if output_json:
            json_path = self.config.output_dir / "guilin_data.json"
            json_result = save_to_json(docs, str(json_path))

        # 3. 存入向量库
        vector_result = save_to_vectorstore(docs, collection=collection)

        return {
            "documents": len(docs),
            "json": json_result,
            "vectorstore": vector_result,
        }


# =============================================================================
# 便捷函数
# =============================================================================

async def crawl_guilin_data(
    collection: str = "lvyou_guilin",
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    一键爬取桂林数据并存入向量库

    使用方式:
        from lvyou_harness.crawler import crawl_guilin_data

        result = await crawl_guilin_data()
        print(f"爬取完成: {result['documents']} 文档")
    """
    crawler = GuilinCrawler()
    return await crawler.run_and_save(
        collection=collection,
        output_json=save_json,
    )
