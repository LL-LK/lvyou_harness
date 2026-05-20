"""
LvyouTools - 旅游领域专用工具函数
=====================================

提供:
- 景点信息查询
- 交通路线规划
- 天气预报获取
- 门票价格查询
- 知识库操作
"""
from __future__ import annotations

import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ScenicSpot:
    """景点"""
    id: str
    name: str
    region: str
    city: str
    level: str  # 5A, 4A, etc
    ticket_price: float
    opening_hours: str
    recommended_duration: int  # 分钟
    best_season: List[str]
    tags: List[str]
    description: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class RouteSegment:
    """路线片段"""
    from_scenic: str
    to_scenic: str
    distance_km: float
    duration_minutes: int
    transport_mode: str  # walk, bike, bus, car
    route_tips: str


@dataclass
class DailyPlan:
    """每日行程"""
    day: int
    date: str
    spots: List[Dict[str, Any]]
    total_duration_minutes: int
    total_distance_km: float
    meals: Dict[str, str]  # breakfast/lunch/dinner -> recommendation
    accommodation: Optional[str] = None
    tips: List[str] = None


# =============================================================================
# 工具函数
# =============================================================================

def load_scenic_database(db_path: str = None) -> Dict[str, ScenicSpot]:
    """
    加载景点数据库

    Args:
        db_path: 景点数据库路径 (JSON格式)

    Returns:
        {scenic_id: ScenicSpot} 字典
    """
    if db_path is None:
        db_path = Path(__file__).parent / "data" / "scenic_db.json"

    if not Path(db_path).exists():
        logger.warning(f"景点数据库不存在: {db_path}")
        return {}

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                k: ScenicSpot(**v) for k, v in data.items()
            }
    except Exception as e:
        logger.error(f"加载景点数据库失败: {e}")
        return {}


def search_scenics(
    keyword: str,
    db: Dict[str, ScenicSpot] = None,
    region: str = None,
    level: str = None,
    max_price: float = None,
    tags: List[str] = None,
) -> List[ScenicSpot]:
    """
    搜索景点

    Args:
        keyword: 搜索关键词
        db: 景点数据库
        region: 地区筛选
        level: 等级筛选 (5A, 4A)
        max_price: 最高门票价格
        tags: 标签筛选

    Returns:
        匹配的景点列表
    """
    if db is None:
        db = load_scenic_database()

    results = []
    kw_lower = keyword.lower()

    for scenic in db.values():
        # 关键词匹配
        if keyword and kw_lower not in scenic.name.lower() and kw_lower not in scenic.description.lower():
            # 检查tags
            if not any(kw_lower in tag.lower() for tag in scenic.tags):
                continue

        # 地区筛选
        if region and region not in scenic.region and region not in scenic.city:
            continue

        # 等级筛选
        if level and scenic.level != level:
            continue

        # 价格筛选
        if max_price and scenic.ticket_price > max_price:
            continue

        # 标签筛选
        if tags and not all(t in scenic.tags for t in tags):
            continue

        results.append(scenic)

    # 按等级和价格排序
    results.sort(key=lambda s: (s.level, s.ticket_price))
    return results


def estimate_transfer_time(
    from_scenic: str,
    to_scenic: str,
    db: Dict[str, ScenicSpot] = None,
    transport_mode: str = "car",
) -> RouteSegment:
    """
    估算景点间转移时间

    Args:
        from_scenic: 起始景点名
        to_scenic: 目标景点名
        db: 景点数据库
        transport_mode: 交通方式

    Returns:
        RouteSegment
    """
    # 简化实现：基于坐标估算
    # 实际应用中应调用地图API

    from_spot = None
    to_spot = None

    if db:
        for spot in db.values():
            if from_scenic in spot.name:
                from_spot = spot
            if to_scenic in spot.name:
                to_spot = spot

    if from_spot and to_spot and from_spot.latitude and to_spot.latitude:
        # 简化距离计算
        import math
        lat_diff = abs(from_spot.latitude - to_spot.latitude)
        lon_diff = abs(from_spot.longitude - to_spot.longitude)
        # 粗略估算: 1度约111km
        distance_km = math.sqrt(lat_diff**2 + lon_diff**2) * 111
    else:
        # 默认估算
        distance_km = 30.0

    # 根据交通方式估算时间
    speed_map = {
        "walk": 5,  # km/h
        "bike": 15,
        "bus": 25,
        "car": 40,
    }
    speed = speed_map.get(transport_mode, 30)
    duration = int(distance_km / speed * 60) + 10  # 加10分钟起步时间

    return RouteSegment(
        from_scenic=from_scenic,
        to_scenic=to_scenic,
        distance_km=distance_km,
        duration_minutes=duration,
        transport_mode=transport_mode,
        route_tips=f"约{distance_km:.1f}公里，预计{duration}分钟",
    )


def optimize_route_order(
    spots: List[str],
    db: Dict[str, ScenicSpot] = None,
    start_point: str = None,
) -> List[str]:
    """
    优化景点顺序，减少总路程

    Args:
        spots: 景点名称列表
        db: 景点数据库
        start_point: 起始点

    Returns:
        优化后的景点顺序
    """
    if len(spots) <= 2:
        return spots

    # 贪心算法: 每次选最近的未访问景点
    ordered = []
    remaining = list(spots)
    current = start_point or remaining.pop(0)
    ordered.append(current)

    while remaining:
        # 找最近的
        min_dist = float("inf")
        nearest = None
        nearest_idx = None

        for i, spot in enumerate(remaining):
            seg = estimate_transfer_time(current, spot, db)
            if seg.distance_km < min_dist:
                min_dist = seg.distance_km
                nearest = spot
                nearest_idx = i

        if nearest:
            ordered.append(nearest)
            current = nearest
            remaining.pop(nearest_idx)
        else:
            break

    return ordered


def calculate_daily_stats(
    spots: List[Dict[str, Any]],
    db: Dict[str, ScenicSpot] = None,
) -> Dict[str, Any]:
    """
    计算每日行程统计

    Args:
        spots: 当日景点列表
        db: 景点数据库

    Returns:
        统计信息
    """
    total_duration = 0
    total_distance = 0.0
    total_ticket = 0.0

    prev_scenic = None

    for spot_info in spots:
        scenic_name = spot_info.get("name", "")
        duration = spot_info.get("duration", 0)

        # 获取景点信息
        if db:
            matches = search_scenics(scenic_name, db)
            if matches:
                scenic = matches[0]
                duration = duration or scenic.recommended_duration
                total_ticket += scenic.ticket_price

        total_duration += duration

        # 计算距离
        if prev_scenic:
            seg = estimate_transfer_time(prev_scenic, scenic_name, db)
            total_distance += seg.distance_km

        prev_scenic = scenic_name

    return {
        "total_duration_hours": round(total_duration / 60, 1),
        "total_distance_km": round(total_distance, 1),
        "total_ticket_cny": total_ticket,
        "avg_time_per_spot": round(total_duration / len(spots)) if spots else 0,
    }


def generate_checklist(
    destination: str,
    days: int,
    season: str,
    style: str = "打卡拍照",
) -> Dict[str, List[str]]:
    """
    生成出行清单

    Args:
        destination: 目的地
        days: 天数
        season: 季节
        style: 旅行风格

    Returns:
        分类清单
    """
    checklist = {
        "证件": [
            "身份证",
            "学生证/老年证/军官证（如有）",
            "护照（出境需）",
        ],
        "钱财": [
            "现金（少量）",
            "信用卡/储蓄卡",
            "手机支付已开通",
        ],
        "电子产品": [
            "手机+充电器",
            "充电宝（20000mAh以内）",
            "自拍杆/三脚架",
            "耳机",
        ],
        "衣物": [],
        "洗漱": [
            "牙刷+牙膏",
            "毛巾/湿巾",
            "防晒霜",
            "洗漱用品（根据住宿）",
        ],
        "药品": [
            "创可贴",
            "肠胃药",
            "晕车药（如需要）",
            "常备药",
        ],
        "防护": [
            "雨伞/雨衣",
            "防晒装备",
            "口罩",
        ],
    }

    # 根据季节调整衣物
    season_clothes = {
        "春季": ["轻薄外套", "长袖", "运动鞋", "换洗衣物"],
        "夏季": ["短袖", "短裤/裙子", "凉鞋", "防晒衣", "换洗衣物"],
        "秋季": ["外套", "长袖", "运动鞋", "换洗衣物"],
        "冬季": ["羽绒服", "保暖内衣", "手套", "围巾", "保暖鞋"],
    }
    checklist["衣物"] = season_clothes.get(season, season_clothes["春季"])

    # 根据天数调整换洗衣物
    extra_days = max(0, days - 3)
    if extra_days > 0:
        checklist["衣物"].append(f"换洗衣物（{days}天需{extra_days + 1}套）")

    # 根据风格调整
    if style == "深度体验":
        checklist["电子产品"].append("相机+存储卡")
        checklist["电子产品"].append("笔记本（如需记录）")
    elif style == "打卡拍照":
        checklist["电子产品"].append("手机外接镜头（可选）")
        checklist["衣物"].append("出片服装")

    # 目的地特定
    if "桂林" in destination or "阳朔" in destination:
        checklist["防护"].append("防蚊虫")
        checklist["洗漱"].append("晒后修复")

    return checklist


def estimate_budget(
    days: int,
    destination: str,
    style: str = "中等",
    include_accommodation: bool = True,
    include_food: bool = True,
) -> Dict[str, Any]:
    """
    估算旅行预算

    Args:
        days: 天数
        destination: 目的地
        style: 预算级别 (节约/中等/奢侈)
        include_accommodation: 是否含住宿
        include_food: 是否含餐饮

    Returns:
        预算明细
    """
    # 基础预算 (中等风格)
    base_per_day = {
        "节约": 300,
        "中等": 600,
        "奢侈": 1200,
    }

    daily_base = base_per_day.get(style, 600)

    # 分配比例
    allocation = {
        "ticket": 0.30,
        "food": 0.25,
        "accommodation": 0.25,
        "transport": 0.15,
        "other": 0.05,
    }

    if not include_accommodation:
        allocation["food"] += 0.15
        allocation["ticket"] += 0.10

    total = daily_base * days

    budget = {}
    for cat, ratio in allocation.items():
        budget[cat] = {
            "amount": round(total * ratio),
            "percentage": int(ratio * 100),
            "daily_avg": round(total * ratio / days),
        }

    # 特殊标注
    if "桂林" in destination or "阳朔" in destination:
        budget["special_note"] = "漓江游船票需另计（约210元/人）"

    return {
        "total": total,
        "daily_avg": daily_base,
        "days": days,
        "style": style,
        "breakdown": budget,
    }
