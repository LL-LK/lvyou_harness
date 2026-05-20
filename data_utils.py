"""
LvyouDataUtils - 旅游数据工具
================================

提供:
- 景点数据结构
- 行程数据结构
- 预算数据结构
- 数据验证
- 数据导出/导入
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举类型
# =============================================================================

class ScenicLevel(Enum):
    """景点级别"""
    AAAAA = "5A"
    AAAA = "4A"
    AAA = "3A"
    AA = "2A"
    A = "1A"
    UNRATED = "未评级"


class TravelStyle(Enum):
    """旅行风格"""
    BUDGET = "节约型"
    PHOTO = "打卡拍照"
    DEEP = "深度体验"
    LUXURY = "奢侈豪华"
    LEISURE = "休闲度假"


class BudgetLevel(Enum):
    """预算级别"""
    LOW = "节约"
    MEDIUM = "中等"
    HIGH = "奢侈"


class TransportMode(Enum):
    """交通方式"""
    WALK = "步行"
    BIKE = "骑行"
    BUS = "公交"
    TAXI = "打车"
    CAR = "自驾"
    TRAIN = "火车"
    PLANE = "飞机"


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Scenic:
    """景点"""
    id: str
    name: str
    level: ScenicLevel = ScenicLevel.UNRATED
    ticket_price: float = 0.0
    opening_hours: str = "09:00-17:00"
    recommended_duration: int = 120  # 分钟
    best_season: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    address: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    phone: str = ""
    website: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Scenic":
        d = d.copy()
        if "level" in d:
            d["level"] = ScenicLevel(d["level"])
        return cls(**d)


@dataclass
class Meal:
    """餐饮"""
    name: str
    type: str  # breakfast, lunch, dinner, snack
    price_range: str = ""  # e.g. "20-50元"
    location: str = ""
    recommended_dishes: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Accommodation:
    """住宿"""
    name: str
    type: str  # hotel, hostel, guesthouse
    price_per_night: float = 0.0
    location: str = ""
    rating: float = 0.0  # 1-5
    notes: str = ""


@dataclass
class RoutePoint:
    """路线点"""
    scenic: Scenic
    arrival_time: str = "09:00"
    departure_time: str = "12:00"
    transport_from_prev: str = ""
    ticket_status: str = "自理"  # 已含/自理/需预约
    notes: str = ""


@dataclass
class DayPlan:
    """每日行程"""
    day: int
    date: str = ""
    theme: str = ""
    route_points: List[RoutePoint] = field(default_factory=list)
    meals: List[Meal] = field(default_factory=list)
    accommodation: Optional[Accommodation] = None
    total_cost: float = 0.0
    tips: List[str] = field(default_factory=list)
    weather_tips: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["route_points"] = [
            {
                "scenic": rp.scenic.to_dict(),
                "arrival_time": rp.arrival_time,
                "departure_time": rp.departure_time,
                "transport_from_prev": rp.transport_from_prev,
                "ticket_status": rp.ticket_status,
                "notes": rp.notes,
            }
            for rp in self.route_points
        ]
        d["meals"] = [asdict(m) for m in self.meals]
        if self.accommodation:
            d["accommodation"] = asdict(self.accommodation)
        return d


@dataclass
class Budget:
    """预算"""
    total: float = 0.0
    currency: str = "CNY"
    daily_allocation: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    spent: float = 0.0
    remaining: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_spent(self, amount: float, category: str = "other") -> None:
        """添加支出"""
        self.spent += amount
        self.remaining = self.total - self.spent
        if category in self.breakdown:
            self.breakdown[category] += amount


@dataclass
class TravelPlan:
    """完整旅行计划"""
    id: str
    destination: str
    days: int
    start_date: str = ""
    end_date: str = ""
    travel_style: TravelStyle = TravelStyle.PHOTO
    budget_level: BudgetLevel = BudgetLevel.MEDIUM
    day_plans: List[DayPlan] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    total_cost: float = 0.0
    companions: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "destination": self.destination,
            "days": self.days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "travel_style": self.travel_style.value,
            "budget_level": self.budget_level.value,
            "day_plans": [d.to_dict() for d in self.day_plans],
            "budget": self.budget.to_dict(),
            "total_cost": self.total_cost,
            "companions": self.companions,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TravelPlan":
        d = d.copy()
        d["travel_style"] = TravelStyle(d.get("travel_style", "打卡拍照"))
        d["budget_level"] = BudgetLevel(d.get("budget_level", "中等"))
        d["day_plans"] = [DayPlan(**day) for day in d.get("day_plans", [])]
        d["budget"] = Budget(**d.get("budget", {}))
        return cls(**d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "TravelPlan":
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# 数据验证
# =============================================================================

def validate_scenic(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    验证景点数据

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    if not data.get("id"):
        errors.append("景点ID不能为空")
    if not data.get("name"):
        errors.append("景点名称不能为空")
    if "ticket_price" in data and data["ticket_price"] < 0:
        errors.append("门票价格不能为负")

    return len(errors) == 0, errors


def validate_travel_plan(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    验证旅行计划

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    if not data.get("id"):
        errors.append("计划ID不能为空")
    if not data.get("destination"):
        errors.append("目的地不能为空")
    if data.get("days", 0) < 1:
        errors.append("天数必须至少为1")
    if data.get("days", 0) > 30:
        errors.append("天数不能超过30")

    # 验证每日行程
    day_plans = data.get("day_plans", [])
    if len(day_plans) != data.get("days", 0):
        errors.append(f"每日行程数量({len(day_plans)})与天数({data.get('days')})不匹配")

    return len(errors) == 0, errors


# =============================================================================
# 数据导出/导入
# =============================================================================

def export_plan(plan: TravelPlan, path: str) -> None:
    """导出旅行计划到文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(plan.to_json(indent=2))
    logger.info(f"旅行计划已导出: {path}")


def import_plan(path: str) -> TravelPlan:
    """从文件导入旅行计划"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, errors = validate_travel_plan(data)
    if not valid:
        raise ValueError(f"数据验证失败: {errors}")

    return TravelPlan.from_dict(data)


def export_to_markdown(plan: TravelPlan) -> str:
    """导出为Markdown格式"""
    lines = [
        f"# {plan.destination} {plan.days}天旅行计划",
        "",
        f"**旅行日期**: {plan.start_date} - {plan.end_date}",
        f"**旅行风格**: {plan.travel_style.value}",
        f"**预算级别**: {plan.budget_level.value}",
        "",
        "## 行程概览",
        "",
    ]

    for day_plan in plan.day_plans:
        lines.append(f"### 第{day_plan.day}天: {day_plan.theme}")
        lines.append("")

        for rp in day_plan.route_points:
            lines.append(
                f"- **{rp.arrival_time}** {rp.scenic.name} "
                f"(游览约{rp.scenic.recommended_duration}分钟)"
            )

        lines.append("")

    lines.append("## 预算总览")
    lines.append("")
    lines.append(f"- 总预算: ¥{plan.budget.total:.2f}")
    lines.append(f"- 已花费: ¥{plan.budget.spent:.2f}")
    lines.append(f"- 剩余: ¥{plan.budget.remaining:.2f}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 数据统计
# =============================================================================

def calculate_plan_stats(plan: TravelPlan) -> Dict[str, Any]:
    """计算旅行计划统计"""
    total_scenics = sum(len(d.route_points) for d in plan.day_plans)
    total_ticket = sum(
        sum(rp.scenic.ticket_price for rp in d.route_points)
        for d in plan.day_plans
    )
    total_meal = sum(len(d.meals) for d in plan.day_plans)

    return {
        "total_days": plan.days,
        "total_scenics": total_scenics,
        "total_ticket_cost": total_ticket,
        "total_meals": total_meal,
        "total_cost": plan.total_cost,
        "avg_cost_per_day": plan.total_cost / plan.days if plan.days > 0 else 0,
        "avg_scenics_per_day": total_scenics / plan.days if plan.days > 0 else 0,
    }


# =============================================================================
# 快捷创建
# =============================================================================

def create_simple_plan(
    destination: str,
    days: int,
    style: str = "中等",
) -> TravelPlan:
    """创建简单旅行计划"""
    plan_id = f"{destination}_{days}days_{datetime.now().strftime('%Y%m%d')}"

    return TravelPlan(
        id=plan_id,
        destination=destination,
        days=days,
        travel_style=TravelStyle(style),
        budget_level=BudgetLevel(style),
        day_plans=[
            DayPlan(day=i + 1, theme=f"第{i + 1}天")
            for i in range(days)
        ],
    )
