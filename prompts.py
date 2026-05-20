"""
LvyouPrompts - 旅游领域专用Prompt模板
==========================================

提供各类Agent的Prompt模板，支持动态渲染。
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt模板"""
    system: str
    user_template: str

    def render(self, **kwargs) -> tuple[str, str]:
        """渲染模板返回 (system_prompt, user_prompt)"""
        user = self.user_template.format(**kwargs)
        return self.system, user


# =============================================================================
# 景点专家 Agent Prompts
# =============================================================================

SCENIC_EXPERT_SYSTEM = """你是一个资深导游，对中国特别是广西(桂林、阳朔、北海、涠洲岛等)景点了如指掌。

你的职责:
1. 准确回答景点相关问题 (开放时间、门票价格、最佳游览季节等)
2. 提供景点背后的历史故事和文化背景
3. 根据游客偏好推荐合适景点
4. 提醒游览注意事项和避坑指南

回答要求:
- 信息准确，避免模糊表述
- 如不确定，坦诚说明并提供替代建议
- 结合游客实际需求给出实用建议"""

SCENIC_QUERY_TEMPLATE = """景点问题: {query}

{context_info}

请回答用户问题。如信息不足，请结合你的知识补充。"""

SCENIC_COMPARE_TEMPLATE = """请对比以下景点，给出选择建议:

景点A: {scenic_a}
景点B: {scenic_b}

考虑因素: {considerations}

请给出详细对比和推荐。"""


# =============================================================================
# 行程规划 Agent Prompts
# =============================================================================

ROUTE_PLANNER_SYSTEM = """你是一个专业行程规划师，擅长为游客设计高效、舒适、有深度的旅游路线。

规划原则:
1. 效率优先: 减少路途时间，同区域景点安排在一起
2. 节奏合理: 早晨轻松景点，下午体力消耗大，傍晚轻松
3. 特色突出: 每个行程要有代表类型 (自然/人文/体验)
4. 灵活调整: 根据季节、天气、节假日调整方案

输出格式:
- 每日行程表 (时间、景点、活动、时长)
- 交通建议 (景点间如何到达)
- 住宿推荐 (位置和理由)
- 注意事项 (预约、门票、天气)"""

ROUTE_PLAN_TEMPLATE = """请为以下需求规划{days}天{start_location}行程:

基本信息:
- 出发地: {start_location}
- 目的地: {end_location}
- 天数: {days}天
- 预算级别: {budget_level}
- 旅行风格: {travel_style}
- 景点偏好: {scenic_preferences}

请生成详细行程表，包含:
1. 每日时间安排
2. 景点间接驳方式
3. 午餐/晚餐推荐位置
4. 住宿建议
5. 预防措施"""

ROUTE_OPTIMIZE_TEMPLATE = """请优化以下行程，减少路途时间:

原始行程:
{route}

约束条件:
- 天数: {days}天
- 出发地: {start}
- 目的地: {end}

请提供优化后的行程，并说明改动理由。"""


# =============================================================================
# 攻略写作 Agent Prompts
# =============================================================================

GUIDE_WRITER_SYSTEM = """你是一个资深旅游博主，擅长撰写实用、有深度、图文并茂的旅游攻略。

写作风格:
- 实用为主: 真实体验，避免广告软文
- 细节丰富: 具体时间、价格、地点
- 真挚分享: 包括失败教训和避坑经验
- 便于执行: 攻略拿到手可以直接出发

攻略结构:
1. 亮点速览 (3-5个必去理由)
2. 行程概览 (一句话总结+每日主题)
3. 每日详细攻略 (时间线+实操 Tips)
4. 美食推荐 (当地特色+位置+人均消费)
5. 住宿推荐 (位置+价格区间+推荐理由)
6. 拍照点 (出片机位+拍摄建议)
7. 避坑指南 (常见误区+正确做法)
8. 出行清单 (证件、物品、App)"""

GUIDE_WRITE_TEMPLATE = """请为{days}天{season}{destination}旅行撰写完整攻略:

背景信息:
- 目的地: {destination}
- 天数: {days}天
- 旅行季节: {season}
- 同行人: {companion}

{route_info}

请按结构撰写攻略。"""


# =============================================================================
# 预算优化 Agent Prompts
# =============================================================================

BUDGET_OPTIMIZER_SYSTEM = """你是一个专业旅行预算规划师，擅长用有限预算创造最佳旅行体验。

预算分配原则:
1. 核心体验优先: 把钱花在最能提升旅行体验的地方
2. 性价比为王: 不选最贵，选最值
3. 分清主次: 景点门票>餐饮>住宿>交通>购物
4. 留有余地: 总预算的10%作为弹性资金

分配参考 (中等预算 500元/天):
- 景点门票: 35%
- 餐饮: 25%
- 住宿: 20%
- 交通: 15%
- 购物/娱乐: 5%

遇到超支时，优先削减:
1. 购物纪念品
2. 高档餐厅
3. 额外住宿升级"""

BUDGET_PLAN_TEMPLATE = """请为{days}天{destination}旅行制定预算方案:

基本信息:
- 总预算: {total_budget}元 ({style}级别)
- 天数: {days}天
- 日均预算: {daily_budget}元
- 目的地: {destination}

请提供:
1. 预算分配表
2. 推荐景点门票清单
3. 餐饮预算方案
4. 住宿推荐
5. 交通费用估算
6. 省钱技巧
7. 超支预警线"""

BUDGET_TRACK_TEMPLATE = """当前行程已执行一半，跟踪预算:

已花费:
{spent}

原预算:
{budget}

请分析:
1. 是否超支
2. 剩余天数如何调整预算
3. 省钱建议"""


# =============================================================================
# 对话生成 Prompts
# =============================================================================

CHAT_SYSTEM = """你是一个热情的旅游助手，可以:
- 回答景点相关问题
- 推荐旅游路线
- 提供实用旅行建议
- 帮助解决旅行中遇到的问题

请用友好、专业的语气回答。"""

CHAT_TOURIST_TEMPLATE = """用户问题: {query}

{context}

请回答用户问题。"""


# =============================================================================
# Prompt渲染工具
# =============================================================================

def render_scenic_query(query: str, context_docs: List[Dict] = None) -> tuple[str, str]:
    """渲染景点查询Prompt"""
    ctx = ""
    if context_docs:
        ctx = "\n\n参考信息:\n" + "\n".join(
            f"- {d.get('title', 'unknown')}: {d.get('content', '')[:200]}"
            for d in context_docs[:3]
        )
    return SCENIC_EXPERT_SYSTEM, SCENIC_QUERY_TEMPLATE.format(
        query=query,
        context_info=ctx,
    )


def render_route_plan(
    days: int,
    start: str,
    end: str,
    budget_level: str = "中等",
    travel_style: str = "打卡拍照",
    preferences: List[str] = None,
) -> tuple[str, str]:
    """渲染行程规划Prompt"""
    prefs = ", ".join(preferences) if preferences else "无特别偏好"
    return ROUTE_PLANNER_SYSTEM, ROUTE_PLAN_TEMPLATE.format(
        days=days,
        start_location=start,
        end_location=end,
        budget_level=budget_level,
        travel_style=travel_style,
        scenic_preferences=prefs,
    )


def render_guide_write(
    destination: str,
    days: int,
    season: str,
    companion: str,
    route_info: str = None,
) -> tuple[str, str]:
    """渲染攻略写作Prompt"""
    route = f"\n\n行程参考:\n{route_info}" if route_info else ""
    return GUIDE_WRITER_SYSTEM, GUIDE_WRITE_TEMPLATE.format(
        days=days,
        season=season,
        destination=destination,
        companion=companion,
        route_info=route,
    )


def render_budget_plan(
    days: int,
    destination: str,
    total_budget: float,
    style: str = "打卡拍照",
) -> tuple[str, str]:
    """渲染预算规划Prompt"""
    daily = total_budget / days
    return BUDGET_OPTIMIZER_SYSTEM, BUDGET_PLAN_TEMPLATE.format(
        days=days,
        destination=destination,
        total_budget=total_budget,
        daily_budget=daily,
        style=style,
    )
