"""
LvyouAgents - 旅游领域专用Agent
=================================

提供4类专用Agent:
- RoutePlannerAgent: 行程规划
- GuideWriterAgent: 攻略写作
- ScenicExpertAgent: 景点知识专家 (RAG)
- BudgetOptimizerAgent: 预算优化
"""
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# 导入共享基类 (延迟导入避免循环)
BaseHarnessAgent = None
AgentRole = None
AgentResponse = None


def _import_shared():
    global BaseHarnessAgent, AgentRole, AgentResponse
    if BaseHarnessAgent is None:
        try:
            from _shared_harness.base_agent import BaseHarnessAgent, AgentRole, AgentResponse
        except ImportError:
            # 开发环境fallback
            from enum import Enum


            class _DummyBase:
                ROLE = None
                SYSTEM_PROMPT = ""

                def __init__(self, name, workspace, model="MiniMax-M2.7",
                             provider="minimax-cn", system_prompt=None,
                             max_iterations=5, timeout_per_call=120):
                    self.name = name
                    self.workspace = workspace
                    self.model = model
                    self.provider = provider
                    self.max_iterations = max_iterations
                    self.timeout_per_call = timeout_per_call
                    self._messages = []

                def _call_llm(self, prompt, attachments=None, tools=None):
                    return {"success": True, "content": f"[模拟] {prompt[:100]}"}

                def run(self, task, context=None):
                    result = self._execute(task, context or {})
                    return result

                def _execute(self, task, context):
                    raise NotImplementedError

                def write_file(self, path, content):
                    from pathlib import Path
                    p = Path(path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")
                    return str(p)

                def read_file(self, path):
                    from pathlib import Path
                    return Path(path).read_text(encoding="utf-8")

                def append_log(self, stage, message):
                    pass

            BaseHarnessAgent = _DummyBase


_import_shared()


class AgentRole(Enum):
    RESEARCHER = "researcher"
    PLANNER = "planner"
    CODER = "coder"
    ANALYZER = "analyzer"
    WRITER = "writer"
    CLASSIFIER = "classifier"
    BOOKING = "booking"
    COMPLIANCE = "compliance"
    SCENIC_EXPERT = "scenic_expert"
    ROUTE_PLANNER = "route_planner"
    BUDGET_OPTIMIZER = "budget_optimizer"


@dataclass
class AgentResponse:
    """Agent执行结果"""
    success: bool
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    attachments: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0


# =============================================================================
# 景点知识专家 Agent
# =============================================================================

class ScenicExpertAgent(BaseHarnessAgent):
    """
    景点知识专家 - 基于RAG检索回答景点相关问题

    职责:
    - 查询景点基本信息 (开放时间、门票、地址)
    - 提供景点历史人文背景
    - 推荐最佳游览路线和季节
    - 对比相似景点
    """

    ROLE = AgentRole.SCENIC_EXPERT
    SYSTEM_PROMPT = """你是一个资深导游，对中国特别是广西(桂林、阳朔、北海等)景点了如指掌。

你的职责:
1. 准确回答景点相关问题 (开放时间、门票价格、最佳游览季节等)
2. 提供景点背后的历史故事和文化背景
3. 根据游客偏好推荐合适景点
4. 提醒游览注意事项和避坑指南

回答要求:
- 信息准确，避免模糊表述
- 如不确定，坦诚说明并提供替代建议
- 结合游客实际需求给出实用建议"""

    def __init__(self, name: str, workspace: str,
                 collection_name: str = "lvyou_guilin",
                 top_k: int = 5, **kwargs):
        super().__init__(name, workspace, **kwargs)
        self.collection_name = collection_name
        self.top_k = top_k
        self._rag_client = None

    def init_rag(self):
        """初始化RAG检索"""
        try:
            import sys
            sys.path.insert(0, "/home/l2140/RAG-Harness")
            from rag_harness.unified_adapter import UnifiedHarnessRAGAdapter, HarnessConfig
            cfg = HarnessConfig.for_lvyou() if hasattr(HarnessConfig, "for_lvyou") else None
            if cfg:
                self._rag_client = UnifiedHarnessRAGAdapter(cfg)
                logger.info(f"RAG初始化完成: {self.collection_name}")
        except Exception as e:
            logger.warning(f"RAG初始化失败: {e}")

    def retrieve_scenic_info(self, query: str) -> List[Dict[str, Any]]:
        """检索景点相关信息"""
        if not self._rag_client:
            return []
        try:
            return self._rag_client.retrieve(query, top_k=self.top_k)
        except Exception as e:
            logger.error(f"RAG检索失败: {e}")
            return []

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行景点知识查询"""
        # 1. RAG检索景点信息
        docs = self.retrieve_scenic_info(task)

        # 2. 构建prompt
        context_info = ""
        if docs:
            context_info = "\n\n参考信息:\n" + "\n".join(
                f"- {d.get('title', d.get('source', 'unknown'))}: {d.get('content', d.get('text', ''))[:200]}"
                for d in docs[:3]
            )

        prompt = f"""问题: {task}
{context_info}

请基于上述信息，回答用户问题。如信息不足，请结合你的知识补充。"""

        # 3. LLM生成回答
        result = self._call_llm(prompt)

        return {
            "content": result.get("content", ""),
            "tool_calls": [],
            "attachments": {"docs": docs},
            "metadata": {"query": task, "docs_count": len(docs)},
        }


# =============================================================================
# 行程规划 Agent
# =============================================================================

class RoutePlannerAgent(BaseHarnessAgent):
    """
    行程规划Agent - 智能规划旅游路线

    职责:
    - 根据天数和偏好生成每日行程
    - 优化景点顺序减少路程时间
    - 确保行程节奏合理 (体力分配)
    - 处理景点时间冲突
    """

    ROLE = AgentRole.ROUTE_PLANNER
    SYSTEM_PROMPT = """你是一个专业行程规划师，擅长为游客设计高效、舒适、有深度的旅游路线。

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

    def __init__(self, name: str, workspace: str,
                 default_days: int = 3,
                 max_daily_spots: int = 6,
                 transfer_time_minutes: int = 30,
                 **kwargs):
        super().__init__(name, workspace, **kwargs)
        self.default_days = default_days
        self.max_daily_spots = max_daily_spots
        self.transfer_time = transfer_time_minutes

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成行程规划"""
        # 解析用户需求
        days = context.get("days", self.default_days)
        start_location = context.get("start_location", "桂林")
        end_location = context.get("end_location", "桂林")
        scenic_preferences = context.get("scenic_preferences", [])  # ["自然风光", "人文历史"]
        budget_level = context.get("budget_level", "中等")  # 节约/中等/奢侈
        travel_style = context.get("travel_style", "打卡拍照")  # 打卡拍照/深度体验/休闲度假

        # 构建prompt
        prompt = f"""请为以下需求规划{days}天{start_location}行程:

基本信息:
- 出发地: {start_location}
- 目的地: {end_location}
- 天数: {days}天
- 预算级别: {budget_level}
- 旅行风格: {travel_style}
- 景点偏好: {', '.join(scenic_preferences) if scenic_preferences else '无特别偏好'}

请生成详细行程表，包含:
1. 每日时间安排 (每个景点的开始时间、游览时长)
2. 景点间接驳方式 (车程时长推荐交通)
3. 午餐/晚餐推荐位置
4. 住宿建议 (性价比/位置)
5. 預防措施 (预约渠道、必备物品)

格式要求: 清晰易读，方便直接使用"""

        result = self._call_llm(prompt)

        return {
            "content": result.get("content", ""),
            "tool_calls": [],
            "attachments": {},
            "metadata": {
                "days": days,
                "start": start_location,
                "end": end_location,
                "preferences": scenic_preferences,
            },
        }


# =============================================================================
# 攻略写作 Agent
# =============================================================================

class GuideWriterAgent(BaseHarnessAgent):
    """
    攻略写作Agent - 生成精美旅游攻略

    职责:
    - 将行程转化为精美图文攻略
    - 提供实用Tips和避坑指南
    - 撰写美食、住宿、拍照点推荐
    - 生成行程清单和预算表
    """

    ROLE = AgentRole.WRITER
    SYSTEM_PROMPT = """你是一个资深旅游博主，擅长撰写实用、有深度、图文并茂的旅游攻略。

写作风格:
- 实用为主: 真实体验，避免广告软文
- 细节丰富: 具体时间、价格、地点，而非泛泛推荐
- 真诚分享: 包括失败教训和避坑经验
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

    def __init__(self, name: str, workspace: str,
                 include_checklist: bool = True,
                 include_budget: bool = True,
                 **kwargs):
        super().__init__(name, workspace, **kwargs)
        self.include_checklist = include_checklist
        self.include_budget = include_budget

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成旅游攻略"""
        route_info = context.get("route", "")  # 行程信息
        destination = context.get("destination", "桂林阳朔")
        days = context.get("days", 3)
        season = context.get("season", "春季")
        travel_companion = context.get("companion", "朋友")  # 家人/朋友/情侣/独行

        prompt = f"""请为{days}天{season}{destination}旅行撰写完整攻略:

背景信息:
- 目的地: {destination}
- 天数: {days}天
- 旅行季节: {season}
- 同行人: {travel_companion}

请按以下结构撰写:
1. 一句话亮点总结
2. 行程概览表
3. 每日详细攻略 (时间线)
4. 美食推荐清单
5. 住宿推荐清单
6. 出片机位推荐
7. 避坑指南 (至少5条)
8. 出行准备清单
9. 估算费用表

要求:
- 内容真实具体，有可操作性
- 适当加入个人体验感受
- 格式清晰美观"""

        result = self._call_llm(prompt)

        return {
            "content": result.get("content", ""),
            "tool_calls": [],
            "attachments": {},
            "metadata": {
                "destination": destination,
                "days": days,
                "season": season,
                "companion": travel_companion,
            },
        }


# =============================================================================
# 预算优化 Agent
# =============================================================================

class BudgetOptimizerAgent(BaseHarnessAgent):
    """
    预算优化Agent - 智能分配旅游预算

    职责:
    - 根据总预算制定分配方案
    - 推荐性价比景点组合
    - 识别可节省和必须花的地方
    - 实时追踪预算执行情况
    """

    ROLE = AgentRole.BUDGET_OPTIMIZER
    SYSTEM_PROMPT = """你是一个专业旅行预算规划师，擅长用有限预算创造最佳旅行体验。

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

    def __init__(self, name: str, workspace: str,
                 default_currency: str = "CNY",
                 default_budget_per_day: float = 500.0,
                 **kwargs):
        super().__init__(name, workspace, **kwargs)
        self.currency = default_currency
        self.budget_per_day = default_budget_per_day

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """优化预算分配"""
        total_budget = context.get("total_budget", self.budget_per_day * 3)
        days = context.get("days", 3)
        destination = context.get("destination", "桂林")
        style = context.get("style", "打卡拍照")  # 节约/打卡/深度

        # 调整预算基准
        style_multiplier = {"节约": 0.6, "打卡拍照": 1.0, "深度体验": 1.5, "奢侈": 2.5}
        multiplier = style_multiplier.get(style, 1.0)
        effective_budget = self.budget_per_day * multiplier * days

        prompt = f"""请为{days}天{destination}旅行制定预算方案:

基本信息:
- 总预算: {effective_budget:.0f}元 ({style}级别)
- 天数: {days}天
- 日均预算: {effective_budget / days:.0f}元
- 目的地: {destination}

请提供:
1. 预算分配表 (景点/餐饮/住宿/交通/其他)
2. 推荐景点门票清单 (含价格)
3. 餐饮预算方案 (早/中/晚各档次)
4. 住宿推荐 (价格区间+推荐理由)
5. 交通费用估算 (大交通+市内交通)
6. 省钱技巧 (至少5条)
7. 超支预警线

格式要求: 清晰量化，方便执行"""

        result = self._call_llm(prompt)

        return {
            "content": result.get("content", ""),
            "tool_calls": [],
            "attachments": {},
            "metadata": {
                "total_budget": effective_budget,
                "daily_budget": effective_budget / days,
                "days": days,
                "style": style,
                "currency": self.currency,
            },
        }
