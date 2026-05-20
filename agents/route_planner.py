"""
行程规划Agent
=============

规划旅行行程
"""
from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentResponse

logger = logging.getLogger(__name__)


ROUTE_PLANNER_SYSTEM_PROMPT = """你是一个专业的旅行规划师，擅长规划广西(桂林、阳朔、北海等)地区的旅行路线。

你的职责:
1. 根据用户需求规划最佳路线
2. 合理安排时间，避免走回头路
3. 推荐景点游览顺序
4. 提供交通建议

规划原则:
- 同一区域景点尽量安排在同一天
- 考虑景点间交通时间
- 留出足够的游览时间
- 早晚安排要合理
"""


@dataclass
class RoutePlan:
    """行程计划"""
    days: List[Dict[str, Any]]


class RoutePlannerAgent(BaseAgent):
    """
    行程规划Agent

    根据目的地、天数、偏好规划行程
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        scenic_agent: Optional[BaseAgent] = None,
        **kwargs,
    ):
        cfg = config or AgentConfig(
            name="RoutePlanner",
            system_prompt=ROUTE_PLANNER_SYSTEM_PROMPT,
        )
        super().__init__(cfg, **kwargs)
        self.scenic_agent = scenic_agent

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        执行行程规划

        Args:
            task: 规划任务，如"帮我们规划一个3天桂林阳朔之旅，预算中等"
            context: 可选上下文

        Returns:
            AgentResponse
        """
        start = time.time()
        self.log("Execute", f"规划行程: {task[:50]}...")

        try:
            # 1. 解析需求
            requirements = self._parse_requirements(task)

            # 2. 查询相关景点信息
            scenic_context = ""
            if self.scenic_agent:
                scenic_query = f"桂林{requirements.get('days', 3)}天旅游推荐景点"
                response = await self.scenic_agent.execute(scenic_query)
                if response.success:
                    scenic_context = f"\n\n【景点信息】\n{response.content}"

            # 3. 生成行程
            prompt = f"""基于以下信息规划行程:

【用户需求】
{task}

【可用景点信息】
{scenic_context or "请基于你的知识规划行程"}

请给出详细的每日行程安排，包括:
1. 每天游览的景点
2. 建议的出发时间和游览时长
3. 景点间的交通方式
4. 用餐建议

格式清晰，便于阅读。"""

            plan = await self._call_llm(prompt)

            return AgentResponse(
                success=True,
                content=plan,
                metadata={
                    "task": task,
                    "requirements": requirements,
                },
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"RoutePlanner执行失败: {e}")
            return AgentResponse(
                success=False,
                content=f"规划失败: {e}",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _parse_requirements(self, task: str) -> Dict[str, Any]:
        """解析用户需求"""
        # 简化实现，实际应该用LLM解析
        requirements = {
            "destination": "桂林",
            "days": 3,
            "budget": "中等",
        }

        # 简单关键词检测
        if "3天" in task:
            requirements["days"] = 3
        elif "4天" in task:
            requirements["days"] = 4
        elif "5天" in task:
            requirements["days"] = 5

        if "节约" in task or "省钱" in task:
            requirements["budget"] = "节约"
        elif "奢侈" in task or "豪华" in task:
            requirements["budget"] = "奢侈"

        return requirements
