"""
行程规划流水线
================

完整的行程规划流程:
1. 理解用户需求
2. 查询景点信息 (RAG)
3. 生成行程规划
4. 优化预算
5. 输出结果
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional

from .base import BasePipeline, PipelineResult
from ..agents.base import BaseAgent, AgentResponse
from ..interfaces.rag import RAGPort
from ..interfaces.llm import LLMPort

logger = logging.getLogger(__name__)


class RoutePlanningPipeline(BasePipeline):
    """
    行程规划流水线

    组合多个Agent完成行程规划任务
    """

    def __init__(
        self,
        scenic_agent: BaseAgent,
        route_agent: BaseAgent,
        budget_agent: Optional[BaseAgent] = None,
    ):
        super().__init__("RoutePlanning")
        self.scenic_agent = scenic_agent
        self.route_agent = route_agent
        self.budget_agent = budget_agent

    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        执行行程规划

        Args:
            input_data: 用户需求描述
            context: 额外上下文

        Returns:
            PipelineResult
        """
        start = time.time()
        result = PipelineResult(success=True, output={})

        try:
            # Step 1: 理解需求
            logger.info(f"[{self.name}] 理解用户需求...")
            requirements = await self._understand_requirements(input_data)
            result.add_step("understand", input_data, requirements, 0)

            # Step 2: 查询景点信息
            scenic_info = ""
            if self.scenic_agent:
                logger.info(f"[{self.name}] 查询景点信息...")
                scenic_response = await self.scenic_agent.execute(
                    f"推荐{requirements.get('destination', '桂林')}"
                    f"{requirements.get('days', 3)}天旅游的热门景点"
                )
                scenic_info = scenic_response.content if scenic_response.success else ""
                result.add_step(
                    "query_scenics",
                    requirements.get("destination"),
                    scenic_info[:200],
                    scenic_response.duration_ms,
                )

            # Step 3: 生成行程
            logger.info(f"[{self.name}] 生成行程规划...")
            route_context = {
                "requirements": requirements,
                "scenic_info": scenic_info,
            }
            route_response = await self.route_agent.execute(
                input_data,
                context=route_context,
            )
            result.add_step(
                "plan_route",
                input_data,
                route_response.content,
                route_response.duration_ms,
            )

            # Step 4: 优化预算 (可选)
            budget_info = {}
            if self.budget_agent and requirements.get("budget"):
                logger.info(f"[{self.name}] 优化预算...")
                budget_response = await self.budget_agent.execute(
                    f"{requirements.get('destination')} "
                    f"{requirements.get('days')}天 "
                    f"预算{requirements.get('budget')}元",
                    context=requirements,
                )
                budget_info = budget_response.metadata if budget_response.success else {}
                result.add_step(
                    "optimize_budget",
                    requirements.get("budget"),
                    budget_info.get("content", "")[:200] if isinstance(budget_info, dict) else "",
                    budget_response.duration_ms,
                )

            # 组合输出
            result.output = {
                "requirements": requirements,
                "route_plan": route_response.content,
                "budget_info": budget_info,
                "scenic_recommendations": scenic_info,
            }

        except Exception as e:
            logger.error(f"RoutePlanningPipeline失败: {e}")
            result.success = False
            result.error = str(e)

        result.duration_ms = (time.time() - start) * 1000
        return result

    async def _understand_requirements(self, task: str) -> Dict[str, Any]:
        """理解用户需求"""
        # 简化实现
        requirements = {
            "destination": "桂林",
            "days": 3,
            "budget": 3000,
            "style": "普通",
            "travelers": 2,
        }

        import re
        # 解析天数
        days_match = re.search(r"(\d+)\s*天", task)
        if days_match:
            requirements["days"] = int(days_match.group(1))

        # 解析目的地
        destinations = ["桂林", "阳朔", "北海", "涠洲岛", "龙脊"]
        for dest in destinations:
            if dest in task:
                requirements["destination"] = dest
                break

        # 解析预算
        budget_match = re.search(r"(\d+)\s*元", task)
        if budget_match:
            requirements["budget"] = int(budget_match.group(1))

        return requirements

    def log(self, stage: str, message: str):
        logger.info(f"[{self.name}][{stage}] {message}")
