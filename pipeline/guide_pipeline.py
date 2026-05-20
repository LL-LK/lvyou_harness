"""
攻略写作流水线
===============

完整的攻略写作流程:
1. 收集行程信息
2. 收集景点信息
3. 撰写攻略
4. 输出结果
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional

from .base import BasePipeline, PipelineResult
from ..agents.base import BaseAgent

logger = logging.getLogger(__name__)


class GuideWritingPipeline(BasePipeline):
    """
    攻略写作流水线
    """

    def __init__(
        self,
        scenic_agent: BaseAgent,
        guide_agent: BaseAgent,
    ):
        super().__init__("GuideWriting")
        self.scenic_agent = scenic_agent
        self.guide_agent = guide_agent

    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        执行攻略写作

        Args:
            input_data: 目的地描述
            context: 行程计划等信息

        Returns:
            PipelineResult
        """
        start = time.time()
        result = PipelineResult(success=True, output={})

        try:
            destination = input_data
            route_plan = context.get("route_plan") if context else None

            # Step 1: 查询目的地信息
            logger.info(f"[{self.name}] 查询目的地信息...")
            scenic_response = await self.scenic_agent.execute(
                f"详细介绍{destination}的旅游景点、美食、交通、住宿"
            )
            scenic_info = scenic_response.content if scenic_response.success else ""
            result.add_step(
                "query_destination",
                destination,
                scenic_info[:200],
                scenic_response.duration_ms,
            )

            # Step 2: 撰写攻略
            logger.info(f"[{self.name}] 撰写攻略...")
            guide_context = {
                "destination": destination,
                "route_plan": route_plan,
                "scenic_info": scenic_info,
            }
            guide_response = await self.guide_agent.execute(
                f"为{destination}撰写完整旅行攻略",
                context=guide_context,
            )
            result.add_step(
                "write_guide",
                destination,
                guide_response.content,
                guide_response.duration_ms,
            )

            # 组合输出
            result.output = {
                "destination": destination,
                "guide": guide_response.content,
                "scenic_info": scenic_info,
                "route_plan": route_plan,
            }

        except Exception as e:
            logger.error(f"GuideWritingPipeline失败: {e}")
            result.success = False
            result.error = str(e)

        result.duration_ms = (time.time() - start) * 1000
        return result

    def log(self, stage: str, message: str):
        logger.info(f"[{self.name}][{stage}] {message}")
