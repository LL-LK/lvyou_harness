"""
LvyouOrchestrator V2
====================

统一编排器 - 整合所有Agent和Pipeline

设计原则:
1. 依赖注入 - 适配器通过构造函数注入
2. 配置驱动 - 行为由配置决定
3. 可组合 - 支持自定义Pipeline
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """编排器配置"""
    name: str = "LvyouOrchestrator"
    enable_scenic: bool = True
    enable_route: bool = True
    enable_guide: bool = True
    enable_budget: bool = True
    enable_rag: bool = True
    enable_pipeline: bool = True


class LvyouOrchestratorV2:
    """
    旅游编排器 V2

    整合所有组件，提供统一的接口

    使用方式:
        orch = LvyouOrchestratorV2(config, rag_adapter, llm_adapter)
        result = await orch.run("帮我们规划一个3天桂林之旅，预算中等")
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        rag_adapter=None,
        llm_adapter=None,
    ):
        self.config = config or OrchestratorConfig()
        self.rag = rag_adapter
        self.llm = llm_adapter

        # Agent实例
        self._agents: Dict[str, Any] = {}
        # Pipeline实例
        self._pipelines: Dict[str, Any] = {}
        # 已初始化
        self._initialized = False

    def initialize(self):
        """初始化所有Agent和Pipeline"""
        if self._initialized:
            return

        # 延迟导入避免循环依赖
        from ..agents.base import AgentConfig
        from ..agents.scenic_expert import ScenicExpertAgent
        from ..agents.route_planner import RoutePlannerAgent
        from ..agents.guide_writer import GuideWriterAgent
        from ..agents.budget_optimizer import BudgetOptimizerAgent
        from ..pipeline.route_pipeline import RoutePlanningPipeline

        # 创建Agent
        if self.config.enable_scenic:
            scenic_cfg = AgentConfig(name="ScenicExpert")
            self._agents["scenic"] = ScenicExpertAgent(
                config=scenic_cfg,
                rag_adapter=self.rag,
                llm_adapter=self.llm,
            )

        if self.config.enable_route:
            route_cfg = AgentConfig(name="RoutePlanner")
            self._agents["route"] = RoutePlannerAgent(
                config=route_cfg,
                scenic_agent=self._agents.get("scenic"),
                llm_adapter=self.llm,
            )

        if self.config.enable_guide:
            guide_cfg = AgentConfig(name="GuideWriter")
            self._agents["guide"] = GuideWriterAgent(
                config=guide_cfg,
                llm_adapter=self.llm,
            )

        if self.config.enable_budget:
            budget_cfg = AgentConfig(name="BudgetOptimizer")
            self._agents["budget"] = BudgetOptimizerAgent(
                config=budget_cfg,
                llm_adapter=self.llm,
            )

        # 创建Pipeline
        if self.config.enable_pipeline and self.config.enable_route:
            self._pipelines["route"] = RoutePlanningPipeline(
                scenic_agent=self._agents.get("scenic"),
                route_agent=self._agents.get("route"),
                budget_agent=self._agents.get("budget"),
            )

        self._initialized = True
        logger.info(f"LvyouOrchestratorV2 初始化完成: {list(self._agents.keys())}")

    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行任务

        Args:
            task: 用户任务
            context: 额外上下文

        Returns:
            执行结果
        """
        self.initialize()
        start = time.time()

        # 判断任务类型并路由
        if self._is_route_task(task):
            return await self._run_route_pipeline(task, context)
        elif self._is_guide_task(task):
            return await self._run_guide_pipeline(task, context)
        elif self._is_scenic_task(task):
            return await self._run_scenic_query(task)
        elif self._is_budget_task(task):
            return await self._run_budget_task(task)
        else:
            return await self._run_default(task, context)

    def _is_route_task(self, task: str) -> bool:
        """判断是否为行程规划任务"""
        route_keywords = ["规划", "行程", "路线", "旅游", "旅行", "玩"]
        return any(k in task for k in route_keywords)

    def _is_guide_task(self, task: str) -> bool:
        """判断是否为攻略写作任务"""
        guide_keywords = ["攻略", "攻略", "指南", "推荐", "介绍"]
        return any(k in task for k in guide_keywords)

    def _is_scenic_task(self, task: str) -> bool:
        """判断是否为景点查询任务"""
        scenic_keywords = ["景点", "开放时间", "门票", "怎么去", "哪个好"]
        return any(k in task for k in scenic_keywords)

    def _is_budget_task(self, task: str) -> bool:
        """判断是否为预算任务"""
        budget_keywords = ["预算", "花多少钱", "省钱", "费用"]
        return any(k in task for k in budget_keywords)

    async def _run_route_pipeline(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """运行行程规划流水线"""
        pipeline = self._pipelines.get("route")
        if pipeline:
            result = await pipeline.execute(task, context)
            return {
                "success": result.success,
                "type": "route",
                "content": result.output,
                "steps": result.steps,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        return {"success": False, "error": "Route pipeline not available"}

    async def _run_guide_pipeline(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """运行攻略写作流水线"""
        guide_agent = self._agents.get("guide")
        scenic_agent = self._agents.get("scenic")

        if guide_agent:
            # 简单实现
            response = await guide_agent.execute(task, context)
            return {
                "success": response.success,
                "type": "guide",
                "content": response.content,
                "duration_ms": response.duration_ms,
                "error": response.error,
            }

        return {"success": False, "error": "Guide agent not available"}

    async def _run_scenic_query(self, task: str) -> Dict[str, Any]:
        """运行景点查询"""
        scenic_agent = self._agents.get("scenic")
        if scenic_agent:
            response = await scenic_agent.execute(task)
            return {
                "success": response.success,
                "type": "scenic",
                "content": response.content,
                "metadata": response.metadata,
                "duration_ms": response.duration_ms,
                "error": response.error,
            }
        return {"success": False, "error": "Scenic agent not available"}

    async def _run_budget_task(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """运行预算任务"""
        budget_agent = self._agents.get("budget")
        if budget_agent:
            response = await budget_agent.execute(task, context)
            return {
                "success": response.success,
                "type": "budget",
                "content": response.content,
                "duration_ms": response.duration_ms,
                "error": response.error,
            }
        return {"success": False, "error": "Budget agent not available"}

    async def _run_default(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """默认处理"""
        # 使用LLM直接回答
        if self.llm:
            response = self.llm.generate(
                prompt=task,
                system_prompt="你是一个桂林旅游助手，请回答用户的问题。",
            )
            return {
                "success": True,
                "type": "default",
                "content": response,
            }
        return {"success": False, "error": "No LLM available"}

    def get_agent(self, name: str):
        """获取Agent"""
        return self._agents.get(name)

    def get_pipeline(self, name: str):
        """获取Pipeline"""
        return self._pipelines.get(name)
