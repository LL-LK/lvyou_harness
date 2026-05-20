"""
LvyouHarness - 旅游领域AGI系统
==============================

模块化架构，支持低耦合、高扩展

核心模块:
- interfaces: 接口抽象层
- adapters: 适配器实现
- agents: Agent实现
- pipeline: 流水线编排
- crawler: 数据爬虫
- orchestrator: 统一编排器
- utils: 工具函数

快速开始:
    from lvyou_harness import LvyouOrchestratorV2, OrchestratorConfig
    from lvyou_harness.adapters import create_rag_adapter, create_llm_adapter

    # 创建适配器
    rag = create_rag_adapter("milvus")
    llm = create_llm_adapter("minimax")

    # 创建编排器
    config = OrchestratorConfig()
    orch = LvyouOrchestratorV2(config, rag, llm)

    # 执行任务
    result = await orch.run("帮我们规划一个3天桂林之旅，预算中等")
"""
from .config import LvyouHarnessConfig

# 接口层
from .interfaces import RAGPort, LLMPort, CrawlerPort, StoragePort

# 适配器层
from .adapters.rag_adapter import MilvusAdapter, SimpleVectorAdapter, create_rag_adapter
from .adapters.llm_adapter import MiniMaxAdapter, MockLLMAdapter, create_llm_adapter

# Agent层
from .agents.base import BaseAgent, AgentConfig, AgentResponse
from .agents.scenic_expert import ScenicExpertAgent
from .agents.route_planner import RoutePlannerAgent
from .agents.guide_writer import GuideWriterAgent
from .agents.budget_optimizer import BudgetOptimizerAgent

# Pipeline层
from .pipeline.base import BasePipeline, PipelineResult, SequentialPipeline, ParallelPipeline
from .pipeline.route_pipeline import RoutePlanningPipeline
from .pipeline.guide_pipeline import GuideWritingPipeline

# 编排器
from .orchestrator.lvyou_orchestrator_v2 import LvyouOrchestratorV2, OrchestratorConfig

# 数据工具
from .data_utils import (
    Scenic,
    RoutePoint,
    DayPlan,
    Budget,
    TravelPlan,
    TravelStyle,
    BudgetLevel,
    ScenicLevel,
    TransportMode,
    validate_scenic,
    validate_travel_plan,
    export_plan,
    import_plan,
    export_to_markdown,
)

__version__ = "2.0.0"

__all__ = [
    # 配置
    "LvyouHarnessConfig",
    # 接口
    "RAGPort",
    "LLMPort",
    "CrawlerPort",
    "StoragePort",
    # 适配器
    "MilvusAdapter",
    "SimpleVectorAdapter",
    "create_rag_adapter",
    "MiniMaxAdapter",
    "MockLLMAdapter",
    "create_llm_adapter",
    # Agent
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    "ScenicExpertAgent",
    "RoutePlannerAgent",
    "GuideWriterAgent",
    "BudgetOptimizerAgent",
    # Pipeline
    "BasePipeline",
    "PipelineResult",
    "SequentialPipeline",
    "ParallelPipeline",
    "RoutePlanningPipeline",
    "GuideWritingPipeline",
    # 编排器
    "LvyouOrchestratorV2",
    "OrchestratorConfig",
    # 数据工具
    "Scenic",
    "RoutePoint",
    "DayPlan",
    "Budget",
    "TravelPlan",
    "TravelStyle",
    "BudgetLevel",
    "ScenicLevel",
    "TransportMode",
    "validate_scenic",
    "validate_travel_plan",
    "export_plan",
    "import_plan",
    "export_to_markdown",
]
