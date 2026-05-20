"""
LvyouHarness V2 演示
====================

展示新架构的使用方式:
1. 依赖注入的适配器
2. 模块化的Agent
3. 可组合的Pipeline
4. 低耦合的接口设计

使用方式:
    python demo_v2.py
"""
import asyncio
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_dependency_injection():
    """演示依赖注入"""
    print("\n" + "=" * 60)
    print("演示1: 依赖注入 - 适配器热插拔")
    print("=" * 60)

    from lvyou_harness.adapters.rag_adapter import MilvusAdapter, SimpleVectorAdapter
    from lvyou_harness.adapters.llm_adapter import MiniMaxAdapter, MockLLMAdapter

    # 可以轻松切换实现
    # 使用SimpleVectorAdapter (内存/测试用)
    rag = SimpleVectorAdapter()
    rag.initialize()
    rag.add_documents([
        {"content": "漓江是桂林最著名的景点", "source": "桂林百科"},
        {"content": "象山景区位于桂林市区", "source": "桂林百科"},
    ])
    print(f"✓ SimpleVectorAdapter: {rag.count()} 条数据")

    # 使用MockLLMAdapter (测试用)
    llm = MockLLMAdapter()
    response = llm.generate("推荐桂林景点")
    print(f"✓ MockLLMAdapter: {response[:50]}...")

    print("\n优势: 可以轻松切换 RAG/LLM 实现，无需修改业务代码")


async def demo_agent_composition():
    """演示Agent组合"""
    print("\n" + "=" * 60)
    print("演示2: Agent组合")
    print("=" * 60)

    from lvyou_harness.agents.base import AgentConfig
    from lvyou_harness.agents.scenic_expert import ScenicExpertAgent
    from lvyou_harness.agents.route_planner import RoutePlannerAgent
    from lvyou_harness.agents.guide_writer import GuideWriterAgent
    from lvyou_harness.agents.budget_optimizer import BudgetOptimizerAgent
    from lvyou_harness.adapters.llm_adapter import MockLLMAdapter

    # 创建LLM适配器
    llm = MockLLMAdapter()
    llm.initialize()

    # 创建各个Agent
    scenic_cfg = AgentConfig(name="ScenicExpert")
    scenic_agent = ScenicExpertAgent(config=scenic_cfg, llm_adapter=llm)

    route_cfg = AgentConfig(name="RoutePlanner")
    route_agent = RoutePlannerAgent(
        config=route_cfg,
        scenic_agent=scenic_agent,
        llm_adapter=llm,
    )

    guide_cfg = AgentConfig(name="GuideWriter")
    guide_agent = GuideWriterAgent(config=guide_cfg, llm_adapter=llm)

    budget_cfg = AgentConfig(name="BudgetOptimizer")
    budget_agent = BudgetOptimizerAgent(config=budget_cfg, llm_adapter=llm)

    print("✓ 创建了4个专业Agent:")
    print(f"  - {scenic_agent.config.name}: 景点知识专家")
    print(f"  - {route_agent.config.name}: 行程规划师")
    print(f"  - {guide_agent.config.name}: 攻略作家")
    print(f"  - {budget_agent.config.name}: 预算优化师")

    # 执行任务
    response = await scenic_agent.execute("漓江漂流应该怎么玩?")
    print(f"\n✓ ScenicExpert执行: {response.success}")

    response = await route_agent.execute("帮我们规划3天桂林之旅")
    print(f"✓ RoutePlanner执行: {response.success}")

    print("\n优势: Agent可独立使用，也可组合使用")


async def demo_pipeline():
    """演示Pipeline"""
    print("\n" + "=" * 60)
    print("演示3: Pipeline编排")
    print("=" * 60)

    from lvyou_harness.agents.base import AgentConfig
    from lvyou_harness.agents.scenic_expert import ScenicExpertAgent
    from lvyou_harness.agents.route_planner import RoutePlannerAgent
    from lvyou_harness.agents.budget_optimizer import BudgetOptimizerAgent
    from lvyou_harness.pipeline.route_pipeline import RoutePlanningPipeline
    from lvyou_harness.adapters.llm_adapter import MockLLMAdapter

    # 创建组件
    llm = MockLLMAdapter()
    llm.initialize()

    scenic_cfg = AgentConfig(name="ScenicExpert")
    scenic_agent = ScenicExpertAgent(config=scenic_cfg, llm_adapter=llm)

    route_cfg = AgentConfig(name="RoutePlanner")
    route_agent = RoutePlannerAgent(
        config=route_cfg,
        scenic_agent=scenic_agent,
        llm_adapter=llm,
    )

    budget_cfg = AgentConfig(name="BudgetOptimizer")
    budget_agent = BudgetOptimizerAgent(config=budget_cfg, llm_adapter=llm)

    # 创建Pipeline
    pipeline = RoutePlanningPipeline(
        scenic_agent=scenic_agent,
        route_agent=route_agent,
        budget_agent=budget_agent,
    )

    print("✓ 创建了RoutePlanningPipeline")
    print("  流程: 理解需求 → 查询景点 → 生成行程 → 优化预算")

    # 执行Pipeline
    result = await pipeline.execute(
        "帮我们规划一个3天桂林阳朔之旅，预算中等"
    )

    print(f"\n✓ Pipeline执行完成: {result.success}")
    print(f"  耗时: {result.duration_ms:.0f}ms")
    print(f"  步骤数: {len(result.steps)}")

    for step in result.steps:
        print(f"  - {step['name']}: {step['duration_ms']:.0f}ms")

    print("\n优势: Pipeline封装了复杂流程，对外只暴露简单接口")


async def demo_orchestrator():
    """演示编排器"""
    print("\n" + "=" * 60)
    print("演示4: LvyouOrchestratorV2 统一编排")
    print("=" * 60)

    from lvyou_harness.orchestrator.lvyou_orchestrator_v2 import (
        LvyouOrchestratorV2,
        OrchestratorConfig,
    )
    from lvyou_harness.adapters.llm_adapter import MockLLMAdapter

    # 创建配置
    config = OrchestratorConfig(
        enable_scenic=True,
        enable_route=True,
        enable_guide=True,
        enable_budget=True,
        enable_pipeline=True,
    )

    # 创建LLM
    llm = MockLLMAdapter()

    # 创建编排器
    orch = LvyouOrchestratorV2(config=config, llm_adapter=llm)
    orch.initialize()

    print("✓ LvyouOrchestratorV2 初始化完成")
    print(f"  启用的Agent: {list(orch._agents.keys())}")
    print(f"  启用的Pipeline: {list(orch._pipelines.keys())}")

    # 测试不同类型任务
    tasks = [
        "漓江漂流应该怎么玩？",  # 景点查询
        "帮我们规划3天桂林之旅",  # 行程规划
        "预算3000元怎么玩桂林？",  # 预算优化
    ]

    for task in tasks:
        result = await orch.run(task)
        print(f"\n任务: {task[:20]}...")
        print(f"类型: {result.get('type', 'unknown')}")
        print(f"成功: {result.get('success', False)}")

    print("\n优势: 统一入口，自动路由到合适的Agent/Pipeline")


def demo_module_structure():
    """演示模块结构"""
    print("\n" + "=" * 60)
    print("模块结构")
    print("=" * 60)

    import os

    def print_tree(directory, prefix=""):
        items = []
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path) and not item.startswith('__'):
                items.append(('dir', item, path))
            elif item.endswith('.py') and not item.startswith('__'):
                items.append(('file', item[:-3], path))

        for i, (type_, name, path) in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "

            if type_ == 'dir':
                print(f"{prefix}{current_prefix}{name}/")
                print_tree(path, prefix + next_prefix)
            else:
                print(f"{prefix}{current_prefix}{name}")

    print_tree("/home/l2140/lvyou_harness/interfaces")
    print()
    print_tree("/home/l2140/lvyou_harness/adapters")
    print()
    print_tree("/home/l2140/lvyou_harness/agents")
    print()
    print_tree("/home/l2140/lvyou_harness/pipeline")


async def main():
    """主函数"""
    print("=" * 60)
    print("LvyouHarness V2 架构演示")
    print("=" * 60)

    await demo_dependency_injection()
    await demo_agent_composition()
    await demo_pipeline()
    await demo_orchestrator()
    demo_module_structure()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
