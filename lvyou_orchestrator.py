"""
LvyouOrchestrator - 旅游领域统一编排器
==========================================

负责任务分解 → Agent调度 → RAG检索 → 结果聚合

使用方式:
    from lvyou_harness import LvyouOrchestrator, LvyouHarnessConfig

    cfg = LvyouHarnessConfig.for_guilin()
    orch = LvyouOrchestrator(cfg)
    result = orch.run("帮我们规划一个3天桂林阳朔之旅，预算中等")

    print(result["result"]["route"])
    print(result["result"]["guide"])
    print(result["result"]["budget"])
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
from pathlib import Path

from .lvyou_agents import (
    RoutePlannerAgent,
    GuideWriterAgent,
    ScenicExpertAgent,
    BudgetOptimizerAgent,
    AgentRole,
)
from .config import LvyouHarnessConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 尝试导入共享基类
# =============================================================================

SharedOrchestrator = None
SharedTaskBoard = None
Task = None
TaskStatus = None
TaskPriority = None

try:
    from _shared_harness.orchestrator import SharedOrchestrator
    from _shared_harness.task_board import SharedTaskBoard, Task, TaskStatus, TaskPriority
except ImportError:
    logger.warning("_shared_harness未安装，使用简化实现")
    SharedOrchestrator = object
    SharedTaskBoard = None
    Task = None
    TaskStatus = None
    TaskPriority = None


@dataclass
class LvyouTask:
    """Lvyou专用Task类型"""
    id: str
    title: str
    description: str
    agent_type: str  # "route" | "guide" | "scenic" | "budget"
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)


# =============================================================================
# Agent类别映射
# =============================================================================

LVYOU_AGENT_CLASSES: Dict[str, Type] = {
    "route": RoutePlannerAgent,
    "guide": GuideWriterAgent,
    "scenic": ScenicExpertAgent,
    "budget": BudgetOptimizerAgent,
}

LVYOU_TASK_TYPE_MAP: Dict[str, str] = {
    "route": "行程规划",
    "guide": "攻略写作",
    "scenic": "景点知识",
    "budget": "预算优化",
}


class LvyouOrchestrator:
    """
    旅游领域统一编排器

    支持4类Agent并行/顺序执行:
    - route: 行程规划
    - guide: 攻略写作 (依赖route)
    - scenic: 景点查询 (可独立，可被route使用)
    - budget: 预算优化 (可独立，可被guide使用)

    工作流程:
    1. 解析用户需求 → 确定需要哪些Agent
    2. 并行执行独立任务 (scenic, budget)
    3. 串行执行依赖任务 (route → guide)
    4. 聚合输出最终结果
    """

    def __init__(
        self,
        config: Optional[LvyouHarnessConfig] = None,
        workspace: Optional[str | Path] = None,
        enable_rag: bool = True,
    ):
        self.config = config or LvyouHarnessConfig()
        self.workspace = Path(workspace) if workspace else self.config.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.enable_rag = enable_rag

        # Agent实例缓存
        self._agents: Dict[str, Any] = {}
        # 任务列表
        self._tasks: Dict[str, LvyouTask] = {}
        # 执行统计
        self._stats: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_duration_ms": 0.0,
        }

    # -------------------------------------------------------------------------
    # Agent获取
    # -------------------------------------------------------------------------

    def get_agent(self, agent_type: str) -> Any:
        """获取或创建指定类型的Agent"""
        if agent_type in self._agents:
            return self._agents[agent_type]

        if agent_type not in LVYOU_AGENT_CLASSES:
            raise ValueError(f"未知Agent类型: {agent_type}")

        agent_cls = LVYOU_AGENT_CLASSES[agent_type]
        agent_kwargs = {
            "name": f"{agent_type}_1",
            "workspace": str(self.workspace / agent_type),
        }

        # 注入配置参数
        if agent_type == "route":
            agent_kwargs.update({
                "default_days": self.config.default_days,
                "max_daily_spots": self.config.max_daily_spots,
                "transfer_time_minutes": self.config.transfer_time_minutes,
            })
        elif agent_type == "budget":
            agent_kwargs.update({
                "default_currency": self.config.default_currency,
                "default_budget_per_day": self.config.budget_per_day,
            })
        elif agent_type == "scenic":
            agent_kwargs.update({
                "collection_name": self.config.collection_name,
                "top_k": self.config.top_k,
            })

        agent = agent_cls(**agent_kwargs)

        # 初始化RAG
        if self.enable_rag and agent_type == "scenic":
            if hasattr(agent, "init_rag"):
                agent.init_rag()

        self._agents[agent_type] = agent
        logger.info(f"创建Agent: {agent_type}")
        return agent

    # -------------------------------------------------------------------------
    # 任务管理
    # -------------------------------------------------------------------------

    def _add_task(
        self,
        task_id: str,
        title: str,
        description: str,
        agent_type: str,
        depends_on: List[str] = None,
    ) -> LvyouTask:
        """添加任务"""
        task = LvyouTask(
            id=task_id,
            title=title,
            description=description,
            agent_type=agent_type,
            depends_on=depends_on or [],
        )
        self._tasks[task_id] = task
        self._stats["total_tasks"] += 1
        return task

    def _execute_task(self, task: LvyouTask) -> Dict[str, Any]:
        """执行单个任务"""
        agent = self.get_agent(task.agent_type)
        context = {}

        # scenic任务直接传递原始query
        if task.agent_type == "scenic":
            context = {"query": task.description}
        # route任务需要解析后的上下文
        elif task.agent_type == "route":
            context = self._parse_route_context(task.description)
        # guide任务需要行程信息
        elif task.agent_type == "guide":
            context = self._parse_guide_context(task.description)
        # budget任务需要预算上下文
        elif task.agent_type == "budget":
            context = self._parse_budget_context(task.description)

        start = time.time()
        try:
            response = agent.run(task.description, context=context)
            duration_ms = (time.time() - start) * 1000

            if response.get("success", True):
                task.status = "done"
                task.result = response
                self._stats["completed_tasks"] += 1
                return {
                    "success": True,
                    "task_id": task.id,
                    "content": response.get("content", ""),
                    "duration_ms": duration_ms,
                }
            else:
                task.status = "failed"
                task.error = response.get("error", "Unknown error")
                self._stats["failed_tasks"] += 1
                return {
                    "success": False,
                    "task_id": task.id,
                    "error": task.error,
                    "duration_ms": duration_ms,
                }
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self._stats["failed_tasks"] += 1
            logger.exception(f"Task {task.id} 执行失败")
            return {"success": False, "task_id": task.id, "error": str(e)}

    # -------------------------------------------------------------------------
    # 上下文解析
    # -------------------------------------------------------------------------

    def _parse_route_context(self, description: str) -> Dict[str, Any]:
        """从描述中解析行程规划上下文"""
        context = {
            "days": self.config.default_days,
            "start_location": "桂林",
            "end_location": "桂林",
            "scenic_preferences": [],
            "budget_level": "中等",
            "travel_style": "打卡拍照",
        }

        # 简单关键词检测
        desc = description.lower()
        if "阳朔" in description:
            context["end_location"] = "阳朔"
        if "3天" in description or "三日" in description:
            context["days"] = 3
        elif "4天" in description or "四日" in description:
            context["days"] = 4
        elif "5天" in description or "五日" in description:
            context["days"] = 5

        if "深度" in description:
            context["travel_style"] = "深度体验"
        elif "休闲" in description or "度假" in description:
            context["travel_style"] = "休闲度假"

        if "节约" in description or "穷游" in description:
            context["budget_level"] = "节约"
        elif "奢侈" in description or "豪华" in description:
            context["budget_level"] = "奢侈"

        return context

    def _parse_guide_context(self, description: str) -> Dict[str, Any]:
        """从描述中解析攻略写作上下文"""
        context = {
            "destination": "桂林阳朔",
            "days": 3,
            "season": "春季",
            "companion": "朋友",
        }

        if "阳朔" in description:
            context["destination"] = "桂林阳朔"
        if "3天" in description:
            context["days"] = 3
        elif "4天" in description:
            context["days"] = 4

        # 季节检测
        if "夏天" in description or "暑期" in description:
            context["season"] = "夏季"
        elif "秋天" in description or "秋季" in description:
            context["season"] = "秋季"
        elif "冬天" in description or "冬季" in description:
            context["season"] = "冬季"

        return context

    def _parse_budget_context(self, description: str) -> Dict[str, Any]:
        """从描述中解析预算优化上下文"""
        context = {
            "total_budget": self.config.budget_per_day * 3,
            "days": 3,
            "destination": "桂林",
            "style": "打卡拍照",
        }

        if "预算" in description:
            # 尝试提取数字
            import re
            numbers = re.findall(r'\d+', description)
            if numbers:
                context["total_budget"] = float(numbers[0]) * 1000  # 假设单位是k

        if "3天" in description:
            context["days"] = 3
        elif "4天" in description:
            context["days"] = 4
        elif "5天" in description:
            context["days"] = 5

        if "节约" in description or "穷游" in description:
            context["style"] = "节约"
        elif "深度" in description:
            context["style"] = "深度体验"
        elif "奢侈" in description:
            context["style"] = "奢侈"

        return context

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        主入口: 分析需求 → 分解任务 → 并行/顺序执行 → 聚合结果

        Args:
            task: 用户任务描述 (如 "帮我们规划3天桂林阳朔之旅")
            context: 额外上下文 (可选)

        Returns:
            {
                "success": bool,
                "result": {
                    "route": {...},    # 行程规划结果
                    "guide": {...},    # 攻略写作结果
                    "scenic": {...},   # 景点查询结果
                    "budget": {...},   # 预算优化结果
                },
                "stats": {...},       # 执行统计
                "tasks": [...],        # 任务列表
            }
        """
        start = time.time()
        context = context or {}

        # 1. 分析需求，确定需要执行哪些任务
        task_plan = self._plan_tasks(task)
        logger.info(f"任务规划: {[t.agent_type for t in task_plan]}")

        # 2. 拓扑排序执行
        # 先执行无依赖的 (scenic, budget)
        # 再执行有依赖的 (route, guide)
        ready_tasks = [t for t in task_plan if not t.depends_on]
        completed = set()

        while ready_tasks:
            # 并行执行当前批次
            results = {}
            for t in ready_tasks:
                result = self._execute_task(t)
                results[t.id] = result
                if result["success"]:
                    completed.add(t.id)

            # 收集下一批就绪任务
            ready_tasks = []
            for t in task_plan:
                if t.id in completed or t.status != "pending":
                    continue
                if all(dep in completed for dep in t.depends_on):
                    ready_tasks.append(t)

        # 3. 聚合结果
        final_result = self._aggregate_results(task_plan)

        self._stats["total_duration_ms"] = (time.time() - start) * 1000

        return {
            "success": self._stats["failed_tasks"] == 0,
            "result": final_result,
            "stats": self._stats.copy(),
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "agent_type": t.agent_type,
                    "status": t.status,
                    "error": t.error,
                }
                for t in task_plan
            ],
        }

    def _plan_tasks(self, task: str) -> List[LvyouTask]:
        """分析用户需求，规划任务"""
        tasks = []
        task_counter = {"route": 0, "guide": 0, "scenic": 0, "budget": 0}

        # 总是执行景点查询 (为其他任务提供知识支持)
        task_counter["scenic"] += 1
        scenic_task = self._add_task(
            task_id=f"scenic_{task_counter['scenic']:03d}",
            title="景点知识查询",
            description=task,
            agent_type="scenic",
        )
        tasks.append(scenic_task)

        # 行程规划
        if any(kw in task for kw in ["规划", "行程", "路线", "几天"]):
            task_counter["route"] += 1
            route_task = self._add_task(
                task_id=f"route_{task_counter['route']:03d}",
                title="行程规划",
                description=task,
                agent_type="route",
                depends_on=[],  # route可独立运行
            )
            tasks.append(route_task)

            # 攻略写作依赖行程
            if any(kw in task for kw in ["攻略", "推荐", "指南"]):
                task_counter["guide"] += 1
                guide_task = self._add_task(
                    task_id=f"guide_{task_counter['guide']:03d}",
                    title="攻略写作",
                    description=task,
                    agent_type="guide",
                    depends_on=[route_task.id],  # 依赖行程
                )
                tasks.append(guide_task)

        # 预算优化
        if any(kw in task for kw in ["预算", "花费", "费用", "省钱", "花费"]):
            task_counter["budget"] += 1
            budget_task = self._add_task(
                task_id=f"budget_{task_counter['budget']:03d}",
                title="预算优化",
                description=task,
                agent_type="budget",
            )
            tasks.append(budget_task)

        return tasks

    def _aggregate_results(self, tasks: List[LvyouTask]) -> Dict[str, Any]:
        """聚合多Agent结果"""
        result = {}

        for task in tasks:
            if task.status != "done":
                continue

            agent_type = task.agent_type
            response = task.result

            result[agent_type] = {
                "content": response.get("content", ""),
                "attachments": response.get("attachments", {}),
                "metadata": response.get("metadata", {}),
            }

        return result


# =============================================================================
# 便捷入口函数
# =============================================================================

def plan_trip(
    destination: str,
    days: int,
    budget: float = None,
    style: str = "打卡拍照",
    config: LvyouHarnessConfig = None,
) -> Dict[str, Any]:
    """
    便捷入口: 快速规划行程

    Args:
        destination: 目的地 (如 "桂林阳朔")
        days: 天数
        budget: 总预算 (可选)
        style: 旅行风格
        config: 配置 (可选)

    Returns:
        行程规划结果
    """
    cfg = config or LvyouHarnessConfig()
    orch = LvyouOrchestrator(cfg)

    budget_str = f"，预算{int(budget)}元" if budget else ""
    task = f"帮我们规划{days}天{destination}之旅，{style}风格{budget_str}"

    result = orch.run(task)
    return result.get("result", {}).get("route", {})


def write_guide(
    destination: str,
    days: int,
    season: str = "春季",
    companion: str = "朋友",
    route_content: str = None,
    config: LvyouHarnessConfig = None,
) -> Dict[str, Any]:
    """
    便捷入口: 生成旅游攻略

    Args:
        destination: 目的地
        days: 天数
        season: 季节
        companion: 同行人
        route_content: 行程内容 (可选，有则更精准)
        config: 配置 (可选)

    Returns:
        攻略内容
    """
    cfg = config or LvyouHarnessConfig()
    orch = LvyouOrchestrator(cfg)

    task = f"为{days}天{season}{destination}旅行撰写完整攻略，同行人是{companion}"
    if route_content:
        task += f"\n\n行程参考:\n{route_content}"

    result = orch.run(task)
    return result.get("result", {}).get("guide", {})
