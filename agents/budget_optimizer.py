"""
预算优化Agent
=============

优化旅行预算
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base import BaseAgent, AgentConfig, AgentResponse

logger = logging.getLogger(__name__)


BUDGET_OPTIMIZER_SYSTEM_PROMPT = """你是一个精明的旅行预算规划师，擅长帮用户优化旅行预算。

你的职责:
1. 根据预算合理分配费用
2. 推荐性价比高的选择
3. 提供省钱建议
4. 避免不必要的开支

预算分配原则:
- 住宿: 30-40%
- 餐饮: 20-25%
- 交通: 15-20%
- 门票/活动: 15-20%
- 购物/备用: 5-10%
"""


@dataclass
class BudgetAllocation:
    """预算分配"""
    accommodation: float
    food: float
    transport: float
    tickets: float
    shopping: float
    total: float


class BudgetOptimizerAgent(BaseAgent):
    """
    预算优化Agent
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        cfg = config or AgentConfig(
            name="BudgetOptimizer",
            system_prompt=BUDGET_OPTIMIZER_SYSTEM_PROMPT,
        )
        super().__init__(cfg, **kwargs)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        优化预算

        Args:
            task: 预算任务
            context: 行程等信息

        Returns:
            AgentResponse
        """
        start = time.time()
        self.log("Execute", f"优化预算: {task}")

        try:
            # 解析预算需求
            budget_info = self._parse_budget(task, context)

            prompt = f"""为以下旅行计划优化预算:

【目的地】{budget_info['destination']}
【天数】{budget_info['days']}天
【总预算】{budget_info['total']}元
【旅行风格】{budget_info.get('style', '普通')}

请给出:
1. 详细的预算分配方案
2. 各部分的花费建议
3. 省钱技巧
4. 推荐选择的对比"""

            result = await self._call_llm(prompt)

            return AgentResponse(
                success=True,
                content=result,
                metadata=budget_info,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"BudgetOptimizer执行失败: {e}")
            return AgentResponse(
                success=False,
                content=f"优化失败: {e}",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _parse_budget(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """解析预算信息"""
        info = {
            "destination": "桂林",
            "days": 3,
            "total": 3000,
            "style": "普通",
        }

        # 简单解析
        if context:
            if context.get("days"):
                info["days"] = context["days"]
            if context.get("budget"):
                info["total"] = context["budget"]
            if context.get("destination"):
                info["destination"] = context["destination"]

        # 从task中提取
        import re
        days_match = re.search(r"(\d+)天", task)
        if days_match:
            info["days"] = int(days_match.group(1))

        budget_match = re.search(r"(\d+)\s*元", task)
        if budget_match:
            info["total"] = int(budget_match.group(1))
        elif "节约" in task or "省钱" in task:
            info["style"] = "节约"
            info["total"] = info["days"] * 300
        elif "奢侈" in task or "豪华" in task:
            info["style"] = "奢侈"
            info["total"] = info["days"] * 1500

        return info

    def calculate_allocation(self, total: float, days: int) -> BudgetAllocation:
        """计算标准预算分配"""
        return BudgetAllocation(
            accommodation=total * 0.35,
            food=total * 0.22,
            transport=total * 0.18,
            tickets=total * 0.15,
            shopping=total * 0.10,
            total=total,
        )
