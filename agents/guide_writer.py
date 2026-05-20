"""
攻略写作Agent
=============

生成旅行攻略
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional

from .base import BaseAgent, AgentConfig, AgentResponse

logger = logging.getLogger(__name__)


GUIDE_WRITER_SYSTEM_PROMPT = """你是一个资深旅行攻略作家，擅长撰写实用、有趣的旅行攻略。

你的职责:
1. 撰写完整的旅行攻略
2. 提供实用的Tips和建议
3. 分享当地特色美食和购物信息
4. 给出避坑指南

写作风格:
- 实用为主，干货满满
- 语言生动，但不过度营销
- 重点信息突出
"""


class GuideWriterAgent(BaseAgent):
    """
    攻略写作Agent

    根据行程生成完整攻略
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        cfg = config or AgentConfig(
            name="GuideWriter",
            system_prompt=GUIDE_WRITER_SYSTEM_PROMPT,
        )
        super().__init__(cfg, **kwargs)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        生成攻略

        Args:
            task: 任务描述
            context: 行程计划等上下文

        Returns:
            AgentResponse
        """
        start = time.time()
        self.log("Execute", f"生成攻略: {task[:50]}...")

        try:
            # 获取行程上下文
            route_context = ""
            if context and context.get("route_plan"):
                route_context = f"\n\n【行程计划】\n{context['route_plan']}"

            prompt = f"""基于以下信息撰写旅行攻略:

【目的地】
{task}

{route_context}

请撰写完整的攻略，包含:
1. 行前准备
2. 每日行程详解
3. 美食推荐
4. 住宿建议
5. 交通指南
6. 注意事项
7. 省钱Tips

格式美观，信息实用。"""

            guide = await self._call_llm(prompt)

            return AgentResponse(
                success=True,
                content=guide,
                metadata={"task": task},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"GuideWriter执行失败: {e}")
            return AgentResponse(
                success=False,
                content=f"生成失败: {e}",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def write_daily_guide(
        self,
        day: int,
        attractions: list,
        tips: Optional[list] = None,
    ) -> str:
        """生成单日攻略"""
        attractions_str = "\n".join([f"- {a}" for a in attractions])
        tips_str = "\n".join([f"- {t}" for t in tips]) if tips else ""

        prompt = f"""为第{day}天撰写攻略:

【游览景点】
{attractions_str}

【小贴士】
{tips_str or "无特别提示"}

请给出详细的游览建议。"""

        return await self._call_llm(prompt)
