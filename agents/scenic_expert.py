"""
景点专家Agent
=============

基于RAG检索回答景点相关问题

职责:
1. 查询景点基本信息 (开放时间、门票、地址)
2. 提供景点历史人文背景
3. 推荐最佳游览路线和季节
4. 对比相似景点
"""
from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional

from .base import BaseAgent, AgentConfig, AgentResponse

logger = logging.getLogger(__name__)


SCENIC_EXPERT_SYSTEM_PROMPT = """你是一个资深导游，对中国特别是广西(桂林、阳朔、北海等)景点了如指掌。

你的职责:
1. 准确回答景点相关问题 (开放时间、门票价格，最佳游览季节等)
2. 提供景点背后的历史故事和文化背景
3. 根据游客偏好推荐合适景点
4. 提醒游览注意事项和避坑指南

回答要求:
- 信息准确，避免模糊表述
- 如不确定，坦诚说明并提供替代建议
- 结合游客实际需求给出实用建议
"""


class ScenicExpertAgent(BaseAgent):
    """
    景点知识专家

    基于RAG检索回答景点相关问题
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        cfg = config or AgentConfig(
            name="ScenicExpert",
            system_prompt=SCENIC_EXPERT_SYSTEM_PROMPT,
        )
        super().__init__(cfg, **kwargs)

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        执行景点查询

        Args:
            task: 查询任务，如"漓江漂流最佳路线是什么？"
            context: 可选上下文

        Returns:
            AgentResponse
        """
        start = time.time()
        self.log("Execute", f"处理查询: {task[:50]}...")

        try:
            # 1. RAG检索相关景点信息
            docs = await self._retrieve(task, top_k=5)

            # 2. 构造上下文
            context_parts = []
            if docs:
                for doc in docs:
                    content = doc.get("content", "")
                    source = doc.get("metadata", {}).get("source", "")
                    context_parts.append(f"【参考信息 - {source}】\n{content[:500]}")

            rag_context = "\n\n".join(context_parts) if context_parts else "暂无相关背景信息"

            # 3. 调用LLM生成回答
            prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足，请基于你的知识回答。

【参考信息】
{rag_context}

【用户问题】
{task}

请给出准确、实用的回答。"""

            answer = await self._call_llm(prompt)

            return AgentResponse(
                success=True,
                content=answer,
                metadata={
                    "query": task,
                    "docs_retrieved": len(docs),
                    "rag_context": rag_context[:200],
                },
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"ScenicExpert执行失败: {e}")
            return AgentResponse(
                success=False,
                content=f"查询失败: {e}",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def query_scenic_info(
        self,
        scenic_name: str,
        info_type: str = "all",
    ) -> AgentResponse:
        """
        查询特定景点信息

        Args:
            scenic_name: 景点名称
            info_type: info_type类型 (basic, ticket, transport, history, all)
        """
        query_map = {
            "basic": f"{scenic_name}的基本信息、开放时间、地址",
            "ticket": f"{scenic_name}的门票价格、优惠政策",
            "transport": f"如何前往{scenic_name}，交通方式",
            "history": f"{scenic_name}的历史背景、文化故事",
            "all": f"{scenic_name}的全面介绍",
        }

        query = query_map.get(info_type, query_map["all"])
        return await self.execute(query)

    async def compare_scenics(
        self,
        scenic_names: List[str],
        criteria: Optional[List[str]] = None,
    ) -> AgentResponse:
        """
        对比多个景点

        Args:
            scenic_names: 景点名称列表
            criteria: 对比标准 (如 ["门票", "景色", "适合人群"])
        """
        scenic_list = "、".join(scenic_names)
        criteria_str = "、".join(criteria) if criteria else "全面对比"

        task = f"对比{scenic_list}，从{criteria_str}等方面进行分析"
        return await self.execute(task)
