"""
Agent基类
==========

所有Lvyou Agent的基类

设计原则:
1. 依赖注入 - 通过构造函数注入RAG/LLM适配器
2. 单一职责 - 每个Agent只做一件事
3. 可组合 - Agent可以互相调用
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces.rag import RAGPort
    from ..interfaces.llm import LLMPort

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent执行结果"""
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentConfig:
    """Agent配置"""
    name: str
    system_prompt: str = ""
    model: str = "MiniMax-M2.7"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_k: int = 5
    max_iterations: int = 3


class BaseAgent(ABC):
    """
    Agent基类

    所有Agent需要实现:
    - execute(): 执行具体任务
    - get_system_prompt(): 返回系统提示
    """

    def __init__(
        self,
        config: AgentConfig,
        rag_adapter: Optional["RAGPort"] = None,
        llm_adapter: Optional["LLMPort"] = None,
    ):
        self.config = config
        self.rag = rag_adapter
        self.llm = llm_adapter
        self._initialized = False

    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        执行任务

        Args:
            task: 任务描述
            context: 执行上下文

        Returns:
            AgentResponse
        """
        pass

    def get_system_prompt(self) -> str:
        """返回系统提示"""
        return self.config.system_prompt

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """调用LLM"""
        if not self.llm:
            return f"[No LLM] {prompt}"

        system = system_prompt or self.get_system_prompt()
        return self.llm.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    async def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索RAG"""
        if not self.rag:
            return []

        k = top_k or self.config.top_k
        return self.rag.retrieve(query, top_k=k, filters=filters)

    def _format_context(self, context: Optional[Dict[str, Any]] = None) -> str:
        """格式化上下文为字符串"""
        if not context:
            return ""

        parts = []
        for key, value in context.items():
            parts.append(f"【{key}】\n{value}")
        return "\n\n".join(parts)

    def log(self, stage: str, message: str):
        """记录日志"""
        logger.info(f"[{self.config.name}][{stage}] {message}")


class SequentialAgent(BaseAgent):
    """
    顺序执行Agent

    将多个Agent顺序组合
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: List[BaseAgent],
    ):
        super().__init__(config)
        self.agents = agents

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """顺序执行所有Agent"""
        start = time.time()
        current_task = task
        results = []

        for agent in self.agents:
            self.log("Sequential", f"执行 {agent.config.name}")
            response = await agent.execute(current_task, context)
            results.append(response)

            if not response.success:
                return AgentResponse(
                    success=False,
                    content=f"{agent.config.name} 执行失败: {response.error}",
                    duration_ms=(time.time() - start) * 1000,
                )

            current_task = response.content

        return AgentResponse(
            success=True,
            content=current_task,
            metadata={"agent_results": [r.to_dict() for r in results]},
            duration_ms=(time.time() - start) * 1000,
        )


class ParallelAgent(BaseAgent):
    """
    并行执行Agent

    将多个Agent并行组合
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: List[BaseAgent],
    ):
        super().__init__(config)
        self.agents = agents

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """并行执行所有Agent"""
        import asyncio
        start = time.time()

        async def run_agent(agent: BaseAgent) -> AgentResponse:
            return await agent.execute(task, context)

        results = await asyncio.gather(
            *[run_agent(a) for a in self.agents],
            return_exceptions=True,
        )

        # 合并结果
        contents = []
        errors = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                errors.append(f"{self.agents[i].config.name}: {r}")
            elif r.success:
                contents.append(r.content)
            else:
                errors.append(f"{self.agents[i].config.name}: {r.error}")

        if errors:
            return AgentResponse(
                success=False,
                content=f"部分Agent失败: {errors}",
                duration_ms=(time.time() - start) * 1000,
            )

        return AgentResponse(
            success=True,
            content="\n\n".join(contents),
            metadata={"agent_results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in results]},
            duration_ms=(time.time() - start) * 1000,
        )
