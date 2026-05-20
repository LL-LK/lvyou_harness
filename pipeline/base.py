"""
Pipeline基类
=============

流水线基类，定义Agent编排模式

支持:
- Sequential: 顺序执行
- Parallel: 并行执行
- Conditional: 条件执行
- Pipeline: 流水线组合
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """流水线执行结果"""
    success: bool
    output: Any
    steps: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def add_step(self, name: str, input_data: Any, output_data: Any, duration_ms: float):
        self.steps.append({
            "name": name,
            "input": input_data,
            "output": output_data,
            "duration_ms": duration_ms,
        })


class BasePipeline(ABC):
    """
    流水线基类

    所有Pipeline需要实现:
    - execute(): 执行流水线
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """执行流水线"""
        pass


class SequentialPipeline(BasePipeline):
    """
    顺序流水线

    按顺序执行一系列步骤
    """

    def __init__(self, name: str, steps: List[Callable]):
        super().__init__(name)
        self.steps = steps

    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        start = time.time()
        result = PipelineResult(success=True, output=input_data)
        current_input = input_data

        for step in self.steps:
            step_start = time.time()
            step_name = getattr(step, "__name__", str(step))

            try:
                if asyncio.iscoroutinefunction(step):
                    step_output = await step(current_input, context or {})
                else:
                    step_output = step(current_input, context or {})

                result.add_step(
                    step_name,
                    current_input,
                    step_output,
                    (time.time() - step_start) * 1000,
                )
                current_input = step_output

            except Exception as e:
                logger.error(f"Pipeline步骤 {step_name} 失败: {e}")
                result.success = False
                result.error = f"{step_name}: {e}"
                break

        result.output = current_input
        result.duration_ms = (time.time() - start) * 1000
        return result


class ParallelPipeline(BasePipeline):
    """
    并行流水线

    并行执行多个任务，然后合并结果
    """

    def __init__(self, name: str, tasks: List[Callable]):
        super().__init__(name)
        self.tasks = tasks

    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        import asyncio
        start = time.time()

        async def run_task(task: Callable):
            if asyncio.iscoroutinefunction(task):
                return await task(input_data, context or {})
            return task(input_data, context or {})

        try:
            results = await asyncio.gather(
                *[run_task(t) for t in self.tasks],
                return_exceptions=True,
            )

            # 处理异常
            outputs = []
            errors = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    errors.append(f"Task {i}: {r}")
                else:
                    outputs.append(r)

            result = PipelineResult(
                success=len(errors) == 0,
                output=outputs,
                duration_ms=(time.time() - start) * 1000,
                error="\n".join(errors) if errors else None,
            )

            return result

        except Exception as e:
            logger.error(f"ParallelPipeline执行失败: {e}")
            return PipelineResult(
                success=False,
                output=None,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# 需要导入asyncio
import asyncio
