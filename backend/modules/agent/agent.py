from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from backend.modules.agent.memory import AgentMemoryStore
from backend.modules.agent.planner import AgentPlanner
from backend.modules.agent.tools import AgentTool, build_default_tools

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(
        self,
        planner: AgentPlanner | None = None,
        tools: dict[str, AgentTool] | None = None,
        memory: AgentMemoryStore | None = None,
    ) -> None:
        self._planner = planner or AgentPlanner()
        self._tools = tools or build_default_tools()
        self._memory = memory or AgentMemoryStore()

    def chat(self, message: str, top_k: int = 3) -> dict[str, Any]:
        start_time = time.perf_counter()
        history = self._memory.recall()
        decision = self._planner.select_tool(message, history)
        tool = self._tools[decision.tool_name]
        tool_start = time.perf_counter()
        try:
            result = tool.run(message, top_k=top_k, history=history)
            tool_seconds = time.perf_counter() - tool_start
            total_seconds = time.perf_counter() - start_time

            self._memory.remember(message, decision.tool_name, result)
            logger.info(
                "User Request=%s Selected Tool=%s Execution Time=%.2fms Success=%s",
                message,
                decision.tool_name,
                total_seconds * 1000,
                True,
            )

            return {
                "tool": decision.tool_name,
                "reason": decision.reason,
                "execution_time_ms": round(total_seconds * 1000, 2),
                "tool_execution_time_ms": round(tool_seconds * 1000, 2),
                "result": result,
            }
        except Exception:
            total_seconds = time.perf_counter() - start_time
            logger.exception(
                "User Request=%s Selected Tool=%s Execution Time=%.2fms Success=%s",
                message,
                decision.tool_name,
                total_seconds * 1000,
                False,
            )
            raise


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService()
