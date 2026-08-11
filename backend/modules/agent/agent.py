from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from backend.config import AGENT_PLANNER_TYPE
from backend.modules.agent.hybrid_planner import HybridPlanner
from backend.modules.agent.memory import AgentMemoryStore
from backend.modules.agent.planner import AgentPlanner, BasePlanner
from backend.modules.agent.tools import AgentTool, build_default_tools

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(
        self,
        planner: BasePlanner | None = None,
        tools: dict[str, AgentTool] | None = None,
        memory: AgentMemoryStore | None = None,
    ) -> None:
        if planner is None:
            planner = self._create_planner()
        self._planner = planner
        self._tools = tools or build_default_tools()
        self._memory = memory or AgentMemoryStore()

    def _create_planner(self) -> BasePlanner:
        if AGENT_PLANNER_TYPE == "llm":
            from backend.modules.agent.llm_planner import LLMPlanner
            return LLMPlanner()
        if AGENT_PLANNER_TYPE == "hybrid":
            return HybridPlanner()
        return AgentPlanner()

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
                "thinking": self._build_thinking(decision, result, total_seconds, tool_seconds),
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

    def _build_thinking(
        self,
        decision,
        result: dict[str, Any],
        total_seconds: float,
        tool_seconds: float,
    ) -> dict[str, Any]:
        generation_mode = result.get("generation_mode", "unknown")
        generation_reason = result.get("generation_reason", "Not provided by tool")

        return {
            "planner": {
                "selected_tool": decision.tool_name,
                "reason": decision.reason,
                "confidence": decision.confidence,
            },
            "generation": {
                "mode": generation_mode,
                "reason": generation_reason,
            },
            "timing": {
                "total_ms": round(total_seconds * 1000, 2),
                "tool_ms": round(tool_seconds * 1000, 2),
            },
        }


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService()
