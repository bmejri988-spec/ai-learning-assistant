from __future__ import annotations

from typing import TYPE_CHECKING

from backend.modules.agent.llm_planner import LLMPlanner
from backend.modules.agent.planner import AgentPlanner, BasePlanner
from backend.modules.agent.schemas import PlanDecision

if TYPE_CHECKING:
    from backend.modules.agent.memory import ConversationTurn


class HybridPlanner(BasePlanner):
    def __init__(
        self,
        llm_planner: BasePlanner | None = None,
        deterministic_planner: BasePlanner | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self._llm_planner = llm_planner or LLMPlanner()
        self._deterministic_planner = deterministic_planner or AgentPlanner()
        self._confidence_threshold = confidence_threshold

    def select_tool(self, message: str, history: list[ConversationTurn] | None = None) -> PlanDecision:
        llm_decision = self._llm_planner.select_tool(message, history)
        if self._should_fallback(llm_decision):
            fallback_decision = self._deterministic_planner.select_tool(message, history)
            return PlanDecision(
                tool_name=fallback_decision.tool_name,
                reason=f"Hybrid fallback used deterministic planner after: {llm_decision.reason}",
                confidence=fallback_decision.confidence,
            )

        return llm_decision

    def _should_fallback(self, decision: PlanDecision) -> bool:
        confidence = decision.confidence if decision.confidence is not None else 0.0
        return confidence < self._confidence_threshold