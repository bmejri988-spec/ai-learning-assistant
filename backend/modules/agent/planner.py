from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend.modules.agent.schemas import PlanDecision

if TYPE_CHECKING:
    from backend.modules.agent.memory import ConversationTurn


class BasePlanner(ABC):
    @abstractmethod
    def select_tool(self, message: str, history: list[ConversationTurn] | None = None) -> PlanDecision:
        pass


class AgentPlanner(BasePlanner):
    def select_tool(self, message: str, history: list[ConversationTurn] | None = None) -> PlanDecision:
        normalized_message = message.lower()
        
        context = ""
        if history:
            last_turn = history[-1]
            context = last_turn.message.lower()

        if any(keyword in normalized_message for keyword in ["quiz", "multiple choice", "mcq"]):
            return PlanDecision(tool_name="quiz", reason="Matched quiz keywords", confidence=1.0)

        if any(keyword in normalized_message for keyword in ["flashcard", "flash cards", "flash cards", "cards"]):
            return PlanDecision(tool_name="flashcards", reason="Matched flashcard keywords", confidence=1.0)

        if any(keyword in normalized_message for keyword in ["summarize", "summary", "summarise"]):
            return PlanDecision(tool_name="summarize", reason="Matched summarize keywords", confidence=1.0)

        if context and any(ref in normalized_message for ref in ["it", "that", "the above", "this", "them"]):
            if history and history[-1].tool_name in ["summarize", "quiz", "flashcards"]:
                return PlanDecision(
                    tool_name=history[-1].tool_name,
                    reason=f"Referenced previous context with tool {history[-1].tool_name}",
                    confidence=0.9,
                )

        return PlanDecision(tool_name="search_docs", reason="Defaulted to document search", confidence=0.6)
