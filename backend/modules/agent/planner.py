from __future__ import annotations

from typing import TYPE_CHECKING

from backend.modules.agent.schemas import PlanDecision

if TYPE_CHECKING:
    from backend.modules.agent.memory import ConversationTurn


class AgentPlanner:
    def select_tool(self, message: str, history: list[ConversationTurn] | None = None) -> PlanDecision:
        normalized_message = message.lower()
        
        context = ""
        if history:
            last_turn = history[-1]
            context = last_turn.message.lower()

        if any(keyword in normalized_message for keyword in ["quiz", "multiple choice", "mcq"]):
            return PlanDecision(tool_name="quiz", reason="Matched quiz keywords")

        if any(keyword in normalized_message for keyword in ["flashcard", "flash cards", "flash cards", "cards"]):
            return PlanDecision(tool_name="flashcards", reason="Matched flashcard keywords")

        if any(keyword in normalized_message for keyword in ["summarize", "summary", "summarise"]):
            return PlanDecision(tool_name="summarize", reason="Matched summarize keywords")

        if context and any(ref in normalized_message for ref in ["it", "that", "the above", "this", "them"]):
            if history and history[-1].tool_name in ["summarize", "quiz", "flashcards"]:
                return PlanDecision(
                    tool_name=history[-1].tool_name,
                    reason=f"Referenced previous context with tool {history[-1].tool_name}"
                )

        return PlanDecision(tool_name="search_docs", reason="Defaulted to document search")
