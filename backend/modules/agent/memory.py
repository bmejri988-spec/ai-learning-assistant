from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone


@dataclass(slots=True)
class ConversationTurn:
    message: str
    tool_name: str
    result: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class AgentMemoryStore:
    history: list[ConversationTurn] = field(default_factory=list)
    max_history: int = 10

    def remember(self, message: str, tool_name: str, result: dict[str, Any]) -> None:
        turn = ConversationTurn(message=message, tool_name=tool_name, result=result)
        self.history.append(turn)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def recall(self, limit: int | None = None) -> list[ConversationTurn]:
        if limit is None or limit >= len(self.history):
            return list(self.history)
        return list(self.history[-limit:])

    def get_conversation_context(self, limit: int | None = None) -> str:
        turns = self.recall(limit)
        if not turns:
            return ""
        
        context_parts = []
        for turn in turns:
            context_parts.append(f"User: {turn.message}")
            context_parts.append(f"Tool: {turn.tool_name}")
        
        return "\n".join(context_parts)

    def clear(self) -> None:
        self.history.clear()
