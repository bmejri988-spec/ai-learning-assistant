from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User request for the agent")
    top_k: int = Field(default=3, ge=1, le=10, description="How many documents to retrieve")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: str
    output_schema: str


@dataclass(frozen=True, slots=True)
class PlanDecision:
    tool_name: Literal["search_docs", "summarize", "quiz", "flashcards"]
    reason: str
    confidence: float | None = None
