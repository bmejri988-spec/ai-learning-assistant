from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from backend.modules.agent.llm import get_agent_llm
from backend.modules.agent.planner import BasePlanner
from backend.modules.agent.schemas import PlanDecision

if TYPE_CHECKING:
    from backend.modules.agent.memory import ConversationTurn


PLANNER_SYSTEM_PROMPT = """You are a tool selection planner for an AI study assistant. 
Your ONLY job is to decide which tool to use based on the user's request.

Available tools:
1. search_docs - Answer a question using the RAG answer API
2. summarize - Summarize retrieved document chunks into a short study note
3. quiz - Create multiple-choice questions from retrieved notes
4. flashcards - Create question-answer flashcards from retrieved notes

Analyze the user's request and return ONLY a JSON object with this exact format:
{
  "tool": "tool_name",
    "confidence": 0.92,
  "reason": "brief explanation of why this tool was chosen"
}

Do NOT answer the user's question. Only select the tool."""


class LLMPlanner(BasePlanner):
    def __init__(self) -> None:
        self._llm = get_agent_llm()

    def select_tool(self, message: str, history: list[ConversationTurn] | None = None) -> PlanDecision:
        prompt = self._build_prompt(message, history)

        try:
            response = self._llm.answer(prompt)
            decision = self._parse_response(response)
            return decision
        except Exception:
            return PlanDecision(
                tool_name="search_docs",
                reason="LLM planner failed, defaulted to search_docs",
                confidence=0.0,
            )

    def _build_prompt(self, message: str, history: list[ConversationTurn] | None = None) -> str:
        prompt = PLANNER_SYSTEM_PROMPT + "\n\n"

        if history:
            context = "\n".join(
                f"Previous: {turn.message} -> {turn.tool_name}" 
                for turn in history[-3:]
            )
            prompt += f"Conversation context:\n{context}\n\n"
        
        prompt += f"User request: {message}\n\n"
        prompt += "Return ONLY the JSON object:"
        
        return prompt

    def _parse_response(self, response: str) -> PlanDecision:
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if not json_match:
            return PlanDecision(
                tool_name="search_docs",
                reason="Failed to parse LLM response, defaulted to search_docs",
                confidence=0.0,
            )
        
        try:
            data = json.loads(json_match.group())
            tool_name = data.get("tool", "search_docs")
            reason = data.get("reason", "LLM selected this tool")
            confidence = data.get("confidence", 0.0)

            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            
            valid_tools = ["search_docs", "summarize", "quiz", "flashcards"]
            if tool_name not in valid_tools:
                tool_name = "search_docs"
                reason = f"Invalid tool '{data.get('tool')}', defaulted to search_docs"
            
            return PlanDecision(tool_name=tool_name, reason=reason, confidence=confidence)
        except json.JSONDecodeError:
            return PlanDecision(
                tool_name="search_docs",
                reason="Failed to parse JSON, defaulted to search_docs",
                confidence=0.0,
            )
