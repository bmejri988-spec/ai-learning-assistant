from fastapi import APIRouter, Depends

from backend.api.utils.responses import success_response
from backend.modules.agent.agent import AgentService, get_agent_service
from backend.modules.agent.memory import ConversationTurn
from backend.modules.agent.schemas import AgentChatRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat")
def chat(
    payload: AgentChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> dict[str, object]:
    result = agent_service.chat(payload.message, payload.top_k)
    return success_response("Agent response generated", result)


@router.get("/history")
def get_history(
    agent_service: AgentService = Depends(get_agent_service),
) -> dict[str, object]:
    history = agent_service._memory.recall()
    history_data = [
        {
            "message": turn.message,
            "tool_name": turn.tool_name,
            "result": turn.result,
            "timestamp": turn.timestamp.isoformat(),
        }
        for turn in history
    ]
    return success_response("Conversation history retrieved", {"history": history_data})


@router.delete("/history")
def clear_history(
    agent_service: AgentService = Depends(get_agent_service),
) -> dict[str, object]:
    agent_service._memory.clear()
    return success_response("Conversation history cleared", {"history": []})
