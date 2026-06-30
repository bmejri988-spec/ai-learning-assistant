from fastapi import APIRouter

from backend.api.utils.responses import success_response

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/chat")
def chat() -> dict[str, object]:
    return success_response("Coming soon")
