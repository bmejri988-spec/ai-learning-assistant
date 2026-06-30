from fastapi import APIRouter

from backend.api.utils.responses import success_response

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/ask")
def ask() -> dict[str, object]:
    return success_response("Coming soon")
