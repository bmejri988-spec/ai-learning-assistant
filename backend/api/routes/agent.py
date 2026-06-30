from fastapi import APIRouter

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/chat")
def chat() -> dict[str, str]:
    return {"status": "Coming soon"}
