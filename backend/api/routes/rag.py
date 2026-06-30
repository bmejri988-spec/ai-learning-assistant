from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/ask")
def ask() -> dict[str, str]:
    return {"status": "Coming soon"}
