from fastapi import APIRouter

router = APIRouter(prefix="/text", tags=["text"])


@router.get("/predict")
def predict() -> dict[str, str]:
    return {"status": "Coming soon"}
