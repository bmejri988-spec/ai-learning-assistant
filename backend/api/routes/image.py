from fastapi import APIRouter

router = APIRouter(prefix="/image", tags=["image"])


@router.get("/predict")
def predict() -> dict[str, str]:
    return {"status": "Coming soon"}
