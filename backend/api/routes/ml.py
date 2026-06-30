from fastapi import APIRouter

router = APIRouter(prefix="/ml", tags=["ml"])


@router.get("/predict")
def predict() -> dict[str, str]:
    return {"status": "Coming soon"}
