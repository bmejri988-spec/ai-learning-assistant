from fastapi import APIRouter

from backend.api.utils.responses import success_response

router = APIRouter(prefix="/image", tags=["image"])


@router.get("/predict")
def predict() -> dict[str, object]:
    return success_response("Coming soon")
