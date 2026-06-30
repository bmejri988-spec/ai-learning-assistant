from fastapi import FastAPI

from backend.api.routes.agent import router as agent_router
from backend.api.routes.image import router as image_router
from backend.api.routes.ml import router as ml_router
from backend.api.routes.rag import router as rag_router
from backend.api.routes.text import router as text_router
from backend.config import PROJECT_NAME, PROJECT_VERSION

app = FastAPI(title=PROJECT_NAME, version=PROJECT_VERSION)

app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(image_router)
app.include_router(text_router)
app.include_router(ml_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Learning Assistant API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
