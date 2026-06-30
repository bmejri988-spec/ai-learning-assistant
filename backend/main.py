import logging
import time

from fastapi import FastAPI, Request

from backend.api.routes.agent import router as agent_router
from backend.api.routes.image import router as image_router
from backend.api.routes.ml import router as ml_router
from backend.api.routes.rag import router as rag_router
from backend.api.routes.text import router as text_router
from backend.api.utils.exceptions import register_exception_handlers
from backend.api.utils.logging import configure_logging
from backend.api.utils.responses import success_response
from backend.config import PROJECT_NAME, PROJECT_VERSION

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title=PROJECT_NAME, version=PROJECT_VERSION)

app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(image_router)
app.include_router(text_router)
app.include_router(ml_router)
register_exception_handlers(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info("Request started: %s %s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed: %s %s -> %s in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def root() -> dict[str, object]:
    return success_response("AI Learning Assistant API", {"version": PROJECT_VERSION})


@app.get("/health")
def health() -> dict[str, object]:
    return success_response("healthy", {"status": "healthy"})
