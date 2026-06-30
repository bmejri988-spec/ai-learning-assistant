from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.utils.responses import success_response
from backend.modules.rag.service import RagIngestionService, get_rag_ingestion_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/ask")
def ask() -> dict[str, object]:
    return success_response("Coming soon")


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    service: RagIngestionService = Depends(get_rag_ingestion_service),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    result = service.ingest_pdf(file.filename, pdf_bytes)
    return success_response("PDF indexed", result)
