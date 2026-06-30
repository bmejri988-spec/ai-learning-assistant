from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.utils.responses import success_response
from backend.modules.rag.uploads import save_uploaded_pdf

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/ask")
def ask() -> dict[str, object]:
    return success_response("Coming soon")


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    saved_path = save_uploaded_pdf(file.filename, pdf_bytes)
    return success_response("PDF uploaded", {"file_name": saved_path.name, "saved_path": str(saved_path)})
