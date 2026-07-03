from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api.utils.responses import success_response
from backend.modules.rag.pipeline import RagAnswerPipeline, RagPipeline, get_rag_answer_pipeline, get_rag_pipeline
from backend.modules.rag.uploads import save_uploaded_pdf
from backend.modules.rag.retriever import RagRetriever, get_rag_retriever

router = APIRouter(prefix="/rag", tags=["rag"])


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, description="Question to search against the knowledge base")
    top_k: int = Field(default=3, ge=1, le=10)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, description="Question to answer from the knowledge base")
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/ask")
def ask(
    payload: AnswerRequest,
    answer_pipeline: RagAnswerPipeline = Depends(get_rag_answer_pipeline),
) -> dict[str, object]:
    result = answer_pipeline.answer_question(payload.question, payload.top_k)
    return success_response("Answer generated", result)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    saved_path = save_uploaded_pdf(file.filename, pdf_bytes)
    index_result = pipeline.index_pdf(saved_path)
    return success_response(
        "PDF uploaded and indexed",
        {
            "file_name": saved_path.name,
            "saved_path": str(saved_path),
            **index_result,
        },
    )


@router.post("/retrieve")
def retrieve_documents(
    payload: RetrievalRequest,
    retriever: RagRetriever = Depends(get_rag_retriever),
) -> dict[str, object]:
    result = retriever.retrieve(payload.query, payload.top_k)
    return success_response("Retrieved documents", result)
