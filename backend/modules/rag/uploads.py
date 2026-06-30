from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from backend.config import UPLOADS_DIR


def save_uploaded_pdf(filename: str, pdf_bytes: bytes) -> Path:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    uploads_dir = Path(UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target_path = uploads_dir / Path(filename).name
    target_path.write_bytes(pdf_bytes)
    return target_path