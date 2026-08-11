from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from backend.config import UPLOADS_DIR


def save_uploaded_pdf(filename: str, pdf_bytes: bytes) -> Path:
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    uploads_dir = Path(UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(filename).name
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()

    safe_filename = f"{stem}_{uuid4().hex[:12]}{suffix}"
    target_path = uploads_dir / safe_filename

    target_path.write_bytes(pdf_bytes)

    return target_path