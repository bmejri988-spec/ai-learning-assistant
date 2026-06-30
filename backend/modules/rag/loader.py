from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_text(pdf_path: Path) -> str:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    return "\n".join(page.page_content for page in pages).strip()