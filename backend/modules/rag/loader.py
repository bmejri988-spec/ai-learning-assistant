from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    return [
        {
            "page_content": page.page_content.strip(),
            "metadata": dict(page.metadata),
        }
        for page in pages
    ]


def load_pdf_text(pdf_path: Path) -> str:
    pages = load_pdf_pages(pdf_path)
    return "\n".join(page["page_content"] for page in pages).strip()