from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "page_content": text,
                "metadata": {
                    "document_name": pdf_path.name,
                    "page": page_number,
                    "source": str(pdf_path),
                },
            }
        )

    return pages


def load_pdf_text(pdf_path: Path) -> str:
    pages = load_pdf_pages(pdf_path)
    return "\n\n".join(page["page_content"] for page in pages)