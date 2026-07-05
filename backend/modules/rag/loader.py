from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def load_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages):
        pages.append(
            {
                "page_content": page.extract_text().strip(),
                "metadata": {"page": page_num, "source": str(pdf_path)},
            }
        )
    return pages


def load_pdf_text(pdf_path: Path) -> str:
    pages = load_pdf_pages(pdf_path)
    return "\n".join(page["page_content"] for page in pages).strip()