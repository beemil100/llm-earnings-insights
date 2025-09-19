import fitz  # PyMuPDF
from typing import List, Tuple


def extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Return list of (page_number, text) for each page in PDF.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append((i, text))
    return pages
