"""Text extraction for uploaded documents, with OCR for scanned PDF pages."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {".pdf", ".docx", ".txt"}

# A page with fewer non-whitespace characters than this is treated as scanned and sent to OCR
MIN_TEXT_CHARS = 20


class UnsupportedFileType(ValueError):
    pass


class OCRUnavailable(RuntimeError):
    pass


@dataclass
class Page:
    number: int | None  # 1-based; None when the format has no pages
    text: str
    ocr: bool = False


def extract_pages(
    path: Path, file_type: str, ocr_enabled: bool = True, ocr_language: str = "eng", ocr_dpi: int = 300
) -> list[Page]:
    if file_type == ".pdf":
        return _extract_pdf(path, ocr_enabled, ocr_language, ocr_dpi)

    if file_type == ".docx":
        import docx2txt

        return [Page(None, docx2txt.process(str(path)) or "")]

    if file_type == ".txt":
        return [Page(None, path.read_bytes().decode("utf-8", errors="replace"))]

    raise UnsupportedFileType(f"Unsupported file type: {file_type}")


def _has_text(text: str) -> bool:
    return len("".join(text.split())) >= MIN_TEXT_CHARS


def _extract_pdf(path: Path, ocr_enabled: bool, ocr_language: str, ocr_dpi: int) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [Page(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    scanned = [p for p in pages if not _has_text(p.text)]
    if not scanned or not ocr_enabled:
        return pages

    try:
        texts = ocr_pdf_pages(path, [p.number for p in scanned], ocr_language, ocr_dpi)
    except OCRUnavailable as exc:
        if len(scanned) == len(pages):
            raise ValueError(f"This PDF has no text layer and OCR is unavailable: {exc}") from exc
        logger.warning("Skipping OCR for %d scanned page(s) of %s: %s", len(scanned), path.name, exc)
        return pages

    for page in scanned:
        text = texts.get(page.number, "")
        if text.strip():
            page.text, page.ocr = text, True
    return pages


@lru_cache
def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_pdf_pages(path: Path, page_numbers: list[int], language: str, dpi: int) -> dict[int, str]:
    """Render the given 1-based pages and run Tesseract on them."""
    try:
        import pymupdf
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailable(f"missing Python package ({exc.name})") from exc
    if not ocr_available():
        raise OCRUnavailable("the tesseract binary is not installed")

    texts = {}
    with pymupdf.open(str(path)) as pdf:
        for number in page_numbers:
            pixmap = pdf[number - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
            texts[number] = pytesseract.image_to_string(image, lang=language)
    return texts
