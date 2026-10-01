"""Text extraction for uploaded documents."""

from pathlib import Path

SUPPORTED_TYPES = {".pdf", ".docx", ".txt"}


class UnsupportedFileType(ValueError):
    pass


def extract_pages(path: Path, file_type: str) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. Page numbers are 1-based; None when the format has no pages."""
    if file_type == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    if file_type == ".docx":
        import docx2txt

        return [(None, docx2txt.process(str(path)) or "")]

    if file_type == ".txt":
        return [(None, path.read_bytes().decode("utf-8", errors="replace"))]

    raise UnsupportedFileType(f"Unsupported file type: {file_type}")
