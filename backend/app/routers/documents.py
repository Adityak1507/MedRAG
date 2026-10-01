import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, get_db
from app.deps import embedder_dep
from app.models import Document
from app.rag.embeddings import Embedder
from app.rag.loaders import SUPPORTED_TYPES
from app.rag.pipeline import ingest_document
from app.schemas import DocumentOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

READ_CHUNK = 1024 * 1024


def _ingest_in_background(document_id: int, path: Path, embedder: Embedder) -> None:
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if document is not None:
                ingest_document(db, document, path, embedder, get_settings())
    finally:
        path.unlink(missing_ok=True)


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(embedder_dep),
):
    filename = Path(file.filename or "").name
    file_type = Path(filename).suffix.lower()
    if file_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{file_type or 'none'}'. Upload one of: {', '.join(sorted(SUPPORTED_TYPES))}",
        )

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    fd, tmp_name = tempfile.mkstemp(suffix=file_type)
    tmp_path = Path(tmp_name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(READ_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        413,  # Content Too Large
                        f"File is larger than {get_settings().max_upload_mb} MB",
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    document = Document(filename=filename[:255], file_type=file_type)
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(_ingest_in_background, document.id, tmp_path, embedder)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.created_at.desc(), Document.id.desc())).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    db.delete(document)
    db.commit()
