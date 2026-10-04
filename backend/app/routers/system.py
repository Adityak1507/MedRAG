from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import current_user
from app.models import Chunk, Document, User
from app.rag.llm import get_llm
from app.rag.loaders import ocr_available
from app.schemas import SystemInfo

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@router.get("/info", response_model=SystemInfo)
def info(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Active configuration, and counts of your documents and chunks."""
    settings = get_settings()
    try:
        llm = get_llm()
        llm_name = llm.name if llm else "retrieval_only"
    except Exception as exc:
        llm_name = f"misconfigured: {exc}"
    return SystemInfo(
        llm=llm_name,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        ocr=settings.ocr_enabled and ocr_available(),
        max_upload_mb=settings.max_upload_mb,
        documents=db.scalar(select(func.count()).select_from(Document).where(Document.user_id == user.id)),
        chunks=db.scalar(
            select(func.count()).select_from(Chunk).join(Document).where(Document.user_id == user.id)
        ),
    )
