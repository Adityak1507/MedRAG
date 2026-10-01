from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Chunk, Document
from app.rag.llm import get_llm
from app.schemas import SystemInfo

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@router.get("/info", response_model=SystemInfo)
def info(db: Session = Depends(get_db)):
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
        max_upload_mb=settings.max_upload_mb,
        documents=db.scalar(select(func.count()).select_from(Document)),
        chunks=db.scalar(select(func.count()).select_from(Chunk)),
    )
