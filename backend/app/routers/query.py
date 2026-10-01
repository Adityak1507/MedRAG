from fastapi import APIRouter, Depends, Query as QueryParam
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import embedder_dep, llm_dep
from app.models import Query
from app.rag.embeddings import Embedder
from app.rag.llm import LLM
from app.rag.pipeline import answer_question
from app.schemas import QueryIn, QueryOut

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryOut)
def ask(
    body: QueryIn,
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(embedder_dep),
    llm: LLM | None = Depends(llm_dep),
):
    result = answer_question(
        db,
        embedder,
        llm,
        body.question,
        body.top_k or get_settings().top_k,
        body.document_ids,
    )
    record = Query(
        question=body.question,
        answer=result.answer,
        llm=result.llm,
        document_ids=body.document_ids,
        sources=[
            {
                "source_id": i,
                "document_id": r.chunk.document_id,
                "filename": r.filename,
                "page": r.chunk.page,
                "chunk_index": r.chunk.chunk_index,
                "similarity": round(r.similarity, 4),
                "content": r.chunk.content,
            }
            for i, r in enumerate(result.retrieved, 1)
        ],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/queries", response_model=list[QueryOut])
def query_history(limit: int = QueryParam(20, ge=1, le=100), db: Session = Depends(get_db)):
    return db.scalars(select(Query).order_by(Query.created_at.desc(), Query.id.desc()).limit(limit)).all()
