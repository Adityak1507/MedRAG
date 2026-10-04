import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, get_db
from app.deps import current_user, embedder_dep, llm_dep
from app.models import Chat, Query, User
from app.rag.embeddings import Embedder
from app.rag.llm import LLM
from app.rag.pipeline import Answer, RetrievedChunk, answer_question, prepare, retrieve, stream_answer
from app.routers.chats import get_own_chat, title_from_question
from app.schemas import QueryIn, QueryOut, SourceOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


def _sources(retrieved: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "source_id": i,
            "document_id": r.chunk.document_id,
            "filename": r.filename,
            "page": r.chunk.page,
            "chunk_index": r.chunk.chunk_index,
            "similarity": round(r.similarity, 4),
            "content": r.chunk.content,
        }
        for i, r in enumerate(retrieved, 1)
    ]


def _chat_for(db: Session, user: User, body: QueryIn, existing: Chat | None) -> Chat:
    """The chat named in the request, or a new one titled after the question. Called once retrieval has
    succeeded, so a failed request doesn't leave an empty chat behind."""
    if existing is not None:
        return existing
    chat = Chat(user_id=user.id, title=title_from_question(body.question))
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def _save(db: Session, user_id: int, chat_id: int, body: QueryIn, answer: Answer, sources: list[dict]) -> Query:
    record = Query(
        user_id=user_id,
        chat_id=chat_id,
        question=body.question,
        answer=answer.answer,
        llm=answer.llm,
        document_ids=body.document_ids,
        sources=sources,
    )
    db.add(record)
    chat = db.get(Chat, chat_id)
    if chat is not None:  # may have been deleted while the answer streamed
        chat.updated_at = func.now()
    db.commit()
    db.refresh(record)
    return record


@router.post("/search", response_model=list[SourceOut])
def search(
    body: QueryIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(embedder_dep),
):
    """The passages a question would retrieve from your documents, best first, without calling an LLM or
    saving anything."""
    top_k = body.top_k or get_settings().top_k
    return _sources(retrieve(db, embedder, body.question, top_k, body.document_ids, user.id))


@router.post("/query", response_model=QueryOut)
def ask(
    body: QueryIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(embedder_dep),
    llm: LLM | None = Depends(llm_dep),
):
    """Answer a question from your documents and save it to a chat (a new one unless chat_id is given)."""
    settings = get_settings()
    existing = get_own_chat(db, user, body.chat_id) if body.chat_id is not None else None
    answer = answer_question(
        db, embedder, llm, body.question, body.top_k or settings.top_k, body.document_ids, settings.min_similarity,
        user_id=user.id,
    )
    chat = _chat_for(db, user, body, existing)
    return _save(db, user.id, chat.id, body, answer, _sources(answer.retrieved))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent events"}},
)
def ask_stream(
    body: QueryIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(embedder_dep),
    llm: LLM | None = Depends(llm_dep),
):
    """Like POST /api/query, but streams server-sent events: `sources` (the chat id and the retrieved
    passages), then `token` events with pieces of the answer as they are generated, then `done` with the
    saved query (same shape as POST /api/query). `error` is sent instead of `done` if saving fails."""
    settings = get_settings()
    # The chat and retrieval are settled before the response starts, so errors are normal HTTP errors
    existing = get_own_chat(db, user, body.chat_id) if body.chat_id is not None else None
    prep = prepare(
        db, embedder, llm, body.question, body.top_k or settings.top_k, body.document_ids, settings.min_similarity,
        user_id=user.id,
    )
    chat = _chat_for(db, user, body, existing)
    sources = _sources(prep.retrieved)
    user_id, chat_id = user.id, chat.id

    def events() -> Iterator[str]:
        yield _sse("sources", {"chat_id": chat_id, "sources": sources})
        for event in stream_answer(prep, llm, body.question):
            if event.kind == "token":
                yield _sse("token", {"text": event.text})
            else:
                try:
                    with SessionLocal() as session:  # the request's session may be closed by now
                        record = _save(session, user_id, chat_id, body, event.answer, sources)
                        yield _sse("done", QueryOut.model_validate(record).model_dump(mode="json"))
                except Exception as exc:
                    logger.exception("Could not save streamed answer")
                    yield _sse("error", {"detail": f"Could not save the answer: {type(exc).__name__}"})

    # X-Accel-Buffering stops nginx from holding back the stream
    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/queries", response_model=list[QueryOut])
def query_history(
    response: Response,
    limit: int = QueryParam(20, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """All your questions across chats, newest first. The total count is in the X-Total-Count header."""
    mine = Query.user_id == user.id
    response.headers["X-Total-Count"] = str(db.scalar(select(func.count()).select_from(Query).where(mine)))
    return db.scalars(
        select(Query).where(mine).order_by(Query.created_at.desc(), Query.id.desc()).offset(offset).limit(limit)
    ).all()


@router.delete("/queries/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_query(query_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    record = db.get(Query, query_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Query not found")
    db.delete(record)
    db.commit()


@router.delete("/queries", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Delete all your chats, questions and answers. Your documents are kept."""
    db.execute(delete(Chat).where(Chat.user_id == user.id))
    db.execute(delete(Query).where(Query.user_id == user.id))
    db.commit()
