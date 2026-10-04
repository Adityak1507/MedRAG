"""Ingestion (extract/OCR -> chunk -> embed -> store) and question answering, blocking or streamed."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import STATUS_FAILED, STATUS_READY, Chunk, Document
from app.rag.embeddings import Embedder
from app.rag.llm import LLM, AllProvidersFailed
from app.rag.loaders import extract_pages

logger = logging.getLogger(__name__)

NO_ANSWER = "I cannot answer this question based on the provided medical document."

PROMPT_TEMPLATE = """You are a medical assistant. Answer the question based ONLY on the provided medical context.
Do not use any external knowledge. If the answer cannot be found in the context, say "{no_answer}"

Context:
{context}

Question: {question}

Answer: """

EMBED_BATCH = 128


def ingest_document(
    db: Session, document: Document, path: Path, embedder: Embedder, settings: Settings
) -> None:
    """Extract, chunk and embed a stored Document row, marking it ready or failed."""
    try:
        pages = extract_pages(
            path, document.file_type, settings.ocr_enabled, settings.ocr_language, settings.ocr_dpi
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        pieces: list[tuple[int | None, str]] = []
        for page in pages:
            text = page.text.replace("\x00", "")  # Postgres text cannot hold NUL bytes
            pieces.extend((page.number, piece) for piece in splitter.split_text(text) if piece.strip())

        if not pieces:
            raise ValueError("No extractable text found, even after OCR.")

        for start in range(0, len(pieces), EMBED_BATCH):
            batch = pieces[start:start + EMBED_BATCH]
            vectors = embedder.embed([content for _, content in batch])
            db.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_index=start + i,
                    page=page,
                    content=content,
                    embedding=vector,
                )
                for i, ((page, content), vector) in enumerate(zip(batch, vectors))
            )

        document.num_pages = len(pages)
        document.num_chunks = len(pieces)
        document.ocr_pages = sum(page.ocr for page in pages)
        document.status = STATUS_READY
        document.error = None
        db.commit()
    except Exception as exc:
        logger.exception("Ingestion failed for document %s", document.id)
        db.rollback()
        try:
            document.status = STATUS_FAILED
            document.error = str(exc)[:1000]
            db.commit()
        except Exception:  # e.g. the document was deleted while processing
            db.rollback()
            logger.warning("Could not record failure for document %s", document.id)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    filename: str
    similarity: float


def retrieve(
    db: Session,
    embedder: Embedder,
    question: str,
    top_k: int,
    document_ids: list[int] | None,
    user_id: int,
) -> list[RetrievedChunk]:
    """The top_k chunks closest to the question, from the user's own ready documents only."""
    query_vector = embedder.embed([question])[0]
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.user_id == user_id, Document.status == STATUS_READY)
        .order_by(distance)
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    # The HNSW index finds nearest neighbours before the user/document filter is applied; iterative scans
    # (pgvector >= 0.8) keep searching until top_k rows pass the filter instead of returning too few
    db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = sorted(db.execute(stmt), key=lambda row: row.distance)  # relaxed order: re-sort exactly
    return [RetrievedChunk(chunk=chunk, filename=filename, similarity=1.0 - float(dist)) for chunk, filename, dist in rows]


def summarize_context(context: str, question: str) -> str:
    """Pick context lines sharing keywords with the question, used when no LLM answer is available."""
    stop_words = {'the', 'and', 'for', 'are', 'what', 'which', 'how', 'why', 'who',
                  'when', 'does', 'with', 'this', 'that', 'from', 'into', 'about'}
    query_words = {w for w in re.findall(r"[a-z0-9]+", question.lower())
                   if len(w) > 2 and w not in stop_words}

    relevant = [
        line.strip()
        for line in context.split("\n")
        if line.strip() and query_words & set(re.findall(r"[a-z0-9]+", line.lower()))
    ]
    if relevant:
        return "Most relevant excerpts:\n\n" + "\n\n".join(
            f"{i}. {line}" for i, line in enumerate(relevant[:5], 1)
        )
    return context[:500] + "..." if len(context) > 500 else context


@dataclass
class Answer:
    answer: str
    llm: str
    retrieved: list[RetrievedChunk]


# Labels stored in Query.llm when no model produced the answer
NO_LLM = "retrieval_only"
LOW_SIMILARITY = "similarity_cutoff"
ALL_FAILED = "all LLMs failed"


@dataclass
class Prepared:
    """Retrieval is done: either the final answer (no LLM call needed) or the prompt to send."""

    retrieved: list[RetrievedChunk]
    context: str = ""
    prompt: str = ""
    final: Answer | None = None


def prepare(
    db: Session,
    embedder: Embedder,
    llm: LLM | None,
    question: str,
    top_k: int,
    document_ids: list[int] | None = None,
    min_similarity: float = 0.0,
    *,
    user_id: int,
) -> Prepared:
    retrieved = retrieve(db, embedder, question, top_k, document_ids, user_id)
    if not retrieved:
        return Prepared([], final=Answer(NO_ANSWER, llm.name if llm else NO_LLM, []))

    # Nothing in the documents is close to the question: refuse without spending an LLM call
    if retrieved[0].similarity < min_similarity:
        return Prepared(retrieved, final=Answer(NO_ANSWER, LOW_SIMILARITY, retrieved))

    context = "\n\n".join(f"[Source {i}]: {r.chunk.content}" for i, r in enumerate(retrieved, 1))
    if llm is None:
        return Prepared(retrieved, context, final=Answer(summarize_context(context, question), NO_LLM, retrieved))

    prompt = PROMPT_TEMPLATE.format(no_answer=NO_ANSWER, context=context, question=question)
    return Prepared(retrieved, context, prompt)


def answer_question(
    db: Session,
    embedder: Embedder,
    llm: LLM | None,
    question: str,
    top_k: int,
    document_ids: list[int] | None = None,
    min_similarity: float = 0.0,
    *,
    user_id: int,
) -> Answer:
    prep = prepare(db, embedder, llm, question, top_k, document_ids, min_similarity, user_id=user_id)
    if prep.final is not None:
        return prep.final
    try:
        text, used = llm.generate(prep.prompt)
        return Answer(text, used, prep.retrieved)
    except Exception:
        logger.exception("LLM generation failed")
        fallback = "All language models failed. " + summarize_context(prep.context, question)
        return Answer(fallback, ALL_FAILED, prep.retrieved)


@dataclass
class StreamEvent:
    kind: str  # "token": a piece of the answer; "final": the complete Answer
    text: str = ""
    answer: Answer | None = None


def stream_answer(prep: Prepared, llm: LLM | None, question: str) -> Iterator[StreamEvent]:
    """Yield answer tokens as the LLM produces them, then one final event with the complete Answer."""
    if prep.final is not None:
        yield StreamEvent("token", prep.final.answer)
        yield StreamEvent("final", answer=prep.final)
        return

    parts: list[str] = []
    used = None
    try:
        for used, piece in llm.stream(prep.prompt):
            parts.append(piece)
            yield StreamEvent("token", piece)
        answer = Answer("".join(parts).strip(), used, prep.retrieved)
    except AllProvidersFailed:
        logger.exception("LLM generation failed")
        text = "All language models failed. " + summarize_context(prep.context, question)
        yield StreamEvent("token", text)
        answer = Answer(text, ALL_FAILED, prep.retrieved)
    except Exception:
        # The provider broke mid-answer; keep what arrived and say so rather than restarting on another model
        logger.exception("LLM stream interrupted")
        note = "\n\n[The answer was interrupted. Relevant excerpts:]\n" + summarize_context(prep.context, question)
        yield StreamEvent("token", note)
        answer = Answer("".join(parts).strip() + note, f"{used} (interrupted)", prep.retrieved)
    yield StreamEvent("final", answer=answer)
