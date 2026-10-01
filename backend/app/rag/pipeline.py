"""Ingestion (extract -> chunk -> embed -> store) and question answering."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import STATUS_FAILED, STATUS_READY, Chunk, Document
from app.rag.embeddings import Embedder
from app.rag.llm import LLM
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
        pages = extract_pages(path, document.file_type)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        pieces: list[tuple[int | None, str]] = []
        for page, text in pages:
            text = text.replace("\x00", "")  # Postgres text cannot hold NUL bytes
            pieces.extend((page, piece) for piece in splitter.split_text(text) if piece.strip())

        if not pieces:
            raise ValueError("No extractable text found (scanned PDFs need OCR first).")

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
    db: Session, embedder: Embedder, question: str, top_k: int, document_ids: list[int] | None
) -> list[RetrievedChunk]:
    query_vector = embedder.embed([question])[0]
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == STATUS_READY)
        .order_by(distance)
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    return [
        RetrievedChunk(chunk=chunk, filename=filename, similarity=1.0 - float(dist))
        for chunk, filename, dist in db.execute(stmt)
    ]


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


def answer_question(
    db: Session,
    embedder: Embedder,
    llm: LLM | None,
    question: str,
    top_k: int,
    document_ids: list[int] | None = None,
) -> Answer:
    retrieved = retrieve(db, embedder, question, top_k, document_ids)
    if not retrieved:
        return Answer(NO_ANSWER, llm.name if llm else "retrieval_only", [])

    context = "\n\n".join(f"[Source {i}]: {r.chunk.content}" for i, r in enumerate(retrieved, 1))

    if llm is None:
        return Answer(summarize_context(context, question), "retrieval_only", retrieved)

    prompt = PROMPT_TEMPLATE.format(no_answer=NO_ANSWER, context=context, question=question)
    try:
        return Answer(llm.generate(prompt), llm.name, retrieved)
    except Exception as exc:
        logger.exception("LLM generation failed")
        fallback = f"The language model failed ({type(exc).__name__}). " + summarize_context(context, question)
        return Answer(fallback, f"{llm.name} (failed)", retrieved)
