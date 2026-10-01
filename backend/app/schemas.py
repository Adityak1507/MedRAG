from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    status: str
    error: str | None
    num_pages: int
    num_chunks: int
    created_at: datetime


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[int] | None = Field(
        default=None, description="Limit the search to these documents; omit to search all ready documents."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    source_id: int
    document_id: int
    filename: str
    page: int | None
    chunk_index: int
    similarity: float
    content: str


class QueryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    llm: str
    sources: list[SourceOut]
    created_at: datetime


class SystemInfo(BaseModel):
    llm: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_mb: int
    documents: int
    chunks: int
