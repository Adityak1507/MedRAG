from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

PASSWORD_MIN = 8


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    status: str
    error: str | None
    num_pages: int
    num_chunks: int
    ocr_pages: int
    created_at: datetime


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: int | None = Field(default=None, description="Add to this chat; omit to start a new one.")
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
    chat_id: int | None
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
    min_similarity: float
    ocr: bool
    max_upload_mb: int
    documents: int
    chunks: int


# --- accounts


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=PASSWORD_MIN, max_length=200)
    name: str | None = Field(default=None, max_length=100)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=200)


class PasswordIn(BaseModel):
    password: str = Field(max_length=200)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str | None
    created_at: datetime


class LoginOut(BaseModel):
    user: UserOut
    token: str = Field(description="Also set as an httpOnly cookie; API clients send it as 'Authorization: Bearer'.")
    expires_at: datetime


class AuthConfig(BaseModel):
    allow_registration: bool
    password_min_length: int


# --- chats


class ChatIn(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
