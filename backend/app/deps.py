"""FastAPI dependencies: the signed-in user, and model loading (problems become 503s instead of 500s)."""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyCookie, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AuthSession, User
from app.rag.embeddings import Embedder, get_embedder
from app.rag.llm import LLM, get_llm
from app.security import token_hash

SESSION_COOKIE = "medrag_session"

# Declared so Swagger (/docs) offers an Authorize button; either one is accepted
_bearer = HTTPBearer(auto_error=False, description="The token returned by POST /api/auth/login")
_cookie = APIKeyCookie(name=SESSION_COOKIE, auto_error=False, description="Set by the web app on login")


def session_token(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
    cookie: str | None = Depends(_cookie),
) -> str | None:
    return bearer.credentials if bearer else cookie


def current_session(token: str | None = Depends(session_token), db: Session = Depends(get_db)) -> AuthSession:
    if token:
        session = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_hash(token)))
        if session is not None and session.expires_at > datetime.now(timezone.utc):
            return session
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Not signed in or the session has expired", headers={"WWW-Authenticate": "Bearer"}
    )


def current_user(session: AuthSession = Depends(current_session), db: Session = Depends(get_db)) -> User:
    return db.get(User, session.user_id)


def embedder_dep() -> Embedder:
    try:
        return get_embedder()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Embedding model unavailable: {exc}")


def llm_dep() -> LLM | None:
    try:
        return get_llm()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"LLM unavailable: {exc}")
