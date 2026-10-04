from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import SESSION_COOKIE, current_session, current_user
from app.models import AuthSession, Chat, Document, Query, User
from app.schemas import PASSWORD_MIN, AuthConfig, LoginIn, LoginOut, PasswordIn, RegisterIn, UserOut
from app.security import hash_password, login_throttle, new_token, token_hash, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

LEGACY_CHAT_TITLE = "Earlier questions"


def _start_session(db: Session, user: User, response: Response) -> LoginOut:
    settings = get_settings()
    token = new_token()
    expires = datetime.now(timezone.utc) + timedelta(hours=settings.session_ttl_hours)
    db.add(AuthSession(user_id=user.id, token_hash=token_hash(token), expires_at=expires))
    # Expired sessions are dropped whenever anyone signs in
    db.execute(delete(AuthSession).where(AuthSession.expires_at < datetime.now(timezone.utc)))
    db.commit()
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,  # not readable from JavaScript
        secure=settings.cookie_secure,
        samesite="lax",  # not sent on cross-site POSTs, which blocks CSRF
        path="/api",
    )
    return LoginOut(user=UserOut.model_validate(user), token=token, expires_at=expires)


def _claim_legacy_data(db: Session, user: User) -> None:
    """Give documents and questions from before accounts existed to the first user, so nothing is lost."""
    db.execute(update(Document).where(Document.user_id.is_(None)).values(user_id=user.id))
    if db.scalar(select(func.count()).select_from(Query).where(Query.user_id.is_(None))):
        latest = db.scalar(select(func.max(Query.created_at)).where(Query.user_id.is_(None)))
        chat = Chat(user_id=user.id, title=LEGACY_CHAT_TITLE, updated_at=latest)
        db.add(chat)
        db.flush()
        db.execute(update(Query).where(Query.user_id.is_(None)).values(user_id=user.id, chat_id=chat.id))


@router.get("/config", response_model=AuthConfig)
def auth_config():
    """Public: what the sign-in page should offer."""
    return AuthConfig(allow_registration=get_settings().allow_registration, password_min_length=PASSWORD_MIN)


@router.post("/register", response_model=LoginOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, response: Response, db: Session = Depends(get_db)):
    """Create an account and sign in."""
    if not get_settings().allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled. Ask an administrator for an account.")
    first_user = db.scalar(select(func.count()).select_from(User)) == 0
    user = User(email=body.email.lower(), name=(body.name or "").strip() or None, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    if first_user:
        _claim_legacy_data(db, user)
    db.commit()
    return _start_session(db, user, response)


@router.post("/login", response_model=LoginOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    settings = get_settings()
    email = body.email.lower()
    key = f"{email}|{request.client.host if request.client else ''}"
    if login_throttle.blocked(key, settings.login_max_failures, settings.login_window_minutes * 60):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many failed sign-in attempts. Try again in {settings.login_window_minutes} minutes.",
        )
    user = db.scalar(select(User).where(User.email == email))
    if not verify_password(body.password, user.password_hash if user else None):
        login_throttle.failed(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password")
    login_throttle.reset(key)
    return _start_session(db, user, response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, session: AuthSession = Depends(current_session), db: Session = Depends(get_db)):
    """End this session (other devices stay signed in)."""
    db.execute(delete(AuthSession).where(AuthSession.id == session.id))
    db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/api")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    body: PasswordIn, response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    """Permanently delete the account with all its documents, chats and sessions. Needs the password."""
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Wrong password")
    db.execute(delete(User).where(User.id == user.id))  # everything else cascades in the database
    db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/api")
