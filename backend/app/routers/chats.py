from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import current_user
from app.models import Chat, Query, User
from app.schemas import ChatIn, ChatOut, ChatUpdate, QueryOut

router = APIRouter(prefix="/api/chats", tags=["chats"])

NEW_CHAT_TITLE = "New chat"


def get_own_chat(db: Session, user: User, chat_id: int) -> Chat:
    """The user's chat, or 404 (also for other users' chats, so ids can't be probed)."""
    chat = db.get(Chat, chat_id)
    if chat is None or chat.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    return chat


def title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 80 else title[:77].rstrip() + "…"


@router.get("", response_model=list[ChatOut])
def list_chats(
    response: Response,
    limit: int = QueryParam(30, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Your chats, most recently active first. The total count is in the X-Total-Count header."""
    mine = Chat.user_id == user.id
    response.headers["X-Total-Count"] = str(db.scalar(select(func.count()).select_from(Chat).where(mine)))
    return db.scalars(
        select(Chat).where(mine).order_by(Chat.updated_at.desc(), Chat.id.desc()).offset(offset).limit(limit)
    ).all()


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
def create_chat(body: ChatIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    chat = Chat(user_id=user.id, title=(body.title or "").strip() or NEW_CHAT_TITLE)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/{chat_id}", response_model=ChatOut)
def get_chat(chat_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return get_own_chat(db, user, chat_id)


@router.patch("/{chat_id}", response_model=ChatOut)
def rename_chat(chat_id: int, body: ChatUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    chat = get_own_chat(db, user, chat_id)
    chat.title = body.title.strip()
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(chat_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Delete the chat and its questions and answers."""
    db.delete(get_own_chat(db, user, chat_id))
    db.commit()


@router.get("/{chat_id}/messages", response_model=list[QueryOut])
def chat_messages(
    chat_id: int,
    response: Response,
    limit: int = QueryParam(20, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """The chat's questions and answers, newest first. The total count is in the X-Total-Count header."""
    get_own_chat(db, user, chat_id)
    in_chat = Query.chat_id == chat_id
    response.headers["X-Total-Count"] = str(db.scalar(select(func.count()).select_from(Query).where(in_chat)))
    return db.scalars(
        select(Query).where(in_chat).order_by(Query.created_at.desc(), Query.id.desc()).offset(offset).limit(limit)
    ).all()
