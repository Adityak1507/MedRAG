from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Enable pgvector and create tables that don't exist yet."""
    from app import models  # noqa: F401  (registers the tables on Base)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Upgrades for databases created by earlier versions (create_all never alters existing tables)
        conn.execute(text("ALTER TABLE queries ALTER COLUMN llm TYPE varchar(255)"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_pages integer NOT NULL DEFAULT 0"))
        # Accounts: owner columns on tables that predate them (existing rows stay NULL until claimed)
        for table, column, target in (
            ("documents", "user_id", "users"),
            ("queries", "user_id", "users"),
            ("queries", "chat_id", "chats"),
        ):
            conn.execute(text(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} integer "
                f"REFERENCES {target}(id) ON DELETE CASCADE"
            ))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{column} ON {table} ({column})"))


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
