import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import init_db
from app.rag.embeddings import get_embedder
from app.routers import auth, chats, documents, query, system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medrag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if get_settings().preload_models:
        try:
            get_embedder()
            logger.info("Embedding model loaded")
        except Exception:
            # Keep serving; upload/query endpoints return 503 until the model can load
            logger.exception("Could not load the embedding model at startup")
    yield


app = FastAPI(
    title="MedRAG API",
    description="Retrieval-augmented question answering over uploaded medical documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,  # the session cookie
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)

app.include_router(system.router)
app.include_router(auth.router)
app.include_router(chats.router)
app.include_router(documents.router)
app.include_router(query.router)
