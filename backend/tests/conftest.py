import hashlib
import os
import re

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/medrag_test"
)
os.environ["PRELOAD_MODELS"] = "false"
os.environ["LLM_PROVIDERS"] = "none"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Base, engine, init_db  # noqa: E402
from app.deps import embedder_dep, llm_dep  # noqa: E402
from app.main import app  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic and needs no model download."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimension
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FakeLLM:
    name = "fake:echo"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "Fake answer.", self.name


@pytest.fixture(scope="session", autouse=True)
def database():
    Base.metadata.drop_all(engine)
    init_db()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def clean_tables(database):
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def client(fake_llm, request):
    embedder = FakeEmbedder(get_settings().embedding_dim)
    use_llm = request.node.get_closest_marker("no_llm") is None
    app.dependency_overrides[embedder_dep] = lambda: embedder
    app.dependency_overrides[llm_dep] = lambda: fake_llm if use_llm else None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def pytest_configure(config):
    config.addinivalue_line("markers", "no_llm: run the request without an LLM (retrieval-only)")
