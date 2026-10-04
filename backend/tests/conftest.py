import hashlib
import os
import re

import pytest

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/medrag_test"
)
os.environ["PRELOAD_MODELS"] = "false"
os.environ["LLM_PROVIDERS"] = "none"
# The bag-of-words test embedder scores lower than real models; tests that need the cutoff set it
os.environ["MIN_SIMILARITY"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Base, engine, init_db  # noqa: E402
from app.deps import embedder_dep, llm_dep  # noqa: E402
from app.main import app  # noqa: E402
from app.security import login_throttle  # noqa: E402


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

    def stream(self, prompt):
        self.prompts.append(prompt)
        yield self.name, "Fake "
        yield self.name, "answer."


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


PASSWORD = "correct horse battery"


def register(test_client: TestClient, email: str, password: str = PASSWORD, name: str | None = None) -> dict:
    """Create an account; the client keeps the session cookie, so later requests are signed in."""
    resp = test_client.post("/api/auth/register", json={"email": email, "password": password, "name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
def anon_client(fake_llm, request):
    """A client that is not signed in."""
    embedder = FakeEmbedder(get_settings().embedding_dim)
    use_llm = request.node.get_closest_marker("no_llm") is None
    app.dependency_overrides[embedder_dep] = lambda: embedder
    app.dependency_overrides[llm_dep] = lambda: fake_llm if use_llm else None
    login_throttle.reset()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client(anon_client):
    """Signed in as alice@example.org."""
    register(anon_client, "alice@example.org", name="Alice")
    return anon_client


@pytest.fixture
def other_client(client):
    """Signed in as bob@example.org, with its own cookies (alice stays signed in on `client`)."""
    with TestClient(app) as bob:
        register(bob, "bob@example.org", name="Bob")
        yield bob


def pytest_configure(config):
    config.addinivalue_line("markers", "no_llm: run the request without an LLM (retrieval-only)")
