"""Accounts, sessions, per-user data isolation, chats and their pagination."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update

from app.config import get_settings
from app.db import SessionLocal, engine
from app.main import app
from app.models import AuthSession, Chat, Document, Query
from tests.conftest import PASSWORD, register
from tests.test_api import ASTHMA_TEXT, CFS_TEXT, upload

PROTECTED = [
    ("GET", "/api/info"),
    ("GET", "/api/documents"),
    ("POST", "/api/documents"),
    ("GET", "/api/documents/1"),
    ("DELETE", "/api/documents/1"),
    ("POST", "/api/query"),
    ("POST", "/api/query/stream"),
    ("POST", "/api/search"),
    ("GET", "/api/queries"),
    ("DELETE", "/api/queries"),
    ("GET", "/api/chats"),
    ("POST", "/api/chats"),
    ("GET", "/api/chats/1/messages"),
    ("GET", "/api/auth/me"),
    ("POST", "/api/auth/logout"),
]


# --- registration, sign-in and sessions


@pytest.mark.parametrize("method,path", PROTECTED)
def test_endpoints_require_sign_in(anon_client, method, path):
    resp = anon_client.request(method, path, json={"question": "q"})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_health_and_auth_config_are_public(anon_client):
    assert anon_client.get("/api/health").status_code == 200
    assert anon_client.get("/api/auth/config").json() == {"allow_registration": True, "password_min_length": 8}


def test_register_signs_in_with_an_httponly_cookie(anon_client):
    body = register(anon_client, "Carol@Example.org", name="  Carol ")
    assert body["user"]["email"] == "carol@example.org"
    assert body["user"]["name"] == "Carol"
    cookie = next(c for c in anon_client.cookies.jar if c.name == "medrag_session")
    assert cookie.value == body["token"]
    assert cookie.path == "/api"
    assert cookie.has_nonstandard_attr("HttpOnly")
    assert anon_client.get("/api/auth/me").json()["email"] == "carol@example.org"


def test_register_validation(anon_client):
    register(anon_client, "dave@example.org")
    duplicate = anon_client.post("/api/auth/register", json={"email": "DAVE@example.org", "password": PASSWORD})
    assert duplicate.status_code == 409
    for body in ({"email": "not-an-email", "password": PASSWORD}, {"email": "e@example.org", "password": "short"}):
        assert anon_client.post("/api/auth/register", json=body).status_code == 422


def test_registration_can_be_disabled(anon_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "allow_registration", False)
    assert anon_client.get("/api/auth/config").json()["allow_registration"] is False
    resp = anon_client.post("/api/auth/register", json={"email": "e@example.org", "password": PASSWORD})
    assert resp.status_code == 403


def test_login_logout_and_bearer_tokens(client):
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401

    assert client.post("/api/auth/login", json={"email": "alice@example.org", "password": "wrong password"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "nobody@example.org", "password": PASSWORD}).status_code == 401
    login = client.post("/api/auth/login", json={"email": "ALICE@example.org", "password": PASSWORD})
    assert login.status_code == 200
    token = login.json()["token"]

    # API clients use the same token as a Bearer header, without cookies
    with TestClient(app) as api_client:
        headers = {"Authorization": f"Bearer {token}"}
        assert api_client.get("/api/auth/me", headers=headers).json()["email"] == "alice@example.org"
        assert api_client.get("/api/auth/me", headers={"Authorization": "Bearer made-up"}).status_code == 401
        # Signing out ends the session for every client using it
        assert api_client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_only_a_hash_of_the_token_is_stored(client):
    token = client.post("/api/auth/login", json={"email": "alice@example.org", "password": PASSWORD}).json()["token"]
    with SessionLocal() as db:
        stored = db.scalars(select(AuthSession.token_hash)).all()
    assert token not in stored and all(len(h) == 64 for h in stored)


def test_expired_sessions_are_rejected(client):
    with engine.begin() as conn:
        conn.execute(update(AuthSession).values(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1)))
    assert client.get("/api/auth/me").status_code == 401


def test_repeated_failed_logins_are_throttled(client):
    bad = {"email": "alice@example.org", "password": "wrong password"}
    for _ in range(get_settings().login_max_failures):
        assert client.post("/api/auth/login", json=bad).status_code == 401
    # Even the right password is refused until the window passes
    resp = client.post("/api/auth/login", json={"email": "alice@example.org", "password": PASSWORD})
    assert resp.status_code == 429
    # Other accounts are unaffected
    register(client, "erin@example.org")
    assert client.post("/api/auth/login", json={"email": "erin@example.org", "password": PASSWORD}).status_code == 200


def test_delete_account_removes_all_its_data(client, other_client):
    upload(client, "cfs.txt", CFS_TEXT)
    client.post("/api/query", json={"question": "core symptoms"})
    upload(other_client, "asthma.txt", ASTHMA_TEXT)

    assert client.request("DELETE", "/api/auth/me", json={"password": "wrong password"}).status_code == 403
    assert client.request("DELETE", "/api/auth/me", json={"password": PASSWORD}).status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"email": "alice@example.org", "password": PASSWORD}).status_code == 401

    with SessionLocal() as db:
        assert db.scalars(select(Document.filename)).all() == ["asthma.txt"]  # bob's document survives
        assert db.scalars(select(Query)).all() == []
        assert db.scalars(select(Chat)).all() == []
        assert db.scalars(select(AuthSession.user_id)).all() != []  # bob is still signed in
    assert other_client.get("/api/auth/me").status_code == 200


# --- every user sees only their own data


def test_documents_are_private(client, other_client):
    alice_doc = upload(client, "cfs.txt", CFS_TEXT).json()["id"]
    upload(other_client, "asthma.txt", ASTHMA_TEXT)

    assert [d["filename"] for d in client.get("/api/documents").json()] == ["cfs.txt"]
    bob_list = other_client.get("/api/documents")
    assert [d["filename"] for d in bob_list.json()] == ["asthma.txt"] and bob_list.headers["x-total-count"] == "1"
    assert other_client.get(f"/api/documents/{alice_doc}").status_code == 404
    assert other_client.delete(f"/api/documents/{alice_doc}").status_code == 404
    assert client.get(f"/api/documents/{alice_doc}").status_code == 200
    assert other_client.get("/api/info").json()["documents"] == 1


def test_retrieval_never_uses_other_users_documents(client, other_client):
    alice_doc = upload(client, "cfs.txt", CFS_TEXT).json()["id"]
    upload(other_client, "asthma.txt", ASTHMA_TEXT)

    question = "post-exertional malaise unrefreshing sleep"
    bob = other_client.post("/api/query", json={"question": question}).json()
    assert {s["filename"] for s in bob["sources"]} == {"asthma.txt"}
    # Naming someone else's document id doesn't help either
    assert other_client.post("/api/query", json={"question": question, "document_ids": [alice_doc]}).json()["sources"] == []
    assert other_client.post("/api/search", json={"question": question, "document_ids": [alice_doc]}).json() == []
    assert {s["filename"] for s in client.post("/api/search", json={"question": question}).json()} == {"cfs.txt"}


def test_history_and_chats_are_private(client, other_client):
    upload(client, "cfs.txt", CFS_TEXT)
    upload(other_client, "asthma.txt", ASTHMA_TEXT)
    alice = client.post("/api/query", json={"question": "alice question"}).json()
    other_client.post("/api/query", json={"question": "bob question"})

    assert [q["question"] for q in client.get("/api/queries").json()] == ["alice question"]
    assert [q["question"] for q in other_client.get("/api/queries").json()] == ["bob question"]
    assert [c["title"] for c in other_client.get("/api/chats").json()] == ["bob question"]

    chat = alice["chat_id"]
    for method, path, body in (
        ("GET", f"/api/chats/{chat}", None),
        ("GET", f"/api/chats/{chat}/messages", None),
        ("PATCH", f"/api/chats/{chat}", {"title": "mine now"}),
        ("DELETE", f"/api/chats/{chat}", None),
        ("DELETE", f"/api/queries/{alice['id']}", None),
        ("POST", "/api/query", {"question": "sneaky", "chat_id": chat}),
        ("POST", "/api/query/stream", {"question": "sneaky", "chat_id": chat}),
    ):
        assert other_client.request(method, path, json=body).status_code == 404, (method, path)

    # Clearing history only touches your own
    assert other_client.delete("/api/queries").status_code == 204
    assert len(client.get("/api/queries").json()) == 1
    assert client.get(f"/api/chats/{chat}/messages").headers["x-total-count"] == "1"


# --- chats


def test_questions_start_a_chat_unless_one_is_given(client):
    upload(client, "cfs.txt", CFS_TEXT)
    long_question = "What are the core symptoms " + "and criteria " * 10 + "?"
    first = client.post("/api/query", json={"question": long_question}).json()
    chat_id = first["chat_id"]
    chat = client.get(f"/api/chats/{chat_id}").json()
    assert chat["title"].startswith("What are the core symptoms") and chat["title"].endswith("…")
    assert len(chat["title"]) <= 80

    follow_up = client.post("/api/query", json={"question": "And treatment?", "chat_id": chat_id}).json()
    assert follow_up["chat_id"] == chat_id
    assert [m["question"] for m in client.get(f"/api/chats/{chat_id}/messages").json()] == ["And treatment?", long_question]
    assert len(client.get("/api/chats").json()) == 1


def test_streamed_answers_belong_to_a_chat(client):
    from tests.test_features import stream

    upload(client, "cfs.txt", CFS_TEXT)
    events = stream(client, "What is pacing?")
    chat_id = events[0][1]["chat_id"]
    assert events[-1][1]["chat_id"] == chat_id
    again = stream(client, "And sleep?", chat_id=chat_id)
    assert again[0][1]["chat_id"] == chat_id
    assert client.get(f"/api/chats/{chat_id}/messages").headers["x-total-count"] == "2"


def test_failed_question_does_not_leave_an_empty_chat(client):
    from app.deps import embedder_dep

    def broken(texts):
        raise RuntimeError("model missing")

    app.dependency_overrides[embedder_dep] = lambda: type("E", (), {"embed": staticmethod(broken), "dimension": 384})()
    with pytest.raises(RuntimeError):
        client.post("/api/query", json={"question": "anything"})
    assert client.get("/api/chats").json() == []


def test_chat_list_is_paginated_by_recent_activity(client):
    upload(client, "cfs.txt", CFS_TEXT)
    ids = [client.post("/api/query", json={"question": f"chat {i}"}).json()["chat_id"] for i in range(5)]
    # Asking in the oldest chat moves it to the top
    client.post("/api/query", json={"question": "back again", "chat_id": ids[0]})

    page = client.get("/api/chats", params={"limit": 2})
    assert page.headers["x-total-count"] == "5"
    assert [c["id"] for c in page.json()] == [ids[0], ids[4]]
    assert [c["id"] for c in client.get("/api/chats", params={"limit": 2, "offset": 4}).json()] == [ids[1]]
    assert client.get("/api/chats", params={"limit": 101}).status_code == 422


def test_chat_messages_are_paginated(client):
    upload(client, "cfs.txt", CFS_TEXT)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    for i in range(5):
        client.post("/api/query", json={"question": f"q{i}", "chat_id": chat_id})
    page = client.get(f"/api/chats/{chat_id}/messages", params={"limit": 2, "offset": 2})
    assert page.headers["x-total-count"] == "5"
    assert [m["question"] for m in page.json()] == ["q2", "q1"]


def test_create_rename_and_delete_chats(client):
    upload(client, "cfs.txt", CFS_TEXT)
    blank = client.post("/api/chats", json={"title": "  "}).json()
    assert blank["title"] == "New chat"
    renamed = client.patch(f"/api/chats/{blank['id']}", json={"title": " Ward round notes "}).json()
    assert renamed["title"] == "Ward round notes"
    assert client.patch(f"/api/chats/{blank['id']}", json={"title": ""}).status_code == 422

    client.post("/api/query", json={"question": "q", "chat_id": blank["id"]})
    assert client.delete(f"/api/chats/{blank['id']}").status_code == 204
    assert client.get(f"/api/chats/{blank['id']}").status_code == 404
    assert client.get("/api/queries").json() == []  # its messages went with it
    assert len(client.get("/api/documents").json()) == 1  # documents are kept


# --- data from before accounts existed


def test_first_user_claims_existing_data(anon_client):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (filename, file_type, status, num_pages, num_chunks) "
                          "VALUES ('old.pdf', '.pdf', 'ready', 1, 0)"))
        for q in ("old question 1", "old question 2"):
            conn.execute(text("INSERT INTO queries (question, answer, llm, sources) VALUES (:q, 'a', 'x', '[]')"), {"q": q})

    register(anon_client, "first@example.org")
    assert [d["filename"] for d in anon_client.get("/api/documents").json()] == ["old.pdf"]
    chats = anon_client.get("/api/chats").json()
    assert [c["title"] for c in chats] == ["Earlier questions"]
    messages = anon_client.get(f"/api/chats/{chats[0]['id']}/messages").json()
    assert sorted(m["question"] for m in messages) == ["old question 1", "old question 2"]

    with TestClient(app) as second:
        register(second, "second@example.org")
        assert second.get("/api/documents").json() == []
        assert second.get("/api/chats").json() == []


def test_cli_creates_users(anon_client, monkeypatch, capsys):
    import sys

    from app import cli

    monkeypatch.setattr("getpass.getpass", lambda prompt="": PASSWORD)
    monkeypatch.setattr(sys, "argv", ["cli", "create-user", "Admin@Example.org", "--name", "Admin"])
    cli.main()
    assert "Created admin@example.org" in capsys.readouterr().out
    login = anon_client.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD})
    assert login.status_code == 200 and login.json()["user"]["name"] == "Admin"
