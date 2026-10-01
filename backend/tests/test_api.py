import io

import pytest

CFS_TEXT = """Chronic Fatigue Syndrome (CFS), also known as Myalgic Encephalomyelitis, is a complex long-term illness.

Diagnosis requires all three core symptoms: a substantial reduction in activity lasting more than six months,
post-exertional malaise, and unrefreshing sleep.

Treatment focuses on symptom management, including pacing, medications for sleep and pain,
and cognitive behavioral therapy.
"""

ASTHMA_TEXT = """Asthma is a chronic inflammatory disease of the airways.
Inhaled corticosteroids are the mainstay of long-term asthma control.
"""


def upload(client, name, text):
    return client.post("/api/documents", files={"file": (name, io.BytesIO(text.encode()), "text/plain")})


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_processes_document(client):
    resp = upload(client, "cfs.txt", CFS_TEXT)
    assert resp.status_code == 202
    doc_id = resp.json()["id"]

    # TestClient runs background tasks before returning, so ingestion is done
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready", doc
    assert doc["filename"] == "cfs.txt"
    assert doc["num_chunks"] >= 1

    assert [d["id"] for d in client.get("/api/documents").json()] == [doc_id]


def test_upload_rejects_unsupported_and_empty(client):
    resp = client.post("/api/documents", files={"file": ("notes.doc", io.BytesIO(b"x"), "application/msword")})
    assert resp.status_code == 415
    resp = client.post("/api/documents", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")})
    assert resp.status_code == 400
    assert client.get("/api/documents").json() == []


def test_upload_rejects_large_file(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    resp = upload(client, "big.txt", "a" * (1024 * 1024 + 1))
    assert resp.status_code == 413


def test_unreadable_document_is_marked_failed(client):
    resp = client.post("/api/documents", files={"file": ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf")})
    assert resp.status_code == 202
    doc = client.get(f"/api/documents/{resp.json()['id']}").json()
    assert doc["status"] == "failed"
    assert doc["error"]


def test_query_uses_llm_with_retrieved_context(client, fake_llm):
    upload(client, "cfs.txt", CFS_TEXT)
    resp = client.post("/api/query", json={"question": "What are the core symptoms for diagnosis?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Fake answer."
    assert body["llm"] == "fake:echo"
    assert body["sources"] and body["sources"][0]["filename"] == "cfs.txt"
    assert "post-exertional malaise" in fake_llm.prompts[0]
    assert -1.0 <= body["sources"][0]["similarity"] <= 1.0


def test_query_ranks_relevant_document_first(client):
    upload(client, "cfs.txt", CFS_TEXT)
    upload(client, "asthma.txt", ASTHMA_TEXT)
    body = client.post("/api/query", json={"question": "inhaled corticosteroids asthma control"}).json()
    assert body["sources"][0]["filename"] == "asthma.txt"


def test_query_filters_by_document(client):
    cfs_id = upload(client, "cfs.txt", CFS_TEXT).json()["id"]
    upload(client, "asthma.txt", ASTHMA_TEXT)
    body = client.post(
        "/api/query", json={"question": "asthma corticosteroids", "document_ids": [cfs_id], "top_k": 10}
    ).json()
    assert body["sources"]
    assert {s["document_id"] for s in body["sources"]} == {cfs_id}


@pytest.mark.no_llm
def test_query_without_llm_returns_excerpts(client):
    upload(client, "cfs.txt", CFS_TEXT)
    body = client.post("/api/query", json={"question": "What treatment uses pacing?"}).json()
    assert body["llm"] == "retrieval_only"
    assert "pacing" in body["answer"]


def test_query_with_no_documents(client):
    body = client.post("/api/query", json={"question": "anything"}).json()
    assert body["sources"] == []
    assert "cannot answer" in body["answer"]


def test_query_validation(client):
    assert client.post("/api/query", json={"question": ""}).status_code == 422
    assert client.post("/api/query", json={"question": "x", "top_k": 0}).status_code == 422


def test_history_and_delete(client):
    doc_id = upload(client, "cfs.txt", CFS_TEXT).json()["id"]
    client.post("/api/query", json={"question": "first"})
    client.post("/api/query", json={"question": "second"})
    history = client.get("/api/queries").json()
    assert [q["question"] for q in history] == ["second", "first"]

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    # Chunks are removed with the document
    assert client.get("/api/info").json()["chunks"] == 0


def test_info(client):
    info = client.get("/api/info").json()
    assert info["llm"] == "retrieval_only"
    assert info["documents"] == 0


class StubChatModel:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def invoke(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return type("Reply", (), {"content": self.reply})()


def chain(*stubs):
    from app.rag.llm import LLM, ChatModel

    return LLM([ChatModel(f"stub{i}", stub) for i, stub in enumerate(stubs)])


def test_llm_chain_uses_first_working_provider():
    first, second = StubChatModel(reply="from first"), StubChatModel(reply="from second")
    assert chain(first, second).generate("q") == ("from first", "stub0")
    assert second.calls == 0


def test_llm_chain_falls_back_on_error_or_empty_reply():
    failing = StubChatModel(error=RuntimeError("quota exceeded"))
    empty = StubChatModel(reply="  ")
    working = StubChatModel(reply="from third")
    assert chain(failing, empty, working).generate("q") == ("from third", "stub2")


def test_llm_chain_raises_when_all_fail():
    from app.rag.llm import AllProvidersFailed

    with pytest.raises(AllProvidersFailed, match="stub0.*quota.*stub1.*down"):
        chain(StubChatModel(error=RuntimeError("quota")), StubChatModel(error=RuntimeError("down"))).generate("q")


def test_query_falls_back_to_second_provider(client):
    from app.deps import llm_dep
    from app.main import app

    app.dependency_overrides[llm_dep] = lambda: chain(
        StubChatModel(error=RuntimeError("503")), StubChatModel(reply="Groq answer.")
    )
    upload(client, "cfs.txt", CFS_TEXT)
    body = client.post("/api/query", json={"question": "core symptoms"}).json()
    assert body["answer"] == "Groq answer."
    assert body["llm"] == "stub1"


def test_query_returns_excerpts_when_all_providers_fail(client):
    from app.deps import llm_dep
    from app.main import app

    app.dependency_overrides[llm_dep] = lambda: chain(StubChatModel(error=RuntimeError("x")))
    upload(client, "cfs.txt", CFS_TEXT)
    body = client.post("/api/query", json={"question": "What treatment uses pacing?"}).json()
    assert body["llm"] == "stub0 (all failed)"
    assert "pacing" in body["answer"]


def test_provider_order_from_settings(monkeypatch):
    from app.config import get_settings
    from app.rag import llm as llm_module

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_providers", "groq,gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "g-key")
    monkeypatch.setattr(settings, "groq_api_key", "")
    llm_module.get_llm.cache_clear()
    try:
        built = llm_module.get_llm()
        assert [m.name for m in built.models] == [f"gemini:{settings.gemini_model}"]  # groq skipped: no key

        monkeypatch.setattr(settings, "llm_providers", "gemini,openai")
        llm_module.get_llm.cache_clear()
        with pytest.raises(RuntimeError, match="openai"):
            llm_module.get_llm()
    finally:
        llm_module.get_llm.cache_clear()
