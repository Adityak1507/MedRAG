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
