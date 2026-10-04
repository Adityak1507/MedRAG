"""Streaming answers, the similarity cutoff, history pagination/deletion and OCR."""

import io
import json

import pytest

from app.config import get_settings
from app.deps import llm_dep
from app.main import app
from app.rag.loaders import ocr_available
from tests.test_api import ASTHMA_TEXT, CFS_TEXT, StubChatModel, chain, upload


def stream(client, question, **body):
    """POST /api/query/stream and return the parsed (event, data) pairs."""
    with client.stream("POST", "/api/query/stream", json={"question": question, **body}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def use_llm(llm):
    app.dependency_overrides[llm_dep] = lambda: llm


# --- streaming


def test_stream_sends_sources_tokens_then_saved_answer(client):
    upload(client, "cfs.txt", CFS_TEXT)
    events = stream(client, "What are the core symptoms?")
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"} and len(kinds) >= 4

    sources = events[0][1]["sources"]
    assert sources and sources[0]["filename"] == "cfs.txt"
    done = events[-1][1]
    assert done["answer"] == "Fake answer." == "".join(d["text"] for e, d in events if e == "token").strip()
    assert done["llm"] == "fake:echo"
    assert done["sources"] == sources
    # The streamed answer is saved like a normal one
    assert client.get("/api/queries").json()[0]["id"] == done["id"]


def test_stream_falls_back_before_first_token(client):
    use_llm(chain(StubChatModel(error=RuntimeError("429 quota")), StubChatModel(reply="From Groq.")))
    upload(client, "cfs.txt", CFS_TEXT)
    done = stream(client, "core symptoms")[-1][1]
    assert done["answer"] == "From Groq."
    assert done["llm"] == "stub1"


def test_stream_interrupted_mid_answer_keeps_partial_text(client):
    first = StubChatModel(reply="Pacing is the main", fail_after=2)
    second = StubChatModel(reply="should not be used")
    use_llm(chain(first, second))
    upload(client, "cfs.txt", CFS_TEXT)
    done = stream(client, "What treatment uses pacing?")[-1][1]
    assert done["answer"].startswith("Pacing is")
    assert "interrupted" in done["answer"]
    assert done["llm"] == "stub0 (interrupted)"
    assert second.calls == 0  # no restart on another model once text has been shown


def test_stream_when_all_providers_fail(client):
    use_llm(chain(StubChatModel(error=RuntimeError("down"))))
    upload(client, "cfs.txt", CFS_TEXT)
    done = stream(client, "What treatment uses pacing?")[-1][1]
    assert done["llm"] == "all LLMs failed"
    assert "pacing" in done["answer"]


@pytest.mark.no_llm
def test_stream_without_llm(client):
    upload(client, "cfs.txt", CFS_TEXT)
    events = stream(client, "What treatment uses pacing?")
    assert events[-1][1]["llm"] == "retrieval_only"
    assert "pacing" in events[-1][1]["answer"]


def test_stream_validation_errors_are_plain_http(client):
    assert client.post("/api/query/stream", json={"question": ""}).status_code == 422


# --- similarity cutoff


def test_similarity_cutoff_refuses_without_calling_llm(client, fake_llm, monkeypatch):
    upload(client, "cfs.txt", CFS_TEXT)
    monkeypatch.setattr(get_settings(), "min_similarity", 0.99)
    for body in (
        client.post("/api/query", json={"question": "Who won the 2018 World Cup?"}).json(),
        stream(client, "Who won the 2018 World Cup?")[-1][1],
    ):
        assert body["llm"] == "similarity_cutoff"
        assert "cannot answer" in body["answer"]
        assert body["sources"]  # the closest passages are still shown
    assert fake_llm.prompts == []


def test_similarity_cutoff_lets_relevant_questions_through(client, fake_llm, monkeypatch):
    upload(client, "cfs.txt", CFS_TEXT)
    monkeypatch.setattr(get_settings(), "min_similarity", 0.05)
    body = client.post("/api/query", json={"question": "post-exertional malaise unrefreshing sleep"}).json()
    assert body["llm"] == "fake:echo"
    assert len(fake_llm.prompts) == 1


# --- history pagination and deletion


def test_history_pagination(client):
    upload(client, "cfs.txt", CFS_TEXT)
    for i in range(5):
        client.post("/api/query", json={"question": f"question {i}"})

    first = client.get("/api/queries", params={"limit": 2})
    assert first.headers["x-total-count"] == "5"
    assert [q["question"] for q in first.json()] == ["question 4", "question 3"]
    rest = client.get("/api/queries", params={"limit": 2, "offset": 4}).json()
    assert [q["question"] for q in rest] == ["question 0"]
    assert client.get("/api/queries", params={"limit": 0}).status_code == 422


def test_delete_one_question_and_clear_history(client):
    upload(client, "cfs.txt", CFS_TEXT)
    ids = [client.post("/api/query", json={"question": f"q{i}"}).json()["id"] for i in range(3)]

    assert client.delete(f"/api/queries/{ids[1]}").status_code == 204
    assert client.delete(f"/api/queries/{ids[1]}").status_code == 404
    assert [q["id"] for q in client.get("/api/queries").json()] == [ids[2], ids[0]]

    assert client.delete("/api/queries").status_code == 204
    resp = client.get("/api/queries")
    assert resp.json() == [] and resp.headers["x-total-count"] == "0"
    # Clearing history leaves documents alone
    assert len(client.get("/api/documents").json()) == 1


def test_document_list_pagination(client):
    for name in ("a.txt", "b.txt", "c.txt"):
        upload(client, name, ASTHMA_TEXT)
    resp = client.get("/api/documents", params={"limit": 2, "offset": 1})
    assert resp.headers["x-total-count"] == "3"
    assert [d["filename"] for d in resp.json()] == ["b.txt", "a.txt"]


# --- OCR


def scanned_pdf(text: str) -> bytes:
    """A PDF whose only page is an image of `text` (no text layer), like a scanner produces."""
    import pymupdf

    with pymupdf.open() as source:
        page = source.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=14)
        image = page.get_pixmap(dpi=200).tobytes("png")
    with pymupdf.open() as scanned:
        scanned.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image)
        return scanned.tobytes()


SCANNED_TEXT = "Warfarin requires regular INR monitoring.\nThe usual target INR range is 2.0 to 3.0."


@pytest.mark.skipif(not ocr_available(), reason="tesseract is not installed")
def test_scanned_pdf_is_read_with_ocr(client):
    resp = client.post("/api/documents", files={"file": ("scan.pdf", io.BytesIO(scanned_pdf(SCANNED_TEXT)), "application/pdf")})
    doc = client.get(f"/api/documents/{resp.json()['id']}").json()
    assert doc["status"] == "ready", doc
    assert doc["ocr_pages"] == 1

    body = client.post("/api/query", json={"question": "What is the target INR range for warfarin?"}).json()
    assert "2.0 to 3.0" in body["sources"][0]["content"]


def test_scanned_pdf_without_ocr_fails_clearly(client, monkeypatch):
    pytest.importorskip("pymupdf")
    monkeypatch.setattr(get_settings(), "ocr_enabled", False)
    resp = client.post("/api/documents", files={"file": ("scan.pdf", io.BytesIO(scanned_pdf(SCANNED_TEXT)), "application/pdf")})
    doc = client.get(f"/api/documents/{resp.json()['id']}").json()
    assert doc["status"] == "failed"
    assert "No extractable text" in doc["error"]
    assert doc["ocr_pages"] == 0


def test_text_pdf_does_not_use_ocr(client):
    pymupdf = pytest.importorskip("pymupdf")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Metformin is first-line therapy for type 2 diabetes.")
        data = pdf.tobytes()
    resp = client.post("/api/documents", files={"file": ("text.pdf", io.BytesIO(data), "application/pdf")})
    doc = client.get(f"/api/documents/{resp.json()['id']}").json()
    assert doc["status"] == "ready" and doc["ocr_pages"] == 0


# --- search


def test_search_returns_passages_without_llm_or_history(client, fake_llm):
    upload(client, "cfs.txt", CFS_TEXT)
    upload(client, "asthma.txt", ASTHMA_TEXT)
    hits = client.post("/api/search", json={"question": "inhaled corticosteroids asthma", "top_k": 1}).json()
    assert len(hits) == 1 and hits[0]["filename"] == "asthma.txt"
    assert fake_llm.prompts == []
    assert client.get("/api/queries").json() == []
