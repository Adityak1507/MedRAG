import { useCallback, useEffect, useState } from "react";
import { api, type MedDocument, type QueryResult, type SystemInfo } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentPanel } from "./components/DocumentPanel";

const POLL_MS = 2000;

export default function App() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [documents, setDocuments] = useState<MedDocument[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [results, setResults] = useState<QueryResult[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  const refreshDocuments = useCallback(async () => {
    const docs = await api.listDocuments();
    setDocuments(docs);
    // Drop selections for documents that were deleted or are not ready
    const ready = new Set(docs.filter((d) => d.status === "ready").map((d) => d.id));
    setSelected((prev) => {
      const next = new Set([...prev].filter((id) => ready.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, []);

  useEffect(() => {
    api.info().then(setInfo).catch(report);
    refreshDocuments().catch(report);
    api
      .history()
      .then((h) => setResults(h.reverse()))
      .catch(report);
  }, [refreshDocuments]);

  // Poll while any document is still being processed
  const processing = documents.some((d) => d.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => refreshDocuments().catch(report), POLL_MS);
    return () => clearInterval(timer);
  }, [processing, refreshDocuments]);

  async function handleUpload(files: File[]) {
    setError(null);
    for (const file of files) {
      try {
        await api.uploadDocument(file);
      } catch (e) {
        report(new Error(`${file.name}: ${e instanceof Error ? e.message : e}`));
      }
    }
    await refreshDocuments().catch(report);
  }

  async function handleDelete(doc: MedDocument) {
    if (!window.confirm(`Delete "${doc.filename}" and its index?`)) return;
    try {
      await api.deleteDocument(doc.id);
      await refreshDocuments();
    } catch (e) {
      report(e);
    }
  }

  async function handleAsk(question: string) {
    setError(null);
    setPending(question);
    try {
      const result = await api.ask(question, selected.size ? [...selected] : null);
      setResults((prev) => [...prev, result]);
    } catch (e) {
      report(e);
    } finally {
      setPending(null);
    }
  }

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const hasReadyDocument = documents.some((d) => d.status === "ready");

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          Med<span>RAG</span>
        </h1>
        {info && (
          <div className="system muted small">
            <span>LLM: {info.llm === "retrieval_only" ? "none (retrieval only)" : info.llm}</span>
            <span>Embeddings: {info.embedding_model}</span>
          </div>
        )}
      </header>

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button className="icon-button" aria-label="Dismiss" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}

      <main className="layout">
        <DocumentPanel
          documents={documents}
          selected={selected}
          onToggle={toggle}
          onUpload={handleUpload}
          onDelete={handleDelete}
          maxUploadMb={info?.max_upload_mb}
        />
        <ChatPanel results={results} pending={pending} canAsk={hasReadyDocument} onAsk={handleAsk} />
      </main>

      <footer className="muted small">
        Answers come only from the uploaded documents and are not medical advice.
      </footer>
    </div>
  );
}
