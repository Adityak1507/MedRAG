import { useEffect, useRef, useState } from "react";
import type { QueryResult } from "../api";

interface Props {
  results: QueryResult[];
  pending: string | null;
  canAsk: boolean;
  onAsk: (question: string) => Promise<void>;
}

export function ChatPanel({ results, pending, canAsk, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [results.length, pending]);

  async function submit(e: { preventDefault(): void }) {
    e.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    setQuestion("");
    await onAsk(q);
  }

  return (
    <section className="panel chat">
      <div className="thread">
        {results.length === 0 && !pending && (
          <div className="welcome">
            <h2>Ask about your documents</h2>
            <p className="muted">
              Answers are based only on the documents you upload, and every answer lists the passages it used.
            </p>
          </div>
        )}
        {results.map((r) => (
          <Exchange key={r.id} result={r} />
        ))}
        {pending && (
          <div className="exchange">
            <div className="question">{pending}</div>
            <div className="answer muted">Searching documents and generating an answer…</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="ask" onSubmit={submit}>
        <textarea
          value={question}
          placeholder={canAsk ? "e.g. What are the core diagnostic symptoms?" : "Upload a document to start asking questions"}
          disabled={!canAsk}
          rows={2}
          maxLength={2000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) void submit(e);
          }}
        />
        <button type="submit" disabled={!canAsk || !question.trim() || pending !== null}>
          Ask
        </button>
      </form>
    </section>
  );
}

function Exchange({ result }: { result: QueryResult }) {
  return (
    <div className="exchange">
      <div className="question">{result.question}</div>
      <div className="answer">
        <p className="answer-text">{result.answer}</p>
        <span className="tag">{result.llm === "retrieval_only" ? "retrieval only (no LLM)" : result.llm}</span>
        {result.sources.length > 0 && (
          <details className="sources">
            <summary>
              {result.sources.length} source{result.sources.length > 1 ? "s" : ""}
            </summary>
            <ol>
              {result.sources.map((s) => (
                <li key={s.source_id}>
                  <div className="source-meta">
                    <strong>{s.filename}</strong>
                    {s.page !== null && <span> · page {s.page}</span>}
                    <span className="muted"> · similarity {s.similarity.toFixed(3)}</span>
                  </div>
                  <blockquote>{s.content}</blockquote>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </div>
  );
}
