import { useEffect, useRef, useState } from "react";
import type { QueryResult, Source } from "../api";

/** An answer that is still streaming in. */
export interface PendingAnswer {
  question: string;
  answer: string;
  sources: Source[];
}

interface Props {
  /** Title of the open chat; null for a new chat that has no questions yet */
  title: string | null;
  results: QueryResult[];
  hasOlder: boolean;
  loadingOlder: boolean;
  pending: PendingAnswer | null;
  /** An answer is being generated (possibly in another chat): no new question until it's done */
  busy?: boolean;
  canAsk: boolean;
  onAsk: (question: string) => Promise<void>;
  onLoadOlder: () => Promise<void>;
  onDeleteQuestion: (result: QueryResult) => Promise<void>;
}

const LLM_LABELS: Record<string, string> = {
  retrieval_only: "retrieval only (no LLM)",
  similarity_cutoff: "no relevant passage found (LLM not called)",
};

export function llmLabel(llm: string): string {
  return LLM_LABELS[llm] ?? llm;
}

export function ChatPanel({
  title,
  results,
  hasOlder,
  loadingOlder,
  pending,
  busy = pending !== null,
  canAsk,
  onAsk,
  onLoadOlder,
  onDeleteQuestion,
}: Props) {
  const [question, setQuestion] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  // Follow new answers (not older history being loaded above)
  const lastId = results.at(-1)?.id;
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [lastId, pending?.question, pending?.answer.length]);

  async function submit(e: { preventDefault(): void }) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    await onAsk(q);
  }

  return (
    <section className="panel chat">
      <div className="chat-toolbar">
        <h2 className="chat-heading" title={title ?? undefined}>
          {title ?? "New chat"}
        </h2>
      </div>

      <div className="thread">
        {hasOlder && (
          <button className="text-button load-older" disabled={loadingOlder} onClick={() => void onLoadOlder()}>
            {loadingOlder ? "Loading…" : "Load earlier questions"}
          </button>
        )}
        {results.length === 0 && !pending && (
          <div className="welcome">
            <h3>Ask about your documents</h3>
            <p className="muted">
              Answers are based only on the documents you upload, and every answer lists the passages it used.
            </p>
          </div>
        )}
        {results.map((r) => (
          <Exchange key={r.id} result={r} onDelete={() => void onDeleteQuestion(r)} />
        ))}
        {pending && (
          <div className="exchange" aria-live="polite" aria-busy="true">
            <div className="question">{pending.question}</div>
            <div className="answer">
              {pending.answer ? (
                <p className="answer-text streaming">{pending.answer}</p>
              ) : (
                <p className="muted">
                  {pending.sources.length ? "Generating an answer…" : "Searching documents…"}
                </p>
              )}
              {pending.sources.length > 0 && <Sources sources={pending.sources} />}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="ask" onSubmit={submit}>
        <textarea
          value={question}
          aria-label="Question"
          placeholder={canAsk ? "e.g. What are the core diagnostic symptoms?" : "Upload a document to start asking questions"}
          disabled={!canAsk}
          rows={2}
          maxLength={2000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) void submit(e);
          }}
        />
        <button type="submit" disabled={!canAsk || !question.trim() || busy}>
          Ask
        </button>
      </form>
    </section>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  return (
    <details className="sources">
      <summary>
        {sources.length} source{sources.length > 1 ? "s" : ""}
      </summary>
      <ol>
        {sources.map((s) => (
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
  );
}

function Exchange({ result, onDelete }: { result: QueryResult; onDelete: () => void }) {
  return (
    <div className="exchange">
      <div className="question-row">
        <button
          className="icon-button delete-question"
          title="Delete this question"
          aria-label={`Delete question: ${result.question}`}
          onClick={onDelete}
        >
          ×
        </button>
        <div className="question">{result.question}</div>
      </div>
      <div className="answer">
        <p className="answer-text">{result.answer}</p>
        <span className="tag">{llmLabel(result.llm)}</span>
        {result.sources.length > 0 && <Sources sources={result.sources} />}
      </div>
    </div>
  );
}
