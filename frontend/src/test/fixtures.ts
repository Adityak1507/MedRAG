import type { Chat, MedDocument, QueryResult, Source, User } from "../api";

export function source(overrides: Partial<Source> = {}): Source {
  return {
    source_id: 1,
    document_id: 1,
    filename: "cfs.txt",
    page: null,
    chunk_index: 0,
    similarity: 0.6123,
    content: "Diagnosis requires three core symptoms.",
    ...overrides,
  };
}

export function result(id: number, overrides: Partial<QueryResult> = {}): QueryResult {
  return {
    id,
    chat_id: 1,
    question: `Question ${id}?`,
    answer: `Answer ${id}.`,
    llm: "gemini:gemini-3.8-flash",
    sources: [source()],
    created_at: "2026-10-04T10:00:00Z",
    ...overrides,
  };
}

export function doc(id: number, overrides: Partial<MedDocument> = {}): MedDocument {
  return {
    id,
    filename: `doc${id}.pdf`,
    file_type: ".pdf",
    status: "ready",
    error: null,
    num_pages: 3,
    num_chunks: 5,
    ocr_pages: 0,
    created_at: "2026-10-04T10:00:00Z",
    ...overrides,
  };
}

/** A fetch Response whose body streams the given string pieces. */
export function streamingResponse(pieces: string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const piece of pieces) controller.enqueue(encoder.encode(piece));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" }, ...init });
}

export const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

export function chat(id: number, overrides: Partial<Chat> = {}): Chat {
  return {
    id,
    title: `Chat ${id}`,
    created_at: "2026-10-04T10:00:00Z",
    updated_at: "2026-10-04T10:00:00Z",
    ...overrides,
  };
}

export const alice: User = { id: 1, email: "alice@example.org", name: "Alice", created_at: "2026-10-04T10:00:00Z" };

export const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" }, ...init });
