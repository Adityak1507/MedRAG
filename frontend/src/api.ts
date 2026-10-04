export type DocumentStatus = "processing" | "ready" | "failed";

export interface MedDocument {
  id: number;
  filename: string;
  file_type: string;
  status: DocumentStatus;
  error: string | null;
  num_pages: number;
  num_chunks: number;
  ocr_pages: number;
  created_at: string;
}

export interface Source {
  source_id: number;
  document_id: number;
  filename: string;
  page: number | null;
  chunk_index: number;
  similarity: number;
  content: string;
}

export interface QueryResult {
  id: number;
  chat_id: number | null;
  question: string;
  answer: string;
  llm: string;
  sources: Source[];
  created_at: string;
}

export interface SystemInfo {
  llm: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  min_similarity: number;
  ocr: boolean;
  max_upload_mb: number;
  documents: number;
  chunks: number;
}

export interface Page<T> {
  items: T[];
  total: number;
}

export type HistoryPage = Page<QueryResult>;

export interface User {
  id: number;
  email: string;
  name: string | null;
  created_at: string;
}

export interface AuthConfig {
  allow_registration: boolean;
  password_min_length: number;
}

export interface Chat {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StreamHandlers {
  /** The chat the answer is saved to (a new one when no chat was given). */
  onChat?: (chatId: number) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (text: string) => void;
}

/** An HTTP error from the API; status 401 means the user is not (or no longer) signed in. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const isUnauthorized = (e: unknown) => e instanceof ApiError && e.status === 401;

const BASE = import.meta.env.VITE_API_URL ?? "";

async function errorFrom(resp: Response): Promise<ApiError> {
  let message = `${resp.status} ${resp.statusText}`;
  try {
    const body = await resp.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON; keep the status text
  }
  return new ApiError(message, resp.status);
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  // The session is an httpOnly cookie; a separate API origin (VITE_API_URL) needs it sent explicitly
  const resp = await fetch(`${BASE}${path}`, { credentials: BASE ? "include" : "same-origin", ...init });
  if (!resp.ok) throw await errorFrom(resp);
  return resp;
}

async function page<T>(path: string): Promise<Page<T>> {
  const resp = await send(path);
  const items: T[] = await resp.json();
  return { items, total: Number(resp.headers.get("X-Total-Count") ?? items.length) };
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await send(path, init);
  return (resp.status === 204 ? undefined : await resp.json()) as T;
}

/** Parse a server-sent event stream, calling onEvent(event, data) for each complete event. */
export async function readEventStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: string, data: string) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = (block: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length) onEvent(event, data.join("\n"));
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: !done });
    buffer = buffer.replace(/\r\n?/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      flush(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
    }
    if (done) break;
  }
  if (buffer.trim()) flush(buffer);
}

function questionBody(question: string, documentIds: number[] | null, chatId: number | null) {
  return json("POST", { question, document_ids: documentIds, chat_id: chatId });
}

export const api = {
  auth: {
    config: () => request<AuthConfig>("/api/auth/config"),
    me: () => request<User>("/api/auth/me"),
    login: (email: string, password: string) =>
      request<{ user: User }>("/api/auth/login", json("POST", { email, password })).then((r) => r.user),
    register: (email: string, password: string, name: string | null) =>
      request<{ user: User }>("/api/auth/register", json("POST", { email, password, name })).then((r) => r.user),
    logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  },

  info: () => request<SystemInfo>("/api/info"),
  listDocuments: () => request<MedDocument[]>("/api/documents?limit=500"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<MedDocument>("/api/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: number) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),

  chats: {
    /** Most recently active first. */
    list: (limit = 30, offset = 0) => page<Chat>(`/api/chats?limit=${limit}&offset=${offset}`),
    get: (id: number) => request<Chat>(`/api/chats/${id}`),
    rename: (id: number, title: string) => request<Chat>(`/api/chats/${id}`, json("PATCH", { title })),
    remove: (id: number) => request<void>(`/api/chats/${id}`, { method: "DELETE" }),
    /** Newest first. */
    messages: (id: number, limit = 20, offset = 0) =>
      page<QueryResult>(`/api/chats/${id}/messages?limit=${limit}&offset=${offset}`),
  },

  ask: (question: string, documentIds: number[] | null, chatId: number | null = null) =>
    request<QueryResult>("/api/query", questionBody(question, documentIds, chatId)),
  /** Ask a question and receive the answer piece by piece; resolves with the saved result. */
  askStream: async (
    question: string,
    documentIds: number[] | null,
    chatId: number | null,
    handlers: StreamHandlers = {},
  ) => {
    const resp = await send("/api/query/stream", questionBody(question, documentIds, chatId));
    if (!resp.body) throw new Error("This browser does not support streaming responses");
    let result: QueryResult | null = null;
    let failure: string | null = null;
    await readEventStream(resp.body, (event, data) => {
      const payload = JSON.parse(data);
      if (event === "sources") {
        handlers.onChat?.(payload.chat_id);
        handlers.onSources?.(payload.sources);
      } else if (event === "token") handlers.onToken?.(payload.text);
      else if (event === "done") result = payload;
      else if (event === "error") failure = payload.detail;
    });
    if (failure) throw new Error(failure);
    if (!result) throw new Error("The answer stream ended unexpectedly");
    return result as QueryResult;
  },
  /** All of the user's questions across chats, newest first. */
  history: (limit = 20, offset = 0): Promise<HistoryPage> => page<QueryResult>(`/api/queries?limit=${limit}&offset=${offset}`),
  deleteQuery: (id: number) => request<void>(`/api/queries/${id}`, { method: "DELETE" }),
  clearHistory: () => request<void>("/api/queries", { method: "DELETE" }),
};
