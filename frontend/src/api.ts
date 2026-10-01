export type DocumentStatus = "processing" | "ready" | "failed";

export interface MedDocument {
  id: number;
  filename: string;
  file_type: string;
  status: DocumentStatus;
  error: string | null;
  num_pages: number;
  num_chunks: number;
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
  max_upload_mb: number;
  documents: number;
  chunks: number;
}

const BASE = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, init);
  if (!resp.ok) {
    let message = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // not JSON; keep the status text
    }
    throw new Error(message);
  }
  return (resp.status === 204 ? undefined : await resp.json()) as T;
}

export const api = {
  info: () => request<SystemInfo>("/api/info"),
  listDocuments: () => request<MedDocument[]>("/api/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<MedDocument>("/api/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: number) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),
  ask: (question: string, documentIds: number[] | null) =>
    request<QueryResult>("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_ids: documentIds }),
    }),
  history: (limit = 20) => request<QueryResult[]>(`/api/queries?limit=${limit}`),
};
