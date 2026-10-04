import { useRef, useState } from "react";
import type { MedDocument } from "../api";

const ACCEPT = ".pdf,.docx,.txt";

interface Props {
  documents: MedDocument[];
  selected: Set<number>;
  onToggle: (id: number) => void;
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (doc: MedDocument) => Promise<void>;
  maxUploadMb?: number;
  /** Whether the server can OCR scanned PDFs */
  ocr?: boolean;
}

export function DocumentPanel({ documents, selected, onToggle, onUpload, onDelete, maxUploadMb, ocr }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  async function handleFiles(list: FileList | null) {
    if (!list || list.length === 0) return;
    setUploading(true);
    try {
      await onUpload(Array.from(list));
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <aside className="panel documents">
      <h2>Documents</h2>

      <label
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => void handleFiles(e.target.files)}
        />
        <strong>{uploading ? "Uploading…" : "Drop files or click to upload"}</strong>
        <span className="muted">
          PDF, DOCX or TXT{maxUploadMb ? `, up to ${maxUploadMb} MB` : ""}
        </span>
        {ocr && <span className="muted small">Scanned PDFs are read with OCR</span>}
      </label>

      {documents.length === 0 ? (
        <p className="muted empty">No documents yet.</p>
      ) : (
        <>
          <p className="muted hint">
            {selected.size === 0
              ? "Questions search all ready documents. Tick documents to narrow the search."
              : `Searching ${selected.size} selected document${selected.size > 1 ? "s" : ""}.`}
          </p>
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.id} className="doc">
                <input
                  type="checkbox"
                  aria-label={`Search only ${doc.filename}`}
                  checked={selected.has(doc.id)}
                  disabled={doc.status !== "ready"}
                  onChange={() => onToggle(doc.id)}
                />
                <div className="doc-body">
                  <span className="doc-name" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className="muted small">
                    <span className={`badge ${doc.status}`}>{doc.status}</span>
                    {doc.status === "ready" &&
                      ` ${doc.num_chunks} chunk${doc.num_chunks === 1 ? "" : "s"}${doc.num_pages > 1 ? ` · ${doc.num_pages} pages` : ""}`}
                    {doc.status === "ready" && doc.ocr_pages > 0 && (
                      <span className="ocr" title="Text was read from page images with OCR; check quotes against the original">
                        {` · OCR ${doc.ocr_pages === doc.num_pages ? "all pages" : `${doc.ocr_pages} page${doc.ocr_pages === 1 ? "" : "s"}`}`}
                      </span>
                    )}
                  </span>
                  {doc.error && <span className="error small">{doc.error}</span>}
                </div>
                <button
                  className="icon-button"
                  title="Delete document"
                  aria-label={`Delete ${doc.filename}`}
                  onClick={() => void onDelete(doc)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
