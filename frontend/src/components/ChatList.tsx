import { useState } from "react";
import type { Chat } from "../api";

interface Props {
  chats: Chat[];
  total: number;
  activeId: number | null;
  loadingMore: boolean;
  onSelect: (id: number) => void;
  onNew: () => void;
  onLoadMore: () => Promise<void>;
  onRename: (id: number, title: string) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}

export function relativeTime(iso: string, now = Date.now()): string {
  const minutes = Math.round((now - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days} d ago`;
  return new Date(iso).toLocaleDateString();
}

export function ChatList({ chats, total, activeId, loadingMore, onSelect, onNew, onLoadMore, onRename, onDelete }: Props) {
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [confirming, setConfirming] = useState<number | null>(null);

  function startRename(chat: Chat) {
    setConfirming(null);
    setEditing(chat.id);
    setDraft(chat.title);
  }

  async function saveRename(chat: Chat) {
    const title = draft.trim();
    setEditing(null);
    if (title && title !== chat.title) await onRename(chat.id, title);
  }

  return (
    <nav className="panel chats" aria-label="Chats">
      <div className="chats-header">
        <h2>Chats</h2>
        <button className="text-button" onClick={onNew}>
          + New chat
        </button>
      </div>

      {chats.length === 0 ? (
        <p className="muted small empty">No chats yet. Ask a question to start one.</p>
      ) : (
        <ul className="chat-list">
          {chats.map((chat) => (
            <li key={chat.id} className={`chat-item${chat.id === activeId ? " active" : ""}`}>
              {editing === chat.id ? (
                <input
                  className="rename"
                  aria-label="Chat title"
                  value={draft}
                  maxLength={200}
                  autoFocus
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={() => void saveRename(chat)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") void saveRename(chat);
                    if (e.key === "Escape") setEditing(null);
                  }}
                />
              ) : (
                <button
                  className="chat-open"
                  aria-label={chat.title}
                  aria-current={chat.id === activeId ? "page" : undefined}
                  onClick={() => onSelect(chat.id)}
                  onDoubleClick={() => startRename(chat)}
                  title={chat.title}
                >
                  <span className="chat-title">{chat.title}</span>
                  <span className="muted small">{relativeTime(chat.updated_at)}</span>
                </button>
              )}

              {confirming === chat.id ? (
                <span className="chat-confirm small">
                  Delete?
                  <button
                    className="text-button danger"
                    onClick={() => {
                      setConfirming(null);
                      void onDelete(chat.id);
                    }}
                  >
                    Yes
                  </button>
                  <button className="text-button" onClick={() => setConfirming(null)}>
                    No
                  </button>
                </span>
              ) : (
                editing !== chat.id && (
                  <span className="chat-actions">
                    <button
                      className="icon-button"
                      title="Rename"
                      aria-label={`Rename chat: ${chat.title}`}
                      onClick={() => startRename(chat)}
                    >
                      ✎
                    </button>
                    <button
                      className="icon-button"
                      title="Delete chat"
                      aria-label={`Delete chat: ${chat.title}`}
                      onClick={() => setConfirming(chat.id)}
                    >
                      ×
                    </button>
                  </span>
                )
              )}
            </li>
          ))}
        </ul>
      )}

      {chats.length < total && (
        <button className="text-button load-more" disabled={loadingMore} onClick={() => void onLoadMore()}>
          {loadingMore ? "Loading…" : `Show more (${total - chats.length})`}
        </button>
      )}
    </nav>
  );
}
