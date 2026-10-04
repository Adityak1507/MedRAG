import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  isUnauthorized,
  type Chat,
  type MedDocument,
  type QueryResult,
  type Source,
  type SystemInfo,
  type User,
} from "./api";
import { AuthScreen } from "./components/AuthScreen";
import { ChatList } from "./components/ChatList";
import { ChatPanel, type PendingAnswer } from "./components/ChatPanel";
import { DocumentPanel } from "./components/DocumentPanel";

const POLL_MS = 2000;
export const HISTORY_PAGE = 20;
export const CHATS_PAGE = 30;

/** Shows the sign-in screen until there is a session, then the workspace. */
export default function App() {
  // undefined: still checking for an existing session
  const [user, setUser] = useState<User | null | undefined>(undefined);
  const [unreachable, setUnreachable] = useState(false);

  const checkSession = useCallback(() => {
    setUnreachable(false);
    api.auth
      .me()
      .then(setUser)
      .catch((e) => {
        // Only a 401 means "not signed in"; anything else (e.g. the backend restarting) keeps the session
        if (isUnauthorized(e)) setUser(null);
        else setUnreachable(true);
      });
  }, []);

  useEffect(checkSession, [checkSession]);

  if (unreachable)
    return (
      <div className="auth-page">
        <div className="panel auth-card" role="alert">
          <p>Can't reach the MedRAG server. It may be starting up.</p>
          <button className="primary" onClick={checkSession}>
            Try again
          </button>
        </div>
      </div>
    );
  if (user === undefined) return <div className="auth-page muted">Loading…</div>;
  if (user === null) return <AuthScreen onSignedIn={setUser} />;
  // key: a different account gets a fresh workspace with none of the previous user's state
  return <Workspace key={user.id} user={user} onSignedOut={() => setUser(null)} />;
}

function Workspace({ user, onSignedOut }: { user: User; onSignedOut: () => void }) {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [documents, setDocuments] = useState<MedDocument[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const [chats, setChats] = useState<Chat[]>([]);
  const [chatsTotal, setChatsTotal] = useState(0);
  const [loadingChats, setLoadingChats] = useState(false);
  // null: a new chat, created on the server with its first question
  const [activeChat, setActiveChat] = useState<number | null>(null);
  const activeChatRef = useRef<number | null>(null);
  activeChatRef.current = activeChat;

  // The open chat's messages, oldest first, as displayed
  const [results, setResults] = useState<QueryResult[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [pending, setPending] = useState<PendingAnswer | null>(null);
  // The chat the pending answer belongs to (null until a new chat gets its id)
  const [pendingChat, setPendingChat] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const report = useCallback(
    (e: unknown) => {
      if (isUnauthorized(e)) onSignedOut(); // session expired or signed out elsewhere
      else setError(e instanceof Error ? e.message : String(e));
    },
    [onSignedOut],
  );

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

  const openChat = useCallback(
    async (id: number | null) => {
      activeChatRef.current = id; // now, so responses for a chat opened earlier are ignored
      setActiveChat(id);
      setResults([]);
      setHistoryTotal(0);
      if (id === null) return;
      try {
        const page = await api.chats.messages(id, HISTORY_PAGE);
        if (activeChatRef.current !== id) return; // another chat was opened meanwhile
        setResults([...page.items].reverse());
        setHistoryTotal(page.total);
      } catch (e) {
        report(e);
      }
    },
    [report],
  );

  useEffect(() => {
    api.info().then(setInfo).catch(report);
    refreshDocuments().catch(report);
    api.chats
      .list(CHATS_PAGE)
      .then((page) => {
        setChats(page.items);
        setChatsTotal(page.total);
        // Reopen the most recent chat
        if (page.items.length) void openChat(page.items[0].id);
      })
      .catch(report);
  }, [refreshDocuments, openChat, report]);

  // Poll while any document is still being processed
  const processing = documents.some((d) => d.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => refreshDocuments().catch(report), POLL_MS);
    return () => clearInterval(timer);
  }, [processing, refreshDocuments, report]);

  async function handleSignOut() {
    try {
      await api.auth.logout();
    } catch {
      // the session may already be gone; signing out locally is what matters
    }
    onSignedOut();
  }

  async function handleUpload(files: File[]) {
    setError(null);
    for (const file of files) {
      try {
        await api.uploadDocument(file);
      } catch (e) {
        if (isUnauthorized(e)) return report(e);
        report(new Error(`${file.name}: ${e instanceof Error ? e.message : e}`));
      }
    }
    await refreshDocuments().catch(report);
  }

  async function handleDeleteDocument(doc: MedDocument) {
    if (!window.confirm(`Delete "${doc.filename}" and its index?`)) return;
    try {
      await api.deleteDocument(doc.id);
      await refreshDocuments();
    } catch (e) {
      report(e);
    }
  }

  /** Put the chat at the top of the list (it was just used), fetching it if it's new. */
  async function bumpChat(id: number) {
    const known = chats.find((c) => c.id === id);
    const chat = known ? { ...known, updated_at: new Date().toISOString() } : await api.chats.get(id);
    setChats((prev) => [chat, ...prev.filter((c) => c.id !== id)]);
    if (!known) setChatsTotal((t) => t + 1);
  }

  async function handleAsk(question: string) {
    setError(null);
    const askedIn = activeChat;
    setPending({ question, answer: "", sources: [] });
    setPendingChat(askedIn);
    try {
      const result = await api.askStream(question, selected.size ? [...selected] : null, askedIn, {
        onChat: (chatId) => {
          setPendingChat(chatId);
          // A new chat now exists on the server: open it so follow-up questions go to it
          if (askedIn === null && activeChatRef.current === null) {
            activeChatRef.current = chatId;
            setActiveChat(chatId);
          }
        },
        onSources: (sources: Source[]) => setPending((p) => p && { ...p, sources }),
        onToken: (text: string) => setPending((p) => p && { ...p, answer: p.answer + text }),
      });
      if (result.chat_id !== null && activeChatRef.current === result.chat_id) {
        setResults((prev) => [...prev, result]);
        setHistoryTotal((t) => t + 1);
      }
      if (result.chat_id !== null) await bumpChat(result.chat_id);
    } catch (e) {
      report(e);
    } finally {
      setPending(null);
    }
  }

  async function handleLoadOlder() {
    if (activeChat === null) return;
    setLoadingOlder(true);
    try {
      // Offset by what is loaded; answers asked this session are already counted in results
      const page = await api.chats.messages(activeChat, HISTORY_PAGE, results.length);
      const known = new Set(results.map((r) => r.id));
      const older = page.items.filter((r) => !known.has(r.id)).reverse();
      setResults((prev) => [...older, ...prev]);
      setHistoryTotal(page.total);
    } catch (e) {
      report(e);
    } finally {
      setLoadingOlder(false);
    }
  }

  async function handleLoadMoreChats() {
    setLoadingChats(true);
    try {
      const page = await api.chats.list(CHATS_PAGE, chats.length);
      setChats((prev) => [...prev, ...page.items.filter((c) => !prev.some((p) => p.id === c.id))]);
      setChatsTotal(page.total);
    } catch (e) {
      report(e);
    } finally {
      setLoadingChats(false);
    }
  }

  async function handleRenameChat(id: number, title: string) {
    try {
      const chat = await api.chats.rename(id, title);
      setChats((prev) => prev.map((c) => (c.id === id ? chat : c)));
    } catch (e) {
      report(e);
    }
  }

  async function handleDeleteChat(id: number) {
    try {
      await api.chats.remove(id);
      setChats((prev) => prev.filter((c) => c.id !== id));
      setChatsTotal((t) => Math.max(0, t - 1));
      if (activeChat === id) void openChat(null);
    } catch (e) {
      report(e);
    }
  }

  async function handleDeleteQuestion(result: QueryResult) {
    try {
      await api.deleteQuery(result.id);
      setResults((prev) => prev.filter((r) => r.id !== result.id));
      setHistoryTotal((t) => Math.max(0, t - 1));
    } catch (e) {
      report(e);
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
  const activeTitle = chats.find((c) => c.id === activeChat)?.title ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          Med<span>RAG</span>
        </h1>
        <div className="system muted small">
          {info && <span>LLM: {info.llm === "retrieval_only" ? "none (retrieval only)" : info.llm}</span>}
          <span className="account">
            <span title={user.email}>{user.name ?? user.email}</span>
            <button className="text-button" onClick={() => void handleSignOut()}>
              Sign out
            </button>
          </span>
        </div>
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
        <ChatList
          chats={chats}
          total={chatsTotal}
          activeId={activeChat}
          loadingMore={loadingChats}
          onSelect={(id) => void openChat(id)}
          onNew={() => void openChat(null)}
          onLoadMore={handleLoadMoreChats}
          onRename={handleRenameChat}
          onDelete={handleDeleteChat}
        />
        <ChatPanel
          title={activeChat === null ? null : activeTitle}
          results={results}
          hasOlder={results.length < historyTotal}
          loadingOlder={loadingOlder}
          pending={pending && pendingChat === activeChat ? pending : null}
          busy={pending !== null}
          canAsk={hasReadyDocument}
          onAsk={handleAsk}
          onLoadOlder={handleLoadOlder}
          onDeleteQuestion={handleDeleteQuestion}
        />
        <DocumentPanel
          documents={documents}
          selected={selected}
          onToggle={toggle}
          onUpload={handleUpload}
          onDelete={handleDeleteDocument}
          maxUploadMb={info?.max_upload_mb}
          ocr={info?.ocr}
        />
      </main>

      <footer className="muted small">
        Answers come only from your uploaded documents and are not medical advice.
      </footer>
    </div>
  );
}
