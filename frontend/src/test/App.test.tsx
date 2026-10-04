import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, type QueryResult, type StreamHandlers } from "../api";
import App, { CHATS_PAGE, HISTORY_PAGE } from "../App";
import { alice, chat, doc, result, source } from "./fixtures";

const info = {
  llm: "gemini:gemini-3.8-flash -> groq:openai/gpt-oss-120b",
  embedding_model: "all-MiniLM-L6-v2",
  chunk_size: 512,
  chunk_overlap: 128,
  top_k: 4,
  min_similarity: 0.35,
  ocr: true,
  max_upload_mb: 25,
  documents: 1,
  chunks: 5,
};

const unauthorized = () => new ApiError("Not signed in", 401);

beforeEach(() => {
  vi.spyOn(api.auth, "me").mockResolvedValue(alice);
  vi.spyOn(api.auth, "config").mockResolvedValue({ allow_registration: true, password_min_length: 8 });
  vi.spyOn(api, "info").mockResolvedValue(info);
  vi.spyOn(api, "listDocuments").mockResolvedValue([doc(1)]);
  vi.spyOn(api.chats, "list").mockResolvedValue({ items: [], total: 0 });
  vi.spyOn(api.chats, "messages").mockResolvedValue({ items: [], total: 0 });
});

/** Mock askStream so the test controls when tokens arrive and when the answer is done. */
function controlledStream() {
  let handlers: StreamHandlers = {};
  let finish: (r: QueryResult) => void = () => {};
  const askStream = vi.spyOn(api, "askStream").mockImplementation((_q, _ids, _chat, h = {}) => {
    handlers = h;
    return new Promise((resolve) => (finish = resolve));
  });
  return { askStream, handlers: () => handlers, finish: (r: QueryResult) => finish(r) };
}

async function questionBox() {
  const box = await screen.findByRole("textbox", { name: "Question" });
  await waitFor(() => expect(box).toBeEnabled());
  return box;
}

describe("signing in and out", () => {
  it("shows the sign-in screen without a session, then the workspace", async () => {
    vi.spyOn(api.auth, "me").mockRejectedValue(unauthorized());
    vi.spyOn(api.auth, "login").mockResolvedValue(alice);
    const user = userEvent.setup();
    render(<App />);

    await user.type(await screen.findByLabelText("Email"), "alice@example.org");
    await user.type(screen.getByLabelText("Password"), "correct horse");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    const header = await screen.findByRole("banner");
    expect(within(header).getByText("Alice")).toBeInTheDocument();
    expect(api.chats.list).toHaveBeenCalledWith(CHATS_PAGE);
  });

  it("offers a retry instead of the sign-in screen when the server is unreachable", async () => {
    const me = vi
      .spyOn(api.auth, "me")
      .mockRejectedValueOnce(new ApiError("502 Bad Gateway", 502))
      .mockResolvedValueOnce(alice);
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText(/Can't reach the MedRAG server/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sign in" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(me).toHaveBeenCalledTimes(2);
  });

  it("signs out", async () => {
    const logout = vi.spyOn(api.auth, "logout").mockResolvedValue();
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(logout).toHaveBeenCalledOnce();
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });

  it("returns to the sign-in screen when the session expires", async () => {
    vi.spyOn(api, "askStream").mockRejectedValue(unauthorized());
    const user = userEvent.setup();
    render(<App />);
    await user.type(await questionBox(), "Q?{Enter}");
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("chats", () => {
  it("opens the most recent chat with its newest messages, oldest first", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7, { title: "Asthma" }), chat(3)], total: 2 });
    const messages = vi
      .spyOn(api.chats, "messages")
      .mockResolvedValueOnce({ items: [result(25), result(24)], total: 3 })
      .mockResolvedValueOnce({ items: [result(23)], total: 3 });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Asthma" })).toBeInTheDocument();
    const questions = await screen.findAllByText(/^Question \d+\?$/);
    expect(questions.map((q) => q.textContent)).toEqual(["Question 24?", "Question 25?"]);
    expect(messages).toHaveBeenCalledWith(7, HISTORY_PAGE);

    await user.click(screen.getByRole("button", { name: "Load earlier questions" }));
    await waitFor(() =>
      expect(screen.getAllByText(/^Question \d+\?$/).map((q) => q.textContent)).toEqual([
        "Question 23?",
        "Question 24?",
        "Question 25?",
      ]),
    );
    expect(messages).toHaveBeenLastCalledWith(7, HISTORY_PAGE, 2);
  });

  it("switches between chats", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7, { title: "Asthma" }), chat(3, { title: "Diabetes" })], total: 2 });
    vi.spyOn(api.chats, "messages").mockImplementation(async (id) => ({
      items: [result(id * 10, { chat_id: id, question: `In chat ${id}?` })],
      total: 1,
    }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("In chat 7?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Diabetes" }));
    expect(await screen.findByText("In chat 3?")).toBeInTheDocument();
    expect(screen.queryByText("In chat 7?")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Diabetes" })).toBeInTheDocument();
  });

  it("starts a new chat with the first question and adds it to the list", async () => {
    const stream = controlledStream();
    vi.spyOn(api.chats, "get").mockResolvedValue(chat(42, { title: "What is PEM?" }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("heading", { name: "New chat" })).toBeInTheDocument();
    await user.type(await questionBox(), "What is PEM?{Enter}");
    expect(stream.askStream).toHaveBeenCalledWith("What is PEM?", null, null, expect.any(Object));
    expect(screen.getByText("Searching documents…")).toBeInTheDocument();

    stream.handlers().onChat?.(42);
    stream.handlers().onSources?.([source()]);
    stream.handlers().onToken?.("Post-exertional ");
    stream.handlers().onToken?.("malaise");
    expect(await screen.findByText("Post-exertional malaise")).toHaveClass("streaming");

    stream.finish(result(9, { chat_id: 42, question: "What is PEM?", answer: "Post-exertional malaise.", llm: "groq:x" }));
    expect(await screen.findByText("Post-exertional malaise.")).not.toHaveClass("streaming");
    const chats = screen.getByRole("navigation", { name: "Chats" });
    expect(await within(chats).findByRole("button", { name: "What is PEM?" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("heading", { name: "What is PEM?" })).toBeInTheDocument();
  });

  it("sends follow-up questions to the open chat and limits the search to ticked documents", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7)], total: 1 });
    const askStream = vi.spyOn(api, "askStream").mockResolvedValue(result(1, { chat_id: 7 }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("checkbox", { name: "Search only doc1.pdf" }));
    await user.type(await questionBox(), "Q?{Enter}");
    expect(askStream).toHaveBeenCalledWith("Q?", [1], 7, expect.any(Object));
  });

  it("keeps a streaming answer in its own chat when another chat is opened", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7, { title: "Asthma" }), chat(3, { title: "Diabetes" })], total: 2 });
    const stream = controlledStream();
    const user = userEvent.setup();
    render(<App />);

    await user.type(await questionBox(), "Slow question?{Enter}");
    stream.handlers().onChat?.(7);
    stream.handlers().onToken?.("Partial");
    expect(await screen.findByText("Partial")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Diabetes" }));
    await waitFor(() => expect(screen.queryByText("Partial")).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();

    stream.finish(result(5, { chat_id: 7, question: "Slow question?" }));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Question" })).toBeEnabled());
    expect(screen.queryByText("Slow question?")).not.toBeInTheDocument();
  });

  it("deletes the open chat and starts a new one", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7, { title: "Asthma" })], total: 1 });
    vi.spyOn(api.chats, "messages").mockResolvedValue({ items: [result(1, { chat_id: 7 })], total: 1 });
    const remove = vi.spyOn(api.chats, "remove").mockResolvedValue();
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Delete chat: Asthma" }));
    await user.click(screen.getByRole("button", { name: "Yes" }));
    expect(remove).toHaveBeenCalledWith(7);
    expect(await screen.findByRole("heading", { name: "New chat" })).toBeInTheDocument();
    expect(screen.getByText("No chats yet. Ask a question to start one.")).toBeInTheDocument();
  });

  it("renames a chat", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7, { title: "Asthma" })], total: 1 });
    vi.spyOn(api.chats, "rename").mockResolvedValue(chat(7, { title: "Asthma clinic" }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Rename chat: Asthma" }));
    await user.type(screen.getByRole("textbox", { name: "Chat title" }), " clinic{Enter}");
    expect(api.chats.rename).toHaveBeenCalledWith(7, "Asthma clinic");
    expect(await screen.findByRole("heading", { name: "Asthma clinic" })).toBeInTheDocument();
  });

  it("loads more chats", async () => {
    vi.spyOn(api.chats, "list")
      .mockResolvedValueOnce({ items: [chat(9)], total: 2 })
      .mockResolvedValueOnce({ items: [chat(8)], total: 2 });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Show more (1)" }));
    expect(await screen.findByRole("button", { name: "Chat 8" })).toBeInTheDocument();
    expect(api.chats.list).toHaveBeenLastCalledWith(CHATS_PAGE, 1);
    expect(screen.queryByRole("button", { name: /Show more/ })).not.toBeInTheDocument();
  });

  it("deletes a single question", async () => {
    vi.spyOn(api.chats, "list").mockResolvedValue({ items: [chat(7)], total: 1 });
    vi.spyOn(api.chats, "messages").mockResolvedValue({ items: [result(2), result(1)], total: 2 });
    const deleteQuery = vi.spyOn(api, "deleteQuery").mockResolvedValue();
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Delete question: Question 1?" }));
    expect(deleteQuery).toHaveBeenCalledWith(1);
    await waitFor(() => expect(screen.queryByText("Question 1?")).not.toBeInTheDocument());
    expect(screen.getByText("Question 2?")).toBeInTheDocument();
  });
});

describe("errors and system info", () => {
  it("shows an error banner when asking fails", async () => {
    vi.spyOn(api, "askStream").mockRejectedValue(new ApiError("Embedding model unavailable", 503));
    const user = userEvent.setup();
    render(<App />);
    await user.type(await questionBox(), "Q?{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Embedding model unavailable");
  });

  it("shows the active LLM chain", async () => {
    render(<App />);
    const header = await screen.findByRole("banner");
    expect(await within(header).findByText(/gemini:gemini-3\.8-flash -> groq/)).toBeInTheDocument();
  });
});
