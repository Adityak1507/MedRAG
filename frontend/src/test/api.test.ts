import { describe, expect, it, vi } from "vitest";
import { ApiError, api, isUnauthorized, readEventStream } from "../api";
import { alice, chat, jsonResponse, result, source, sse, streamingResponse } from "./fixtures";

async function collect(pieces: string[]) {
  const events: [string, string][] = [];
  await readEventStream(streamingResponse(pieces).body!, (event, data) => events.push([event, data]));
  return events;
}

describe("readEventStream", () => {
  it("parses events split across network chunks", async () => {
    const events = await collect(["event: tok", 'en\ndata: {"text":"Hel', 'lo"}\n', "\nevent: done\ndata: {}\n\n"]);
    expect(events).toEqual([
      ["token", '{"text":"Hello"}'],
      ["done", "{}"],
    ]);
  });

  it("handles CRLF line endings, multi-line data and a missing final blank line", async () => {
    const events = await collect(["event: a\r\ndata: one\r\ndata: two\r\n\r\n", "data: last"]);
    expect(events).toEqual([
      ["a", "one\ntwo"],
      ["message", "last"],
    ]);
  });

  it("decodes multi-byte characters split between chunks", async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"text":"Post‑exertional"}\n\n');
    const cut = bytes.indexOf(0xe2) + 1; // inside the 3-byte non-breaking hyphen
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, cut));
        c.enqueue(bytes.slice(cut));
        c.close();
      },
    });
    const events: string[] = [];
    await readEventStream(body, (_, data) => events.push(JSON.parse(data).text));
    expect(events).toEqual(["Post‑exertional"]);
  });
});

describe("api.askStream", () => {
  it("reports sources and tokens as they arrive and resolves with the saved result", async () => {
    const saved = result(7, { answer: "Hello world" });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      streamingResponse([
        sse("sources", { chat_id: 12, sources: [source()] }),
        sse("token", { text: "Hello " }),
        sse("token", { text: "world" }),
        sse("done", saved),
      ]),
    );
    const onSources = vi.fn();
    const onChat = vi.fn();
    const tokens: string[] = [];

    const out = await api.askStream("Q?", [3], 12, { onChat, onSources, onToken: (t: string) => tokens.push(t) });

    expect(out).toEqual(saved);
    expect(onChat).toHaveBeenCalledWith(12);
    expect(onSources).toHaveBeenCalledWith([source()]);
    expect(tokens).toEqual(["Hello ", "world"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/query/stream");
    expect(init!.credentials).toBe("same-origin");
    expect(JSON.parse(init!.body as string)).toEqual({ question: "Q?", document_ids: [3], chat_id: 12 });
  });

  it("throws the server's error event", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      streamingResponse([sse("sources", { sources: [] }), sse("error", { detail: "Could not save the answer" })]),
    );
    await expect(api.askStream("Q?", null, null)).rejects.toThrow("Could not save the answer");
  });

  it("throws if the stream ends without a result", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(streamingResponse([sse("token", { text: "partial" })]));
    await expect(api.askStream("Q?", null, null)).rejects.toThrow("ended unexpectedly");
  });

  it("surfaces HTTP errors before streaming starts", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Embedding model unavailable" }), { status: 503 }),
    );
    await expect(api.askStream("Q?", null, null)).rejects.toThrow("Embedding model unavailable");
  });
});

describe("history endpoints", () => {
  it("reads the page and the total count header", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify([result(2), result(1)]), { headers: { "X-Total-Count": "42" } }));
    const page = await api.history(2, 4);
    expect(page.total).toBe(42);
    expect(page.items.map((r) => r.id)).toEqual([2, 1]);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/queries?limit=2&offset=4");
  });

  it("deletes one question or the whole history", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(null, { status: 204 }));
    await api.deleteQuery(5);
    await api.clearHistory();
    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method])).toEqual([
      ["/api/queries/5", "DELETE"],
      ["/api/queries", "DELETE"],
    ]);
  });
});

describe("errors", () => {
  it("carries the HTTP status, so a lost session can be told apart", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse({ detail: "Not signed in" }, { status: 401 }));
    const error = await api.info().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toBe("Not signed in");
    expect(isUnauthorized(error)).toBe(true);
    expect(isUnauthorized(new ApiError("Server error", 500))).toBe(false);
  });
});

describe("auth endpoints", () => {
  it("signs in, registers and signs out", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (url) =>
        String(url).endsWith("logout") ? new Response(null, { status: 204 }) : jsonResponse({ user: alice, token: "t" }),
      );
    expect(await api.auth.login("alice@example.org", "pw")).toEqual(alice);
    expect(await api.auth.register("bob@example.org", "pw", null)).toEqual(alice);
    await api.auth.logout();
    const calls = fetchMock.mock.calls.map(([url, init]) => [url, init?.method, init?.body && JSON.parse(String(init.body))]);
    expect(calls).toEqual([
      ["/api/auth/login", "POST", { email: "alice@example.org", password: "pw" }],
      ["/api/auth/register", "POST", { email: "bob@example.org", password: "pw", name: null }],
      ["/api/auth/logout", "POST", undefined],
    ]);
  });
});

describe("chat endpoints", () => {
  it("pages chats and messages with the total from X-Total-Count", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (url) =>
        String(url).includes("messages")
          ? jsonResponse([result(4)], { headers: { "X-Total-Count": "9" } })
          : jsonResponse([chat(1), chat(2)], { headers: { "X-Total-Count": "31" } }),
      );
    expect(await api.chats.list(2, 4)).toEqual({ items: [chat(1), chat(2)], total: 31 });
    expect(await api.chats.messages(7, 1, 3)).toEqual({ items: [result(4)], total: 9 });
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/chats?limit=2&offset=4",
      "/api/chats/7/messages?limit=1&offset=3",
    ]);
  });

  it("renames and deletes chats", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (_url, init) =>
        init?.method === "DELETE" ? new Response(null, { status: 204 }) : jsonResponse(chat(3, { title: "Renamed" })),
      );
    expect((await api.chats.rename(3, "Renamed")).title).toBe("Renamed");
    await api.chats.remove(3);
    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method])).toEqual([
      ["/api/chats/3", "PATCH"],
      ["/api/chats/3", "DELETE"],
    ]);
  });
});
