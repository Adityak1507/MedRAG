import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatList, relativeTime } from "../components/ChatList";
import { chat } from "./fixtures";

function setup(overrides: Partial<Parameters<typeof ChatList>[0]> = {}) {
  const props = {
    chats: [chat(1, { title: "Asthma questions" }), chat(2, { title: "Diabetes" })],
    total: 2,
    activeId: 1 as number | null,
    loadingMore: false,
    onSelect: vi.fn(),
    onNew: vi.fn(),
    onLoadMore: vi.fn().mockResolvedValue(undefined),
    onRename: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
  render(<ChatList {...props} />);
  return { props, user: userEvent.setup() };
}

describe("ChatList", () => {
  it("lists chats, marks the open one and opens others", async () => {
    const { props, user } = setup();
    expect(screen.getByRole("button", { name: "Asthma questions" })).toHaveAttribute("aria-current", "page");
    await user.click(screen.getByRole("button", { name: "Diabetes" }));
    expect(props.onSelect).toHaveBeenCalledWith(2);
  });

  it("starts a new chat", async () => {
    const { props, user } = setup();
    await user.click(screen.getByRole("button", { name: "+ New chat" }));
    expect(props.onNew).toHaveBeenCalledOnce();
  });

  it("shows an empty state", () => {
    setup({ chats: [], total: 0, activeId: null });
    expect(screen.getByText("No chats yet. Ask a question to start one.")).toBeInTheDocument();
  });

  it("loads more chats when there are more on the server", async () => {
    const { props, user } = setup({ total: 32 });
    await user.click(screen.getByRole("button", { name: "Show more (30)" }));
    expect(props.onLoadMore).toHaveBeenCalledOnce();
  });

  it("hides the load button when every chat is shown", () => {
    setup({ total: 2 });
    expect(screen.queryByRole("button", { name: /Show more/ })).not.toBeInTheDocument();
  });

  it("renames with Enter, and Escape cancels", async () => {
    const { props, user } = setup();
    await user.click(screen.getByRole("button", { name: "Rename chat: Diabetes" }));
    const input = screen.getByRole("textbox", { name: "Chat title" });
    await user.clear(input);
    await user.type(input, "Type 2 diabetes{Enter}");
    expect(props.onRename).toHaveBeenCalledWith(2, "Type 2 diabetes");

    await user.dblClick(screen.getByRole("button", { name: "Asthma questions" }));
    await user.type(screen.getByRole("textbox", { name: "Chat title" }), " edited{Escape}");
    expect(props.onRename).toHaveBeenCalledTimes(1);
  });

  it("doesn't save an empty or unchanged title", async () => {
    const { props, user } = setup();
    await user.click(screen.getByRole("button", { name: "Rename chat: Diabetes" }));
    await user.type(screen.getByRole("textbox", { name: "Chat title" }), "{Enter}");
    await user.click(screen.getByRole("button", { name: "Rename chat: Diabetes" }));
    await user.clear(screen.getByRole("textbox", { name: "Chat title" }));
    await user.keyboard("{Enter}");
    expect(props.onRename).not.toHaveBeenCalled();
  });

  it("asks before deleting a chat", async () => {
    const { props, user } = setup();
    await user.click(screen.getByRole("button", { name: "Delete chat: Diabetes" }));
    expect(props.onDelete).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "No" }));
    expect(screen.queryByText("Delete?")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete chat: Diabetes" }));
    await user.click(screen.getByRole("button", { name: "Yes" }));
    expect(props.onDelete).toHaveBeenCalledWith(2);
  });
});

describe("relativeTime", () => {
  const now = Date.parse("2026-10-04T12:00:00Z");
  it.each([
    ["2026-10-04T11:59:50Z", "just now"],
    ["2026-10-04T11:45:00Z", "15 min ago"],
    ["2026-10-04T09:00:00Z", "3 h ago"],
    ["2026-10-02T12:00:00Z", "2 d ago"],
  ])("%s → %s", (iso, expected) => expect(relativeTime(iso, now)).toBe(expected));
});
