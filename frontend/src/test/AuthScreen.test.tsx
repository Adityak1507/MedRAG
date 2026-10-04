import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api";
import { AuthScreen } from "../components/AuthScreen";
import { alice } from "./fixtures";

beforeEach(() => {
  vi.spyOn(api.auth, "config").mockResolvedValue({ allow_registration: true, password_min_length: 8 });
});

function setup() {
  const onSignedIn = vi.fn();
  render(<AuthScreen onSignedIn={onSignedIn} />);
  return { onSignedIn, user: userEvent.setup() };
}

describe("AuthScreen", () => {
  it("signs in", async () => {
    const login = vi.spyOn(api.auth, "login").mockResolvedValue(alice);
    const { onSignedIn, user } = setup();
    await user.type(screen.getByLabelText("Email"), " alice@example.org ");
    await user.type(screen.getByLabelText("Password"), "correct horse");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(login).toHaveBeenCalledWith("alice@example.org", "correct horse");
    expect(onSignedIn).toHaveBeenCalledWith(alice);
  });

  it("shows the server's error and stays on the form", async () => {
    vi.spyOn(api.auth, "login").mockRejectedValue(new ApiError("Wrong email or password", 401));
    const { onSignedIn, user } = setup();
    await user.type(screen.getByLabelText("Email"), "alice@example.org");
    await user.type(screen.getByLabelText("Password"), "nope nope");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Wrong email or password");
    expect(onSignedIn).not.toHaveBeenCalled();
  });

  it("creates an account", async () => {
    const register = vi.spyOn(api.auth, "register").mockResolvedValue(alice);
    const { onSignedIn, user } = setup();
    await user.click(await screen.findByRole("button", { name: "Create an account" }));
    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/Name/), "Alice");
    await user.type(screen.getByLabelText("Email"), "alice@example.org");
    await user.type(screen.getByLabelText("Password"), "correct horse");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(register).toHaveBeenCalledWith("alice@example.org", "correct horse", "Alice");
    expect(onSignedIn).toHaveBeenCalledWith(alice);
  });

  it("enforces the minimum password length when registering", async () => {
    const register = vi.spyOn(api.auth, "register").mockResolvedValue(alice);
    const { user } = setup();
    await user.click(await screen.findByRole("button", { name: "Create an account" }));
    await user.type(screen.getByLabelText("Email"), "alice@example.org");
    await user.type(screen.getByLabelText("Password"), "short");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect(register).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("at least 8 characters");
  });

  it("hides registration when the server has it turned off", async () => {
    vi.spyOn(api.auth, "config").mockResolvedValue({ allow_registration: false, password_min_length: 8 });
    setup();
    expect(await screen.findByText("Ask your administrator for an account.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create an account" })).not.toBeInTheDocument();
  });
});
