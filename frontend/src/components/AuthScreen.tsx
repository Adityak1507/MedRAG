import { useEffect, useState } from "react";
import { api, type AuthConfig, type User } from "../api";

interface Props {
  onSignedIn: (user: User) => void;
}

export function AuthScreen({ onSignedIn }: Props) {
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.auth.config().then(setConfig).catch(() => setConfig({ allow_registration: false, password_min_length: 8 }));
  }, []);

  const registering = mode === "register";
  const minLength = config?.password_min_length ?? 8;

  async function submit(e: { preventDefault(): void }) {
    e.preventDefault();
    if (registering && password.length < minLength) {
      setError(`Use a password of at least ${minLength} characters.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const user = registering
        ? await api.auth.register(email.trim(), password, name.trim() || null)
        : await api.auth.login(email.trim(), password);
      onSignedIn(user);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function switchMode(next: "login" | "register") {
    setMode(next);
    setError(null);
  }

  return (
    <div className="auth-page">
      <form className="panel auth-card" onSubmit={submit}>
        <h1>
          Med<span>RAG</span>
        </h1>
        <p className="muted">
          {registering
            ? "Create an account. Your documents and chats are private to you."
            : "Sign in to your documents and chats."}
        </p>

        {registering && (
          <label>
            Name <span className="muted small">(optional)</span>
            <input value={name} maxLength={100} autoComplete="name" onChange={(e) => setName(e.target.value)} />
          </label>
        )}
        <label>
          Email
          <input
            type="email"
            required
            value={email}
            autoComplete="email"
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>
        <label>
          Password
          <input
            type="password"
            required
            minLength={registering ? minLength : undefined}
            value={password}
            autoComplete={registering ? "new-password" : "current-password"}
            aria-describedby={registering ? "password-hint" : undefined}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {registering && (
          <span id="password-hint" className="muted small password-hint">
            At least {minLength} characters.
          </span>
        )}

        {error && (
          <p className="error small" role="alert">
            {error}
          </p>
        )}

        <button type="submit" className="primary" disabled={busy}>
          {busy ? "Please wait…" : registering ? "Create account" : "Sign in"}
        </button>

        {registering ? (
          <p className="small">
            Already have an account?{" "}
            <button type="button" className="text-button" onClick={() => switchMode("login")}>
              Sign in
            </button>
          </p>
        ) : config?.allow_registration ? (
          <p className="small">
            New here?{" "}
            <button type="button" className="text-button" onClick={() => switchMode("register")}>
              Create an account
            </button>
          </p>
        ) : (
          config && <p className="muted small">Ask your administrator for an account.</p>
        )}
      </form>
    </div>
  );
}
