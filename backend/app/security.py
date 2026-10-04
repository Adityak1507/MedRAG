"""Password hashing, session tokens and login throttling."""

import hashlib
import secrets
import threading
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()
# Verified when the email is unknown, so a login attempt takes as long whether or not the account exists
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Sessions are looked up by this hash, so a leaked database doesn't leak usable tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


class LoginThrottle:
    """Counts recent failed logins per key in memory. Per process: run one backend worker, or put a shared
    limiter (e.g. at the reverse proxy) in front when scaling out."""

    def __init__(self) -> None:
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str, window: float) -> deque[float]:
        failures = self._failures[key]
        while failures and failures[0] < time.monotonic() - window:
            failures.popleft()
        return failures

    def blocked(self, key: str, max_failures: int, window: float) -> bool:
        with self._lock:
            return len(self._recent(key, window)) >= max_failures

    def failed(self, key: str) -> None:
        with self._lock:
            self._failures[key].append(time.monotonic())

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._failures.clear()
            else:
                self._failures.pop(key, None)


login_throttle = LoginThrottle()
