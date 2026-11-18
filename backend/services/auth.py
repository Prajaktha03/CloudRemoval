from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path


SESSION_COOKIE_NAME = "orbit_session"
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
SESSION_TOKEN_BYTES = 32
PASSWORD_SCRYPT_N = 2**14
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 10
_DUMMY_PASSWORD_HASH = None


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=PASSWORD_SCRYPT_N, r=8, p=1)
    return "scrypt${}${}${}${}${}".format(
        PASSWORD_SCRYPT_N,
        8,
        1,
        _encode(salt),
        _encode(digest),
    )


def _encode(value: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    import base64

    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
        )
    except (ValueError, TypeError, UnicodeError):
        return False
    return hmac.compare_digest(actual, _decode(expected))


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class AuthStore:
    def __init__(self, path: Path, session_max_age_seconds: int = 12 * 60 * 60) -> None:
        self.path = path
        self.session_max_age_seconds = session_max_age_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE INDEX IF NOT EXISTS auth_sessions_expiry ON auth_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS auth_failures (
                    source_hash TEXT NOT NULL,
                    attempted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS auth_failures_window ON auth_failures(source_hash, attempted_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_user(self, email: str, password: str) -> None:
        normalized = normalize_email(email)
        if (
            not normalized
            or len(normalized) > 254
            or normalized.count("@") != 1
            or normalized.startswith("@")
            or normalized.endswith("@")
        ):
            raise ValueError("Enter a valid email address")
        if len(password) < MIN_PASSWORD_LENGTH or len(password) > MAX_PASSWORD_LENGTH:
            raise ValueError(f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters")
        password_hash = _password_hash(password)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (normalized, password_hash),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with that email already exists") from exc

    def login(self, email: str, password: str, source: str) -> tuple[str, dict[str, object]] | None:
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        window_start = (now - timedelta(seconds=LOGIN_WINDOW_SECONDS)).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM auth_failures WHERE attempted_at < ?", (window_start,))
            failure_count = connection.execute(
                "SELECT COUNT(*) FROM auth_failures WHERE source_hash = ? AND attempted_at >= ?",
                (source_hash, window_start),
            ).fetchone()[0]
            if failure_count >= LOGIN_MAX_FAILURES:
                connection.commit()
                return None

            user = connection.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ? COLLATE NOCASE",
                (normalize_email(email),),
            ).fetchone()
            password_hash = user["password_hash"] if user is not None else _dummy_hash()
            password_valid = _verify_password(password, password_hash)
            valid = user is not None and password_valid
            if not valid:
                connection.execute(
                    "INSERT INTO auth_failures (source_hash, attempted_at) VALUES (?, ?)",
                    (source_hash, now.isoformat()),
                )
                connection.commit()
                return None

            connection.execute("DELETE FROM auth_failures WHERE source_hash = ?", (source_hash,))
            connection.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now.isoformat(),))
            token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
            expires_at = (now + timedelta(seconds=self.session_max_age_seconds)).isoformat()
            connection.execute(
                "INSERT INTO auth_sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (_token_digest(token), user["id"], expires_at),
            )
            connection.commit()
        return token, {"id": user["id"], "email": user["email"]}

    def get_user(self, token: str | None) -> dict[str, object] | None:
        if not token:
            return None
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """SELECT users.id, users.email
                   FROM auth_sessions
                   JOIN users ON users.id = auth_sessions.user_id
                   WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?""",
                (_token_digest(token), now),
            ).fetchone()
        return {"id": row["id"], "email": row["email"]} if row else None

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_digest(token),))


def _dummy_hash() -> str:
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        _DUMMY_PASSWORD_HASH = _password_hash(secrets.token_urlsafe(32))
    return _DUMMY_PASSWORD_HASH
