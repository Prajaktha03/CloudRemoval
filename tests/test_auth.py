from pathlib import Path

import pytest

from backend.services.auth import AuthStore


def test_auth_store_hashes_passwords_and_revokes_sessions(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.create_user("Researcher@example.org", "correct horse battery 42")

    assert store.login("researcher@example.org", "not the password", "127.0.0.1") is None
    authenticated = store.login("RESEARCHER@example.org", "correct horse battery 42", "127.0.0.1")
    assert authenticated is not None
    token, user = authenticated
    assert user["email"] == "researcher@example.org"
    assert store.get_user(token) == user
    store.logout(token)
    assert store.get_user(token) is None


@pytest.mark.parametrize("password", ["short", "x" * 129])
def test_auth_store_rejects_weak_or_oversized_passwords(tmp_path: Path, password: str):
    store = AuthStore(tmp_path / "auth.sqlite3")
    with pytest.raises(ValueError, match="Password must"):
        store.create_user("researcher@example.org", password)
