"""Browser sessions are revocable credentials, separate from the API key."""
import hashlib
import hmac
import logging
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.test_ui_routes import client, ui_app  # noqa: F401

KEY = "test-key-xyz"
COOKIE = "cortex_ui"


@pytest.fixture
def sessions():
    from src.ui.session_store import store
    return store


def login(client):
    response = client.post("/ui/login", data={"key": KEY}, follow_redirects=False)
    assert response.status_code == 303
    return client.cookies.get(COOKIE), response


def test_cookie_is_a_unique_session(client):
    first, _ = login(client)
    assert KEY not in first and len(first) >= 40
    second, _ = login(client)
    assert second != first
    from src.ui.session_store import store
    for token in (first, second):
        assert store.valid(token, KEY)
        client.cookies.clear()
        client.cookies.set(COOKIE, token)
        for path in ("/ui/novo", "/ui/data", "/ui/effects"):
            assert client.get(path).status_code == 200


def test_raw_key_cookie_is_refused(client):
    client.cookies.set(COOKIE, KEY)
    for path in ("/ui/novo", "/ui/data"):
        assert client.get(path).status_code == 401


def test_logout_revokes_only_its_session(client, sessions):
    other, _ = login(client)
    token, _ = login(client)
    assert client.get("/ui/logout").status_code == 200
    client.cookies.clear()
    client.cookies.set(COOKIE, token)
    assert client.get("/ui/data").status_code == 401
    assert not sessions.valid(token, KEY)
    assert sessions.valid(other, KEY)


def test_expiry_is_read_only_and_next_issue_purges(client, sessions):
    token, _ = login(client)
    conn = sessions._conn
    row = conn.execute("SELECT * FROM sessions").fetchone()
    created, expires = map(datetime.fromisoformat, (row["created_at"], row["expires_at"]))
    assert created.utcoffset() == timedelta(0) and expires - created == timedelta(days=30)
    with conn:
        conn.execute("UPDATE sessions SET expires_at = ?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),))
    assert client.get("/ui/data").status_code == 401
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    new, _ = login(client)
    assert new != token
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    assert sessions.valid(new, KEY) and not sessions.valid(token, KEY)


def test_key_rotation(client, monkeypatch):
    login(client)
    monkeypatch.setenv("CORTEX_API_KEY", "rotated-key")
    assert client.get("/ui/data").status_code == 401
    assert client.get("/ui/data", headers={"X-Cortex-Key": "rotated-key"}).status_code == 200


def test_secure_cookie_follows_request_scheme(client, ui_app):
    with TestClient(ui_app, base_url="https://testserver") as https:
        _, secure = login(https)
        _, plain = login(client)
        assert "; secure" in secure.headers["set-cookie"].lower()
        assert "; secure" not in plain.headers["set-cookie"].lower()
        for response in (secure, plain):
            cookie = response.headers["set-cookie"].lower()
            assert "httponly" in cookie and "samesite=lax" in cookie
            assert "max-age=2592000" in cookie and "path=/" in cookie
        deleted = https.get("/ui/logout").headers["set-cookie"].lower()
        assert "; secure" in deleted and "max-age=0" in deleted


def test_only_keyed_digests_at_rest(client, sessions):
    tokens = [login(client)[0] for _ in range(2)]
    rows = sessions._conn.execute("SELECT digest FROM sessions").fetchall()
    assert {r[0] for r in rows} == {
        hmac.new(KEY.encode(), token.encode(), hashlib.sha256).hexdigest() for token in tokens}
    path = Path(sessions.db_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for suffix in ("", "-wal", "-journal"):
        file = Path(str(path) + suffix)
        if file.exists():
            content = file.read_bytes()
            assert all(secret.encode() not in content for secret in [KEY, *tokens])


def test_headers_and_ask_still_require_the_key(client, monkeypatch):
    token, _ = login(client)
    client.cookies.clear()
    assert client.get("/ui/data", headers={"X-Cortex-Key": KEY}).status_code == 200
    assert client.get("/ui/data", headers={"X-Cortex-Key": token}).status_code == 401
    import src.api as api
    monkeypatch.setattr(api, "SECRET_API_KEY", KEY)
    request = SimpleNamespace(client=SimpleNamespace(host="session-test"),
                              url=SimpleNamespace(path="/ask"))
    api._failed_auth.clear()
    try:
        assert api.verify_access(request, api_key=KEY) == KEY
        with pytest.raises(HTTPException) as exc:
            api.verify_access(request, api_key=token)
        assert exc.value.status_code == 403
    finally:
        api._failed_auth.clear()


def test_storage_failure_is_soft_and_auth_fails_closed(client, sessions, tmp_path, monkeypatch):
    token, _ = login(client)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(sessions, "db_path", str(blocked / "sessions.db"))
    monkeypatch.setattr(sessions, "_conn", None)
    response = client.post("/ui/login", data={"key": KEY}, follow_redirects=False)
    assert response.status_code == 503 and "<form" in response.text
    assert "set-cookie" not in response.headers
    assert client.get("/ui/data").status_code == 401
    assert client.get("/ui/data", headers={"X-Cortex-Key": KEY}).status_code == 200
    assert sessions.revoke(token, KEY) is False
    assert client.get("/ui/logout").status_code == 200


def test_logs_and_cross_origin_login(client, sessions, caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("cortex.auth"), "propagate", True)
    caplog.set_level(logging.INFO, logger="cortex.auth")
    token, _ = login(client)
    client.cookies.clear()
    bad = "invalid-session-that-must-not-appear-in-logs"
    client.cookies.set(COOKIE, bad)
    assert client.get("/ui/data").status_code == 401
    client.cookies.set(COOKIE, token)
    client.get("/ui/logout")
    before = sessions._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    response = client.post("/ui/login", data={"key": KEY},
                           headers={"Origin": "https://foreign.example"})
    assert response.status_code == 403 and "set-cookie" not in response.headers
    assert sessions._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == before
    assert "reason=ui-bad-session" in caplog.text
    assert all(secret not in caplog.text for secret in (KEY, token, bad))


def test_invalid_tokens_never_open_storage(sessions, monkeypatch):
    def unexpected_connect():
        pytest.fail("invalid token touched storage")
    monkeypatch.setattr(sessions, "_connect", unexpected_connect)
    for token in (None, 123, b"bytes", "", "x" * 257):
        assert sessions.valid(token, KEY) is False
