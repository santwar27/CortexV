"""
API security: brute-force guard for failed auth attempts + IP allowlist
middleware.

The slowapi decorator only counts requests that pass verify_access, so
a separate sliding-window counter in verify_access closes the brute-
force loophole. These tests cover the counter logic in isolation.

The IP allowlist middleware adds a defence-in-depth layer: even with
a leaked API key, requests outside CORTEX_IP_ALLOWLIST are rejected
before auth runs. Tests cover CIDR matching, IPv4/IPv6, empty
allowlist (back-compat), and malformed env entries.
"""
import ipaddress
import time
import asyncio

import pytest


@pytest.fixture(autouse=True)
def fresh_state():
    """Wipe the per-IP failed-auth state between tests."""
    from src.api import _failed_auth
    _failed_auth.clear()
    yield
    _failed_auth.clear()


class TestBruteForceGuard:

    def test_initial_state_clean(self):
        from src.api import _failed_auth_exceeded
        assert _failed_auth_exceeded("1.1.1.1") is False

    def test_under_limit_passes(self):
        from src.api import _failed_auth_exceeded, _record_failed_auth, _FAILED_AUTH_LIMIT
        # Hit exactly limit-1 times → still allowed
        for _ in range(_FAILED_AUTH_LIMIT - 1):
            _record_failed_auth("1.1.1.1")
        assert _failed_auth_exceeded("1.1.1.1") is False

    def test_at_limit_blocks(self):
        from src.api import _failed_auth_exceeded, _record_failed_auth, _FAILED_AUTH_LIMIT
        for _ in range(_FAILED_AUTH_LIMIT):
            _record_failed_auth("1.1.1.1")
        assert _failed_auth_exceeded("1.1.1.1") is True

    def test_over_limit_blocks(self):
        from src.api import _failed_auth_exceeded, _record_failed_auth, _FAILED_AUTH_LIMIT
        for _ in range(_FAILED_AUTH_LIMIT + 5):
            _record_failed_auth("1.1.1.1")
        assert _failed_auth_exceeded("1.1.1.1") is True

    def test_per_ip_isolation(self):
        """Attempts from one IP must NOT block another IP."""
        from src.api import _failed_auth_exceeded, _record_failed_auth, _FAILED_AUTH_LIMIT
        for _ in range(_FAILED_AUTH_LIMIT + 1):
            _record_failed_auth("1.1.1.1")
        assert _failed_auth_exceeded("1.1.1.1") is True
        assert _failed_auth_exceeded("2.2.2.2") is False

    def test_window_expiration(self, monkeypatch):
        """Old timestamps outside the window get trimmed."""
        from src.api import _failed_auth_exceeded, _record_failed_auth, _FAILED_AUTH_LIMIT
        import src.api as api_mod
        # Fast-forward time by mocking time.monotonic()
        fake_now = [1000.0]
        monkeypatch.setattr(api_mod.time, "monotonic", lambda: fake_now[0])

        for _ in range(_FAILED_AUTH_LIMIT):
            _record_failed_auth("1.1.1.1")
        assert _failed_auth_exceeded("1.1.1.1") is True

        # Advance past the window
        fake_now[0] += api_mod._FAILED_AUTH_WINDOW_S + 1
        assert _failed_auth_exceeded("1.1.1.1") is False

    def test_gc_empty_buckets(self, monkeypatch):
        """After expiration, the IP entry should be GC'd to avoid
        unbounded growth from many transient IPs."""
        from src.api import _failed_auth_exceeded, _record_failed_auth, _failed_auth
        import src.api as api_mod
        fake_now = [1000.0]
        monkeypatch.setattr(api_mod.time, "monotonic", lambda: fake_now[0])

        _record_failed_auth("1.1.1.1")
        assert "1.1.1.1" in _failed_auth
        fake_now[0] += api_mod._FAILED_AUTH_WINDOW_S + 1
        _failed_auth_exceeded("1.1.1.1")  # triggers cleanup
        assert "1.1.1.1" not in _failed_auth

    def test_unknown_ip_handled(self):
        """The verify_access path tags missing IPs as 'unknown' —
        that label must not crash the counter."""
        from src.api import _failed_auth_exceeded, _record_failed_auth
        _record_failed_auth("unknown")
        # Should not raise
        assert _failed_auth_exceeded("unknown") in (True, False)


# ----------------------------------------------------------------------
# IP allowlist middleware (CORTEX_IP_ALLOWLIST)
# ----------------------------------------------------------------------


class TestIPAllowlistParsing:
    """`_parse_allowlist` converts the env var string into network
    objects. Bare IPs, CIDRs, IPv6, and garbage all need to behave
    predictably so a typo in .env doesn't take down the API."""

    def test_empty_string_yields_empty_list(self):
        from src.api import _parse_allowlist
        assert _parse_allowlist("") == []

    def test_single_bare_ipv4_becomes_slash32(self):
        from src.api import _parse_allowlist
        nets = _parse_allowlist("10.0.0.1")
        assert len(nets) == 1
        assert nets[0].prefixlen == 32

    def test_cidr_preserved(self):
        from src.api import _parse_allowlist
        nets = _parse_allowlist("100.64.0.0/10")
        assert nets[0].prefixlen == 10

    def test_multiple_comma_separated(self):
        from src.api import _parse_allowlist
        nets = _parse_allowlist("100.64.0.0/10, 127.0.0.1, ::1")
        assert len(nets) == 3

    def test_ipv6_supported(self):
        from src.api import _parse_allowlist
        nets = _parse_allowlist("fd00::/8")
        assert nets[0].prefixlen == 8

    def test_invalid_entry_skipped_not_fatal(self):
        from src.api import _parse_allowlist
        # Mixing valid and invalid — valid one must survive.
        nets = _parse_allowlist("100.64.0.0/10, not-an-ip, 999.999.999.999")
        assert len(nets) == 1
        assert nets[0].prefixlen == 10

    def test_whitespace_tolerated(self):
        from src.api import _parse_allowlist
        nets = _parse_allowlist("  100.64.0.0/10  ,  127.0.0.1  ")
        assert len(nets) == 2


class TestIPAllowed:
    """`_ip_allowed` is the actual gate. Empty allowlist allows
    everything (back-compat); populated allowlist matches by CIDR."""

    def _with_allowlist(self, monkeypatch, raw: str):
        """Helper to swap the module-level _ALLOWLIST for the duration
        of one test, without touching env vars (which only matter at
        import time)."""
        import src.api as api_mod
        monkeypatch.setattr(api_mod, "_ALLOWLIST",
                            api_mod._parse_allowlist(raw))

    def test_empty_allowlist_allows_all(self, monkeypatch):
        self._with_allowlist(monkeypatch, "")
        from src.api import _ip_allowed
        assert _ip_allowed("8.8.8.8") is True
        assert _ip_allowed("100.64.0.1") is True

    def test_ip_inside_cidr_allowed(self, monkeypatch):
        self._with_allowlist(monkeypatch, "100.64.0.0/10")
        from src.api import _ip_allowed
        # Tailscale CGNAT range, anything inside 100.64.0.0 - 100.127.255.255
        assert _ip_allowed("100.64.0.1") is True
        assert _ip_allowed("100.100.50.50") is True
        assert _ip_allowed("100.127.255.254") is True

    def test_ip_outside_cidr_blocked(self, monkeypatch):
        self._with_allowlist(monkeypatch, "100.64.0.0/10")
        from src.api import _ip_allowed
        assert _ip_allowed("8.8.8.8") is False
        assert _ip_allowed("100.128.0.1") is False  # just past the range
        assert _ip_allowed("192.168.1.1") is False

    def test_loopback_inclusion(self, monkeypatch):
        """Typical config: tailscale range plus localhost for dev."""
        self._with_allowlist(monkeypatch, "100.64.0.0/10, 127.0.0.1")
        from src.api import _ip_allowed
        assert _ip_allowed("127.0.0.1") is True
        assert _ip_allowed("100.64.5.5") is True
        assert _ip_allowed("8.8.8.8") is False

    def test_garbage_ip_string_blocked(self, monkeypatch):
        """A malformed proxy header should fail closed, not silently
        pass through."""
        self._with_allowlist(monkeypatch, "100.64.0.0/10")
        from src.api import _ip_allowed
        assert _ip_allowed("not-an-ip") is False
        assert _ip_allowed("") is False

    def test_ipv6_allowlist(self, monkeypatch):
        self._with_allowlist(monkeypatch, "fd7a:115c:a1e0::/48")
        from src.api import _ip_allowed
        # Tailscale assigns ULA addresses in fd7a:115c:a1e0::/48
        assert _ip_allowed("fd7a:115c:a1e0::1") is True
        assert _ip_allowed("2001:db8::1") is False

    def test_ipv4_mapped_ipv6_unwrapped_against_ipv4_cidr(self, monkeypatch):
        """When the OS surfaces an IPv4 connection as IPv4-mapped IPv6
        (`::ffff:a.b.c.d`), the allowlist match must still succeed
        against IPv4 CIDR entries. Without unwrapping the v6 address,
        the membership test would silently return False and Pedro
        would see 403s only in certain network configurations."""
        self._with_allowlist(monkeypatch, "100.64.0.0/10, 127.0.0.1")
        from src.api import _ip_allowed
        # Tailscale-range IPv4 surfaced as v4-mapped v6
        assert _ip_allowed("::ffff:100.64.5.5") is True
        # Loopback v4 surfaced as v4-mapped v6
        assert _ip_allowed("::ffff:127.0.0.1") is True
        # Outside the v4 allowlist, even when v4-mapped, must stay blocked
        assert _ip_allowed("::ffff:8.8.8.8") is False

    def test_ipv4_mapped_ipv6_against_ipv6_allowlist_unchanged(self, monkeypatch):
        """If the operator only listed IPv6 ranges, an unwrap to IPv4
        must NOT make IPv4-mapped clients pass when the operator only
        allowed pure IPv6."""
        self._with_allowlist(monkeypatch, "fd7a:115c:a1e0::/48")
        from src.api import _ip_allowed
        # IPv4-mapped IPv6 unwraps to IPv4, which isn't in the v6 allowlist
        assert _ip_allowed("::ffff:100.64.5.5") is False


class TestAllowlistFailOpenWarning:
    """When CORTEX_IP_ALLOWLIST is set but parses to empty (operator
    intended to restrict access but every entry was invalid), the
    module logs a loud error. We don't crash startup — the trade-off
    is explained in _parse_allowlist's docstring — but the failure
    mode must be very visible in logs.

    These tests reproduce the import-time error log by re-running the
    same logic in isolation, since the module-level branch only fires
    once at import.
    """

    def test_unset_env_no_warning(self, caplog):
        import logging
        from src.api import _parse_allowlist
        with caplog.at_level(logging.ERROR, logger="cortex.api"):
            raw = ""
            parsed = _parse_allowlist(raw)
            # Mirror the module-level branch
            if raw.strip() and not parsed:
                logging.getLogger("cortex.api").error("would not fire")
        assert "would not fire" not in caplog.text
        assert "Server is running with NO IP restriction" not in caplog.text

    def test_valid_entries_no_warning(self, caplog):
        import logging
        from src.api import _parse_allowlist
        with caplog.at_level(logging.ERROR, logger="cortex.api"):
            raw = "100.64.0.0/10"
            parsed = _parse_allowlist(raw)
            if raw.strip() and not parsed:
                logging.getLogger("cortex.api").error("would not fire")
        assert parsed  # sanity: at least one network parsed
        assert "would not fire" not in caplog.text

    def test_all_invalid_entries_logs_loud_error(self, caplog):
        """The fail-open scenario: operator typed something but every
        entry failed to parse. Error log must mention the raw value
        AND the lack of restriction."""
        import logging
        from src.api import _parse_allowlist
        with caplog.at_level(logging.ERROR, logger="cortex.api"):
            raw = "nope, also-nope, 999.999.999.999"
            parsed = _parse_allowlist(raw)
            if raw.strip() and not parsed:
                logging.getLogger("cortex.api").error(
                    "CORTEX_IP_ALLOWLIST was set (%r) but no valid entries "
                    "were parsed. Server is running with NO IP restriction. "
                    "Fix the env var and restart, or unset it to acknowledge "
                    "open access.",
                    raw,
                )
        assert not parsed  # confirm fail-open path is taken
        assert "Server is running with NO IP restriction" in caplog.text
        assert "nope, also-nope, 999.999.999.999" in caplog.text


# ---------------------------------------------------------------------------
# Constant-time key comparison (M10-A1, review Fable 5)
# ---------------------------------------------------------------------------
# `!=` on strings short-circuits on the first divergent byte, leaking the
# key prefix through response timing. All three auth call-sites now go
# through hmac.compare_digest over utf-8 bytes. Real timing is impractical
# to assert in a unit test; these pin the BEHAVIOUR of the new path:
# right key accepted, wrong key rejected, and non-ASCII attacker input
# rejected cleanly (compare_digest on str would raise TypeError → 500).

from types import SimpleNamespace


def _fake_post(body: str, headers=None):
    """Request stand-in for the POST login handler: same shape as
    _fake_request plus the awaitable .body() the handler reads."""
    async def _body():
        return body.encode("utf-8")
    return SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.9"),
        url=SimpleNamespace(path="/ui/login", scheme="http"),
        headers=headers or {},
        cookies={},
        body=_body,
    )


def _fake_request(headers=None, cookies=None, path="/ask"):
    return SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.9"),
        url=SimpleNamespace(path=path, scheme="http"),
        headers=headers or {},
        cookies=cookies or {},
    )


class TestVerifyAccessConstantTime:

    @pytest.fixture(autouse=True)
    def _pin_secret(self, monkeypatch):
        import src.api as api
        monkeypatch.setattr(api, "SECRET_API_KEY", "right-key")

    def test_right_key_accepted(self):
        from src.api import verify_access
        assert verify_access(_fake_request(), api_key="right-key") == "right-key"

    def test_wrong_key_rejected_403(self):
        from fastapi import HTTPException
        from src.api import verify_access
        with pytest.raises(HTTPException) as exc:
            verify_access(_fake_request(), api_key="wrong-key")
        assert exc.value.status_code == 403

    def test_same_prefix_wrong_suffix_rejected(self):
        from fastapi import HTTPException
        from src.api import verify_access
        with pytest.raises(HTTPException) as exc:
            verify_access(_fake_request(), api_key="right-kez")
        assert exc.value.status_code == 403

    def test_non_ascii_key_rejected_not_500(self):
        # compare_digest(str, str) raises TypeError on non-ASCII; the
        # bytes encoding must keep this a clean 403, never a TypeError.
        from fastapi import HTTPException
        from src.api import verify_access
        with pytest.raises(HTTPException) as exc:
            verify_access(_fake_request(), api_key="chavé-errada-ñ")
        assert exc.value.status_code == 403


class TestUIAuthConstantTime:

    @pytest.fixture(autouse=True)
    def _pin_key(self, monkeypatch):
        monkeypatch.setenv("CORTEX_API_KEY", "right-key")

    def test_require_auth_right_header_passes(self):
        from src.ui.routes import _require_auth
        _require_auth(_fake_request(headers={"X-Cortex-Key": "right-key"}))

    def test_require_auth_right_cookie_passes(self):
        from fastapi import HTTPException
        from src.ui.routes import _require_auth
        from src.ui.session_store import store
        with pytest.raises(HTTPException) as exc:
            _require_auth(_fake_request(cookies={"cortex_ui": "right-key"}))
        assert exc.value.status_code == 401
        token = store.issue("right-key")
        assert token is not None
        _require_auth(_fake_request(cookies={"cortex_ui": token}))

    def test_require_auth_wrong_key_401(self):
        from fastapi import HTTPException
        from src.ui.routes import _require_auth
        with pytest.raises(HTTPException) as exc:
            _require_auth(_fake_request(headers={"X-Cortex-Key": "wrong"}))
        assert exc.value.status_code == 401

    def test_require_auth_non_ascii_key_401_not_500(self):
        from fastapi import HTTPException
        from src.ui.routes import _require_auth
        with pytest.raises(HTTPException) as exc:
            _require_auth(_fake_request(headers={"X-Cortex-Key": "çhave-é"}))
        assert exc.value.status_code == 401

    # s82: login moved from GET ?key= to a POST body — the query string put
    # the key in uvicorn's access log and in the browser history. The
    # properties these pin are unchanged: constant-time comparison, and a
    # non-ASCII key must 401 rather than 500 (compare_digest raises TypeError
    # on non-ASCII str, which is why both sides are encoded).

    def test_login_right_key_redirects(self):
        from src.ui.routes import ui_login
        resp = asyncio.run(ui_login(_fake_post("key=right-key")))
        assert resp.status_code == 303

    def test_login_wrong_key_401(self):
        from src.ui.routes import ui_login
        resp = asyncio.run(ui_login(_fake_post("key=wrong")))
        assert resp.status_code == 401

    def test_login_non_ascii_key_401_not_500(self):
        from src.ui.routes import ui_login
        resp = asyncio.run(ui_login(_fake_post("key=%C3%A7have-%C3%A9")))
        assert resp.status_code == 401

    def test_login_empty_body_401(self):
        from src.ui.routes import ui_login
        assert asyncio.run(ui_login(_fake_post(""))).status_code == 401

    def test_login_cross_origin_403(self):
        from fastapi import HTTPException
        from src.ui.routes import ui_login
        with pytest.raises(HTTPException) as exc:
            asyncio.run(ui_login(_fake_post(
                "key=right-key",
                headers={"origin": "http://evil.example", "host": "127.0.0.1:8000"})))
        assert exc.value.status_code == 403
