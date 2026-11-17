"""Leaked model calls fail immediately, as if Ollama were switched off."""
import errno
import socket
from time import monotonic

import ollama
import pytest
import requests


@pytest.mark.parametrize("family, address", [
    (socket.AF_INET, ("127.0.0.1", 11434)),
    (socket.AF_INET, ("localhost", 11434)),
    (socket.AF_INET6, ("::1", 11434)),
])
@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_raw_ollama_connections_are_refused(family, address, method):
    with socket.socket(family) as client:
        client.settimeout(1)
        started = monotonic()
        if method == "connect":
            with pytest.raises(ConnectionRefusedError,
                               match="real Ollama blocked in tests") as error:
                client.connect(address)
            assert error.value.errno == errno.ECONNREFUSED
        else:
            assert client.connect_ex(address) == errno.ECONNREFUSED
        assert monotonic() - started < 1


def test_the_default_ollama_client_fails_immediately():
    started = monotonic()
    with pytest.raises(ConnectionError):
        ollama.chat(model="x", messages=[{"role": "user", "content": "oi"}])
    assert monotonic() - started < 2


def test_requests_to_ollama_fail_immediately():
    started = monotonic()
    with pytest.raises(requests.ConnectionError):
        requests.post("http://localhost:11434/api/embeddings", json={}, timeout=5)
    assert monotonic() - started < 2


def test_other_ports_still_accept_connections():
    with socket.socket() as listener, socket.socket() as client:
        listener.settimeout(1)
        client.settimeout(1)
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("local sockets are blocked in this sandbox")
        listener.listen(1)
        client.connect(listener.getsockname())
        accepted, _ = listener.accept()
        with accepted:
            assert accepted.getpeername() == client.getsockname()
