"""Approved source writes: exact recovery bytes, recorded intent and no retry."""
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_proposals, effect_store
from src.core.action_ledger import ledger
from src.core.staging_mixin import Delivery
from tests.test_code_apply_approval import proposal
from tests.test_code_proposal import OLD, NEW, setup  # noqa: F401


def apply(o, lang="pt"):
    *_, (reply, _, mode, _) = o.process_stream("sim", language=lang)
    return reply, mode


def success(pid, path, lang="pt"):
    return ((f"Apliquei a proposta {pid} em {path} (−3 +3 linhas). Antes de gravar, guardei uma cópia exata do original. Para voltar ao original, diga 'desfaz'."
             if lang == "pt" else
             f"Applied proposal {pid} to {path} (−3 +3 lines). Before writing, I saved an exact copy of the original. To go back to the original, say 'undo'."), "AGENT_ACTION_DONE")


@pytest.mark.parametrize("origin", [("terminal", None), ("ui", "A")])
@pytest.mark.parametrize("lang", ["pt", "en"])
def test_applied_and_undo(setup, monkeypatch, origin, lang):
    o, llm, path = setup
    # CRLF proves the recovery copy contains bytes, not reconstructed text.
    original = OLD.replace("\n", "\r\n").encode()
    path.write_bytes(original)
    path.chmod(0o640)
    o.set_turn_origin(*origin)
    _, args = proposal(setup, stream=True, lang=lang)
    eid, pid = o._pending_action["id"], args["id"]
    ledger.record("create_reminder", {"title": "older"}, result_id="older")
    older = ledger.load_entries()[0]
    save = code_proposals.save_original

    def copy(pid, data):
        assert effect_store.store.get(eid).state == "executando"
        assert path.read_bytes() == original
        save(pid, data)

    monkeypatch.setattr(code_proposals, "save_original", copy)
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    assert apply(o, lang) == success(pid, path, lang)
    assert path.read_bytes() == NEW.encode() and path.stat().st_mode & 0o7777 == 0o640
    assert code_proposals.load_original(pid) == original
    directory = Path.home() / ".cortex/code_proposals"
    assert (directory / (pid + ".orig")).stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    effect, = effect_store.store.recent()
    assert effect.id == eid and effect.action == "apply_code_change" and effect.state == "feito"
    assert effect.args == {k: args[k] for k in ("id", "path", "original_sha256", "new_sha256")}
    assert effect.origin == o._turn_origin()
    first, entry = ledger.load_entries()
    assert first == older and entry["action"] == "apply_code_change"
    assert entry["args"] == dict(id=pid, path=str(path)) and entry["result_id"] == pid
    assert entry["reversible"] is True and entry["source"] == "gate"
    assert list(path.parent.iterdir()) == [path]
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "apply")] == [dict(id=pid, outcome="applied")]
    assert o._process_core("desfaz" if lang == "pt" else "undo", language=lang) == ((
        f"Pronto, desfiz: a mudança em {path} — o arquivo voltou a ser o original (proposta {pid})." if lang == "pt" else
        f"Done, undid: the change to {path} — the file is back to the original (proposal {pid})."), "UNDO_DONE")
    assert path.read_bytes() == original
    assert ledger.load_entries()[:2] == [older, entry]
    assert ledger.load_entries()[2]["undone_ts"] == entry["ts"]
    llm.assert_called_once()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["changed_early", "changed_late", "symlink", "copy", "write", "unknown", "stop", "store"])
def test_write_boundaries(setup, monkeypatch, case, lang):
    o, llm, path = setup
    _, args = proposal(setup, lang=lang)
    eid, pid = o._pending_action["id"], args["id"]
    delivery = Delivery()
    o.set_turn_delivery(delivery)
    save, create, replace = code_proposals.save_original, effect_store.store.create, os.replace
    other = b"# another edit\n"
    outcome, state, mode = "changed", "falhou", "AGENT_ACTION_ERROR"
    expected = ("labels.py mudou desde a proposta — peça uma proposta nova. Nada foi aplicado." if lang == "pt" else
                "labels.py changed since the proposal — ask for a new one. Nothing was applied.")
    if case in ("changed_early", "stop", "store"):
        def before_write(*a, **kw):
            if case == "store":
                return None
            effect = create(*a, **kw)
            path.write_bytes(other) if case == "changed_early" else delivery.cancelled.set()
            return effect
        monkeypatch.setattr(effect_store.store, "create", before_write)
        if case == "stop":
            outcome, state, mode = None, "cancelado", "AGENT_ACTION_CANCELLED"
            expected = "Parei antes de executar — não fiz nada." if lang == "pt" else "Stopped before running it — nothing was done."
        elif case == "store":
            outcome = None
            expected = ("Não consegui registrar essa ação antes de executar — não fiz nada." if lang == "pt" else
                        "I couldn't record that action before running it — I did nothing.")
    elif case in ("changed_late", "symlink", "copy"):
        def copy(pid, data):
            if case == "copy":
                raise OSError("full")
            save(pid, data)
            if case == "symlink":
                target = path.parent.parent / "other.py"
                target.write_bytes(OLD.encode())
                path.unlink()
                path.symlink_to(target)
            else:
                path.write_bytes(other)
        monkeypatch.setattr(code_proposals, "save_original", copy)
        if case == "copy":
            outcome = "copy_failed"
            expected = ("Não consegui guardar a cópia do original — não gravei nada." if lang == "pt" else
                        "I couldn't save the copy of the original — I wrote nothing.")
    else:
        def replacing(src, dst, **kw):
            if os.path.basename(dst) == path.name:
                assert code_proposals.load_original(pid) == OLD.encode()
                if case == "write":
                    raise OSError("full")
                replace(src, dst, **kw)
                path.write_bytes(other)  # The result cannot be verified after replacement.
            else:
                replace(src, dst, **kw)
        monkeypatch.setattr(os, "replace", replacing)
        if case == "write":
            outcome = "write_failed"
            expected = ("Não consegui gravar labels.py (full). O arquivo continua como estava." if lang == "pt" else
                        "I couldn't write labels.py (full). The file is unchanged.")
        else:
            outcome, state = "unknown", "desconhecido"
            expected = ("Não sei se labels.py ficou como a proposta. Confere antes de pedir de novo — não vou repetir sozinho." if lang == "pt" else
                        "I don't know whether labels.py ended up as the proposal. Please check before asking again — I won't retry on my own.")
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    assert apply(o, lang) == (expected, mode)
    effects = effect_store.store.recent()
    if case == "store":
        assert effects == []
    else:
        effect, = effects
        assert effect.id == eid and effect.state == state
    assert path.read_bytes() == (other if case in ("changed_early", "changed_late", "unknown") else OLD.encode())
    assert not list(path.parent.glob(".*.cortex-*"))
    if case == "unknown":
        entry, = ledger.load_entries()
        assert entry["action"] == "apply_code_change" and entry["result_id"] == pid
    else:
        assert ledger.load_entries() == []
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "apply")] == ([] if outcome is None else [dict(id=pid, outcome=outcome)])
    assert o._pending_action is None and not Path(o._PENDING_ACTION_FILE).exists()
    if case == "unknown":
        apply(o, lang)
        assert effect_store.store.recent() == effects and path.read_bytes() == other
    llm.assert_called_once()


def test_original_store(setup, monkeypatch):
    _, args = proposal(setup)
    pid = args["id"]
    assert code_proposals.load_original(pid) is None
    for bad_id, data in [(pid, NEW.encode()), ("../x", OLD.encode()), ("0" * 16, OLD.encode())]:
        with pytest.raises(ValueError):
            code_proposals.save_original(bad_id, data)
    code_proposals.save_original(pid, OLD.encode())
    assert code_proposals.load_original(pid) == OLD.encode()
    original = Path.home() / ".cortex/code_proposals" / (pid + ".orig")
    original.write_bytes(b"edited")
    assert code_proposals.load_original(pid) is None
    original.unlink()
    assert code_proposals.load_original(pid) is None and code_proposals.load_original("../x") is None
    directory = Path.home() / "elsewhere"
    monkeypatch.setenv("CORTEX_CODE_PROPOSALS_DIR", str(directory))
    pid = code_proposals.save(dict(original_sha256=args["original_sha256"]))
    code_proposals.save_original(pid, OLD.encode())
    assert code_proposals.load_original(pid) == OLD.encode() and (directory / (pid + ".orig")).exists()
