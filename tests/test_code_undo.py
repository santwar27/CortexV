"""Exact-byte code recovery through the real undo effect, without model decisions."""
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core import code_proposals, code_roots, effect_store
from src.core.action_ledger import ledger
from tests.test_code_apply_approval import proposal
from tests.test_code_apply_write import apply
from tests.test_code_proposal import OLD, NEW, setup  # noqa: F401


def undo(o, lang="pt"):
    *_, (reply, _, mode, _) = o.process_stream("desfaz" if lang == "pt" else "undo", language=lang)
    return reply, mode


def restored(pid, path, lang):
    return ((f"Pronto, desfiz: a mudança em {path} — o arquivo voltou a ser o original (proposta {pid})."
             if lang == "pt" else
             f"Done, undid: the change to {path} — the file is back to the original (proposal {pid})."), "UNDO_DONE")


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("origin", [("terminal", None), ("ui", "A")])
def test_restored(setup, monkeypatch, lang, origin):
    o, llm, path = setup
    original = OLD.replace("\n", "\r\n").encode()
    path.write_bytes(original)
    path.chmod(0o640)
    o.set_turn_origin(*origin)
    _, args = proposal(setup, stream=True, lang=lang)
    assert apply(o, lang)[1] == "AGENT_ACTION_DONE"
    assert path.read_bytes() == NEW.encode()
    entry, = ledger.load_entries()
    assert entry["reversible"] and entry["reverse"] == {
        "tool": "restore_code_change", "args": dict(id=args["id"], path=str(path))}
    replace = Mock(wraps=os.replace)
    log = Mock()
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    assert undo(o, lang) == restored(args["id"], path, lang)
    assert path.read_bytes() == original and path.stat().st_mode & 0o7777 == 0o640
    applied, undone = ledger.load_entries()
    assert applied == entry and undone["action"] == "undo" and undone["undone_ts"] == entry["ts"]
    effect, = [e for e in effect_store.store.recent() if e.action == "undo"]
    assert effect.state == "feito"
    assert list(path.parent.iterdir()) == [path]
    calls = replace.call_count
    assert undo(o, lang)[1] == "UNDO_NONE"
    assert replace.call_count == calls and path.read_bytes() == original
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "restore")] == [dict(id=args["id"], outcome="restored")]
    llm.assert_called_once()


@pytest.mark.parametrize("lang", ["pt", "en"])
@pytest.mark.parametrize("case", ["edited", "deleted", "symlink", "already", "missing", "corrupt", "record", "path_mismatch", "voice", "revoked", "write", "unknown", "unreadable", "late_edit", "parent_link"])
def test_restore_boundaries(setup, monkeypatch, lang, case):
    o, llm, path = setup
    _, args = proposal(setup, lang=lang)
    pid = args["id"]
    assert apply(o, lang)[1] == "AGENT_ACTION_DONE" and path.read_bytes() == NEW.encode()
    outcome, state, mode = "changed", "falhou", "AGENT_ACTION_ERROR"
    expected = ("labels.py mudou depois da mudança — não desfiz, para não apagar o que veio depois." if lang == "pt" else
                "labels.py changed after the change — I didn't undo it, so nothing that came later is lost.")
    other = b"# later edit\r\n"
    if case == "edited":
        path.write_bytes(other)
    elif case == "deleted":
        path.unlink()
    elif case == "symlink":
        target = path.parent.parent / "other.py"
        target.write_bytes(other)
        path.unlink()
        path.symlink_to(target)
    elif case == "parent_link":
        target = path.parent.with_name("moved")
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    elif case == "already":
        path.write_bytes(OLD.encode())
        outcome, state, mode = "already", "feito", "UNDO_DONE"
        expected = ("Pronto, desfiz: nada a mexer — labels.py já estava como o original." if lang == "pt" else
                    "Done, undid: nothing to change — labels.py was already the original.")
    elif case in ("missing", "corrupt", "record", "path_mismatch"):
        copy = Path.home() / ".cortex/code_proposals" / (pid + ".orig")
        if case == "missing":
            copy.unlink()
        elif case == "corrupt":
            copy.write_bytes(other)
        elif case == "record":
            copy.with_suffix(".json").unlink()
        else:
            load = code_proposals.load
            monkeypatch.setattr(code_proposals, "load", lambda pid: {**load(pid), "path": str(path) + "x"})
        outcome = "no_copy"
        expected = ("Não achei a cópia exata do original de labels.py — não desfiz." if lang == "pt" else
                    "I couldn't find the exact copy of labels.py's original — I didn't undo it.")
    elif case == "voice":
        o.set_turn_origin("voice")
        load = code_proposals.load
        monkeypatch.setattr(code_proposals, "load", Mock(side_effect=AssertionError("voice read")))
        outcome = "typed"
        expected = ("Desfazer mudança de código só digitando, no terminal do Cortex ou na tela." if lang == "pt" else
                    "A code change can only be undone by typing, in the Cortex terminal or on screen.")
    elif case == "revoked":
        code_roots.remove(str(path.parent))
        outcome = "path"
        why = "unsupported format — .py only inside a folder authorised for code"
        expected = f"não posso mudar labels.py: {why}." if lang == "pt" else f"I can't change labels.py: {why}."
    else:
        replace = os.replace
        def replacing(src, dst, **kw):
            if os.path.basename(dst) == path.name:
                if case == "write":
                    raise OSError("full")
                replace(src, dst, **kw)
                path.unlink() if case == "unreadable" else path.write_bytes(other)
            else:
                replace(src, dst, **kw)
        if case == "late_edit":
            fsync = os.fsync
            def syncing(fd):
                fsync(fd)
                path.write_bytes(other)
            monkeypatch.setattr(os, "fsync", syncing)
        else:
            monkeypatch.setattr(os, "replace", replacing)
            if case == "write":
                outcome = "write_failed"
                expected = ("Não consegui desfazer em labels.py (full). O arquivo continua com a mudança." if lang == "pt" else
                            "I couldn't undo labels.py (full). The file still has the change.")
            else:
                outcome, state = "unknown", "desconhecido"
                expected = ("Não sei se labels.py voltou ao original. Confere antes de pedir de novo — não vou repetir sozinho." if lang == "pt" else
                            "I don't know whether labels.py went back to the original. Please check before asking again — I won't retry on my own.")
    log = Mock()
    monkeypatch.setattr("src.core.staging_mixin.audit.log", log)
    assert undo(o, lang) == (expected, mode)
    if case in ("deleted", "unreadable"):
        assert not path.exists()
    else:
        assert path.read_bytes() == (other if case in ("edited", "symlink", "unknown", "late_edit") else OLD.encode() if case == "already" else NEW.encode())
    if case == "symlink":
        assert path.is_symlink() and target.read_bytes() == other
    assert not list(path.parent.glob(".*.cortex-*"))
    effect, = [e for e in effect_store.store.recent() if e.action == "undo"]
    assert effect.state == state
    assert len(ledger.load_entries()) == (2 if case == "already" else 1)
    assert [c.kwargs["payload"] for c in log.call_args_list if c.args == ("–", "code", "restore")] == [dict(id=pid, outcome=outcome)]
    if outcome == "unknown":
        assert undo(o, lang)[1] == "UNDO_NONE"
        assert [e for e in effect_store.store.recent() if e.action == "undo"] == [effect]
    if case == "voice":
        monkeypatch.setattr(code_proposals, "load", load)
        o.set_turn_origin("terminal")
        assert undo(o, lang) == restored(pid, path, lang)
    llm.assert_called_once()


def test_unknown_apply_still_targets_code(setup, monkeypatch):
    o, _, path = setup
    ledger.record("create_reminder", {"title": "older"}, result_id="older")
    _, args = proposal(setup)
    open_file, replace = os.open, os.replace
    replaced = False
    def replacing(src, dst, **kw):
        nonlocal replaced
        replace(src, dst, **kw)
        if os.path.basename(dst) == path.name:
            replaced = True
    def opening(name, flags, *a, **kw):
        if replaced and name == path.name:
            assert ledger.load_entries()[-1]["action"] == "apply_code_change"
            raise OSError("read-back failed")
        return open_file(name, flags, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replacing)
        patch.setattr(os, "open", opening)
        assert apply(o)[1] == "AGENT_ACTION_ERROR"
    assert path.read_bytes() == NEW.encode()
    assert effect_store.store.recent()[0].state == "desconhecido"
    assert ledger.load_entries()[-1]["action"] == "apply_code_change"
    assert undo(o) == restored(args["id"], path, "pt")
    assert path.read_bytes() == OLD.encode()
    older, applied, undone = ledger.load_entries()
    assert older["action"] == "create_reminder" and undone["undone_ts"] == applied["ts"]
