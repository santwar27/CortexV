"""
list_reminders payload parsing + AppleScript cap (M4-A11).

The old version separated fields with "||" and records with linefeed,
then sliced in Python: a reminder name containing "||" was silently
dropped, a name with a newline split into two bogus rows, and the cap
ran AFTER reading every incomplete reminder's properties. The rewrite
caps inside AppleScript (O(limit)) and uses ASCII control separators
(US 0x1F / RS 0x1E) that can't appear in user-typed names.

Everything is unit-level: `_run` (the osascript call) is monkeypatched
to return a canned payload, so no Reminders.app / no AppleScript runs.
"""
import src.integrations.reminders_client as rem

FS = "\x1f"
RS = "\x1e"


def _payload(rows):
    """Build the control-separated blob the AppleScript would emit. Each row is
    (name, due) or (name, due, due_iso); due_iso defaults to ''."""
    out = []
    for row in rows:
        name, due = row[0], row[1]
        due_iso = row[2] if len(row) > 2 else ""
        out.append(f"{name}{FS}{due}{FS}{due_iso}{RS}")
    return "".join(out)


def _patch_run(monkeypatch, payload):
    captured = {}

    def fake_run(script):
        captured["script"] = script
        return payload

    monkeypatch.setattr(rem, "_run", fake_run)
    return captured


def test_parses_name_due_and_iso(monkeypatch):
    _patch_run(monkeypatch, _payload([
        ("comprar leite", "Saturday, 14 June 2026 at 10:00:00", "2026-06-14T10:00:00"),
        ("ligar pro dentista", "", ""),
    ]))
    out = rem.list_reminders()
    assert out == [
        {"name": "comprar leite", "due": "Saturday, 14 June 2026 at 10:00:00",
         "due_iso": "2026-06-14T10:00:00"},
        {"name": "ligar pro dentista", "due": "", "due_iso": ""},
    ]


def test_old_two_field_payload_still_parses(monkeypatch):
    # Defensive: a payload without the due_iso field → due_iso defaults to "".
    _patch_run(monkeypatch, f"comprar leite{FS}sabado{RS}")
    assert rem.list_reminders() == [
        {"name": "comprar leite", "due": "sabado", "due_iso": ""}]


def test_iso_built_from_components_in_script(monkeypatch):
    cap = _patch_run(monkeypatch, "")
    rem.list_reminders()
    # The AppleScript builds the ISO from numeric date components, not the
    # locale `as string` form (which is unparseable across regions).
    assert "month of dd as integer" in cap["script"]
    assert 'yr & "-" & mo & "-" & dy & "T"' in cap["script"]


def test_empty_payload_returns_empty_list(monkeypatch):
    _patch_run(monkeypatch, "   \n")
    assert rem.list_reminders() == []


def test_name_with_double_pipe_survives(monkeypatch):
    # The exact byte that used to silently drop the row.
    _patch_run(monkeypatch, _payload([("orçamento || final", "")]))
    out = rem.list_reminders()
    assert out == [{"name": "orçamento || final", "due": "", "due_iso": ""}]


def test_name_with_newline_does_not_split_into_two_rows(monkeypatch):
    _patch_run(monkeypatch, _payload([("linha um\nlinha dois", "")]))
    out = rem.list_reminders()
    assert len(out) == 1
    assert "linha um" in out[0]["name"] and "linha dois" in out[0]["name"]


def test_cap_is_requested_in_the_script(monkeypatch):
    cap = _patch_run(monkeypatch, _payload([("a", ""), ("b", "")]))
    rem.list_reminders(limit=3)
    # The AppleScript must carry the early-exit cap, not rely on Python.
    assert "if cnt ≥ 3 then exit repeat" in cap["script"]


def test_limit_floored_at_one(monkeypatch):
    cap = _patch_run(monkeypatch, _payload([("a", "")]))
    rem.list_reminders(limit=0)
    assert "if cnt ≥ 1 then exit repeat" in cap["script"]


def test_python_side_also_caps(monkeypatch):
    # Defensive: even if the script over-returns, the Python slice holds.
    _patch_run(monkeypatch, _payload([("a", ""), ("b", ""), ("c", "")]))
    out = rem.list_reminders(limit=2)
    assert len(out) == 2


def test_named_list_is_escaped_into_target(monkeypatch):
    cap = _patch_run(monkeypatch, "")
    rem.list_reminders(list_name="Trabalho")
    assert 'list "Trabalho"' in cap["script"]


def test_default_list_when_unnamed(monkeypatch):
    cap = _patch_run(monkeypatch, "")
    rem.list_reminders()
    assert "default list" in cap["script"]
