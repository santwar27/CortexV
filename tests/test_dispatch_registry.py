"""
Tests for src/core/dispatch/registry.py — the declarative routing table that
replaced the inline Onda 4/5 + Fase 2 dispatch blocks in _process_core.

Uses a fake orchestrator carrying the real regex class attrs + stub handlers,
so we test routing/order/None-fallthrough without the full orchestrator.
"""
import re
from unittest.mock import MagicMock

from src.core.dispatch import registry as reg
from src.core.dispatch.spotify import SPOTIFY_DEVICE_HINT_RE


class _FakeOrch:
    """Minimal stand-in: stub handlers that record calls and return a marker
    tuple (or None). The registry now carries its own compiled regexes, so the
    fake only needs to supply the handlers `run` dispatches to."""

    def __init__(self, *, deep=None, dream=None, note_create=("note", "NOTE_CREATE"),
                 calendar=("event", "CALENDAR_STAGED"), reminder=("rem", "REMINDER_STAGED"),
                 contact_info=("card", "CONTACT_INFO")):
        self.calls = []
        # Per-turn reply language the real orchestrator sets before dispatch
        # (I18nMixin). Spotify wrappers pass it through; "pt" = default.
        self._reply_lang = "pt"
        self._deep = deep    # what _handle_deep_research returns
        self._dream = dream
        self._note_create = note_create  # None → unparseable body → fall through
        self._calendar = calendar        # None → unparseable body → fall through
        self._reminder = reminder        # None → unparseable body → fall through
        self._contact_info = contact_info  # None → no contact match → fall through
        # macOS + Spotify routes call into controllers; mocks record the calls.
        self.macos = MagicMock()
        self.spotify = MagicMock()
        # The Spotify play/playlist/radio routes search the full input for a
        # device hint via this regex — use the REAL one so it's faithful.
        self._SPOTIFY_DEVICE_HINT_RE = SPOTIFY_DEVICE_HINT_RE

    def _handle_approve_study(self, ref):
        self.calls.append(("approve", ref)); return ("ok", "STUDIER_APPROVE")

    def _handle_list_emails(self, count):
        self.calls.append(("list_emails", count)); return ("inbox", "EMAIL_LIST")

    def _handle_contact_add(self, fields):
        self.calls.append(("contact_add", fields)); return ("added", "CONTACT_ADD")

    def _handle_contact_info(self, name):
        # Configurable: None models "name matches no contact" → fall through.
        self.calls.append(("contact_info", name)); return self._contact_info

    def _handle_mac_quit_regex(self, app):
        self.calls.append(("mac_quit", app)); return ("fechar?", "MAC_QUIT_STAGED")

    def _handle_silence(self):
        self.calls.append(("silence",)); return ("shh", "SILENCE")

    def _handle_mac_sleep_regex(self, phrased_as_shutdown=False):
        self.calls.append(("mac_sleep", phrased_as_shutdown))
        return ("sleeping?", "MAC_SLEEP")

    def _handle_reminder_regex(self, body, now):
        self.calls.append(("reminder", body)); return self._reminder

    def _handle_calendar_regex(self, body, now):
        self.calls.append(("calendar", body)); return self._calendar

    def _handle_note_list(self):
        self.calls.append(("note_list",)); return ("notes", "NOTE_LIST")

    def _handle_note_search(self, query):
        self.calls.append(("note_search", query)); return (f"found:{query}", "NOTE_SEARCH")

    def _handle_note_create(self, body):
        self.calls.append(("note_create", body)); return self._note_create

    def _handle_undo(self):
        self.calls.append(("undo",)); return ("undone", "UNDO_DONE")

    def _handle_whats_changed(self):
        self.calls.append(("whats_changed",)); return ("delta", "WHATS_CHANGED")

    def _handle_system_status(self):
        self.calls.append(("system_status",)); return ("status", "SYSTEM_STATUS")

    def _handle_deep_research(self, topic):
        self.calls.append(("deep", topic)); return self._deep

    def _handle_dream_review(self, action):
        self.calls.append(("dream", action)); return self._dream

    def _handle_last_contact(self, name):
        self.calls.append(("last_contact", name)); return (f"last:{name}", "ENTITY_LAST_CONTACT")

    def _handle_recent_people(self):
        self.calls.append(("recent",)); return ("recent", "ENTITY_RECENT_PEOPLE")


# --- routing ---------------------------------------------------------------

def test_routes_undo():
    o = _FakeOrch()
    assert reg.run(o, "desfaz isso") == ("undone", "UNDO_DONE")
    assert o.calls == [("undo",)]


def test_routes_whats_changed():
    o = _FakeOrch()
    assert reg.run(o, "o que mudou")[1] == "WHATS_CHANGED"


def test_routes_system_status():
    o = _FakeOrch()
    assert reg.run(o, "status do sistema")[1] == "SYSTEM_STATUS"
    assert o.calls == [("system_status",)]


def test_routes_last_contact_passes_name():
    o = _FakeOrch()
    reg.run(o, "quando falei com João pela última vez")
    assert ("last_contact", "João") in o.calls


def test_routes_recent_people():
    o = _FakeOrch()
    assert reg.run(o, "com quem andei falando")[1] == "ENTITY_RECENT_PEOPLE"


def test_dream_action_derived():
    o = _FakeOrch(dream=("ok", "DREAM_KEPT"))
    reg.run(o, "guarda esse sonho")
    assert ("dream", "keep") in o.calls


# --- None fall-through (dormant gated handlers) ----------------------------

def test_deep_dormant_falls_through():
    # Handler returns None (flag off) → registry returns None → caller continues.
    o = _FakeOrch(deep=None)
    assert reg.run(o, "pesquise a fundo sobre X") is None
    assert ("deep", "X") in o.calls          # it WAS tried


def test_deep_enabled_routes():
    o = _FakeOrch(deep=("done", "DEEP_RESEARCH"))
    assert reg.run(o, "pesquise a fundo sobre X") == ("done", "DEEP_RESEARCH")


def test_no_match_returns_none():
    o = _FakeOrch()
    assert reg.run(o, "oi cortex tudo bem") is None
    assert o.calls == []


# --- second segment: post-planner macOS app-control (s59) ------------------

def test_post_planner_open_app_passes_name():
    o = _FakeOrch()
    assert reg.run(o, "abre o Notion", reg.POST_PLANNER_ROUTES)[1] == "MAC_OPEN_APP"
    o.macos.open_app.assert_called_once_with("Notion", lang="pt")


def test_post_planner_quit_stages():
    o = _FakeOrch()
    assert reg.run(o, "quit Spotify", reg.POST_PLANNER_ROUTES)[1] == "MAC_QUIT_STAGED"
    assert ("mac_quit", "Spotify") in o.calls


def test_simulate_post_planner_segment():
    sim = reg.simulate("fecha o Spotify", reg.POST_PLANNER_ROUTES)
    assert sim is not None
    assert sim.mode == "MAC_CLOSE_WINDOW"
    assert sim.matcher == "MAC_CLOSE_WINDOW_RE"
    assert sim.capture == "Spotify"


def test_post_planner_verbs_not_in_first_segment():
    # "abre o Notion" must NOT match the pre-planner DISPATCH_ROUTES — open is a
    # broad verb that only fires in the second segment (after the planner).
    o = _FakeOrch()
    assert reg.run(o, "abre o Notion") is None
    assert reg.simulate("abre o Notion") is None


# --- migrated families: studier / contacts / email (s59, final block) ------

def test_routes_studier_approve_passes_ref():
    o = _FakeOrch()
    assert reg.run(o, "aprovar estudo 2")[1] == "STUDIER_APPROVE"
    assert ("approve", "2") in o.calls


def test_routes_list_emails_parses_and_clamps_count():
    o = _FakeOrch()
    assert reg.run(o, "meus últimos 3 emails")[1] == "EMAIL_LIST"
    assert ("list_emails", 3) in o.calls  # count_post parsed to int


def test_contact_info_none_falls_through():
    # The name matches no contact → handler returns None → registry returns None
    # so the caller continues (memory/knowledge routing) — e.g. "telefone do meu
    # passaporte" must not be hijacked.
    o = _FakeOrch(contact_info=None)
    assert reg.run(o, "qual o telefone do Lucas") is None
    assert any(c[0] == "contact_info" for c in o.calls)  # it WAS tried


def test_contact_add_executes_but_is_not_simulated():
    o = _FakeOrch()
    assert reg.run(o, "adiciona contato nome: Joao")[1] == "CONTACT_ADD"  # run fires
    # ...but CONTACT_ADD is execute-only (mode=None) → invisible to the simulator
    # (it was never in the old _PRE_STUDIER_TABLE, so behaviour is preserved).
    assert reg.simulate("adiciona contato nome: Joao") is None


def test_simulate_reports_studier_and_list_emails():
    assert reg.simulate("revisar estudos").mode == "STUDIER_REVIEW"
    sim = reg.simulate("meus últimos 3 emails")
    assert sim.mode == "LIST_EMAILS" and sim.matcher == "LIST_EMAILS_RE"


# --- migrated families: silence + mac_sleep (s59) --------------------------

def test_routes_silence():
    o = _FakeOrch()
    assert reg.run(o, "silêncio")[1] == "SILENCE"
    assert o.calls == [("silence",)]


def test_routes_mac_sleep():
    o = _FakeOrch()
    assert reg.run(o, "põe o mac pra dormir")[1] == "MAC_SLEEP"
    assert o.calls == [("mac_sleep", False)]


def test_routes_desliga_o_mac_as_sleep_flagged():
    """"Desliga o mac" sleeps (Pedro, 2026-09-19) — the flag is what makes
    the confirmation say Cortex is not shutting it down."""
    o = _FakeOrch()
    assert reg.run(o, "desliga o mac")[1] == "MAC_SLEEP"
    assert o.calls == [("mac_sleep", True)]


def test_simulate_reports_silence_and_sleep():
    assert reg.simulate("para de falar").mode == "SILENCE"
    assert reg.simulate("põe o mac pra dormir").mode == "MAC_SLEEP"


# --- migrated families: Spotify (s58) --------------------------------------

def test_routes_spotify_pause():
    o = _FakeOrch()
    assert reg.run(o, "pausa")[1] == "SPOTIFY_PAUSE"
    o.spotify.pause.assert_called_once()


def test_routes_spotify_play_passes_target_and_device_hint():
    o = _FakeOrch()
    assert reg.run(o, "toca djonga no iphone")[1] == "SPOTIFY_PLAY"
    o.spotify.play_anything.assert_called_once_with("djonga", device_hint="iphone", lang="pt")


def test_spotify_playlist_wins_before_generic_play():
    # "toca a playlist foco" matches BOTH PLAY and PLAYLIST; order must route it
    # to the playlist handler (the load-bearing precedence).
    o = _FakeOrch()
    assert reg.run(o, "toca a playlist foco")[1] == "SPOTIFY_PLAY_PLAYLIST"
    o.spotify.play_playlist.assert_called_once()
    o.spotify.play_anything.assert_not_called()


def test_simulate_reports_spotify_play_capture():
    sim = reg.simulate("toca djonga no iphone")
    assert sim is not None
    assert sim.mode == "SPOTIFY_PLAY"
    assert sim.matcher == "SPOTIFY_PLAY_RE"
    assert sim.capture == "djonga"  # group(1); device hint is a side-channel


# --- migrated families: macOS control (s58) --------------------------------

def test_routes_mac_mute():
    o = _FakeOrch()
    assert reg.run(o, "mudo")[1] == "MAC_MUTE"
    o.macos.mute.assert_called_once()


def test_routes_mac_volume_set_passes_int():
    o = _FakeOrch()
    assert reg.run(o, "volume sistema 30")[1] == "MAC_VOLUME"
    o.macos.set_system_volume.assert_called_once_with(30, lang="pt")  # str→int conversion


def test_simulate_reports_mac_routing():
    sim = reg.simulate("volume sistema 30")
    assert sim is not None
    assert sim.mode == "MAC_VOLUME"
    assert sim.matcher == "MAC_VOLUME_SET_RE"
    assert sim.capture == "30"  # group(1) reported as capture


# --- migrated families: reminder (s58) -------------------------------------

def test_routes_reminder_passes_body():
    o = _FakeOrch()
    assert reg.run(o, "me lembre de comprar leite hoje às 18h")[1] == "REMINDER_STAGED"
    assert ("reminder", "comprar leite hoje às 18h") in o.calls


def test_reminder_unparseable_falls_through():
    o = _FakeOrch(reminder=None)
    assert reg.run(o, "me lembre de comprar leite hoje às 18h") is None
    assert any(c[0] == "reminder" for c in o.calls)  # it WAS tried


def test_simulate_reports_reminder_routing():
    sim = reg.simulate("me lembre de comprar leite hoje às 18h")
    assert sim is not None
    assert sim.mode == "REMINDER"
    assert sim.matcher == "REMINDER_TRIGGER_RE"


# --- migrated families: calendar (s58) -------------------------------------

def test_routes_calendar_passes_body():
    o = _FakeOrch()
    assert reg.run(o, "adiciona ao calendário: reunião amanhã às 15h")[1] == "CALENDAR_STAGED"
    assert ("calendar", "reunião amanhã às 15h") in o.calls


def test_calendar_unparseable_falls_through():
    o = _FakeOrch(calendar=None)
    assert reg.run(o, "adiciona ao calendário: reunião amanhã às 15h") is None
    assert any(c[0] == "calendar" for c in o.calls)  # it WAS tried


def test_calendar_content_route_executes_but_is_not_simulated():
    # The content-first regex feeds the same handler (run executes it)...
    o = _FakeOrch()
    assert reg.run(o, "marca reunião às 15h no calendário")[1] == "CALENDAR_STAGED"
    # ...but it's execute-only (mode=None) → invisible to the simulator,
    # preserving the eval simulator's prior coverage (TRIGGER only).
    assert reg.simulate("marca reunião às 15h no calendário") is None


def test_simulate_reports_calendar_trigger_routing():
    sim = reg.simulate("adiciona ao calendário: reunião amanhã às 15h")
    assert sim is not None
    assert sim.mode == "CALENDAR"
    assert sim.matcher == "CALENDAR_TRIGGER_RE"


# --- migrated families: notes (s58) ----------------------------------------

def test_routes_note_list():
    o = _FakeOrch()
    assert reg.run(o, "minhas notas")[1] == "NOTE_LIST"
    assert o.calls == [("note_list",)]


def test_routes_note_create_passes_body():
    o = _FakeOrch()
    reg.run(o, "anota comprar leite")
    assert ("note_create", "comprar leite") in o.calls


def test_note_create_unparseable_falls_through():
    # Handler returns None (unparseable body) → registry returns None → caller
    # continues _process_core (the LLM tool-call path).
    o = _FakeOrch(note_create=None)
    assert reg.run(o, "anota comprar leite") is None
    assert ("note_create", "comprar leite") in o.calls  # it WAS tried


# --- simulate (eval-facing routing, no execution) --------------------------

def test_simulate_reports_note_routing():
    sim = reg.simulate("anota comprar leite")
    assert sim is not None
    assert sim.mode == "NOTE_CREATE"
    assert sim.matcher == "NOTE_CREATE_RE"


def test_simulate_skips_dormant_routes():
    # Routes with mode=None (the s57 dormant/new ones) are invisible to the
    # simulator: "desfaz isso" matches the undo route in `run`, but `simulate`
    # returns None (undo has no routing mode → not eval-simulated).
    assert reg.simulate("desfaz isso") is None


def test_simulate_no_match_returns_none():
    assert reg.simulate("oi cortex tudo bem") is None


# --- registry shape --------------------------------------------------------

def test_routes_have_unique_names_and_valid_shape():
    names = [r.name for r in reg.DISPATCH_ROUTES]
    assert len(names) == len(set(names))
    for r in reg.DISPATCH_ROUTES:
        assert hasattr(r.regex, "match")          # a compiled pattern
        assert callable(r.run)
        assert r.mode is None or isinstance(r.mode, str)


class TestRouteBarrier:
    """M1-A11: every handler PROMISES to be fail-soft, but a promise isn't a
    contract. A route handler that raises must become a speakable reply +
    route_crash audit, never an exception escaping to the API (HTTP 500)."""

    def _crashing_routes(self):
        return [reg.Route(
            "CRASH_RE", re.compile(r"^boom$"),
            lambda o, m: (_ for _ in ()).throw(RuntimeError("handler bug")),
            mode="CRASH",
        )]

    def test_crash_becomes_speakable_reply_pt(self, monkeypatch):
        events = []
        monkeypatch.setattr(reg.audit, "log",
                            lambda *a, **k: events.append((a, k)))
        orch = _FakeOrch()
        result = reg.run(orch, "boom", self._crashing_routes())
        assert result is not None, "crashed route must still answer"
        reply, mode = result
        assert mode == "ROUTE_ERROR"
        assert "Não consegui executar esse comando" in reply
        assert "RuntimeError" not in reply  # no raw repr spoken aloud
        assert any(a[2] == "route_crash" for a, _ in events)
        crash = next(k for a, k in events if a[2] == "route_crash")
        assert crash["payload"]["route"] == "CRASH_RE"
        assert "RuntimeError" in crash["payload"]["error"]

    def test_crash_reply_follows_turn_language(self, monkeypatch):
        monkeypatch.setattr(reg.audit, "log", lambda *a, **k: None)
        orch = _FakeOrch()
        orch._reply_lang = "en"
        reply, mode = reg.run(orch, "boom", self._crashing_routes())
        assert mode == "ROUTE_ERROR"
        assert "couldn't run that command" in reply

    def test_orch_without_reply_lang_defaults_pt(self, monkeypatch):
        monkeypatch.setattr(reg.audit, "log", lambda *a, **k: None)
        class _Bare:
            pass
        reply, mode = reg.run(_Bare(), "boom", self._crashing_routes())
        assert mode == "ROUTE_ERROR"
        assert "Não consegui executar esse comando" in reply

    def test_no_match_still_returns_none(self):
        orch = _FakeOrch()
        assert reg.run(orch, "unrelated text", self._crashing_routes()) is None
