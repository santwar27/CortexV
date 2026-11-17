"""
listener.choose_voice_lang — the TTS voice must match the reply's WORDS.

Regression guard for the s53 bug: an English voice read Portuguese words
because the voice was picked from what the user *spoke* (stt_lang, which
Whisper mislabelled) instead of from the generated reply. The voice now
follows the reply text; stt_lang only breaks the rare accent-less-PT tie.
"""
from src.voice.listener import choose_voice_lang


# --- the bug: PT reply must get a PT voice even if STT said "en" -----------

def test_pt_reply_gets_pt_voice_despite_stt_en():
    # The user spoke accent-less PT → Whisper labelled it "en"; the reply is PT
    # with accents. The voice must be PT (this is the live bug, fixed).
    reply = "Eu não tenho preferências pessoais, mas você gosta de funk."
    assert choose_voice_lang(reply, stt_lang="en") == "pt"


# --- straightforward cases -------------------------------------------------

def test_en_reply_gets_en_voice():
    assert choose_voice_lang("The system is functioning normally.", stt_lang="pt") == "en"


def test_pt_reply_with_accents():
    assert choose_voice_lang("São 20:01.", stt_lang=None) == "pt"


# --- accent-less PT caught via common-word hints ---------------------------

def test_accentless_pt_reply_caught_by_hint():
    # No accents, but "feito"/"pronto" are PT → PT voice (not stt-dependent).
    assert choose_voice_lang("ok, feito", stt_lang="en") == "pt"
    assert choose_voice_lang("pronto, tudo certo", stt_lang=None) == "pt"


def test_english_reply_never_promoted_to_pt():
    # The dangerous case: EN reply while the user spoke PT — must STAY EN
    # (a PT voice reading English is just the mirror of the original bug).
    assert choose_voice_lang("ok, done", stt_lang="pt") == "en"
    assert choose_voice_lang("The system is fine.", stt_lang="pt") == "en"


# --- robustness ------------------------------------------------------------

def test_empty_reply_uses_stt_as_last_resort():
    assert choose_voice_lang("", stt_lang=None) == "en"
    assert choose_voice_lang(None, stt_lang="pt") == "pt"   # no words → trust stt


# --- s60: localised deterministic command replies must voice correctly -----
# Now that command handlers reply in the spoken language, the short accent-less
# PT confirmations must voice PT and their EN counterparts must voice EN.
import pytest


@pytest.mark.parametrize("reply", [
    "Pausado.", "Continuando.", "Voltando.", 'Anotado: "leite".',
    "Abrindo o Spotify.", "Fechando o Notion.", "Foco em Notion.",
    "Sistema silenciado.", "Tela travada.", "Curtida: X.",
    "Mac indo dormir.", "Contato salvo: X.", "Lembrete removido.",
    "Suas notas:", "Fila de estudos (1 pendente):",
])
def test_pt_command_replies_voice_pt(reply):
    assert choose_voice_lang(reply, stt_lang=None) == "pt"


@pytest.mark.parametrize("reply", [
    "Paused.", "Resuming.", "Going back.", 'Noted: "milk".',
    "Opening Spotify.", "Closing Notion.", "Focused on Notion.",
    "System muted.", "Screen locked.", "Liked: X.",
    "Mac going to sleep.", "Contact saved: X.", "Reminder removed.",
    "Your notes:", "Volume 50%.",
])
def test_en_command_replies_voice_en(reply):
    # EN command replies must never be promoted to PT by the hint list.
    assert choose_voice_lang(reply, stt_lang="pt") == "en"


# s63: lease replies. The live incident of 2026-06-10 was a stale pre-C8
# listener voicing "Internet desligada." in EN — the heuristic fallback
# must know the lease vocabulary even though the server announces the
# language since C8 (a lagging listener is exactly when the fallback runs).
@pytest.mark.parametrize("reply", [
    "Internet desligada.",
    "Internet liberada até 21:30 (WEST).",
])
def test_lease_replies_voice_pt_without_server_lang(reply):
    assert choose_voice_lang(reply, stt_lang=None) == "pt"


@pytest.mark.parametrize("reply", [
    "Internet disconnected.",
    "Internet granted until 21:30 (WEST).",
])
def test_lease_replies_en_stay_en(reply):
    assert choose_voice_lang(reply, stt_lang="pt") == "en"


def test_server_lang_overrides_heuristic_for_lease_reply():
    # The C8 contract: when the prelude announces the language, believe it.
    assert choose_voice_lang("Internet desligada.", "en", server_lang="pt") == "pt"


# --- s76 smoke: PT words outvote an "en" announcement -----------------------
# On the streaming chat path the server announces the INPUT's language
# before generation; the 8B can disobey the directive and stream PT text
# into an "en" turn — the alba voice read "Não consegui abrir o arquivo
# de notas..." live. Positive PT evidence in the words wins; the reverse
# direction (pt announcement, EN-looking text) is NEVER overridden,
# because absence of accents is not proof of EN (accent-less PT exists).

def test_pt_reply_overrides_en_server_lang():
    reply = "Não consegui abrir o arquivo de notas, pois ele não existe."
    assert choose_voice_lang(reply, "en", server_lang="en") == "pt"


def test_accentless_pt_hint_overrides_en_server_lang():
    assert choose_voice_lang("Feito, tudo certo agora.", "en", server_lang="en") == "pt"


def test_en_reply_keeps_en_server_lang():
    reply = "It seems you're asking for clarification on something."
    assert choose_voice_lang(reply, "en", server_lang="en") == "en"


def test_pt_server_lang_is_never_overridden():
    # Accent-less reply that LOOKS en-ish: the pt announcement stands.
    assert choose_voice_lang("Spotify: paused", "pt", server_lang="pt") == "pt"


def test_empty_reply_trusts_en_server_lang():
    assert choose_voice_lang("", "pt", server_lang="en") == "en"
