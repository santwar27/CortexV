"""Spotify control-verb regexes (pause/resume/next/prev/now/like).

Focus: the EN coverage added in sessão 11 — the control verbs used to
be PT-heavy and natural English phrasings ("next song", "skip this
track", "what song is this", "save this song") fell through to the LLM.
PT forms are asserted too so the EN additions don't regress them.

The play regex has its own coverage in test_orchestrator.py; volume is
bilingual already. Both are out of scope here.
"""
import pytest

from src.core.dispatch.spotify import (
    SPOTIFY_LIKE_RE,
    SPOTIFY_NEXT_RE,
    SPOTIFY_NOW_RE,
    SPOTIFY_PAUSE_RE,
    SPOTIFY_PLAY_PLAYLIST_EN_RE,
    SPOTIFY_PLAY_PLAYLIST_RE,
    SPOTIFY_PLAY_RADIO_EN_RE,
    SPOTIFY_PLAY_RADIO_RE,
    SPOTIFY_PREV_RE,
    SPOTIFY_RESUME_RE,
    SPOTIFY_TRANSFER_RE,
)


class TestPause:
    @pytest.mark.parametrize("text", [
        # PT regression
        "pausa", "pausar", "pausa a música", "para", "parar",
        "para a música", "stop",
        # EN additions
        "pause", "pause the music", "pause the song", "pause spotify",
        "pause it", "stop the music", "stop the track",
    ])
    def test_matches(self, text):
        assert SPOTIFY_PAUSE_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "pause my workout playlist",  # has trailing args → not a bare pause
        "stop the car",
        "pausa o vídeo do youtube",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_PAUSE_RE.match(text), text


class TestResume:
    @pytest.mark.parametrize("text", [
        # PT regression
        "continua", "continuar", "continua a música", "despausa",
        "despausar", "volta a tocar", "voltar a tocar",
        # EN additions
        "resume", "resume the music", "resume playback", "unpause",
        "continue", "continue playing", "keep playing",
        # Bare verb = press play / resume. Without these, "play" / "toca"
        # alone fell to the LLM, which leaked a raw tool-call object.
        "play", "toca", "toque", "PLAY", "Toca",
    ])
    def test_matches(self, text):
        assert SPOTIFY_RESUME_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "continue reading the book",
        "resume my career",
        # Bare verb routes to resume, but verb + argument must NOT —
        # that is a play-this-thing request handled downstream.
        "play matue", "toca djonga", "toque a próxima",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_RESUME_RE.match(text), text


class TestPlayPlaylist:
    # group(1) must be the clean playlist name — trailing "no spotify" /
    # device hints get stripped so the name matches the user's real list
    # (regression: "toque minha playlist T(rap) no spotify" captured
    # "T(rap) no spotify" and fell through to a junk public search).
    @pytest.mark.parametrize("text,name", [
        ("toque minha playlist T(rap) no spotify", "T(rap)"),
        ("toca playlist gym", "gym"),
        ("coloca a playlist foco no meu iphone", "foco"),
        ("play my workout playlist", "workout"),
        ("play the chill playlist on spotify", "chill"),
    ])
    def test_captures_clean_name(self, text, name):
        m = SPOTIFY_PLAY_PLAYLIST_RE.match(text) or \
            SPOTIFY_PLAY_PLAYLIST_EN_RE.match(text)
        assert m, text
        assert m.group(1).strip().lower() == name.lower(), m.group(1)


class TestPlayRadio:
    # group(1) is the artist name; "radio"/"rádio" keyword + optional
    # de/do/da connector + trailing fluff all stripped.
    @pytest.mark.parametrize("text,name", [
        ("toque radio Matue", "Matue"),
        ("toca rádio do matuê", "matuê"),
        ("põe uma radio de djonga", "djonga"),
        ("play radio queen", "queen"),
        ("toca radio matue no iphone", "matue"),
        # EN trailing-noun: "radio" AFTER the name.
        ("play matue radio", "matue"),
        ("start a djonga radio", "djonga"),
    ])
    def test_captures_artist(self, text, name):
        m = SPOTIFY_PLAY_RADIO_RE.match(text) or \
            SPOTIFY_PLAY_RADIO_EN_RE.match(text)
        assert m, text
        assert m.group(1).strip().lower() == name.lower(), m.group(1)

    @pytest.mark.parametrize("text", [
        # No "radio" word — must NOT be a radio command.
        "toca djonga",
        "toca minha playlist gym",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_PLAY_RADIO_RE.match(text), text
        assert not SPOTIFY_PLAY_RADIO_EN_RE.match(text), text


class TestNext:
    @pytest.mark.parametrize("text", [
        # PT regression
        "próxima", "proxima", "próxima música", "pula", "pular",
        "pula essa", "pula essa música",
        # EN additions
        "skip", "next", "next song", "next track", "next one",
        "skip this", "skip song", "skip this song", "skip the track",
    ])
    def test_matches(self, text):
        assert SPOTIFY_NEXT_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "next week",
        "skip the meeting",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_NEXT_RE.match(text), text


class TestPrev:
    @pytest.mark.parametrize("text", [
        # PT regression
        "anterior", "música anterior", "volta", "volta uma música",
        # EN additions
        "previous", "prev", "previous song", "previous track",
        "go back", "last song", "the last track",
    ])
    def test_matches(self, text):
        assert SPOTIFY_PREV_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "previous meeting notes",
        "go back to sleep",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_PREV_RE.match(text), text


class TestNowPlaying:
    @pytest.mark.parametrize("text", [
        # PT regression
        "o que está tocando", "o que tá tocando", "que música é essa",
        "qual a música", "qual é a música",
        # s75: natural PT caught falling to chat in the locked-screen battery
        "que música está tocando", "que música tá tocando",
        "qual música está tocando agora",
        # EN additions
        "now playing", "what's playing", "what is playing",
        "what song is this", "what's this song", "what track is this",
        "what's the song", "current song", "current track",
    ])
    def test_matches(self, text):
        assert SPOTIFY_NOW_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "what song should I learn next",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_NOW_RE.match(text), text


class TestLike:
    @pytest.mark.parametrize("text", [
        # PT regression
        "curte", "curtir", "salva", "salvar", "favorito",
        "adiciona", "adiciona aos favoritos", "curte essa música",
        # EN additions
        "like", "save", "favorite", "favourite", "like this song",
        "save this song", "save this track", "add to favorites",
        "add to my favourites", "add to library",
    ])
    def test_matches(self, text):
        assert SPOTIFY_LIKE_RE.match(text), text

    @pytest.mark.parametrize("text", [
        "save the document",
        "like a boss",
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_LIKE_RE.match(text), text


class TestTransfer:
    @pytest.mark.parametrize("text, device", [
        # PT
        ("troca pro iphone", "iphone"),
        ("troca para o macbook", "macbook"),
        ("passa pro celular", "celular"),
        ("passa pra tv", "tv"),
        ("transfere pro speaker", "speaker"),
        ("manda pro homepod", "homepod"),
        ("joga pro mac", "mac"),
        ("muda pro iphone", "iphone"),
        ("troca o som pro homepod", "homepod"),
        ("passa a música pro celular", "celular"),
        # EN
        ("transfer to my phone", "phone"),
        ("move to the macbook", "macbook"),
        ("switch to the speaker", "speaker"),
        ("cast to the tv", "tv"),
        ("transfer playback to my iphone", "iphone"),
    ])
    def test_matches_and_captures_device(self, text, device):
        m = SPOTIFY_TRANSFER_RE.match(text)
        assert m, text
        assert m.group(1).lower() == device

    @pytest.mark.parametrize("text", [
        "troca pro trabalho",         # not a device noun
        "passa o sal",                # unrelated
        "transfer the money to john", # not a device
        "toca matue no iphone",       # play-with-target, NOT transfer
        "move to london",             # not a device
    ])
    def test_no_false_positive(self, text):
        assert not SPOTIFY_TRANSFER_RE.match(text), text


class TestGoingToTheNextIsSkipping:
    """"vai para a próxima" went to MAC_APP_SWITCH as an app named "próxima":
    a focus error, and the track never changed (s88, Codex lote 9)."""

    @pytest.mark.parametrize("text,mode", [
        ("vai para a próxima", "SPOTIFY_NEXT"),
        ("vai pra próxima música", "SPOTIFY_NEXT"),
        ("passa pra próxima", "SPOTIFY_NEXT"),
        ("vai pra anterior", "SPOTIFY_PREV"),
        ("volta pra música anterior", "SPOTIFY_PREV"),
    ])
    def test_routes_to_the_track(self, text, mode):
        from src.core.dispatch.registry import simulate
        m = simulate(text)
        assert m is not None and m.mode == mode, text

    def test_never_an_app(self):
        from src.core.dispatch.macos import MAC_APP_SWITCH_RE
        assert not MAC_APP_SWITCH_RE.match("vai para a próxima")
        assert MAC_APP_SWITCH_RE.match("vai para o safari").group("app") == "safari"
