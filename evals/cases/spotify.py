"""Spotify dispatch behaviour."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- s83: "por" is a preposition before it is a verb -------
    EvalCase(
        name="adversarial_por_que_is_a_question_not_a_song",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="por que azul?",
        expected_mode="OFFLINE",
        description="Pedro asked why Cortex had picked blue and got a pagode "
                    "album: bare 'por' was read as the verb 'pôr', so every "
                    "'por que …?' played music.",
    ),
    EvalCase(
        name="adversarial_toca_o_barco_is_an_idiom",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca o barco",
        expected_mode="OFFLINE",
        description="Found by the s83 adversarial sweep: an idiom opening with "
                    "a play verb actually started music mid-conversation.",
    ),
    EvalCase(
        name="por_still_plays_a_track",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="por Djavan",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="Djavan",
        description="The verb must survive: 'pode por X' reaches the regex as "
                    "'por X' once the courtesy stripper has run.",
    ),
    # ------- imperative play (baseline) -------
    EvalCase(
        name="play_imperative_pt",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca matue",
        expected_mode="SPOTIFY_PLAY",
        expected_matcher="SPOTIFY_PLAY_RE",
        expected_capture="matue",
        description="Direct PT imperative is the dispatch sweet spot.",
    ),
    EvalCase(
        name="play_imperative_en",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="play queen on my mac",
        expected_mode="SPOTIFY_PLAY",
        expected_matcher="SPOTIFY_PLAY_RE",
        expected_capture="queen",
    ),

    # ------- task 150: sending verbs require an artist object -------
    EvalCase(
        name="play_manda_artist_object",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um som do veigh",
        expected_mode="SPOTIFY_PLAY",
        expected_matcher="SPOTIFY_PLAY_RE",
        expected_capture="veigh",
        tags=["regression", "task150"],
    ),
    EvalCase(
        name="play_solta_artist_with_device",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="solta aí a música da Anitta pra mim no meu iphone",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="Anitta",
        tags=["regression", "task150"],
    ),
    EvalCase(
        name="manda_artist_to_person_is_sending",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda o som do veigh pro João",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task150"],
    ),
    EvalCase(
        name="manda_artist_via_email_is_sending",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda o som do veigh por email",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task150", "task150b"],
    ),
    EvalCase(
        name="manda_without_artist_object_falls_through",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um veigh aí",
        expected_mode="OFFLINE",
        tags=["regression", "task150"],
    ),
    EvalCase(
        name="solta_device_sound_is_not_an_artist",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="solta o som do computador",
        expected_mode="OFFLINE",
        tags=["regression", "conflict", "task150"],
    ),

    # Task 154: library membership is runtime-only; neither the simulator
    # nor hallucinated-command offers may assume a bare artist is saved.
    EvalCase(
        name="solta_bare_artist_needs_runtime_library",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="solta aí um veigh ai",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "task154"],
    ),
    EvalCase(
        name="manda_bare_artist_with_recipient_is_sending",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um veigh pro João",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task154"],
    ),
    EvalCase(
        name="manda_summary_is_not_music",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um resumo aí",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task154"],
    ),

    # ------- task 153: title tails and plural artist objects -------
    EvalCase(
        name="play_bare_title_keeps_ta_ai",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca tá aí",
        expected_mode="SPOTIFY_PLAY",
        expected_matcher="SPOTIFY_PLAY_RE",
        expected_capture="tá aí",
        tags=["regression", "task153"],
    ),
    EvalCase(
        name="play_bare_title_keeps_ai",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="play tô nem aí",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="tô nem aí",
        tags=["regression", "task153"],
    ),
    EvalCase(
        name="play_manda_plural_artist_strips_filler",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um som dos Racionais aí",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="Racionais",
        tags=["regression", "task153"],
    ),
    EvalCase(
        name="play_bota_plural_artist",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="bota uma música das Spice Girls",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="Spice Girls",
        tags=["regression", "task153"],
    ),
    EvalCase(
        name="manda_plural_artist_to_person_is_sending",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda um som dos Racionais pro João",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task153"],
    ),
    EvalCase(
        name="manda_plural_device_is_not_an_artist",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="manda o som dos fones",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        tags=["regression", "conflict", "task153"],
    ),

    # ------- polite forms (bloco F regression) -------
    EvalCase(
        name="play_polite_pode_tocar",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="pode tocar matue",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description="'pode tocar' was the original Matuê bug — must stay green.",
        tags=["regression"],
    ),
    EvalCase(
        name="play_polite_quero_ouvir",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="quero ouvir djonga",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="djonga",
    ),
    EvalCase(
        name="play_polite_can_you_play",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="can you play matue",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
    ),

    # ------- vocative strip (bloco F) -------
    EvalCase(
        name="play_vocative_ola_cortex",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="ola cortex, pode tocar matue no spotify?",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description="The original failing input, end-to-end.",
        tags=["regression"],
    ),
    EvalCase(
        name="play_vocative_ei_jarvis",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="ei jarvis, play queen",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="queen",
    ),

    # ------- trailing punctuation (bloco F) -------
    EvalCase(
        name="play_trailing_question_mark",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca matue?",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
    ),

    # ------- single-word verbs win over PLAY -------
    EvalCase(
        name="pause",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="pausa",
        expected_mode="SPOTIFY_PAUSE",
    ),
    EvalCase(
        name="next",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="próxima",
        expected_mode="SPOTIFY_NEXT",
    ),
    EvalCase(
        name="volume_bare_routes_mac",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="volume 50",
        expected_mode="MAC_VOLUME",
        expected_capture="50",
        description="Bare 'volume N' controls the Mac; Spotify must be named.",
        tags=["regression", "conflict"],
    ),
    EvalCase(
        name="volume_named_spotify",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="volume do spotify 50",
        expected_mode="SPOTIFY_VOLUME",
        expected_capture="50",
        description="Explicitly naming Spotify routes volume to Spotify.",
        tags=["regression", "conflict"],
    ),

    # ------- transfer playback (sessão 14, continuação do bug 3) -------
    EvalCase(
        name="transfer_pt_iphone",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="troca pro iphone",
        expected_mode="SPOTIFY_TRANSFER",
        expected_matcher="SPOTIFY_TRANSFER_RE",
        expected_capture="iphone",
        description="Move active playback to another device.",
        tags=["jou-s14", "bug3"],
    ),
    EvalCase(
        name="transfer_en_macbook",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="transfer to the macbook",
        expected_mode="SPOTIFY_TRANSFER",
        expected_capture="macbook",
        tags=["jou-s14", "bug3"],
    ),
    EvalCase(
        name="transfer_does_not_steal_play_with_target",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca matue no iphone",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description=(
            "play-with-target ('toca X no iphone') must stay SPOTIFY_PLAY — "
            "transfer only fires on explicit move verbs, not play."
        ),
        tags=["jou-s14", "conflict"],
    ),

    # ------- playlist play (sessão 17 bloco F) -------
    EvalCase(
        name="play_playlist_minha_pt",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca minha playlist gym",
        expected_mode="SPOTIFY_PLAY_PLAYLIST",
        expected_matcher="SPOTIFY_PLAY_PLAYLIST_RE",
        expected_capture="gym",
        description="The common 'toca minha playlist X' must reach play_playlist.",
    ),
    EvalCase(
        name="play_playlist_bare_pt",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca a playlist sertanejo",
        expected_mode="SPOTIFY_PLAY_PLAYLIST",
        expected_capture="sertanejo",
        description="No 'minha' prefix — must still route to playlist.",
    ),
    EvalCase(
        name="play_playlist_en_trailing",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="play my workout playlist",
        expected_mode="SPOTIFY_PLAY_PLAYLIST",
        expected_matcher="SPOTIFY_PLAY_PLAYLIST_EN_RE",
        expected_capture="workout",
        description="EN natural phrasing: name BEFORE 'playlist'.",
    ),
    EvalCase(
        name="play_playlist_does_not_steal_artist",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca djonga",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="djonga",
        description=(
            "Bare 'toca djonga' must stay SPOTIFY_PLAY (no playlist verb). "
            "The artist-context branch is decided inside play_anything, not "
            "by the dispatcher."
        ),
    ),
    EvalCase(
        name="play_playlist_strips_no_spotify_suffix",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toque minha playlist T(rap) no spotify",
        expected_mode="SPOTIFY_PLAY_PLAYLIST",
        expected_matcher="SPOTIFY_PLAY_PLAYLIST_RE",
        expected_capture="T(rap)",
        description=(
            "Regression: the trailing 'no spotify' must be stripped from "
            "the playlist name (was captured as 'T(rap) no spotify', which "
            "never matched the user's real list and fell to a junk public "
            "search)."
        ),
    ),
    # ------- artist radio -------
    EvalCase(
        name="play_radio_pt",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toque radio Matue",
        expected_mode="SPOTIFY_PLAY_RADIO",
        expected_matcher="SPOTIFY_PLAY_RADIO_RE",
        expected_capture="Matue",
        description="'toque radio X' must reach play_artist_radio with a clean name.",
    ),
    EvalCase(
        name="play_radio_pt_connector",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="põe uma radio de djonga",
        expected_mode="SPOTIFY_PLAY_RADIO",
        expected_capture="djonga",
        description="'de/do/da' connector after 'radio' is stripped.",
    ),
    EvalCase(
        name="play_radio_en_trailing",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="play matue radio",
        expected_mode="SPOTIFY_PLAY_RADIO",
        expected_matcher="SPOTIFY_PLAY_RADIO_EN_RE",
        expected_capture="matue",
        description="EN natural phrasing: 'radio' AFTER the artist name.",
    ),
    EvalCase(
        name="play_radio_does_not_steal_plain_play",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="toca matue",
        expected_mode="SPOTIFY_PLAY",
        expected_capture="matue",
        description="No 'radio' word — bare play must stay SPOTIFY_PLAY.",
    ),
    # ------- bare verb = resume (mirrors Spotify play button) -------
    EvalCase(
        name="bare_play_resumes",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="play",
        expected_mode="SPOTIFY_RESUME",
        expected_matcher="SPOTIFY_RESUME_RE",
        description=(
            "Bare 'play' must resume deterministically. Without this it "
            "fell to the LLM, which leaked a raw tool-call object to the "
            "user instead of acting."
        ),
    ),

    # ------- M4-A1: network out → speakable reply, never an exception -------
    # The fake orch carries a REAL SpotifyPlayer whose client raises
    # ConnectionError (not a SpotifyException) on every call. Pre-fix the 8
    # short methods only caught SpotifyException, so a spoken "pausa" on
    # flaky Wi-Fi became an HTTP 500 / dead SSE turn.
    EvalCase(
        name="pause_network_down_pt",
        kind=EvalKind.DISPATCH_REPLY,
        category="spotify",
        input="pausa",
        language="pt",
        expected_contains="Sem conexão com o Spotify",
        description="'pausa' with the network out must come back speakable PT.",
    ),
    EvalCase(
        name="pause_network_down_en",
        kind=EvalKind.DISPATCH_REPLY,
        category="spotify",
        input="pause the music",
        language="en",
        expected_contains="Can't reach Spotify",
        description="EN turn with the network out gets the EN line.",
    ),
    EvalCase(
        name="next_network_down_pt",
        kind=EvalKind.DISPATCH_REPLY,
        category="spotify",
        input="próxima",
        language="pt",
        expected_contains="Sem conexão com o Spotify",
        description="The wake-on-404 helper path also surfaces the guard line.",
    ),
    EvalCase(
        name="now_playing_que_musica_tocando",
        kind=EvalKind.DISPATCH,
        category="spotify",
        input="que música está tocando",
        expected_mode="SPOTIFY_NOW",
        expected_matcher="SPOTIFY_NOW_RE",
        description=(
            "s75 locked-screen battery finding: natural PT now-playing "
            "phrasing fell to chat; pinned after widening SPOTIFY_NOW_RE."
        ),
    ),
]
