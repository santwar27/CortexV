"""Priority / conflict guarantees between competing dispatchers.

These cases check ORDER, not just individual regex correctness:
when two regexes can both match an input, the earlier dispatcher
must win. If priority ever flips silently, these fail loudly.
"""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # Spotify single-word verbs win over PLAY (which has broad capture)
    EvalCase(
        name="pause_beats_play",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="pausa",
        expected_mode="SPOTIFY_PAUSE",
        description=(
            "Single-word 'pausa' must not be swallowed by the broader "
            "SPOTIFY_PLAY_RE capture group."
        ),
    ),
    EvalCase(
        name="next_beats_play",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="próxima",
        expected_mode="SPOTIFY_NEXT",
    ),

    # Bare volume controls the Mac; Spotify volume requires its name.
    EvalCase(
        name="bare_volume_routes_mac",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="volume 80",
        expected_mode="MAC_VOLUME",
        expected_capture="80",
        description="An unqualified volume number must route to the Mac.",
    ),
    EvalCase(
        name="volume_sistema_routes_mac",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="volume sistema 80",
        expected_mode="MAC_VOLUME",
        expected_capture="80",
    ),

    # Mac dark TOGGLE beats ON/OFF when verb is "alterna"/"troca"
    EvalCase(
        name="alterna_tema_is_toggle_not_on",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="alterna o tema",
        expected_mode="MAC_DARK",
        expected_matcher="MAC_DARK_TOGGLE_RE",
    ),

    # Mac close-window ("fecha") beats the generic OPEN — distinct from
    # quit ("encerra"/"quit"). the user's 2026-06-03 distinction.
    EvalCase(
        name="fecha_routes_close_not_open",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="fecha o Notion",
        expected_mode="MAC_CLOSE_WINDOW",
    ),

    # Fallthrough TOOL_HINT_RE coverage (bloco H ext): "ouvir queen"
    # must be tagged would_use_tools so the LLM gets the tool list
    # even when the regex dispatch doesn't fire.
    EvalCase(
        name="natural_music_phrasing_routes_via_tools",
        kind=EvalKind.DISPATCH,
        category="conflict",
        input="vamos ouvir queen",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Phrase fell through dispatch BUT mentions music — "
            "TOOL_HINT_RE should mark this for chat_with_tools (not "
            "pure chat), giving the LLM a chance to call spotify_control."
        ),
        tags=["jou-s11-blocoH"],
    ),
]
