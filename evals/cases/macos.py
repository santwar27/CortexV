"""macOS system control dispatch behaviour (bloco G + bloco A sessão 12)."""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    # ------- volume sistema (requires 'sistema'/'system' keyword) -------
    EvalCase(
        name="volume_sistema_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="volume sistema 30",
        expected_mode="MAC_VOLUME",
        expected_matcher="MAC_VOLUME_SET_RE",
        expected_capture="30",
    ),
    EvalCase(
        name="volume_sistema_do",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="volume do sistema 80",
        expected_mode="MAC_VOLUME",
        expected_capture="80",
    ),
    EvalCase(
        name="volume_system_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="system volume to 60",
        expected_mode="MAC_VOLUME",
        expected_capture="60",
    ),

    # ------- volume up/down/mute/unmute (no spotify counterpart) -------
    EvalCase(
        name="volume_up",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="aumenta o volume",
        expected_mode="MAC_VOLUME",
        expected_matcher="MAC_VOLUME_UP_RE",
    ),
    EvalCase(
        name="volume_down",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="abaixa o volume",
        expected_mode="MAC_VOLUME",
        expected_matcher="MAC_VOLUME_DOWN_RE",
    ),
    EvalCase(
        name="mute",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="mudo",
        expected_mode="MAC_MUTE",
    ),
    EvalCase(
        name="unmute",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="tira o mudo",
        expected_mode="MAC_UNMUTE",
    ),

    # ------- brightness -------
    EvalCase(
        name="brightness_up",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="aumenta o brilho",
        expected_mode="MAC_BRIGHTNESS",
        expected_matcher="MAC_BRIGHTNESS_UP_RE",
    ),
    EvalCase(
        name="brightness_set",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="brilho 50",
        expected_mode="MAC_BRIGHTNESS",
        expected_matcher="MAC_BRIGHTNESS_SET_RE",
        expected_capture="50",
    ),

    # ------- dark mode -------
    EvalCase(
        name="dark_toggle",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="alterna o tema",
        expected_mode="MAC_DARK",
        expected_matcher="MAC_DARK_TOGGLE_RE",
    ),
    EvalCase(
        name="dark_on",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="modo escuro",
        expected_mode="MAC_DARK",
        expected_matcher="MAC_DARK_ON_RE",
    ),
    EvalCase(
        name="dark_off",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="modo claro",
        expected_mode="MAC_DARK",
        expected_matcher="MAC_DARK_OFF_RE",
    ),

    # ------- lock + screenshot -------
    EvalCase(
        name="lock",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="trava a tela",
        expected_mode="MAC_LOCK",
    ),
    EvalCase(
        name="screenshot",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="tira print",
        expected_mode="MAC_SCREENSHOT",
    ),

    # ------- app open / quit (bloco A sessão 12 added quit) -------
    EvalCase(
        name="open_app",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="abre Notion",
        expected_mode="MAC_OPEN_APP",
        expected_capture="Notion",
    ),
    # s60: "open"/"abrir" must reach the macOS open route, not be hijacked by
    # the studier SHOW verb (which used to list "open"/"abrir" → "open spotify"
    # routed to STUDIER_SHOW "spotify" not-found). Regression guard.
    EvalCase(
        name="open_app_en",
        kind=EvalKind.DISPATCH, category="macos",
        input="open Spotify",
        expected_mode="MAC_OPEN_APP", expected_capture="Spotify",
        description="EN 'open X' must route to macOS open, not studier-show.",
    ),
    EvalCase(
        name="open_app_en_notion",
        kind=EvalKind.DISPATCH, category="macos",
        input="open Notion",
        expected_mode="MAC_OPEN_APP", expected_capture="Notion",
    ),
    EvalCase(
        name="open_app_abrir_infinitive",
        kind=EvalKind.DISPATCH, category="macos",
        input="abrir o Spotify",
        expected_mode="MAC_OPEN_APP",
        description="PT infinitive 'abrir X' must route to macOS open, not studier-show.",
    ),
    EvalCase(
        name="quit_app_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="encerra o Notion",
        expected_mode="MAC_QUIT_APP",
        expected_capture="Notion",
        description="Quit (encerra/quit) is the destructive Cmd+Q path — "
                    "distinct from close-window. 'fecha' no longer routes here.",
    ),
    EvalCase(
        name="quit_app_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="quit Notion",
        expected_mode="MAC_QUIT_APP",
        expected_capture="Notion",
    ),
    # ------- M6-A1 adversarial: revoke verbs vs quit verbs (M9-A5) -------
    # "desliga a internet" used to stage quitting an app called "internet"
    # (MAC_QUIT_APP_RE ate it ~50 lines before evaluate_gate); the
    # gatekeeper's revoke branch was dead code on that phrasing. The
    # lookahead in MAC_QUIT_APP_RE + the gate state-change step in the
    # simulator pin the collision both ways.
    EvalCase(
        name="adversarial_desliga_internet_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga a internet",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
        description="The M6-A1 collision: a lease-revoke phrasing must NEVER "
                    "route to MAC_QUIT_APP (quitting an app called 'internet').",
    ),
    EvalCase(
        name="adversarial_desliga_internet_ai_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga a internet aí",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
    ),
    EvalCase(
        name="adversarial_corta_internet_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="corta a internet",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
    ),
    EvalCase(
        name="adversarial_desliga_net_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga a net",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
        description="Pedro's real phrasing — 'net' counts as internet, but "
                    "the word boundary keeps 'netflix' an app.",
    ),
    EvalCase(
        name="adversarial_desligar_tela_is_lock",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desligar tela",
        expected_mode="MAC_LOCK",
        expected_matcher="MAC_LOCK_RE",
        description="Pedro's real phrasing over Blink (2026-09-19): it staged "
                    "quitting an app called 'tela'. It's a lock.",
    ),
    EvalCase(
        name="adversarial_desliga_o_mac_is_sleep",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga o mac",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
        description="Cortex never shuts the Mac down — it sleeps, and the "
                    "confirmation says so. Before s83 it staged quitting an "
                    "app called 'mac'.",
    ),
    EvalCase(
        name="adversarial_desconecta_net_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desconecta a net",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
    ),
    EvalCase(
        name="adversarial_fica_offline_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="fica offline",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
    ),
    EvalCase(
        name="adversarial_desliga_wifi_is_revoke",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga o wi-fi",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_revoke",
    ),
    # The grant mirror (s63 follow-up — live gap: "Liga a internet!" fell
    # through to plain chat; there was no spoken way to reopen the gate).
    EvalCase(
        name="liga_internet_is_grant",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="liga a internet",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_lease_grant",
        description="Bare grant verb + net noun opens the default lease "
                    "window — the spoken mirror of 'desliga a internet'.",
    ),
    EvalCase(
        name="conecta_net_is_grant",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="conecta a net",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_lease_grant",
    ),
    EvalCase(
        name="fica_online_is_grant",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="fica online",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_lease_grant",
    ),
    EvalCase(
        name="liga_internet_with_hours_is_grant",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="liga a internet por 2 horas",
        expected_mode="SYSTEM_UPDATE",
        expected_matcher="gate_lease_grant",
    ),
    EvalCase(
        name="generic_liga_is_not_grant",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="liga pra ele",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description="'liga' without a net noun is generic PT (phone him, "
                    "turn on the speakers) — must never open the gate.",
    ),

    EvalCase(
        name="adversarial_desliga_netflix_is_still_quit",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga a Netflix",
        expected_mode="MAC_QUIT_APP",
        expected_capture="Netflix",
        description="The other side of the boundary: 'netflix' starts with "
                    "'net' but must remain an app quit, never a revoke.",
    ),
    EvalCase(
        name="adversarial_desliga_spotify_is_still_quit",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="desliga o Spotify",
        expected_mode="MAC_QUIT_APP",
        expected_capture="Spotify",
    ),

    EvalCase(
        name="close_window_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="fecha o Notion",
        expected_mode="MAC_CLOSE_WINDOW",
        expected_capture="Notion",
        description="'fecha o X' closes the window (red button), not quit "
                    "(Pedro's distinction, 2026-06-03).",
    ),
    EvalCase(
        name="close_window_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="close Spotify",
        expected_mode="MAC_CLOSE_WINDOW",
        expected_capture="Spotify",
    ),

    # ------- vocative still works for mac -------
    EvalCase(
        name="dark_with_vocative",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="cortex, modo escuro",
        expected_mode="MAC_DARK",
        description="Vocative strip applies uniformly across dispatches.",
        tags=["regression"],
    ),

    # ------- app list / switch (batch 3) -------
    EvalCase(
        name="app_list_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="quais apps estão abertos",
        expected_mode="MAC_APP_LIST",
        expected_matcher="MAC_APP_LIST_RE",
        tags=["jou-s16"],
    ),
    EvalCase(
        name="app_list_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="what apps are open",
        expected_mode="MAC_APP_LIST",
        tags=["jou-s16"],
    ),
    # ------- reply to the email in front (inovação #3, Bloco A) -------
    EvalCase(
        name="reply_email_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="responde esse email",
        expected_mode="MAC_REPLY_EMAIL",
        expected_matcher="MAC_REPLY_EMAIL_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="reply_email_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="reply to this email",
        expected_mode="MAC_REPLY_EMAIL",
        tags=["inov-3"],
    ),
    # ------- summarise what's in front (inovação #3, Bloco B) -------
    EvalCase(
        name="summarise_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="resume isso",
        expected_mode="MAC_SUMMARISE",
        expected_matcher="MAC_SUMMARISE_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="summarise_page_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="o que tem nessa página",
        expected_mode="MAC_SUMMARISE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="summarise_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="summarise this",
        expected_mode="MAC_SUMMARISE",
        tags=["inov-3"],
    ),
    # s76 live smoke: these natural phrasings leaked to the LLM (and got
    # hallucinated) because the old regex missed the verb "resumir" and the
    # objects pdf/conteúdo/"na minha frente". Lock them onto the screen path.
    EvalCase(
        name="summarise_resumir_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="resumir isso",
        expected_mode="MAC_SUMMARISE",
        expected_matcher="MAC_SUMMARISE_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="summarise_pdf_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="resume esse pdf",
        expected_mode="MAC_SUMMARISE",
        expected_matcher="MAC_SUMMARISE_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="summarise_conteudo_frente_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="resume o conteúdo na minha frente",
        expected_mode="MAC_SUMMARISE",
        expected_matcher="MAC_SUMMARISE_RE",
        tags=["inov-3"],
    ),
    # ------- "o que é isso?" — identify what's in front (inovação #3, C) -------
    EvalCase(
        name="what_is_this_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="o que é isso",
        expected_mode="MAC_WHAT_IS",
        expected_matcher="MAC_WHAT_IS_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="what_is_this_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="what is this",
        expected_mode="MAC_WHAT_IS",
        tags=["inov-3"],
    ),
    # ------- voice toggle of the screen-reading kill-switch (inovação #3) -------
    EvalCase(
        name="eyes_off_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="para de ver minha tela",
        expected_mode="MAC_EYES_OFF",
        expected_matcher="MAC_EYES_OFF_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        name="eyes_off_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="stop watching my screen",
        expected_mode="MAC_EYES_OFF",
        tags=["inov-3"],
    ),
    EvalCase(
        name="eyes_on_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="volta a ver minha tela",
        expected_mode="MAC_EYES_ON",
        expected_matcher="MAC_EYES_ON_RE",
        tags=["inov-3"],
    ),
    EvalCase(
        # s76 cont.31 live smoke: the user spoke the imperative "VOLTE" and the
        # first regex (volta/voltar only) let it fall through to chat. Pin the
        # exact Whisper transcription (capitalised + trailing period).
        name="eyes_on_volte_imperative",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="Volte a ver minha tela.",
        expected_mode="MAC_EYES_ON",
        tags=["inov-3"],
    ),
    EvalCase(
        name="eyes_on_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="start watching my screen again",
        expected_mode="MAC_EYES_ON",
        tags=["inov-3"],
    ),
    EvalCase(
        name="app_switch_pt",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="vai pro Notion",
        expected_mode="MAC_APP_SWITCH",
        expected_matcher="MAC_APP_SWITCH_RE",
        expected_capture="Notion",
        tags=["jou-s16"],
    ),
    EvalCase(
        name="app_switch_foca",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="foca no Spotify",
        expected_mode="MAC_APP_SWITCH",
        expected_capture="Spotify",
        tags=["jou-s16"],
    ),
    EvalCase(
        name="app_switch_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="switch to Safari",
        expected_mode="MAC_APP_SWITCH",
        expected_matcher="MAC_APP_SWITCH_RE",
        expected_capture="Safari",
        description="English app-switch — parallels the PT cases.",
    ),
    EvalCase(
        name="app_switch_does_not_steal_open",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="abre Notion",
        expected_mode="MAC_OPEN_APP",
        expected_capture="Notion",
        description="switch verbs are disjoint from open — 'abre' stays OPEN.",
        tags=["jou-s16", "conflict"],
    ),

    # ----- system sleep (sessão 21, S3 staged) -----
    EvalCase(
        name="sleep_pt_dorme",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="dorme o mac",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
    ),
    EvalCase(
        name="sleep_pt_poe_pra_dormir",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="põe o mac pra dormir",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
    ),
    EvalCase(
        name="sleep_pt_poe_pra_dormi_dropped_r",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="põe o mac pra dormi",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
        description=(
            "BR accent / STT drops the final r of 'dormir' → 'dormi'; "
            "must still route to MAC_SLEEP (same family as the 'Meck' aliases)."
        ),
    ),
    EvalCase(
        name="sleep_pt_suspende",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="suspende o mac",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
    ),
    EvalCase(
        name="sleep_en",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="sleep the mac",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
    ),
    EvalCase(
        name="sleep_en_put_to_sleep",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="put the mac to sleep",
        expected_mode="MAC_SLEEP",
        expected_matcher="MAC_SLEEP_RE",
    ),
    EvalCase(
        name="sleep_does_not_steal_conversational_dormir",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="hoje vou dormir cedo",
        expected_mode="OFFLINE",
        expected_matcher="fallthrough",
        description=(
            "Conversational 'dormir' (no 'mac/sistema' qualifier) must NOT "
            "fire the sleep regex — false-positive risk would be very bad "
            "(literally puts the mac to sleep on a chat reply)."
        ),
    ),
    EvalCase(
        name="sleep_does_not_steal_close_mac",
        kind=EvalKind.DISPATCH,
        category="macos",
        input="fecha o mac",
        # NOTE: genuinely ambiguous — "fecha o mac" reads as close-window
        # of an app called "mac" (nonsense; mac isn't an app). The
        # close-window `<app>` capture is permissive and matches; the
        # downstream MacOSController surfaces an honest failure. We pin it
        # so the SLEEP regex (dorme/suspende) doesn't accidentally steal it.
        expected_mode="MAC_CLOSE_WINDOW",
        expected_capture="mac",
        description=(
            "Sleep verbs (dorme/suspende/sleep) are disjoint from the close "
            "verb (fecha) — 'fecha o mac' routes to close-window, never sleep."
        ),
    ),
    # ---- s75 (Onda 8 A): courtesy wrapper must not defeat dispatch ----
    # Smoke finding 1 (s74): "Can you open Spotify please?" fell to the
    # OFFLINE chat path while "Open the Spotify" routed. strip_courtesy
    # now peels the polite envelope before the verb-first regexes.
    EvalCase(
        name="open_app_courtesy_en",
        kind=EvalKind.DISPATCH, category="macos",
        input="Can you open Spotify please?",
        expected_mode="MAC_OPEN_APP", expected_capture="Spotify",
        description="The exact s74 smoke phrase must route, not chat.",
    ),
    EvalCase(
        name="open_app_courtesy_pt",
        kind=EvalKind.DISPATCH, category="macos",
        input="você pode abrir o Spotify por favor?",
        expected_mode="MAC_OPEN_APP", expected_capture="Spotify",
        description="PT courtesy wrapper around open-app must route.",
    ),
    EvalCase(
        name="open_app_stt_alias_spot5",
        kind=EvalKind.DISPATCH, category="macos",
        input="Can you open spot 5 please?",
        expected_mode="MAC_OPEN_APP", expected_capture="spotify",
        description=(
            "Whisper small + BR accent hears 'spotify' as 'spot 5' "
            "(audit 2026-06-12). normalise_stt_aliases rewrites the "
            "routing text so the open-app route still fires."
        ),
    ),
    EvalCase(
        name="open_app_courtesy_pra_mim",
        kind=EvalKind.DISPATCH, category="macos",
        input="abre o Spotify pra mim",
        expected_mode="MAC_OPEN_APP", expected_capture="Spotify",
        description="Trailing 'pra mim' must not pollute the app capture.",
    ),
]
