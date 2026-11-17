"""Ordinary conversation must not trigger a command (s83).

Every other dispatch test proves a command WORKS. None proved that talking
normally is left alone — which is how "por que azul?" came to be answered by
putting on a pagode album, and "desliga a tela" by trying to quit an app
called "tela". The regexes are written to catch a phrasing; this file is
written to catch what else that phrasing catches.

Two lists, and the second matters as much as the first:

  SMALL_TALK        — must stay in chat. A failure here is a real bug.
  TOLERATED         — fires today, and we decided that is acceptable. Each
                      line carries WHY. A failure here is not necessarily a
                      bug, but it is a decision someone changed, so it has to
                      be looked at and this list updated.

Adding to SMALL_TALK is cheap and is the point: when a false positive turns
up in real use, it belongs here before it is fixed.
"""
import pytest

from evals.dispatch import simulate_dispatch


SMALL_TALK = [
    # questions — the "por que" family, which used to play music (s83)
    "por que azul?",
    "por que você escolheu azul?",
    "por que um gato?",
    "por que não?",
    "por que o céu é azul?",
    "por favor me explica",
    "por acaso tu sabe?",
    "por enquanto não",
    "por isso que eu falei",
    "por exemplo, o que é RAG?",
    "por volta das 3",
    "como assim?",
    "o que você acha disso?",
    "quem é você?",
    "tudo bem com você?",
    "e aí, beleza?",
    # play verbs used as idiom — these STARTED music before s83
    "toca o barco",
    "toca pra frente",
    "toca em frente",
    "toca fundo",
    "toca no assunto",
    "a música toca fundo",
    "isso me toca profundamente",
    "play it cool",
    "play it safe",
    "play along",
    "play hardball",
    "play the fool",
    "play devil's advocate",
    # short verbs that are also commands
    "liga pra minha mãe quando puder",
    "me liga amanhã",
    "isso liga uma coisa na outra",
    "conta uma piada",
    "me conta como foi teu dia",
    "conta comigo",
    "a conta do restaurante veio errada",
    "manda um abraço pra ela",
    "quem manda aqui é você",
    "o chefe manda e a gente obedece",
    "tira uma dúvida minha",
    "tira o dia pra descansar",
    "ele tira sarro de mim",
    "tira essa ideia da cabeça",
    "a flor abre de manhã",
    "isso abre precedente",
    "corta essa",
    "o filme corta pra outra cena",
    "corta o barato dele",
    "para de brincadeira",
    "o ônibus para aqui",
    "volta pra casa",
    "a moda volta sempre",
    "volta e meia ele some",
    "sobe o astral",
    "o preço sobe todo ano",
    "desce a ladeira",
    "muda de assunto",
    "o tempo muda rápido",
    "apaga essa lembrança",
    "a vela apaga com o vento",
    "dorme bem",
    "quem dorme cedo acorda cedo",
    "trava a discussão",
    "meu joelho trava às vezes",
    "o jogo trava no meu celular",
    "guarda esse segredo",
    "procura um emprego melhor",
    "ele procura briga",
    "procura saber antes de falar",
    "acha graça nisso",
    "acha que vai chover?",
    "escreve aí que eu falei",
    "ele escreve bem",
    "ninguém lembra mais disso",
    "a agenda do país mudou",
    "pesquisa mostra que o brasil cresceu",
    "ele vai travar a minha bike na rua",
    "hoje vou dormir cedo",
    "o monitor desligou sozinho",
    "vou de repouso médico amanhã",
    "quero ler um livro hoje",
    # English
    "call me later",
    "that calls for a celebration",
    "the shop opens at nine",
    "read between the lines",
    "it finds a way",
    "turn off the lights in the story",
    # "lembra" as remembering, not as "remind me" — class 2 of the s83 sweep,
    # fixed s84. Voice drops the "?", so the bare forms are here too.
    "lembra daquele dia?",
    "lembra daquele dia",
    "lembra de mim?",
    "lembra de mim",
    "lembra quando fomos à praia?",
    "lembra quando fomos à praia",
    "lembra de quando a gente se conheceu",
    "lembra do que eu te falei?",
    "lembra como se faz bolo?",
    "me lembra da minha avó",
    "me lembra de quando eu era criança",
]


# (phrase, mode it routes to, why we accept that)
TOLERATED = [
    ("manda ver", "APPROVAL",
     "'manda ver' IS an approval in Portuguese — routing it to yes is right"),
    ("lembra que amanhã tem jogo", "REMINDER",
     "genuinely ambiguous — it can be a reminder request, and it carries a "
     "time; create_reminder is S3, so it asks before creating anything"),
    ("agenda cheia essa semana", "CALENDAR",
     "'agenda' as a noun read as the verb; the calendar path only reads"),
    ("fecha negócio com eles", "MAC_CLOSE_WINDOW",
     "close_window checks the app is running first and answers 'não está "
     "aberto' — a silly reply, not an action (verified s83)"),
    ("fecha o ano com lucro", "MAC_CLOSE_WINDOW", "same as above"),
    ("close enough", "MAC_CLOSE_WINDOW", "same as above"),
    ("abre o jogo comigo", "MAC_OPEN_APP",
     "open_app answers 'não consegui abrir' for a name macOS doesn't know"),
    ("open your mind", "MAC_OPEN_APP", "same as above"),
]


@pytest.mark.parametrize("text", SMALL_TALK)
def test_conversation_is_not_a_command(text):
    outcome = simulate_dispatch(text)
    assert outcome.mode == "OFFLINE", (
        f"{text!r} was routed to {outcome.mode} via {outcome.matcher} "
        f"(captured {outcome.capture!r}) — ordinary conversation must reach "
        f"the chat path untouched."
    )


@pytest.mark.parametrize("text, mode, why", TOLERATED)
def test_known_false_positives_still_behave_as_documented(text, mode, why):
    outcome = simulate_dispatch(text)
    assert outcome.mode == mode, (
        f"{text!r} used to route to {mode} ({why}); it now goes to "
        f"{outcome.mode}. If that was deliberate, update TOLERATED — and if "
        f"it is now chat, move the line to SMALL_TALK."
    )


class TestPuttingSomethingInAPlaceIsNotMusic:
    """"coloca no calendário: dentista amanhã às 10" and "coloca na nota
    Viagem: passaporte" were searched on Spotify: SPOTIFY_PLAY_RE takes
    "coloca" + anything and runs before the calendar and notes routes (s88,
    Codex lote 9)."""

    @pytest.mark.parametrize("text,mode", [
        ("coloca no calendário: dentista amanhã às 10", "CALENDAR"),
        ("coloca na agenda: reunião sexta às 15h", "CALENDAR"),
        ("coloca na nota Viagem: passaporte", "NOTE_APPEND"),
    ])
    def test_it_goes_where_it_was_put(self, text, mode):
        from src.core.dispatch.registry import simulate
        m = simulate(text)
        assert m is not None and m.mode == mode, text

    @pytest.mark.parametrize("text", ["põe um lembrete pra amanhã às 9",
                                      "põe na lista de compras arroz"])
    def test_never_music(self, text):
        from src.core.dispatch.registry import simulate
        m = simulate(text)
        assert m is None or not m.mode.startswith("SPOTIFY"), text

    @pytest.mark.parametrize("text", ["coloca matue", "põe um pagode aí",
                                      "coloca a música do calendário maia"])
    def test_music_is_still_music(self, text):
        from src.core.dispatch.registry import simulate
        m = simulate(text)
        assert m is not None and m.mode.startswith("SPOTIFY"), text
