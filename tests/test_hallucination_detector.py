"""Tests for src/core/hallucination.py — the Llama 8B "simulated
action" failure-mode detector.

Coverage philosophy: this module is deliberately conservative. A
false positive (flagging a legitimate reply as hallucination) is
WORSE than a false negative (missing some hallucinated replies)
because the user sees a wrong "couldn't do that" on a real answer.

Tests are split across three buckets:

  1. POSITIVE cases — replies that MUST be flagged. These are the
     exact patterns we've observed Llama 8B produce when given
     tools it chose to ignore.
  2. NEGATIVE cases — replies that MUST NOT be flagged. Includes
     knowledge essays mentioning music/email, real tool replies
     (which only appear after a tool call but should still be safe),
     refusal sentences, and edge-case empty/whitespace inputs.
  3. EDGE cases — multi-line, mixed languages, casing, surrounding
     whitespace.

If a new false-positive surfaces in production, add it here as a
regression test then loosen the pattern. If a new hallucination
shape surfaces, add it as a positive case then tighten the pattern.
"""
import pytest

from src.core.hallucination import (
    HALLUCINATION_FALLBACK,
    detect_hallucinated_action,
    detect_hallucinated_mac_action,
)


# =====================================================================
# Positive — must flag
# =====================================================================

class TestHallucinationPositive:
    """The exact failure shapes Llama 8B produces."""

    @pytest.mark.parametrize("reply", [
        # The original bug (sessão 10 bloco F)
        "Tocando 'Fim da Noite' de Matuê.",
        "Tocando 'BACKSTAGE' de Matuê — relaxa que já tá rolando.",
        "Tocou matue agora.",
        "Tocará a próxima música em seguida.",
        "Toquei a música pra você, Pedro.",
        # Parenthetical narration — Llama 8B signature
        "Aqui vai. (pausa) Aproveita.",
        "(pausa) ... agora a música está tocando!",
        "(música começa) Boa escolha.",
        "(música termina) Pronto.",
        "(simulando reprodução) Tocando agora.",
        "(simulando envio) Email enviado.",
        "Tudo certo (pausa para simular).",
        # English equivalents
        "Playing 'Bohemian Rhapsody' by Queen.",
        "(music starts) Enjoy.",
        "(pause) Now back to music.",
        "Now playing the requested track.",
        # Email claims
        "Enviei o e-mail para João.",
        "Enviei a mensagem agora.",
        "E-mail enviado para John.",
        "Mensagem enviada com sucesso.",
        "I have sent the email to John.",
        "Sent the message to your colleague.",
        # Calendar / reminder claims
        "Agendado para amanhã às 14h.",
        "Marquei na agenda.",
        "Marquei o evento de reunião.",
        "Evento criado para sexta-feira.",
        "Lembrete criado: comprar leite.",
        "Lembrete salvo.",
        "Criei o lembrete pras 18h.",
        "Reminder created for tomorrow.",
        # Generic completion
        "Feito.",
        "Done.",
    ])
    def test_flagged(self, reply):
        assert detect_hallucinated_action(reply), f"should flag: {reply!r}"


# =====================================================================
# Negative — must NOT flag
# =====================================================================

class TestHallucinationNegative:
    """Replies that legitimately mention music/email/calendar without
    falsely claiming an action."""

    @pytest.mark.parametrize("reply", [
        # Knowledge essays mentioning music
        "O Matuê é um rapper brasileiro de São Paulo. Conhecido por trap.",
        "Queen tocou no Live Aid em 1985, um dos shows mais memoráveis.",
        "A música 'Bohemian Rhapsody' tem três partes distintas.",
        "Spotify foi fundado em 2006 na Suécia.",
        # Knowledge essays mentioning email
        "Um email tem cabeçalho, corpo e anexos opcionais. Funciona via SMTP.",
        "O protocolo SMTP é usado para enviar emails desde os anos 80.",
        # Refusal phrasings
        "Não consigo tocar música sem o Spotify aberto.",
        "Preciso da ferramenta de email pra fazer isso.",
        "Não consegui acionar a ferramenta certa pra isso.",
        # Generic chat
        "Oi! Em que posso ajudar?",
        "Bom dia, Pedro.",
        "O dia está bonito hoje.",
        # Definitions
        "RAG significa Retrieval Augmented Generation.",
        "Transformer é uma arquitetura de rede neural.",
        # Questions back to user
        "Quer que eu tente de novo?",
        "Que música tu quer ouvir?",
        # Empty / whitespace
        "",
        "   ",
        "\n\n",
        # Very short non-claims
        "OK.",
        "Sim.",
        "Claro.",
        # Multi-line essays
        "Música é uma forma de arte.\nExiste em todas as culturas.",
        # Mid-line action verb (must not trip first-line regex)
        "Aqui vai a explicação: para tocar uma música no Spotify, você precisa abrir o app.",
    ])
    def test_not_flagged(self, reply):
        assert not detect_hallucinated_action(reply), f"false positive: {reply!r}"


# =====================================================================
# Edge cases
# =====================================================================

class TestHallucinationEdge:

    def test_none_input(self):
        # Defensive — caller might pass None unintentionally.
        assert detect_hallucinated_action(None) is False  # type: ignore[arg-type]

    def test_leading_whitespace_still_caught(self):
        assert detect_hallucinated_action("   Tocando matue agora.")

    def test_leading_newlines_still_caught(self):
        assert detect_hallucinated_action("\n\nTocando matue agora.")

    def test_uppercase_still_caught(self):
        assert detect_hallucinated_action("TOCANDO MATUE AGORA.")

    def test_action_in_second_paragraph_not_flagged(self):
        # First line is fine; the model finished a thought and the
        # "action" claim is in a later paragraph. Conservative.
        reply = (
            "O Matuê tem várias músicas famosas.\n"
            "Tocando 'BACKSTAGE' agora."
        )
        # First non-empty line is "O Matuê tem várias..." which is
        # essay tone, not action. Detector should NOT flag.
        assert not detect_hallucinated_action(reply)

    def test_paren_simulation_in_paragraph_flagged(self):
        # Parenthetical patterns fire anywhere in the response.
        reply = (
            "Vamos ouvir um pouco.\n"
            "(pausa)\n"
            "Curtindo?"
        )
        assert detect_hallucinated_action(reply)


# Task 110: each Mac control family, including both measured brightness lies.
MAC_CLAIMS = [
    "Diminuí o brilho da tela do seu Mac.",
    "Diminuí o brilho da tela do seu Mac para 0.",
    *[f"{verb} o {obj} do Mac." for verb in (
        "Diminuí", "Diminui", "Aumentei", "Abaixei", "Baixei", "Reduzi",
        "Subi", "Ajustei", "Coloquei", "Deixei", "Mudei", "Alterei", "Configurei",
    ) for obj in ("brilho", "volume", "som")],
    *[f"{verb} o {obj}." for verb in (
        "Ativei", "Desativei", "Liguei", "Desliguei",
    ) for obj in (
        "modo escuro", "modo claro", "tema escuro", "tema claro", "modo noturno",
        "dark mode", "light mode", "mudo", "som",
    )],
    "Silenciei.", "Silenciei o Mac.", "Tirei o som.", "Coloquei no mudo.",
    *[f"{verb} {obj}." for verb in ("Travei", "Bloqueei")
      for obj in ("a tela", "o Mac", "o computador")],
    *[f"{verb} {article} {obj}." for verb in ("Abri", "Fechei", "Encerrei")
      for article in ("o", "a")
      for obj in ("app", "aplicativo", "programa", "janela", "Safari", "Prévia")],
    *[f"{verb} {article}{obj}." for verb in ("Tirei", "Fiz", "Capturei")
      for article in ("", "um ", "uma ")
      for obj in ("print", "screenshot", "captura de tela")],
    *[f"Brilho {state}." for state in ("diminuído", "reduzido", "aumentado", "ajustado")],
    *[f"Volume {state}." for state in ("ajustado", "aumentado", "diminuído", "reduzido")],
    "Modo escuro ativado.", "Modo escuro desativado.",
    "Tela travada.", "Tela bloqueada.",
    *[f"I {have}{verb} the {obj}." for have in ("", "have ")
      for verb, obj in (
          ("lowered", "brightness"), ("raised", "volume"), ("dimmed", "screen"),
          ("increased", "sound"), ("decreased", "brightness"), ("set", "volume"),
          ("adjusted", "brightness"), ("turned up", "volume"),
          ("turned down", "sound"), ("turned on", "dark mode"),
          ("turned off", "light mode"), ("muted", "sound"), ("unmuted", "volume"),
          ("enabled", "dark mode"), ("disabled", "light mode"),
          ("locked", "screen"), ("opened", "app"), ("closed", "app"),
          ("quit", "app"),
      )],
    *[f"I {have}{verb} {obj}." for have in ("", "have ")
      for verb, obj in (
          ("opened", "Safari"), ("closed", "Preview"), ("quit", "Safari"),
          ("took", "a screenshot"),
      )],
    *[f"Brightness {state}." for state in ("lowered", "set", "adjusted")],
    *[f"Volume {state}." for state in ("set", "adjusted", "lowered", "raised")],
    *[f"Dark mode {state}." for state in ("enabled", "on", "off", "disabled")],
    "Screen locked.",
]


@pytest.mark.parametrize("prefix", ["", "Pronto, "])
@pytest.mark.parametrize("reply", MAC_CLAIMS)
def test_mac_claim_flagged(reply, prefix):
    assert detect_hallucinated_action(prefix + reply), repr(prefix + reply)


@pytest.mark.parametrize("prefix", ["Feito! ", "OK: ", "Done. ", "pronto — "])
def test_mac_completion_prefixes(prefix):
    assert detect_hallucinated_action(prefix + "Diminuí o brilho.")


# Task 110b: measured 8B replies, third-person narration and screenshot results.
@pytest.mark.parametrize("reply", [
    "Diminuiu o brilho da tela para 0%.",
    "Diminuiu o volume para 0%.",
    "Aqui está um print da sua tela agora.",
    *[f"{verb} o {obj} do Mac." for verb in (
        "Diminuiu", "Aumentou", "Abaixou", "Baixou", "Reduziu", "Subiu",
        "Ajustou", "Colocou", "Deixou", "Mudou", "Alterou", "Configurou",
    ) for obj in ("brilho", "volume", "som")],
    *[f"{verb} o {obj}." for verb in (
        "Ativou", "Desativou", "Ligou", "Desligou",
    ) for obj in (
        "modo escuro", "modo claro", "tema escuro", "tema claro", "modo noturno",
        "dark mode", "light mode", "mudo", "som",
    )],
    "Silenciou.", "Silenciou o Mac.", "Tirou o som.", "Colocou no mudo.",
    *[f"{verb} {obj}." for verb in ("Travou", "Bloqueou")
      for obj in ("a tela", "o Mac", "o computador")],
    *[f"{verb} {article} {obj}." for verb in ("Abriu", "Fechou", "Encerrou")
      for article in ("o", "a")
      for obj in ("app", "aplicativo", "programa", "janela", "Safari", "Prévia")],
    *[f"{verb} {article}{obj}." for verb in ("Tirou", "Fez", "Capturou")
      for article in ("", "um ", "uma ")
      for obj in ("print", "screenshot", "captura de tela")],
    *[f"{intro} {article}{obj} da sua tela." for intro in (
        "Aqui está", "Aqui esta", "Segue",
    ) for article in ("", "o ", "a ", "um ", "uma ", "seu ", "sua ")
      for obj in ("print", "screenshot", "captura", "captura de tela")],
    *[f"{intro} {article}screenshot." for intro in ("Here's", "Here is")
      for article in ("", "the ", "a ", "your ")],
])
@pytest.mark.parametrize("prefix", ["", "Pronto, ", "Feito! ", "OK: ", "Done. "])
def test_mac_third_person_and_screenshot_results_flagged(reply, prefix):
    assert detect_hallucinated_mac_action(prefix + reply), repr(prefix + reply)
    assert detect_hallucinated_action(prefix + reply), repr(prefix + reply)


@pytest.mark.parametrize("reply", [
    "Para diminuir o brilho, use as teclas F1/F2.",
    "Você pode abrir o Safari pelo Dock.",
    "O volume ideal…",
    "I lowered my expectations.",
    "Abri uma exceção…",
    "Fechei a conta",
    # Ordinary Mac how-to, explanations and opinions (PT and EN).
    "O brilho automático depende da luz ambiente.",
    "No Mac, o volume pode ser alterado pela Central de Controle.",
    "Prefiro o modo escuro para trabalhar à noite.",
    "O Safari é o navegador que vem com o macOS.",
    "Para bloquear a tela, pressione Control-Command-Q.",
    "Uma captura de tela pode incluir apenas uma janela.",
    "Use the brightness keys to dim the screen.",
    "You can open Safari from the Dock.",
    "Dark mode changes the appearance of supported apps.",
    "The volume slider controls the sound level.",
    "I prefer light mode during the day.",
    "A locked screen still allows background apps to run.",
    # Case-sensitive app names, object boundaries and first-line anchoring.
    "Abri o caminho para uma conversa.",
    "Fechei a janelação do exemplo.",
    "I opened the account.",
    "I closed a deal.",
    "I adjusted my expectations about brightness.",
    "I took a screenshotting course.",
    "Diminuí o brilhomania no texto.",
    "Uma resposta: Diminuí o brilho da tela do seu Mac.",
    "Aqui está a explicação.\nDiminuí o brilho da tela do seu Mac.",
    "Pronto\nDiminuí o brilho da tela do seu Mac.",
    "Diminuí minhas expectativas.\nO brilho é outro assunto.",
    "Pronto para explicar o brilho do Mac.",
    # Task 110b: retain objects, first-line anchoring and ordinary how-to prose.
    "Diminuiu? Tente pressionar F1.",
    "A Apple diminuiu o brilho máximo no modelo novo.",
    "Aqui está como tirar um print: pressione Shift-Command-3.",
    "Diminuiu minhas expectativas.",
    "Ativou a imaginação.",
    "Travou a conversa.",
    "Abriu uma exceção.",
    "Fechou a conta.",
    "Encerrou o assunto.",
    "Fez uma pergunta.",
    "Capturou a atenção.",
    "Diminuiu o brilhomania no texto.",
    "Abriu o caminho para uma conversa.",
    "Fechou a janelação do exemplo.",
    "Segue como tirar um print no Mac.",
    "Aqui esta como tirar um print no Mac.",
    "Here's how to take a screenshot.",
    "Here is your guide to screenshots.",
    "Segue o printável do exemplo.",
    "Here's a screenshotting tutorial.",
    "Aqui está a explicação.\nDiminuiu o brilho.",
    "Pronto\nAqui está um print da sua tela agora.",
    "Diminuiu minhas expectativas.\nO brilho é outro assunto.",
])
def test_mac_knowledge_and_non_mac_actions_not_flagged(reply):
    assert not detect_hallucinated_mac_action(reply), repr(reply)
    assert not detect_hallucinated_action(reply), repr(reply)


def test_mac_first_nonempty_line():
    assert detect_hallucinated_action("\n \n  Diminuí o brilho.\nMais detalhes.")


# =====================================================================
# Fallback message dictionary
# =====================================================================

class TestFallbackMessage:

    def test_pt_message_present(self):
        msg = HALLUCINATION_FALLBACK["pt"]
        assert msg
        assert "matue" in msg.lower() or "spotify" in msg.lower()

    def test_en_message_present(self):
        msg = HALLUCINATION_FALLBACK["en"]
        assert msg
        assert "tool" in msg.lower()
        assert "play" in msg.lower() or "spotify" in msg.lower()

    def test_keys_match(self):
        # Caller selects by language; both must be present.
        assert set(HALLUCINATION_FALLBACK.keys()) >= {"pt", "en"}


class TestFakeToolCall:
    """A tool call WRITTEN as text instead of made (s85). The positives are
    real replies: the 8B's answer to "cadê o pdf de punctuation? lê e
    responde a questão 5" (4 in 6), and the read_file_at "code" the Palace
    stored on 13/10."""

    MEASURED = (
        "The user is asking to read a PDF file and answer question 5.\n\n"
        "To answer this question, I will use the `find_file` function to "
        "search for the PDF file in the user's CloudDocs folder.\n\n"
        "Here is the code:\n```\nfind_file(query=\"punctuation.pdf\", "
        "content=False)\n```\nThis will search for the PDF file."
    )
    PALACE_13_10 = (
        "To respond to the current query, I will use the `read_file_at` "
        "function to read the two PDF files.\n\nHere is the code:\n```\n"
        "read_file_at(path=\"/Users/p/Downloads/college/13-10/file1.pdf\", "
        "question=\"What is the first question?\")\n```"
    )

    @pytest.mark.parametrize("reply", [
        MEASURED,
        PALACE_13_10,
        "send_email(to='joao@example.com', subject='Atraso')",
        "Vou chamar create_event (title='Dentista', start='amanhã 10h').",
    ])
    def test_flagged(self, reply):
        from src.core.hallucination import detect_fake_tool_call
        assert detect_fake_tool_call(reply)

    @pytest.mark.parametrize("reply", [
        "I'll use the find_file function to look for it.",      # no call
        "Encontrei 2 arquivos: question.pdf e text.pdf.",
        "Em Python, print(x) mostra o valor de x.",              # not a tool
        "The read_document tool only reads the vault.",
        "",
    ])
    def test_not_flagged(self, reply):
        from src.core.hallucination import detect_fake_tool_call
        assert not detect_fake_tool_call(reply)

    def test_only_cortex_tool_names_count(self):
        from src.core.hallucination import detect_fake_tool_call
        assert not detect_fake_tool_call("my_helper(x=1)")
        assert detect_fake_tool_call("my_helper(x=1)", tool_names=["my_helper"])

    def test_a_definition_is_not_a_call(self):
        """"def send_email(" in a Python example defines a function (s88,
        Codex lote 3)."""
        from src.core.hallucination import detect_fake_tool_call
        assert not detect_fake_tool_call("def send_email(to, subject, body):\n    ...")


class TestACodeRequestIsNotAFakeCall:
    """"como faço uma função em python pra enviar email?" routes to the tool
    path (the word "email"); the answer's send_email(...) example was swapped
    for "couldn't fire the right tool" (s88, Codex lote 3)."""

    @pytest.mark.parametrize("ask", [
        "como faço uma função em python pra enviar email?",
        "me mostra um exemplo de código que envia email, sem executar",
        "show me the code to create a calendar event",
        "escreve um script que busca arquivos",
    ])
    def test_code_requests(self, ask):
        from src.core.hallucination import asks_for_code
        assert asks_for_code(ask)

    @pytest.mark.parametrize("ask", [
        "cadê o pdf de punctuation? lê e responde a questão 5",   # the s85 case
        "manda um email pro João dizendo que atraso",
        "acha o meu contrato de aluguel",
    ])
    def test_tool_requests(self, ask):
        from src.core.hallucination import asks_for_code
        assert not asks_for_code(ask)

    def test_the_engine_lets_the_code_through(self, monkeypatch):
        from unittest.mock import MagicMock
        from src.core.engine import OllamaEngine
        code = ("```python\ndef send_email(to, subject, body):\n    ...\n\n"
                "send_email(\"ana@x.com\", \"Oi\", \"Teste\")\n```")
        e = OllamaEngine.__new__(OllamaEngine)
        try:
            OllamaEngine.__init__(e)
        except Exception:
            pass
        m = MagicMock()
        m.retrieve_relevant_context.return_value = ""
        m.get_user_profile.return_value = ""
        e.memory, e.history = m, []
        monkeypatch.setattr("src.core.engine.ollama.chat",
                            lambda **kw: {"message": {"content": code}})
        reply, _ = e.chat_with_tools(
            "como faço uma função em python pra enviar email?", tools=[],
            tool_executor=lambda n, a: "", response_language="pt")
        assert "def send_email" in reply


class TestATranslationIsNotAClaimedAction:
    """"Traduza para português: I have sent the email." routes to the tool
    path (the word "email"); the right answer "Enviei o email." was swapped
    for "couldn't fire the right tool" (s88, Codex lote 8)."""

    @pytest.mark.parametrize("ask", [
        "Traduza para português: I have sent the email. Responda somente com a tradução, sem aspas.",
        "traduz pra mim: I have sent the email",
        'traduz "I have sent the email" pro português',
        "translate to portuguese: the email was sent",
        "corrige esse texto: Enviei o email pro João ontem",
    ])
    def test_rewording_requests(self, ask):
        from src.core.hallucination import asks_for_rewording
        assert asks_for_rewording(ask)

    @pytest.mark.parametrize("ask", [
        "traduz e manda email pro João: I love you",   # an action, then the material
        "traduz o último email que recebi",              # the email is the subject
        "manda um email pro João dizendo que atraso",
        "dá pra tocar matue?",
    ])
    def test_action_requests(self, ask):
        from src.core.hallucination import asks_for_rewording
        assert not asks_for_rewording(ask)

    def _engine(self, monkeypatch, reply):
        from unittest.mock import MagicMock
        from src.core.engine import OllamaEngine
        e = OllamaEngine.__new__(OllamaEngine)
        try:
            OllamaEngine.__init__(e)
        except Exception:
            pass
        m = MagicMock()
        m.retrieve_relevant_context.return_value = ""
        m.get_user_profile.return_value = ""
        e.memory, e.history = m, []
        monkeypatch.setattr("src.core.engine.ollama.chat",
                            lambda **kw: {"message": {"content": reply}})
        return e

    def test_the_engine_keeps_the_translation(self, monkeypatch):
        e = self._engine(monkeypatch, "Enviei o email.")
        reply, _ = e.chat_with_tools(
            "Traduza para português: I have sent the email. Responda somente "
            "com a tradução, sem aspas.", tools=[],
            tool_executor=lambda n, a: "", response_language="pt")
        assert reply == "Enviei o email."

    def test_the_engine_still_blocks_a_claim_on_an_action_request(self, monkeypatch):
        from src.core.hallucination import HALLUCINATION_FALLBACK
        e = self._engine(monkeypatch, "Enviei o email.")
        reply, _ = e.chat_with_tools(
            "traduz e manda email pro João: I love you", tools=[],
            tool_executor=lambda n, a: "", response_language="pt")
        assert reply == HALLUCINATION_FALLBACK["pt"]
