"""A bracketed paste is one terminal turn; EOF closes the session."""
import io
from types import SimpleNamespace

import pytest

import src.main as cli
from src.ui import chat_store as cs


@pytest.mark.parametrize('line', [
    'adicione na sua lista de estudos: Hoje e\x1b[200~Principais acontecimentos de 2024 a 2026 no Reino Unido e no Brasil.',
    'Adicionei na sua lista de estudos:\x1b[200~Novidades do macOS dos últimos dois anos.\x1b[201~',
    'typed\x1b[200~one\x1b[201~ and \x1b[200~two\x1b[201~',
    'typed\x1b[201~',
])
def test_mid_line_paste_markers_are_removed(line):
    def read():
        nonlocal line
        if line is None:
            raise EOFError
        result, line = line, None
        return result
    expected = line.replace('\x1b[200~', '').replace('\x1b[201~', '')
    assert cli.read_message(read) == expected


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    store = cs.ChatStore(db_path=str(tmp_path / "chat.db"))
    monkeypatch.setattr(cs, "store", store)
    calls = []

    class FakeOrch:
        engine_analytical = engine_creative = SimpleNamespace(model_name="m")
        origin = None

        def set_turn_origin(self, origin):
            self.origin = origin

        def process(self, text, image_b64=None, conversation_history=None):
            calls.append((text, self.origin))
            return ("Resposta.", "OFFLINE")

    orch = FakeOrch()
    monkeypatch.setattr(cli, "CortexOrchestrator", lambda: orch)
    stdin, stdout = io.StringIO(), io.StringIO()
    state = SimpleNamespace(calls=calls, orch=orch, store=store, reads=0, prompts=[],
                            stdin=stdin, stdout=stdout)

    def run(lines):
        monkeypatch.setattr(cli.sys, "stdin", stdin)
        monkeypatch.setattr(cli.sys, "stdout", stdout)
        events = iter(lines)

        def read(_prompt=""):
            state.prompts.append(_prompt)
            state.reads += 1
            event = next(events, None)
            if event is None:
                pytest.fail("read past the expected session end")
            if isinstance(event, BaseException):
                raise event
            return event

        monkeypatch.setattr("builtins.input", read)
        with pytest.raises(SystemExit) as exc:
            cli.run()
        assert exc.value.code == 0

    state.run = run
    return state


def test_pedros_block_is_one_turn(terminal):
    terminal.run(["\x1b[200~Read /Users/x/Downloads/college/test/questions1.pdf",
                  "&&", "/Users/x/Downloads/college/test/text1.pdf", "",
                  "And answer question 5\x1b[201~", "sair"])
    assert terminal.prompts == ["User > ", "", "", "", "", "User > "]
    text = ("Read /Users/x/Downloads/college/test/questions1.pdf\n&&\n"
            "/Users/x/Downloads/college/test/text1.pdf\n\nAnd answer question 5")
    assert terminal.calls == [(text, "terminal")]
    assert terminal.orch.origin is None
    convos = terminal.store.list_conversations()
    assert len(convos) == 1
    turns = terminal.store.get_turns(convos[0]["id"])
    assert [(t["user"], t["source"]) for t in turns] == [(text, "terminal")]


def test_text_after_paste_belongs_to_same_message(terminal):
    terminal.run(["\x1b[200~foo", "bar\x1b[201~ baz", "sair"])
    assert terminal.calls == [("foo\nbar baz", "terminal")]


def test_plain_typing_and_blank_line(terminal):
    terminal.run([" one ", "", "two", "three", "sair"])
    assert terminal.prompts == ["User > "] * 5
    assert terminal.calls == [(text, "terminal") for text in ("one", "two", "three")]


@pytest.mark.parametrize("lines,expected,reads", [
    ([EOFError()], [], 1),
    (["\x1b[200~foo", "bar", EOFError(), EOFError()], [("foo\nbar", "terminal")], 4),
])
def test_eof_ends_session(terminal, lines, expected, reads):
    terminal.run(lines)
    assert terminal.calls == expected
    assert terminal.reads == reads
    assert "CORTEX: Shutting down" in terminal.stdout.getvalue()
    assert "CRITICAL ERROR" not in terminal.stdout.getvalue()


def fake_termios(terminal, monkeypatch, tty=True, failure=None):
    monkeypatch.setattr(cli.sys, "stdin", terminal.stdin)
    monkeypatch.setattr(cli.sys, "stdout", terminal.stdout)
    monkeypatch.setattr(terminal.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(terminal.stdout, "isatty", lambda: tty)
    monkeypatch.setattr(terminal.stdin, "fileno", lambda: 7)
    saved = [0, 0, 0, 0x208, 0, 0, [b"x"]]
    calls = []

    def get(fd):
        calls.append(("get", fd))
        if failure == "get":
            raise OSError("no terminal attributes")
        return saved

    def set_(fd, when, attrs):
        calls.append(("set", fd, when, attrs))
        if failure == "set":
            raise OSError("cannot set terminal attributes")

    monkeypatch.setattr(cli, "termios", SimpleNamespace(
        ECHOCTL=0x200, TCSANOW=0, error=OSError, tcgetattr=get, tcsetattr=set_),
        raising=False)
    return saved, calls


@pytest.mark.parametrize("ending", ["sair", KeyboardInterrupt(), EOFError(), "boot failure"])
def test_terminal_mode_is_restored(terminal, monkeypatch, ending):
    saved, calls = fake_termios(terminal, monkeypatch)
    if ending == "boot failure":
        def boot(_orch):
            raise RuntimeError("boot failed")
        monkeypatch.setattr(cli, "boot_sequence", boot)
        with pytest.raises(RuntimeError, match="boot failed"):
            cli.run()
    else:
        terminal.run([ending])
    assert calls == [("get", 7), ("set", 7, 0, [0, 0, 0, 8, 0, 0, [b"x"]]),
                     ("set", 7, 0, saved)]
    assert saved[3] == 0x208
    output = terminal.stdout.getvalue()
    assert output.startswith("\x1b[?2004h")
    assert output.endswith("\x1b[?2004l")


@pytest.mark.parametrize("tty_stream", [None, "stdin", "stdout"])
def test_non_tty_has_no_terminal_controls(terminal, monkeypatch, tty_stream):
    _, calls = fake_termios(terminal, monkeypatch, tty=False)
    if tty_stream:
        monkeypatch.setattr(getattr(terminal, tty_stream), "isatty", lambda: True)
    terminal.run(["hello", "sair"])
    assert terminal.calls == [("hello", "terminal")]
    assert calls == []
    assert "\x1b[?2004" not in terminal.stdout.getvalue()


@pytest.mark.parametrize("failure", ["get", "set"])
def test_termios_failure_is_soft(terminal, monkeypatch, failure):
    fake_termios(terminal, monkeypatch, failure=failure)
    terminal.run(["hello", "sair"])
    assert terminal.calls == [("hello", "terminal")]
