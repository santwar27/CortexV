import sys
import os
import base64
import re
import termios
from contextlib import contextmanager

from src.core.orchestrator import CortexOrchestrator
from src.core.input_text import strip_paste_markers
from src.core import turn_metrics
from src.ui.chat_store import TERMINAL_LABEL, append_daily
from src.ui.routes import _daily_history_limit, _daily_history_msgs


EXIT_COMMANDS = {"exit", "quit", "sair", "desligar"}
IMAGE_CMD_RE = re.compile(r"^image\s+(\S+)(?:\s+(.*))?$", re.IGNORECASE)


def read_message(read_line) -> str | None:
    try:
        line = read_line()
    except EOFError:
        return None
    if "\x1b[200~" not in line:
        return strip_paste_markers(line).strip()
    # Canonical input delivers a paste one line at a time, up to the final Enter.
    lines = [line]
    while "\x1b[201~" not in lines[-1]:
        try:
            lines.append(read_line())
        except EOFError:
            break
    return strip_paste_markers("\n".join(lines)).strip()


@contextmanager
def terminal_mode():
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    saved = None
    try:
        if tty:
            try:
                saved = termios.tcgetattr(sys.stdin.fileno())
                attrs = saved.copy()
                # Keep the paste markers from echoing as ^[[200~ / ^[[201~.
                attrs[3] &= ~termios.ECHOCTL
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attrs)
            except (OSError, termios.error, ValueError, AttributeError):
                pass
            sys.stdout.write("\x1b[?2004h")
            sys.stdout.flush()
        yield
    finally:
        if tty:
            try:
                if saved is not None:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, saved)
            except (OSError, termios.error, ValueError, AttributeError):
                pass
            finally:
                sys.stdout.write("\x1b[?2004l")
                sys.stdout.flush()


def boot_sequence(orchestrator: CortexOrchestrator) -> None:
    print("\n" + "=" * 50)
    print("  CORTEX SYSTEM INITIALISED")
    print("  Hardware: Apple Silicon (M1 Pro)")
    print(f"  Engine: {orchestrator.engine_analytical.model_name} (analytical)")
    print(f"          {orchestrator.engine_creative.model_name} (creative — prefix with 'criativo:')")
    print("  Modules: Gatekeeper, RAG (read/write), Web Search, Vision")
    print("  Tip: 'image <path> [prompt]' to analyse a local image")
    print("=" * 50 + "\n")


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@terminal_mode()
def run() -> None:
    orchestrator = CortexOrchestrator()
    boot_sequence(orchestrator)

    while True:
        try:
            # Show the prompt once per message, including multiline pastes.
            prompts = iter(["User > "])
            user_input = read_message(lambda: input(next(prompts, "")))
            if user_input is None or user_input.lower() in EXIT_COMMANDS:
                print("\nCORTEX: Shutting down cognitive processes. Goodbye.")
                sys.exit(0)
            if not user_input:
                continue

            image_b64 = None
            text = user_input
            image_match = IMAGE_CMD_RE.match(user_input)
            if image_match:
                image_path = image_match.group(1)
                if not os.path.exists(image_path):
                    print(f"\n[SYSTEM ERROR] Image not found: {image_path}\n")
                    continue
                image_b64 = load_image_b64(image_path)
                text = image_match.group(2) or ""

            set_origin = getattr(orchestrator, "set_turn_origin", None)
            turn_metrics.reset()
            try:
                if set_origin is not None:
                    set_origin("terminal")
                reply, status_mode = orchestrator.process(
                    text, image_b64=image_b64,
                    conversation_history=_daily_history_msgs(
                        TERMINAL_LABEL, _daily_history_limit(orchestrator)))
            finally:
                usage = turn_metrics.collect()
                if set_origin is not None:
                    set_origin(None)
            print(f"\n[{status_mode}]")
            print(f"CORTEX > {reply}\n")
            # Into the chat history too (s85): this conversation used to live
            # only in the terminal and vanish when it closed.
            append_daily("terminal", TERMINAL_LABEL, user_input, reply, status_mode, usage=usage)

        except KeyboardInterrupt:
            print("\n\nCORTEX: Manual interrupt detected. Shutting down.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] The orchestrator encountered a failure: {e}\n")


if __name__ == "__main__":
    run()
