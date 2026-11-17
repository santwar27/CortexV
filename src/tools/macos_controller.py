"""macOS system control tool — volume / brightness / dark mode / lock /
screenshot / app launch via osascript + shell utilities.

All operations are local — no network, no LLM. Screen brightness uses
DisplayServices via ctypes; keyboard light uses CoreBrightness via PyObjC.
Other controls shell out via subprocess
(timeout=5s). Each returns a short user-facing string. Errors degrade
gracefully: a missing framework, TCC permission or an
unknown app surfaces a friendly hint, never an unhandled exception
that would crash the orchestrator pipeline.

Risk tier: most actions are S2 (audit, no confirmation) — reversible
(volume, dark mode, screen lock) or purely additive (screenshot, app
open/switch). `quit_app` is S3: it can lose unsaved work, so the
orchestrator stages it behind a human sim/não gate (graceful quit,
never killall). Full sleep stays deferred; see `docs/JOURNAL.md`.

Permissions: the first run of an osascript that targets System
Events triggers a macOS Accessibility prompt. The hosting process
(python / uvicorn) must be approved in System Settings → Privacy &
Security → Accessibility. The error path in `_run_osascript` detects
the canonical -1743 error and surfaces a directed hint.

Localisation (s60): every public method takes `lang="pt"` and replies in
the spoken language (the dispatch wrappers pass `o._reply_lang`). PT stays
the default so any caller that doesn't pass `lang` is unchanged.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from src.core.i18n import t
from src.integrations.applescript import escape as _escape_applescript


_DEFAULT_TIMEOUT_S = 5
_SCREENSHOT_DIR = Path("data/screenshots")

# launchd runs uvicorn with a minimal PATH (no /usr/sbin), so use the
# fixed locations for these Apple-shipped tools (M4-A6).
_PMSET = "/usr/bin/pmset"
_SCREENCAPTURE = "/usr/sbin/screencapture"
_IOREG = "/usr/sbin/ioreg"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

class _DisplayServicesBrightness:
    """Lazy, fail-soft access to main-display brightness, without TCC or a CLI.

    The private framework can be absent or unsupported on a display. Native
    return codes and invalid readings are failures, never assumed successes.
    Optional framework objects keep unit tests entirely off the real screen.
    """

    def __init__(self, services=None, core_graphics=None):
        self._services = services
        self._core_graphics = core_graphics
        self._ready = False

    def _load(self) -> None:
        if self._ready:
            return
        if self._services is None:
            self._services = ctypes.CDLL(
                "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices"
            )
        if self._core_graphics is None:
            path = ctypes.util.find_library("CoreGraphics")
            if not path:
                raise OSError("CoreGraphics unavailable")
            self._core_graphics = ctypes.CDLL(path)
        self._core_graphics.CGMainDisplayID.argtypes = []
        self._core_graphics.CGMainDisplayID.restype = ctypes.c_uint32
        self._services.DisplayServicesGetBrightness.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
        ]
        self._services.DisplayServicesGetBrightness.restype = ctypes.c_int
        self._services.DisplayServicesSetBrightness.argtypes = [ctypes.c_uint32, ctypes.c_float]
        self._services.DisplayServicesSetBrightness.restype = ctypes.c_int
        self._ready = True

    def read(self) -> Optional[int]:
        """Read a rounded percentage, or None when no reliable value exists."""
        try:
            self._load()
            value = ctypes.c_float(float("nan"))
            rc = self._services.DisplayServicesGetBrightness(
                self._core_graphics.CGMainDisplayID(), ctypes.byref(value),
            )
            if rc != 0 or not math.isfinite(value.value) or not 0 <= value.value <= 1:
                return None
            return round(value.value * 100)
        except Exception:
            return None

    def set(self, level: int) -> bool:
        """Attempt an absolute write; the caller must still read it back."""
        try:
            self._load()
            return self._services.DisplayServicesSetBrightness(
                self._core_graphics.CGMainDisplayID(), ctypes.c_float(_clamp(level) / 100),
            ) == 0
        except Exception:
            return False


class _KeyboardBacklightBrightness:
    """Lazy CoreBrightness access, with an injectable client for hardware-free tests.

    This private framework is optional. Never change automatic brightness;
    the controller reports what macOS actually leaves behind after a write.
    """

    def __init__(self, client=None):
        self._client = client
        self._native_client = client is None
        self._suppression_metadata_ready = False
        self._dimmed_metadata_ready = False
        self._keyboard_id = None
        self._ready = False

    def _load(self) -> None:
        if self._ready:
            return
        if self._client is None:
            import objc
            from Foundation import NSBundle

            bundle = NSBundle.bundleWithPath_(
                "/System/Library/PrivateFrameworks/CoreBrightness.framework"
            )
            if bundle is None or not bundle.load():
                raise OSError("CoreBrightness unavailable")
            # Private selectors need explicit ABI metadata: float is not double,
            # and the keyboard ID must retain its unsigned 64-bit width.
            objc.registerMetaDataForSelector(
                b"KeyboardBrightnessClient", b"brightnessForKeyboard:",
                {"retval": {"type": b"f"}, "arguments": {2: {"type": b"Q"}}},
            )
            objc.registerMetaDataForSelector(
                b"KeyboardBrightnessClient", b"setBrightness:forKeyboard:",
                {"retval": {"type": objc._C_NSBOOL},
                 "arguments": {2: {"type": b"f"}, 3: {"type": b"Q"}}},
            )
            self._client = objc.lookUpClass("KeyboardBrightnessClient").alloc().init()
        ids = self._client.copyKeyboardBacklightIDs()
        if not ids:
            raise OSError("No keyboard backlight")
        self._keyboard_id = int(ids[0])
        self._ready = True

    def read(self) -> Optional[int]:
        """Read a rounded percentage, or None when unavailable/invalid."""
        try:
            self._load()
            value = float(self._client.brightnessForKeyboard_(self._keyboard_id))
            if not math.isfinite(value) or not 0 <= value <= 1:
                return None
            return round(value * 100)
        except Exception:
            return None

    def set(self, level: int) -> bool:
        """Attempt an absolute write; the caller must still read it back."""
        try:
            self._load()
            return bool(self._client.setBrightness_forKeyboard_(
                _clamp(level) / 100, self._keyboard_id,
            ))
        except Exception:
            return False

    def suppressed(self) -> Optional[bool]:
        """Read macOS suppression without changing any automatic settings."""
        try:
            self._load()
            # A missing suppression selector/ABI must not disable 118's read/write path.
            if self._native_client and not self._suppression_metadata_ready:
                import objc

                objc.registerMetaDataForSelector(
                    b"KeyboardBrightnessClient", b"isBacklightSuppressedOnKeyboard:",
                    {"retval": {"type": objc._C_NSBOOL},
                     "arguments": {2: {"type": b"Q"}}},
                )
                self._suppression_metadata_ready = True
            value = self._client.isBacklightSuppressedOnKeyboard_(self._keyboard_id)
            if isinstance(value, (bool, int)) and value in (0, 1):
                return bool(value)
            return None
        except Exception:
            return None

    def dimmed(self) -> Optional[bool]:
        """Read idle dimming without changing macOS automatic settings."""
        try:
            self._load()
            # Keep this optional selector's ABI independent of reads and writes.
            if self._native_client and not self._dimmed_metadata_ready:
                import objc

                objc.registerMetaDataForSelector(
                    b"KeyboardBrightnessClient", b"isBacklightDimmedOnKeyboard:",
                    {"retval": {"type": objc._C_NSBOOL},
                     "arguments": {2: {"type": b"Q"}}},
                )
                self._dimmed_metadata_ready = True
            value = self._client.isBacklightDimmedOnKeyboard_(self._keyboard_id)
            if isinstance(value, (bool, int)) and value in (0, 1):
                return bool(value)
            return None
        except Exception:
            return None


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    """Clamp `value` into [lo, hi]. Used for volume + brightness so a
    typo'd 'volume sistema 500' becomes 100, not an API error."""
    return max(lo, min(int(value), hi))


def already_closed_text(name: str, lang: str = "pt") -> str:
    """Shared with the ledger so an already-closed app is never reopened."""
    return t(lang, f"{name} já estava fechado.", f"{name} was already closed.")


def _run(args, timeout: int = _DEFAULT_TIMEOUT_S, lang: str = "pt") -> Tuple[Optional[bool], str]:
    """Run a subprocess, return (ok, output_or_error). Never raises.

    A timeout has an unknown outcome (None); other errors return False,
    with a short explanation so callers can compose a reply."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, t(lang, f"comando não encontrado: {args[0]}",
                        f"command not found: {args[0]}")
    except subprocess.TimeoutExpired:
        return None, t(lang, f"timeout após {timeout}s", f"timed out after {timeout}s")
    except Exception as exc:  # last-resort safety net
        return False, t(lang, f"erro inesperado: {exc}", f"unexpected error: {exc}")

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        err_lower = err.lower()
        # macOS Accessibility (System Events / osascript) — error
        # -1743 or the literal "not allowed assistive access".
        if "-1743" in err or "not allowed assistive access" in err_lower:
            return False, t(
                lang,
                "permissão negada. Abre System Settings → Privacy & "
                "Security → Accessibility e adiciona o processo que "
                "roda o Cortex (python/uvicorn).",
                "permission denied. Open System Settings → Privacy & "
                "Security → Accessibility and add the process that runs "
                "Cortex (python/uvicorn).",
            )
        # Screen recording (screencapture) — surfaces as "could not
        # create image from display" when the calling process lacks
        # Screen & System Audio Recording permission. Different TCC
        # bucket from Accessibility, different panel in Settings.
        if "could not create image from display" in err_lower:
            return False, t(
                lang,
                "permissão de gravação de tela negada. Abre System "
                "Settings → Privacy & Security → Screen & System "
                "Audio Recording e adiciona o processo do Cortex "
                "(python/uvicorn), depois restart do uvicorn "
                "(launchctl unload && load).",
                "screen recording permission denied. Open System "
                "Settings → Privacy & Security → Screen & System "
                "Audio Recording and add the Cortex process "
                "(python/uvicorn), then restart uvicorn "
                "(launchctl unload && load).",
            )
        return False, err or t(lang, f"exit {result.returncode}", f"exit {result.returncode}")
    return True, (result.stdout or "").strip()


def _osascript(script: str, timeout: int = _DEFAULT_TIMEOUT_S, lang: str = "pt") -> Tuple[Optional[bool], str]:
    """Convenience wrapper for `osascript -e <script>`. Handles the
    same error surface as _run with TCC permission detection."""
    return _run(["osascript", "-e", script], timeout=timeout, lang=lang)


# ----------------------------------------------------------------------
# Public class
# ----------------------------------------------------------------------

class MacOSController:
    """Wrapper around system-level macOS controls.

    Each public method:
      - takes Python-typed args (no shell strings from the caller),
      - returns a short user-facing reply string in `lang` (PT default),
      - never raises (errors come back as the reply text starting
        with a lower-case hint, by convention).
    """

    def __init__(self):
        self._brightness = _DisplayServicesBrightness()
        self._keyboard_brightness = _KeyboardBacklightBrightness()

    # The screenshot directory is created lazily on first use so that
    # importing this module on a fresh checkout (CI, tests) doesn't
    # create stray empty dirs.
    def _ensure_screenshot_dir(self) -> Path:
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        return _SCREENSHOT_DIR

    # ---------- system volume ----------

    def set_system_volume(self, level: int, lang: str = "pt") -> str:
        """Set a clamped percentage and report only the verified read-back."""
        return self._change_volume(_clamp(level), self._get_system_volume(), lang)

    def volume_up(self, step: int = 10, lang: str = "pt") -> str:
        return self._step_volume(_clamp(step, lo=1, hi=50), lang)

    def volume_down(self, step: int = 10, lang: str = "pt") -> str:
        return self._step_volume(-_clamp(step, lo=1, hi=50), lang)

    def _step_volume(self, step: int, lang: str) -> str:
        current = self._get_system_volume()
        if current is None:
            return self._volume_failure(lang)
        if current == 0 and step < 0:
            return t(lang, "O volume já está no mínimo (0%).",
                     "Volume is already at the minimum (0%).")
        if current == 100 and step > 0:
            return t(lang, "O volume já está no máximo (100%).",
                     "Volume is already at the maximum (100%).")
        return self._change_volume(_clamp(current + step), current, lang)

    def _change_volume(self, target: int, before: Optional[int], lang: str) -> str:
        if before is None:
            return self._volume_failure(lang)
        ok, _ = _osascript(f"set volume output volume {target}", lang=lang)
        actual = self._get_system_volume()  # A failed write can still change the output.
        if ok and actual is not None and abs(actual - target) <= 2:
            return t(lang, f"Volume do Mac em {actual}%.", f"Mac volume at {actual}%.")
        if actual is not None and actual != before:
            return t(lang, f"Não consegui deixar em {target}% — o volume do Mac está em {actual}%.",
                     f"I couldn't set it to {target}% — Mac volume is at {actual}%.")
        return self._volume_failure(lang)

    @staticmethod
    def _volume_failure(lang: str) -> str:
        return t(lang, "Não consegui mudar o volume do Mac — nada mudou.",
                 "I couldn't change the Mac volume — nothing changed.")

    def mute(self, lang: str = "pt") -> str:
        ok, err = _osascript("set volume with output muted", lang=lang)
        if not ok:
            return t(lang, f"não consegui silenciar: {err}",
                     f"couldn't mute: {err}")
        return t(lang, "Sistema silenciado.", "System muted.")

    def unmute(self, lang: str = "pt") -> str:
        ok, err = _osascript("set volume without output muted", lang=lang)
        if not ok:
            return t(lang, f"não consegui tirar o mudo: {err}",
                     f"couldn't unmute: {err}")
        return t(lang, "Sistema desmutado.", "System unmuted.")

    def _get_system_volume(self) -> Optional[int]:
        """Internal helper: parse `output volume of (get volume
        settings)` into an int. Returns None on parse failure rather
        than raising, so up/down can surface a graceful error."""
        ok, out = _osascript("output volume of (get volume settings)")
        if not ok:
            return None
        try:
            level = int(out.strip())
            return level if 0 <= level <= 100 else None
        except (ValueError, AttributeError):
            return None

    # ---------- brightness ----------

    def brightness_up(self, lang: str = "pt") -> str:
        return self._step_brightness(10, lang)

    def brightness_down(self, lang: str = "pt") -> str:
        return self._step_brightness(-10, lang)

    def set_brightness(self, level: int, lang: str = "pt") -> str:
        """Set a clamped percentage and report only the verified read-back."""
        return self._change_brightness(_clamp(level), self._brightness.read(), lang)

    def _step_brightness(self, step: int, lang: str) -> str:
        current = self._brightness.read()
        if current is None:
            return self._brightness_failure(lang)
        if current == 0 and step < 0:
            return t(lang, "O brilho já está no mínimo (0%).",
                     "Brightness is already at the minimum (0%).")
        if current == 100 and step > 0:
            return t(lang, "O brilho já está no máximo (100%).",
                     "Brightness is already at the maximum (100%).")
        return self._change_brightness(_clamp(current + step), current, lang)

    def _change_brightness(self, target: int, before: Optional[int], lang: str) -> str:
        if before is None:
            return self._brightness_failure(lang)
        ok = self._brightness.set(target)
        actual = self._brightness.read()  # Even a failed write may have changed the screen.
        if ok and actual is not None and abs(actual - target) <= 2:
            return t(lang, f"Brilho em {actual}%.", f"Brightness at {actual}%.")
        if actual is not None and actual != before:
            return t(lang, f"Não consegui deixar em {target}% — o brilho está em {actual}%.",
                     f"I couldn't set it to {target}% — brightness is at {actual}%.")
        return self._brightness_failure(lang)

    @staticmethod
    def _brightness_failure(lang: str) -> str:
        return t(lang, "Não consegui mudar o brilho desta tela — nada mudou.",
                 "I couldn't change this screen's brightness — nothing changed.")

    # ---------- keyboard backlight ----------

    def keyboard_brightness_up(self, lang: str = "pt") -> str:
        return self._step_keyboard_brightness(10, lang)

    def keyboard_brightness_down(self, lang: str = "pt") -> str:
        return self._step_keyboard_brightness(-10, lang)

    def keyboard_brightness_set(self, level: int, lang: str = "pt") -> str:
        target = _clamp(level)
        lid_reply = self._keyboard_lid_reply(target, lang)
        if lid_reply is not None:
            return lid_reply
        return self._change_keyboard_brightness(
            target, self._keyboard_brightness.read(), lang,
        )

    def _step_keyboard_brightness(self, step: int, lang: str) -> str:
        lid_reply = self._keyboard_lid_reply(None, lang)
        if lid_reply is not None:
            return lid_reply
        current = self._keyboard_brightness.read()
        if current is None:
            return self._keyboard_brightness_failure(lang)
        return self._change_keyboard_brightness(_clamp(current + step), current, lang)

    @staticmethod
    def _lid_closed() -> Optional[bool]:
        """Read the clamshell state; absent/failed readings remain unknown."""
        try:
            ok, output = _run([_IOREG, "-r", "-k", "AppleClamshellState", "-d", "1"])
            if ok is not True:
                return None
            match = re.search(r'"AppleClamshellState"\s*=\s*(Yes|No)\s*$', output, re.MULTILINE)
            return match.group(1) == "Yes" if match else None
        except Exception:
            return None

    def _keyboard_lid_reply(self, target: Optional[int], lang: str) -> Optional[str]:
        if self._lid_closed() is not True:
            return None
        if target == 0:
            return t(lang, "A luz do teclado já está apagada.", "The keyboard light is already off.")
        return t(lang,
                 "A tampa do Mac está fechada — com ela fechada o macOS deixa a luz do teclado apagada. Nada mudou.",
                 "The Mac's lid is closed — with it closed macOS keeps the keyboard light off. Nothing changed.")

    def _change_keyboard_brightness(self, target: int, before: Optional[int], lang: str) -> str:
        if before is None:
            return self._keyboard_brightness_failure(lang)
        if before == target:
            if target == 100:
                return t(lang, "A luz do teclado já está no máximo (100%).",
                         "Keyboard light is already at the maximum (100%).")
            return t(lang, f"A luz do teclado já está em {before}%.",
                     f"Keyboard light is already at {before}%.")
        ok = self._keyboard_brightness.set(target)
        actual = self._keyboard_brightness.read()  # Also verify failed writes.
        if ok and actual == target:
            return t(lang, f"Luz do teclado em {actual}%.", f"Keyboard light at {actual}%.")
        # These flags can be true even when a write sticks. After a miss, idle
        # dimming explains the cause; suppression alone does not establish one.
        if actual != target:
            if self._keyboard_brightness.dimmed() is True:
                return t(lang,
                         "O Mac está sem uso há alguns minutos, então o macOS apagou a luz do teclado. "
                         "Ela volta quando você mexer no teclado ou no trackpad.",
                         "The Mac has been idle for a few minutes, so macOS turned the keyboard light off. "
                         "It comes back when you touch the keyboard or trackpad.")
            if self._keyboard_brightness.suppressed() is True:
                return t(lang,
                         "O macOS está deixando a luz do teclado apagada agora. Nada mudou.",
                         "macOS is keeping the keyboard light off right now. Nothing changed.")
        # macOS can restore even the original value after an accepted write.
        # Report that observed value rather than hiding it.
        if actual is not None and (ok or actual != before):
            return t(lang, f"Não consegui deixar em {target}% — a luz do teclado está em {actual}%.",
                     f"I couldn't set it to {target}% — keyboard light is at {actual}%.")
        return self._keyboard_brightness_failure(lang)

    @staticmethod
    def _keyboard_brightness_failure(lang: str) -> str:
        return t(lang, "Não consegui mudar a luz do teclado — nada mudou.",
                 "I couldn't change the keyboard light — nothing changed.")

    # ---------- dark mode ----------

    def _set_dark_mode(self, enabled: bool, lang: str = "pt") -> str:
        flag = "true" if enabled else "false"
        ok, err = _osascript(
            'tell application "System Events" to tell appearance '
            f'preferences to set dark mode to {flag}',
            lang=lang,
        )
        if not ok:
            return t(lang, f"não consegui mudar o modo: {err}",
                     f"couldn't change the mode: {err}")
        if enabled:
            return t(lang, "Modo escuro ativado.", "Dark mode on.")
        return t(lang, "Modo claro ativado.", "Light mode on.")

    def enable_dark_mode(self, lang: str = "pt") -> str:
        return self._set_dark_mode(True, lang=lang)

    def disable_dark_mode(self, lang: str = "pt") -> str:
        return self._set_dark_mode(False, lang=lang)

    def toggle_dark_mode(self, lang: str = "pt") -> str:
        """Read current state, then flip. Avoids depending on the
        client knowing which mode is active."""
        ok, out = _osascript(
            'tell application "System Events" to tell appearance '
            'preferences to get dark mode',
            lang=lang,
        )
        if not ok:
            return t(lang, f"não consegui ler o estado do tema: {out}",
                     f"couldn't read the theme state: {out}")
        currently_dark = out.strip().lower() == "true"
        return self._set_dark_mode(not currently_dark, lang=lang)

    # ---------- lock screen ----------

    def lock_screen(self, lang: str = "pt") -> str:
        """Lock the screen.

        The old CGSession `-suspend` binary
        (/System/Library/CoreServices/Menu Extras/User.menu/.../CGSession)
        was removed in recent macOS — it 404'd live on 2026-06-03. We use
        the documented Lock Screen shortcut Ctrl+Cmd+Q via System Events,
        and fall back to sleeping the display (which locks when the user has
        'require password after sleep' on — Pedro does). The keystroke path
        needs Accessibility permission; the pmset fallback needs none, so
        between them one always lands."""
        ok, err = _run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "q" '
            "using {control down, command down}",
        ], lang=lang)
        if ok:
            return t(lang, "Tela travada.", "Screen locked.")
        # Fallback: sleep the display (no special permission needed).
        # Absolute path — under launchd this fallback fired with a bare
        # "pmset" and silently no-op'd, so a denied-Accessibility lock
        # looked successful but never locked (M4-A6).
        ok2, err2 = _run([_PMSET, "displaysleepnow"], lang=lang)
        if ok2:
            return t(lang, "Tela travada (display em sleep).",
                     "Screen locked (display asleep).")
        return t(lang, f"não consegui travar a tela: {err or err2}",
                 f"couldn't lock the screen: {err or err2}")

    # ---------- sleep ----------

    def sleep_mac(self, lang: str = "pt") -> str:
        return self.sleep_mac_outcome(lang=lang)[1]

    def sleep_mac_outcome(self, lang: str = "pt") -> Tuple[Optional[bool], str]:
        """Put the mac to sleep via `pmset sleepnow`. Apple-shipped,
        no TCC prompt, works under launchd. Destructive (active session
        is paused; unsaved focus may be interrupted), so the
        orchestrator stages this S3 — only fires after the user
        explicitly says 'sim' to the confirmation prompt."""
        # Absolute path because the launchd PATH for uvicorn omits
        # /usr/bin in some configurations (same reason screencapture
        # / pmset don't resolve cleanly without it).
        ok, err = _run([_PMSET, "sleepnow"], lang=lang)
        if ok is None:
            return None, t(lang, f"não sei se o Mac foi dormir ({err}).",
                           f"I don't know whether the Mac went to sleep ({err}).")
        if not ok:
            return False, t(lang, f"não consegui colocar o Mac pra dormir: {err}",
                            f"couldn't put the Mac to sleep: {err}")
        # The reply is rendered BEFORE the system actually goes down —
        # depending on how fast pmset enters sleep, the user may not
        # see it (TTS gets cut). That's expected.
        return True, t(lang, "Mac indo dormir.", "Mac going to sleep.")

    # ---------- screenshot ----------

    def _screen_is_locked(self) -> bool:
        """Best-effort: is the login screen / lock currently up?

        `screencapture` happily writes a SOLID BLACK png when the screen is
        locked and still reports success — so a remote "tira um print"
        (iPhone, Mac locked) used to return "Print salvo em…" pointing at a
        useless file (verified live, s75). We read `IOConsoleLocked` from
        the IORegistry (no pyobjc/Quartz dependency — those aren't
        installed, and shelling out matches this module's stateless design;
        ~24ms). The key is always present, so absence of "Yes" is a real
        "unlocked", not a parse miss.

        FAIL-OPEN: any error or unexpected output returns False (not
        locked) so the screenshot still runs. Denying a legitimate capture
        on a detection glitch is worse than the occasional black png this
        guard exists to prevent — and False reproduces today's behaviour."""
        ok, out = _run(["ioreg", "-n", "Root", "-d1"])
        if not ok:
            return False
        return bool(re.search(r'"IOConsoleLocked"\s*=\s*Yes', out))

    def screenshot(self, filename: Optional[str] = None, lang: str = "pt") -> str:
        """Capture the full screen to data/screenshots/. Uses -x to
        silence the shutter so it can be triggered from chat without
        startling Pedro."""
        # A locked screen yields a black png + a false "saved" reply, so
        # tell the truth instead of capturing nothing (s75 polish).
        if self._screen_is_locked():
            return t(lang,
                     "Não dá pra tirar print com a tela trancada — sairia "
                     "preto. Destranca o Mac e tenta de novo.",
                     "Can't take a screenshot while the screen is locked — "
                     "it'd come out black. Unlock the Mac and try again.")
        out_dir = self._ensure_screenshot_dir()
        # Defence: strip directory traversal; we only write to
        # data/screenshots/ regardless of what the caller asks. A filename
        # that reduces to nothing ("/", "..", "") falls back to the
        # timestamp name instead of producing a hidden ".png" (M4-A14f).
        safe = os.path.basename(filename or "").strip().strip(".")
        if safe:
            if not safe.lower().endswith(".png"):
                safe += ".png"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = f"cortex_screenshot_{ts}.png"
        target = out_dir / safe
        # Absolute path because the launchd PATH for uvicorn omits
        # /usr/sbin (where screencapture lives on every macOS
        # release since 10.0). Plain "screencapture" works in an
        # interactive shell but fails under the service.
        ok, err = _run([_SCREENCAPTURE, "-x", str(target)], lang=lang)
        if not ok:
            return t(lang, f"não consegui tirar o print: {err}",
                     f"couldn't take the screenshot: {err}")
        # screencapture writes silently; we trust the file is there.
        return t(lang, f"Print salvo em {target}.", f"Screenshot saved to {target}.")

    # ---------- app open ----------

    def open_app(self, app_name: str, lang: str = "pt") -> str:
        return self.open_app_outcome(app_name, lang=lang)[1]

    def open_app_outcome(self, app_name: str, lang: str = "pt") -> Tuple[Optional[bool], str]:
        """Launch the named app via `open -a`. The macOS launcher
        resolves common name variations (e.g. 'notion' → Notion.app)
        case-insensitively. Returns the system error string when the
        app isn't found, so Pedro sees exactly what's wrong."""
        name = (app_name or "").strip()
        if not name:
            return False, t(lang, "preciso do nome do app pra abrir.",
                            "I need the app name to open it.")
        ok, err = _run(["open", "-a", name], lang=lang)
        if not ok:
            # `open -a` writes its error to stderr; pass it through.
            return ok, t(lang,
                         f"não consegui abrir '{name}': {err or 'app não encontrado'}",
                         f"couldn't open '{name}': {err or 'app not found'}")
        return True, t(lang, f"Abrindo {name}.", f"Opening {name}.")

    def open_file(self, path: str, lang: str = "pt") -> str:
        """Open a file in its default app (`open <path>`, s87). The caller
        passes only a path the file search returned — inside the home, with
        the same exclusions — and an absolute path can't be read as a flag."""
        target = (path or "").strip()
        name = os.path.basename(target)
        if not target.startswith("/") or not os.path.isfile(target):
            return t(lang, f"não achei o arquivo {name or target}.",
                     f"couldn't find the file {name or target}.")
        ok, err = _run(["open", target], lang=lang)
        if not ok:
            return t(lang, f"não consegui abrir {name}: {err}",
                     f"couldn't open {name}: {err}")
        return t(lang, f"Abrindo {name}.", f"Opening {name}.")

    # ---------- app quit ----------

    def quit_app(self, app_name: str, lang: str = "pt") -> str:
        return self.quit_app_outcome(app_name, lang=lang)[1]

    def quit_app_outcome(self, app_name: str, lang: str = "pt") -> Tuple[Optional[bool], str]:
        """Gracefully quit the named app via `osascript 'tell app X to
        quit'`. Graceful (not killall) so the app's own unsaved-changes
        dialog still fires — we never silently discard work. If the app
        isn't running, AppleScript raises; we treat that as a friendly
        'já estava fechado' rather than an error.

        Reaches here only after the orchestrator's S3 confirmation gate
        (mac_quit_app is staged in _pending_action and executed on a
        user 'sim'), so no extra guard is needed at this layer."""
        name = (app_name or "").strip()
        if not name:
            return False, t(lang, "preciso do nome do app pra fechar.",
                            "I need the app name to close it.")
        # Full literal escape (backslash first, then quotes) — a name ending
        # in `\` would otherwise terminate the string and run as code (M4-A3).
        safe = _escape_applescript(name)
        # `tell application X to quit` exits 0 for ANY name, real or not
        # (checked 2026-09-19: "tela" and "zzqq_nao_existe" both rc 0), so on
        # its own it answered "Fechando tela." for an app that doesn't exist.
        # `id of application` asks LaunchServices instead: -1728 for an
        # unknown name, case-insensitive, and it never launches the app.
        # Any other failure falls through to the quit attempt below.
        found, err = _osascript(f'id of application "{safe}"', lang=lang)
        if not found and "-1728" in err:
            return False, t(lang, f"não achei nenhum app chamado '{name}'.",
                            f"I couldn't find an app called '{name}'.")
        ok, err = _osascript(f'tell application "{safe}" to quit', lang=lang)
        if ok is None:
            return None, t(lang, f"não sei se {name} fechou ({err}).",
                           f"I don't know whether {name} closed ({err}).")
        if not ok:
            low = err.lower()
            # AppleScript -600 / "isn't running" when the target app has
            # no live process. Not a failure from the user's view.
            if "-600" in err or "isn't running" in low or "is not running" in low:
                return True, already_closed_text(name, lang)
            return False, t(lang, f"não consegui fechar '{name}': {err}",
                            f"couldn't close '{name}': {err}")
        return True, t(lang, f"Fechando {name}.", f"Closing {name}.")

    def close_window(self, app_name: str, lang: str = "pt") -> str:
        """Close the app's front window — the red button / Cmd+W — WITHOUT
        quitting it (the app keeps running). This is NOT destructive, so
        unlike quit_app it is not gated behind a sim/não confirmation. If the
        app has unsaved work its own save dialog fires, exactly like clicking
        the red button.

        Implemented as activate + Cmd+W via System Events, so it needs the
        same Accessibility grant as the lock/volume keystrokes."""
        name = (app_name or "").strip()
        if not name:
            return t(lang, "preciso do nome do app pra fechar a janela.",
                     "I need the app name to close its window.")
        safe = _escape_applescript(name)
        # M4-A14a: `activate` LAUNCHES an app that isn't running ("close the
        # Notion window" with Notion closed would START Notion and then send
        # Cmd+W to whatever ended up frontmost). Check first; "isn't running"
        # is an answer, not an action.
        ok, out = _osascript(f'application "{safe}" is running', lang=lang)
        if ok and out.strip().lower() != "true":
            return t(lang, f"{name} não está aberto — nenhuma janela pra fechar.",
                     f"{name} isn't running — no window to close.")
        # The keystroke goes to whatever app is frontmost. After a fixed
        # `delay 0.1` it went out unchecked: switching to Safari in that
        # tenth of a second closed a Safari tab while Cortex said it had
        # closed Spotify's window (s88, Codex lote 12). It is sent only once
        # the app's own process is frontmost, checked right before it.
        ok, out = _osascript(
            f'set bid to id of application "{safe}"\n'
            f'tell application "{safe}" to activate\n'
            'tell application "System Events"\n'
            '    set p to first application process whose bundle identifier is bid\n'
            '    repeat 20 times\n'
            '        if frontmost of p then exit repeat\n'
            '        delay 0.05\n'
            '    end repeat\n'
            '    if not (frontmost of p) then return "notfront"\n'
            '    set n to count of windows of p\n'
            '    if n is 0 then return "nowindow"\n'
            '    keystroke "w" using command down\n'
            '    delay 0.3\n'
            '    if (count of windows of p) < n then return "closed"\n'
            '    try\n'
            '        if exists sheet 1 of window 1 of p then return "sheet"\n'
            '    end try\n'
            '    return "open"\n'
            'end tell',
            lang=lang,
        )
        if not ok:
            return t(lang, f"não consegui fechar a janela do {name}: {out}",
                     f"couldn't close {name}'s window: {out}")
        if out.strip() == "notfront":
            return t(lang, f"Não fechei: o {name} não veio para a frente, e o "
                           "atalho fecharia a janela de outro app.",
                     f"I didn't close it: {name} didn't come to the front, and "
                     "the shortcut would have closed another app's window.")
        # "fechada" only when a window went away. A document with unsaved
        # changes answers Cmd+W with a save sheet and stays open; Cortex
        # used to say it had closed it (s88, Codex lote 12 — invariant 7).
        result = out.strip()
        if result == "closed":
            return t(lang, f"Janela do {name} fechada (o app continua aberto).",
                     f"{name}'s window closed (the app stays open).")
        if result == "nowindow":
            return t(lang, f"O {name} não tem janela aberta pra fechar.",
                     f"{name} has no open window to close.")
        if result == "sheet":
            return t(lang, f"O {name} abriu um aviso antes de fechar (provavelmente "
                           "pra salvar) — a janela continua aberta até você responder.",
                     f"{name} asked something before closing (probably to save) — "
                     "the window stays open until you answer.")
        return t(lang, f"Mandei fechar a janela do {name}, mas ela continua aberta.",
                 f"I asked {name} to close its window, but it is still open.")

    # ---------- app list / switch ----------

    def list_apps(self, lang: str = "pt") -> str:
        """List the foreground (non-background) apps currently running.

        `background only is false` filters out daemons/agents so we only
        report apps the user could actually switch to. Reads via System
        Events → needs the Accessibility TCC grant (same as volume/dark);
        the canonical -1743 hint is surfaced by _run on denial."""
        ok, out = _osascript(
            'tell application "System Events" to get name of '
            '(every process whose background only is false)',
            lang=lang,
        )
        if not ok:
            return t(lang, f"não consegui listar os apps: {out}",
                     f"couldn't list the apps: {out}")
        # osascript returns a comma-space separated list; normalise it.
        names = [n.strip() for n in (out or "").split(",") if n.strip()]
        if not names:
            return t(lang, "Nenhum app em primeiro plano agora.",
                     "No foreground apps right now.")
        names.sort(key=str.lower)
        return t(lang, "Apps abertos: " + ", ".join(names) + ".",
                 "Open apps: " + ", ".join(names) + ".")

    def switch_to_app(self, app_name: str, lang: str = "pt") -> str:
        """Bring an app to the front via `tell app X to activate`.

        Distinct from open_app (which is `open -a`, the 'launch it'
        verb): switch is the 'put it in front of me now' verb. activate
        launches the app too if it wasn't running, so a switch to a
        closed app still does the sensible thing."""
        name = (app_name or "").strip()
        if not name:
            return t(lang, "preciso do nome do app pra focar.",
                     "I need the app name to focus it.")
        safe = _escape_applescript(name)
        ok, err = _osascript(f'tell application "{safe}" to activate', lang=lang)
        if not ok:
            return t(lang,
                     f"não consegui focar '{name}': {err or 'app não encontrado'}",
                     f"couldn't focus '{name}': {err or 'app not found'}")
        return t(lang, f"Foco em {name}.", f"Focused on {name}.")
