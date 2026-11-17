"""
Tool Risk Levels + World Model Dry-Run — Tier 3 / 3.4 + 3.3.

Single source of truth for how risky each tool is and how its "what
would happen if I run this" preview looks. Replaces the inline
`_DESTRUCTIVE_TOOLS` frozenset in the orchestrator and centralises the
per-tool branches that used to live inside `_human_confirmation_prompt`
and `_format_confirmation_brief`.

Three risk levels, mapped to design-reference ch. 8 conventions:

  S1  read-only. Examples: read_document, list_emails, list_events.
      No approval, no extra logging. Safe to call freely from any LLM
      tool round-trip.

  S2  state-modifying but RECOVERABLE. Examples: write_document
      (append-only to the local vault), web_search (network read with
      lease gating). Executes immediately, but the audit log tags the
      event with the risk level so post-hoc review can find them.

  S3  IRREVERSIBLE or external-facing. Examples: send_email (can't be
      unsent), create_event (lands on shared Google Calendar),
      create_reminder (iCloud-synced to iPhone). Mandatory dry-run +
      human gate: the orchestrator stages the call in
      `_pending_action`, shows the user a deterministic preview, and
      only executes on explicit "sim".

Unknown tools default to S3 — fail-closed (M6-A5). A new tool added
without an entry here gets the human gate until someone classifies it;
the old default (S2 = executes immediately with just an audit tag)
meant an unclassified external/irreversible tool ran with no human in
the loop. The friction is deliberate and temporary: add the entry. A
CI test additionally pins that every tool in TOOL_SPECS is classified.

The dry-run preview builder (`build_preview`) generates BOTH surfaces
the orchestrator needs from a single per-tool spec:

  - "human": the pt-BR prompt the user sees and replies sim/não to.
  - "llm":   the English "CONFIRMATION REQUIRED" brief that goes back
             to the LLM as the tool result so the model doesn't claim
             the action ran.

Both surfaces are deterministic — the LLM never paraphrases the
preview text. This matters because the LLM was observed to translate
"Reunión" → "Reunião" between staging and confirmation, making the
user agree to something different from what would actually run.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from src.core.i18n import t
from src.core.clock import ensure_event_end

# Risk level tokens — plain strings rather than an Enum so they
# serialise cleanly into audit logs / JSON without `.value` ceremony.
S1 = "S1"
S2 = "S2"
S3 = "S3"

# Default for unknown tools: fail-closed (M6-A5) — an unclassified tool
# requires the human gate. Classify it in TOOL_RISK to remove the friction.
DEFAULT_RISK = S3


# ---------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------


TOOL_RISK: Dict[str, str] = {
    # S1 — read-only
    "read_document":  S1,
    "list_emails":    S1,
    "list_events":    S1,
    # find_file returns PATHS only (never contents), scoped to $HOME with
    # credential/key paths filtered out of the results — see file_finder.
    "find_file":      S1,

    # S2 — state-modifying but recoverable
    "write_document": S2,
    "web_search":     S2,

    # S3 — irreversible / external-facing
    # read_file_at destroys nothing: it is gated because it pulls the content
    # of an arbitrary file into the prompt, which is the door indirect prompt
    # injection comes through. The human gate is the architectural defence the
    # security review named (prompt-level defences only raise the bar), and S3
    # is how this codebase spells "ask first" (s83).
    "read_file_at":   S3,
    # Several files under one "sim" (s85) — same door, same gate. Staged only
    # by the deterministic path routes, never offered to the 8B as a tool.
    "read_files":     S3,
    "read_code_project": S3,
    "propose_code_change": S3,
    "apply_code_change": S3,
    "run_code_tests": S3,
    "pdf_merge": S3,
    "pdf_split": S3,
    "pdf_png": S3,
    "pdf_images": S3,
    "code_root_add":  S3,  # Typed phrase only; never a model tool.
    # Creates a NEW file beside the paper (s85): nothing existing is changed,
    # but it writes into the user's folders, so it asks first like any S3.
    "write_answers_pdf": S3,
    "send_email":     S3,
    "create_event":   S3,
    "create_reminder": S3,
    # quitting an app can drop unsaved work — gate it like the others
    "mac_quit_app":   S3,
    # putting the Mac to sleep interrupts active focus / unsaved work
    # and breaks any in-flight downloads or streaming — gate it S3.
    "mac_sleep":      S3,
    "ps5_standby":    S3,
}


def risk_level(name: str) -> str:
    """Return the risk level for a tool. Unknown names get DEFAULT_RISK."""
    return TOOL_RISK.get(name, DEFAULT_RISK)


def is_destructive(name: str) -> bool:
    """True iff the tool needs the human gate (S3). Replacement for
    the inline `_DESTRUCTIVE_TOOLS` frozenset that used to live in
    the orchestrator."""
    return risk_level(name) == S3


# ---------------------------------------------------------------------
# dry-run preview builder
# ---------------------------------------------------------------------


# Type for the ISO-formatter callback the orchestrator injects. The
# orchestrator's _format_iso_for_display does timezone-aware rendering
# we don't want to duplicate here.
_IsoFormatter = Callable[[str], str]


def _identity_iso(iso: str) -> str:
    """Default formatter — returns the ISO string as-is. Used when the
    caller doesn't inject a friendlier renderer."""
    return iso or "?"


def _preview_send_email(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                        lang: str = "pt") -> Dict[str, str]:
    to = args.get("to", "")
    subject = args.get("subject", "")
    body = args.get("body", "") or ""
    # Whole, never cut: "sim" sends ALL of it — a cut at 200 hid what came
    # after it (s88, Codex lote 2). Only the LLM's brief stays short.
    body_human = body
    body_llm = body[:300] + ("…" if len(body) > 300 else "")
    return {
        "human": t(
            lang,
            "Confirmar envio do email?\n"
            f"  Para: {to}\n"
            f"  Assunto: {subject}\n"
            f"  Corpo: {body_human}\n"
            "Responda 'sim' para enviar ou 'não' para cancelar.",
            "Confirm sending the email?\n"
            f"  To: {to}\n"
            f"  Subject: {subject}\n"
            f"  Body: {body_human}\n"
            "Reply 'yes' to send or 'no' to cancel.",
        ),
        "llm": (
            "CONFIRMATION REQUIRED — the email has NOT been sent yet.\n"
            "Staged email:\n"
            f"  To: {to}\n"
            f"  Subject: {subject}\n"
            f"  Body: {body_llm}\n"
            "Now ask the user (in their language, in your own words) "
            "to reply 'sim' to send it or 'não' to cancel. Do not "
            "claim it was sent. Do not call any other tool."
        ),
        "cancel_label": t(lang, "o envio do email", "sending the email"),
    }


def _preview_create_event(args: Dict[str, Any], fmt_iso: _IsoFormatter,
                          lang: str = "pt") -> Dict[str, str]:
    try:
        args = ensure_event_end(args)
    except ValueError:
        return {"human": t(lang, "Horário inicial inválido.", "Invalid start time."),
                "llm": "[SYSTEM ERROR] create_event: invalid start hour",
                "cancel_label": t(lang, "a criação do evento", "creating the event")}
    if lang == "en":
        lines = [
            "Confirm creating the event?",
            f"  Title: {args.get('summary', '')}",
            f"  Start: {fmt_iso(args.get('start_iso', ''))}",
            f"  End:   {fmt_iso(args.get('end_iso', ''))}",
        ]
        if args.get("location"):
            lines.append(f"  Location: {args['location']}")
        if args.get("attendees"):
            lines.append(f"  Attendees: {', '.join(args['attendees'])}")
        lines.append("Reply 'yes' to create or 'no' to cancel.")
    else:
        lines = [
            "Confirmar criação do evento?",
            f"  Título: {args.get('summary', '')}",
            f"  Início: {fmt_iso(args.get('start_iso', ''))}",
            f"  Fim:    {fmt_iso(args.get('end_iso', ''))}",
        ]
        if args.get("location"):
            lines.append(f"  Local:  {args['location']}")
        if args.get("attendees"):
            lines.append(f"  Participantes: {', '.join(args['attendees'])}")
        lines.append("Responda 'sim' para criar ou 'não' para cancelar.")
    return {
        "human": "\n".join(lines),
        "llm": (
            "CONFIRMATION REQUIRED — the event has NOT been created yet.\n"
            "Staged event:\n"
            f"  Title: {args.get('summary', '')}\n"
            f"  Start: {args.get('start_iso', '')}\n"
            f"  End:   {args.get('end_iso', '')}\n"
            f"  Location: {args.get('location') or '—'}\n"
            f"  Attendees: {', '.join(args.get('attendees') or []) or '—'}\n"
            "Now ask the user (in their language, in your own words) "
            "to reply 'sim' to create it or 'não' to cancel. Do not "
            "claim it was created. Do not call any other tool."
        ),
        "cancel_label": t(lang, "a criação do evento", "creating the event"),
    }


def _preview_create_reminder(args: Dict[str, Any], fmt_iso: _IsoFormatter,
                             lang: str = "pt") -> Dict[str, str]:
    if lang == "en":
        lines = [
            "Confirm creating the reminder?",
            f"  Title: {args.get('title', '')}",
        ]
        if args.get("due_iso"):
            lines.append(f"  When: {fmt_iso(args['due_iso'])}")
        else:
            lines.append("  When: no date set")
        lines.append("  (syncs to your iPhone via iCloud)")
        lines.append("Reply 'yes' to create or 'no' to cancel.")
    else:
        lines = [
            "Confirmar criação do lembrete?",
            f"  Título: {args.get('title', '')}",
        ]
        if args.get("due_iso"):
            lines.append(f"  Quando: {fmt_iso(args['due_iso'])}")
        else:
            lines.append("  Quando: sem data definida")
        lines.append("  (sincroniza com seu iPhone via iCloud)")
        lines.append("Responda 'sim' para criar ou 'não' para cancelar.")
    when = args.get("due_iso") or "—"
    return {
        "human": "\n".join(lines),
        "llm": (
            "CONFIRMATION REQUIRED — the reminder has NOT been created yet.\n"
            "Staged reminder:\n"
            f"  Title: {args.get('title', '')}\n"
            f"  Due:   {when}\n"
            "Now ask the user (in their language, in your own words) "
            "to reply 'sim' to create it or 'não' to cancel. Do not "
            "claim it was created. Do not call any other tool."
        ),
        "cancel_label": t(lang, "a criação do lembrete", "creating the reminder"),
    }


def _preview_mac_quit_app(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                          lang: str = "pt") -> Dict[str, str]:
    app = args.get("app_name", "") or ""
    return {
        "human": t(
            lang,
            f"Fechar o {app}? Pode perder trabalho não salvo.\n"
            "Responda 'sim' para fechar ou 'não' para cancelar.",
            f"Quit {app}? You may lose unsaved work.\n"
            "Reply 'yes' to quit or 'no' to cancel.",
        ),
        "llm": (
            f"CONFIRMATION REQUIRED — the app '{app}' has NOT been quit "
            "yet.\nNow ask the user (in their language, in your own "
            "words) to reply 'sim' to quit it or 'não' to cancel. Warn "
            "that unsaved work may be lost. Do not claim it was closed. "
            "Do not call any other tool."
        ),
        "cancel_label": t(lang, f"o fechamento do {app}".rstrip(),
                          f"quitting {app}".rstrip()),
    }


def _preview_ps5_standby(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                         lang: str = "pt") -> Dict[str, str]:
    return {
        "human": t(lang,
            "Colocar o PS5 em repouso? Um jogo aberto pode perder o que não foi salvo. Responda 'sim' para confirmar.",
            "Put the PS5 in rest mode? An open game may lose unsaved progress. Reply 'yes' to confirm."),
        "llm": "CONFIRMATION REQUIRED — the PS5 has NOT been put in rest mode. Ask for confirmation; unsaved progress may be lost.",
        "cancel_label": t(lang, "o repouso do PS5", "putting the PS5 in rest mode"),
    }


def _preview_mac_sleep(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                       lang: str = "pt") -> Dict[str, str]:
    # "desliga o mac" routes here on purpose — Cortex never shuts the Mac
    # down. Say so before the 'sim', so the user isn't told one thing and
    # given another.
    prefix = t(
        lang,
        "Eu não desligo o Mac — vou colocar pra dormir (acorda "
        "abrindo a tampa ou apertando uma tecla). ",
        "I don't shut the Mac down — I'll put it to sleep (it wakes "
        "with the lid or a key press). ",
    ) if args.get("phrased_as_shutdown") else ""
    return {
        "human": prefix + t(
            lang,
            "Colocar o Mac pra dormir? Sessão atual vai pausar e "
            "downloads/streams interrompem.\n"
            "Responda 'sim' para dormir ou 'não' para cancelar.",
            "Put the Mac to sleep? The current session pauses and "
            "downloads/streams will break.\n"
            "Reply 'yes' to sleep or 'no' to cancel.",
        ),
        "llm": (
            "CONFIRMATION REQUIRED — the Mac has NOT been put to sleep "
            "yet.\nNow ask the user (in their language, in your own "
            "words) to reply 'sim' to sleep or 'não' to cancel. Warn "
            "that the current session pauses and any in-flight "
            "downloads/streams will break. Do not claim it slept. Do "
            "not call any other tool."
        ),
        "cancel_label": t(lang, "o sleep do Mac", "Mac sleep"),
    }


def _preview_read_file_at(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                          lang: str = "pt") -> Dict[str, str]:
    import os
    path = str(args.get("path", "") or "")
    name = os.path.basename(path) or path
    where = os.path.dirname(path) or "?"
    return {
        "human": t(
            lang,
            f"Posso abrir e ler '{name}'?\nEstá em {where}. Só leitura — "
            "nada é alterado.\nResponda 'sim' para ler ou 'não' para cancelar.",
            f"May I open and read '{name}'?\nIt is in {where}. Read-only — "
            "nothing is changed.\nReply 'yes' to read or 'no' to cancel.",
        ),
        "llm": (
            f"CONFIRMATION REQUIRED — the file '{name}' has NOT been read "
            "yet.\nNow ask the user (in their language, in your own words) to "
            "reply 'sim' to read it or 'não' to cancel. Do not claim to know "
            "what the file says. Do not call any other tool."
        ),
        "cancel_label": t(lang, f"a leitura de {name}".rstrip(),
                          f"reading {name}".rstrip()),
    }


def _preview_propose_code_change(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                                lang: str = "pt") -> Dict[str, str]:
    import os
    path = args.get("path", "")
    name, folder = os.path.basename(path), os.path.dirname(path)
    request = args.get("request", "")
    return {
        "human": t(lang,
            f"Posso ler '{name}' e pedir ao modelo uma proposta de mudança?\n"
            + (f"Pedido: {request}\n" if request else "")
            + f"Está em {folder}. Nada é alterado no arquivo e nenhum comando roda.\n"
            "Responda 'sim' para continuar ou 'não' para cancelar.",
            f"May I read '{name}' and ask the model for a proposed change?\n"
            + (f"Request: {request}\n" if request else "")
            + f"It is in {folder}. Nothing is changed in the file and no command runs.\n"
            "Reply 'yes' to continue or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] propose_code_change requires the typed phrase.",
        "cancel_label": t(lang, f"a proposta de mudança em {name}", f"change proposal for {name}"),
    }


def _preview_apply_code_change(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                               lang: str = "pt") -> Dict[str, str]:
    import os
    path, pid, diff = args.get("path", ""), args.get("id", ""), args.get("diff", "")
    name = os.path.basename(path)
    return {
        "human": t(lang,
            f"Aplicar a proposta {pid} em {path}?\n```diff\n{diff}\n```\n"
            "Só aplico se o arquivo não mudou desde a proposta.\n"
            "Responda 'sim' (exatamente essa palavra) para aplicar ou 'não' para cancelar.",
            f"Apply proposal {pid} to {path}?\n```diff\n{diff}\n```\n"
            "I only apply it if the file has not changed since the proposal.\n"
            "Reply 'yes' (exactly that word) to apply it or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] apply_code_change requires a typed proposal.",
        "cancel_label": t(lang, f"a mudança em {name}", f"change to {name}"),
    }


def _preview_pdf_merge(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                       lang: str = "pt") -> Dict[str, str]:
    import os

    inputs, name = args.get("inputs", []), args.get("name", "")
    if not isinstance(inputs, list):
        inputs = []
    pages = 0
    lines = [t(lang, "Juntar estes PDFs, nesta ordem?", "Merge these PDFs, in this order?")]
    for number, item in enumerate(inputs, 1):
        item = item if isinstance(item, dict) else {}
        path, count = item.get("path", ""), item.get("pages")
        path = path if isinstance(path, str) else ""
        page_count = ""
        if type(count) is int:
            pages += count
            unit = t(lang, "página" if count == 1 else "páginas", "page" if count == 1 else "pages")
            page_count = f" ({count} {unit})"
        lines.append(f"{number}. {os.path.basename(path)}{page_count} — {os.path.dirname(path)}")
    lines.append(t(lang,
        f"Resultado: um PDF de {pages} páginas em ~/Documents/Cortex/PDF/{name}/\n"
        "Os originais não mudam.\n"
        "Responda 'sim' (exatamente essa palavra) para juntar ou 'não' para cancelar.",
        f"Result: a PDF of {pages} pages in ~/Documents/Cortex/PDF/{name}/\n"
        "The originals don't change.\n"
        "Reply 'yes' (exactly that word) to merge them or 'no' to cancel."))
    return {"human": "\n".join(lines),
            "llm": "[SYSTEM ERROR] pdf_merge requires the typed phrase.",
            "cancel_label": t(lang, "a junção dos PDFs", "merge the PDFs")}


def _preview_pdf_split(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                       lang: str = "pt") -> Dict[str, str]:
    import os

    args = args if isinstance(args, dict) else {}
    item = args.get("input")
    item = item if isinstance(item, dict) else {}
    path = item.get("path")
    path = path if isinstance(path, str) else ""
    source, folder = os.path.basename(path), os.path.dirname(path)
    count = item.get("pages")
    count = count if type(count) is int else 0
    keep = args.get("keep")
    pages = len(keep) if isinstance(keep, list) else 0
    label, name = args.get("label", ""), args.get("name", "")
    first = (t(lang, f"Fazer uma cópia de {source} sem as páginas {label}?",
                    f"Make a copy of {source} without pages {label}?")
             if args.get("mode") == "remove" else
             t(lang, f"Fazer um PDF só com as páginas {label} de {source}?",
                     f"Make a PDF with only pages {label} of {source}?"))
    return {"human": first + t(lang,
        f"\nOrigem: {source} ({count} {'página' if count == 1 else 'páginas'}) — {folder}\n"
        f"Resultado: um PDF de {pages} {'página' if pages == 1 else 'páginas'} em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\n"
        "Responda 'sim' (exatamente essa palavra) para fazer ou 'não' para cancelar.",
        f"\nSource: {source} ({count} {'page' if count == 1 else 'pages'}) — {folder}\n"
        f"Result: a PDF of {pages} {'page' if pages == 1 else 'pages'} in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\n"
        "Reply 'yes' (exactly that word) to do it or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] pdf_split requires the typed phrase.",
        "cancel_label": t(lang, "a separação das páginas", "split the pages")}


def _preview_pdf_images(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                        lang: str = "pt") -> Dict[str, str]:
    import os

    args = args if isinstance(args, dict) else {}
    item = args.get("input")
    item = item if isinstance(item, dict) else {}
    path = item.get("path")
    path = path if isinstance(path, str) else ""
    source, folder = os.path.basename(path), os.path.dirname(path)
    pages = item.get("pages")
    pages = pages if type(pages) is int else 0
    keep = args.get("keep")
    count = len(keep) if isinstance(keep, list) else 0
    label = args.get("label")
    label = label if isinstance(label, str) else ""
    name = args.get("name")
    name = name if isinstance(name, str) else ""
    return {"human": t(lang,
        f"Tirar as imagens{' das páginas ' + label if label else ''} de {source}?\n"
        f"Origem: {source} ({pages} {'página' if pages == 1 else 'páginas'}) — {folder}\n"
        f"Encontrei {count} {'imagem' if count == 1 else 'imagens'} (fora máscaras e ícones menores que 64 px). JPEG fica .jpg, o resto .png.\n"
        f"Resultado: {count} {'arquivo' if count == 1 else 'arquivos'} em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\n"
        "Responda 'sim' (exatamente essa palavra) para tirar ou 'não' para cancelar.",
        f"Take the images out of {'pages ' + label + ' of ' if label else ''}{source}?\n"
        f"Source: {source} ({pages} {'page' if pages == 1 else 'pages'}) — {folder}\n"
        f"Found {count} {'image' if count == 1 else 'images'} (leaving out masks and icons under 64 px). JPEG stays .jpg, the rest .png.\n"
        f"Result: {count} {'file' if count == 1 else 'files'} in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\n"
        "Reply 'yes' (exactly that word) to take them out or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] pdf_images requires the typed phrase.",
        "cancel_label": t(lang, "a extração das imagens", "take the images out")}


def _preview_pdf_png(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                     lang: str = "pt") -> Dict[str, str]:
    import os

    args = args if isinstance(args, dict) else {}
    item = args.get("input")
    item = item if isinstance(item, dict) else {}
    path = item.get("path")
    path = path if isinstance(path, str) else ""
    source, folder = os.path.basename(path), os.path.dirname(path)
    count = item.get("pages")
    count = count if type(count) is int else 0
    keep = args.get("keep")
    pages = len(keep) if isinstance(keep, list) else 0
    label, name = args.get("label", ""), args.get("name", "")
    selection = (f"a página {label}" if pages == 1 else f"as páginas {label}") if lang == "pt" else (
        f"page {label}" if pages == 1 else f"pages {label}")
    return {"human": t(lang,
        f"Salvar como imagem (PNG, 2048 px) {selection} de {source}?\n"
        f"Origem: {source} ({count} {'página' if count == 1 else 'páginas'}) — {folder}\n"
        f"Resultado: {pages} {'imagem' if pages == 1 else 'imagens'} em ~/Documents/Cortex/PDF/{name}/\n"
        "O original não muda.\n"
        "Responda 'sim' (exatamente essa palavra) para salvar ou 'não' para cancelar.",
        f"Save as image (PNG, 2048 px) {selection} of {source}?\n"
        f"Source: {source} ({count} {'page' if count == 1 else 'pages'}) — {folder}\n"
        f"Result: {pages} {'image' if pages == 1 else 'images'} in ~/Documents/Cortex/PDF/{name}/\n"
        "The original doesn't change.\n"
        "Reply 'yes' (exactly that word) to save them or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] pdf_png requires the typed phrase.",
        "cancel_label": t(lang, "a gravação das imagens", "save the images")}


def _preview_run_code_tests(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                            lang: str = "pt") -> Dict[str, str]:
    import os
    from src.core.code_commands import render

    profile = args.get("profile") or {}
    cwd = profile.get("cwd", "")
    name, command = os.path.basename(cwd), render(profile) if "argv" in profile else ""
    return {
        "human": t(lang,
            f"Rodar os testes de {name}?\nComando: {command}\nNa pasta: {cwd}\nSem internet; para em 30 s.\n"
            "Responda 'sim' (exatamente essa palavra) para rodar ou 'não' para cancelar.",
            f"Run the tests of {name}?\nCommand: {command}\nIn the folder: {cwd}\nNo network; stops after 30 s.\n"
            "Reply 'yes' (exactly that word) to run them or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] run_code_tests requires the typed phrase.",
        "cancel_label": t(lang, f"os testes de {name}", f"tests of {name}"),
    }


def _preview_code_root_add(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                           lang: str = "pt") -> Dict[str, str]:
    real = args.get("path", "")
    return {
        "human": t(lang,
            f"Autorizar a pasta {real} para código?\n"
            "O Cortex vai poder LER os arquivos .py dessa pasta e das subpastas. "
            "Nada é alterado e nenhum comando roda.\n"
            "Responda 'sim' (exatamente essa palavra) para autorizar ou 'não' para cancelar.",
            f"Authorise the folder {real} for code?\n"
            "Cortex will be able to READ the .py files in this folder and its subfolders. "
            "Nothing is changed and no command runs.\n"
            "Reply 'yes' (exactly that word) to authorise or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] code_root_add requires a typed fixed phrase.",
        "cancel_label": t(lang, f"a autorização de {real}", f"authorisation of {real}"),
    }


def _preview_read_files(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                        lang: str = "pt") -> Dict[str, str]:
    import os
    paths = [str(p) for p in (args.get("paths") or [])]
    names = [os.path.basename(p) for p in paths] or ["?"]
    joiner = t(lang, " e ", " and ")
    listed = (", ".join(names[:-1]) + joiner + names[-1]
              if len(names) > 1 else names[0])
    folders = sorted({os.path.dirname(p) for p in paths})
    where = (t(lang, f"Estão em {folders[0]}.", f"They are in {folders[0]}.")
             if len(folders) == 1
             else t(lang, "Estão em pastas diferentes.", "They are in different folders."))
    n = len(paths)
    return {
        "human": t(
            lang,
            f"Posso abrir e ler os {n} arquivos: {listed}?\n{where} Só leitura "
            "— nada é alterado.\nResponda 'sim' para ler ou 'não' para cancelar.",
            f"May I open and read these {n} files: {listed}?\n{where} "
            "Read-only — nothing is changed.\nReply 'yes' to read or 'no' to "
            "cancel.",
        ),
        "llm": (
            f"CONFIRMATION REQUIRED — the files {listed} have NOT been read "
            "yet.\nNow ask the user (in their language, in your own words) to "
            "reply 'sim' to read them or 'não' to cancel. Do not claim to know "
            "what the files say. Do not call any other tool."
        ),
        "cancel_label": t(lang, f"a leitura de {listed}", f"reading {listed}"),
    }


def _preview_read_code_project(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                               lang: str = "pt") -> Dict[str, str]:
    import os
    real = args.get("root", "")
    name, files = os.path.basename(real), args.get("files") or []
    listed = (", ".join(files[:-1]) + t(lang, " e ", " and ") + files[-1]
              if len(files) > 1 else ", ".join(files))
    evidence = (t(lang,
        "Junto com o fim da saída da última falha dos testes, para explicar o que falhou.\n",
        "Together with the end of the last failed test run's output, to explain what failed.\n")
        if "evidence" in args else "")
    return {
        "human": t(lang,
            f"Posso abrir e ler o projeto {name}? São {len(files)} arquivos .py em {real}: {listed}.\n"
            + evidence +
            "Só leitura — nada é alterado e nenhum comando roda.\n"
            "Responda 'sim' para ler ou 'não' para cancelar.",
            f"May I open and read the project {name}? {len(files)} .py files in {real}: {listed}.\n"
            + evidence +
            "Read-only — nothing is changed and no command runs.\n"
            "Reply 'yes' to read or 'no' to cancel."),
        "llm": "[SYSTEM ERROR] read_code_project requires a typed folder read.",
        "cancel_label": t(lang, f"a leitura de {name}", f"reading {name}"),
    }


def _preview_write_answers_pdf(args: Dict[str, Any], _fmt_iso: _IsoFormatter,
                               lang: str = "pt") -> Dict[str, str]:
    import os
    out = str(args.get("out", "") or "")
    paper = os.path.basename(str(args.get("paper", "") or "")) or "?"
    name = os.path.basename(out) or "?"
    folder = os.path.basename(os.path.dirname(out)) or "?"
    n = len(args.get("answers") or {})
    return {
        "human": t(
            lang,
            f"Posso criar '{name}' na pasta {folder}?\nEscrevo as {n} respostas "
            f"numa cópia — o original ({paper}) não é alterado.\nResponda 'sim' "
            "para criar ou 'não' para cancelar.",
            f"May I create '{name}' in the folder {folder}?\nI'll write the {n} "
            f"answers on a copy — the original ({paper}) is not changed.\nReply "
            "'yes' to create it or 'no' to cancel.",
        ),
        "llm": (
            f"CONFIRMATION REQUIRED — '{name}' has NOT been written yet.\nNow ask "
            "the user (in their language, in your own words) to reply 'sim' to "
            "create it or 'não' to cancel. Do not call any other tool."
        ),
        "cancel_label": t(lang, "a cópia com as respostas", "the answered copy"),
    }


# Dispatch table — keeps the per-tool dictionaries co-located with the
# tool's risk classification.
_PREVIEW_BUILDERS: Dict[str, Callable[..., Dict[str, str]]] = {
    "pdf_merge": _preview_pdf_merge,
    "pdf_split": _preview_pdf_split,
    "pdf_png": _preview_pdf_png,
    "pdf_images": _preview_pdf_images,
    "run_code_tests": _preview_run_code_tests,
    "apply_code_change": _preview_apply_code_change,
    "propose_code_change": _preview_propose_code_change,
    "code_root_add":  _preview_code_root_add,
    "read_file_at":   _preview_read_file_at,
    "read_files":     _preview_read_files,
    "read_code_project": _preview_read_code_project,
    "write_answers_pdf": _preview_write_answers_pdf,
    "send_email":     _preview_send_email,
    "create_event":   _preview_create_event,
    "create_reminder": _preview_create_reminder,
    "mac_quit_app":   _preview_mac_quit_app,
    "ps5_standby":    _preview_ps5_standby,
    "mac_sleep":      _preview_mac_sleep,
}


def build_preview(
    name: str,
    args: Dict[str, Any],
    fmt_iso: Optional[_IsoFormatter] = None,
    lang: str = "pt",
) -> Dict[str, str]:
    """Build the dry-run preview surfaces for a staged S3 action.

    Returns a dict with at least:
      - "human":         the prompt the user sees and replies to, in `lang`
                         (PT default — the spoken-language fix, s60)
      - "llm":           English brief that goes back to the LLM as the
                         tool result so the model doesn't claim it ran
                         (always English — internal, the LLM reframes it)
      - "cancel_label":  complete noun phrase used in the cancel message,
                         localised to `lang` (PT includes its article)

    Unknown tool names get a generic fallback. The fallback never
    invents a specific cancel label — the orchestrator's cancel
    message will say just "a ação" / "the action".
    """
    builder = _PREVIEW_BUILDERS.get(name)
    if builder is None:
        return {
            "human": t(lang, "Confirmar a ação? Responda 'sim' ou 'não'.",
                       "Confirm the action? Reply 'yes' or 'no'."),
            "llm": "CONFIRMATION REQUIRED. Ask the user to reply 'sim' or 'não'.",
            "cancel_label": t(lang, "a ação", "action"),
        }
    return builder(args or {}, fmt_iso or _identity_iso, lang)
