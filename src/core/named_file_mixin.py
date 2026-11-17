"""A file asked for by name — the route (s87, 4.3). The words and the search
live in src/core/named_file.py; this answers them, always with a fixed text.

Measured before (real 8B, both entry points): 1 useful reply in 28 — see the
module docstring there. Now:
  one file   → "lê": the staged read (confirmation first — the same gate as a
               typed path); "cadê": where it is, and "quer que eu leia?";
               "abre": opened in its app (Pedro, 16/10: opening changes nothing)
  several    → the newest five, numbered: "qual deles?" — "2", "o segundo",
               "abre o primeiro" answer it next turn
  none       → said so, with the synonyms searched; when the request was not
               surely about a file ("acha o melhor restaurante"), the turn
               goes on to the other routes instead

Two calls in _process_core: the answer to the list BEFORE the yes-with-
nothing-pending check (a "sim" here answers this list), the request AFTER the
dispatch registry ("lê meu email" is the email route's). process_stream's
reflex and pure-chat shortcuts step aside through `_is_named_file_turn`.
"""
from __future__ import annotations

import os
from typing import List, Optional

from src.core import named_file
from src.core.audit import audit
from src.core.dispatch.heuristics import multi_intent_match
from src.core.dispatch.normalisation import strip_courtesy
from src.core.named_file import LOCATE, OPEN, READ

_ICLOUD = os.path.join("Library", "Mobile Documents", "com~apple~CloudDocs")


class NamedFileMixin:
    """Mixed into the orchestrator; uses the host's `_offer`, `_t`,
    `_APPROVAL_RE`, `_remember_seen_path`, `_stage_file_read`, `macos`."""

    _NAMED_SHOW_MAX = 5

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------
    def _is_named_file_turn(self, own: str) -> bool:
        """Will this turn be a named-file request, or the answer to its
        list? Regex only — cheap enough for the stream's probes."""
        text = strip_courtesy((own or "").strip())
        if named_file.parse_request(text) is not None:
            return True
        return self._named_offer_answer(text, getattr(self, "_offer", None)) is not None

    def _answer_named_file_offer(self, own: str, offer: Optional[dict]):
        """The answer to "qual deles?" / "quer que eu leia?" — or None."""
        text = strip_courtesy((own or "").strip())
        picked = self._named_offer_answer(text, offer)
        if picked is None:
            return None
        path, verb = picked
        self._reply_lang = offer.get("lang") or getattr(self, "_reply_lang", "pt")
        audit.log("–", "named_file", "picked", payload={"verb": verb})
        return self._act_on_named(verb, path, offer.get("question") or own)

    def _handle_named_file(self, own: str):
        """"lê meu currículo", "cadê o pdf de X", "abre meu perfil" — or None."""
        text = strip_courtesy((own or "").strip())
        request = named_file.parse_request(text)
        if request is None:
            return None
        # "lê o documento agentes e me manda um resumo por email" is two
        # requests — the planner's (it runs right after this route).
        if multi_intent_match(text):
            return None
        found, error = named_file.find(request, readable_only=request.verb != OPEN)
        audit.log("–", "named_file", "search", payload={
            "verb": request.verb, "terms": len(request.terms),
            "found": len(found), "strong": request.strong, "error": bool(error)})
        if not found:
            if error:
                return (self._t(f"a busca de arquivos falhou agora: {error}",
                                f"the file search failed just now: {error}"),
                        "FILE_ERROR")
            if not request.strong:
                return None             # maybe not a file at all
            return (self._named_not_found(request), "FILE_ERROR")
        for hit in found[: self._NAMED_SHOW_MAX]:
            self._remember_seen_path(hit.path)
        if len(found) == 1:
            return self._act_on_named(request.verb, found[0].path, own)
        return self._list_named(request, found, own)

    # ------------------------------------------------------------------
    # answers
    # ------------------------------------------------------------------
    def _act_on_named(self, verb: str, path: str, question: str):
        name = os.path.basename(path)
        if verb == READ:
            return self._stage_file_read(path, question)
        if verb == OPEN:
            return (self.macos.open_file(path, lang=self._reply_lang), "FILE_OPEN")
        self._named_offer([path], READ, question)
        return (self._t(
            f"Está em {self._short_folder(path)}: {name}"
            f"{self._modified_note(path)}. Quer que eu leia? (sim)",
            f"It's in {self._short_folder(path)}: {name}"
            f"{self._modified_note(path)}. Shall I read it? (yes)"), "FILE_LIST")

    def _list_named(self, request, found: list, question: str):
        shown = found[: self._NAMED_SHOW_MAX]
        lines = "\n".join(
            f"{i}. {h.name} — {self._short_folder(h.path)}"
            + (f" ({h.modified})" if h.modified else "")
            for i, h in enumerate(shown, 1))
        n = len(found)
        newest = (self._t(f", os {len(shown)} mais recentes",
                          f", the {len(shown)} newest") if n > len(shown) else "")
        self._named_offer([h.path for h in shown],
                          READ if request.verb == LOCATE else request.verb, question)
        return (self._t(
            f"Achei {n} arquivos com \"{request.target}\" no nome"
            f"{self._synonyms_note(request)}{newest}:\n{lines}\n"
            "Qual deles? (o número)",
            f"I found {n} files with \"{request.target}\" in the name"
            f"{self._synonyms_note(request)}{newest}:\n{lines}\n"
            "Which one? (the number)"), "FILE_LIST")

    def _named_not_found(self, request) -> str:
        return self._t(
            f"Não achei nenhum arquivo com \"{request.target}\" no nome"
            f"{self._synonyms_note(request)}. Me diz outra palavra do nome, ou "
            "o caminho.",
            f"I couldn't find a file with \"{request.target}\" in the name"
            f"{self._synonyms_note(request)}. Tell me another word in its "
            "name, or the path.")

    # ------------------------------------------------------------------
    # the list's question, for one turn
    # ------------------------------------------------------------------
    def _named_offer(self, paths: List[str], verb: str, question: str) -> None:
        self._offer = {"kind": "find_pick", "paths": list(paths), "verb": verb,
                       "question": question,
                       "lang": getattr(self, "_reply_lang", "pt")}

    def _named_offer_answer(self, text: str, offer: Optional[dict]):
        """(path, verb) when `text` answers the open list, else None."""
        if (offer or {}).get("kind") != "find_pick":
            return None
        paths = offer.get("paths") or []
        pick = named_file.parse_pick(text)
        if pick is not None:
            index, verb = pick
            if index < len(paths):
                return paths[index], verb or offer.get("verb") or READ
            return None
        if len(paths) == 1 and self._APPROVAL_RE.match(text):
            return paths[0], offer.get("verb") or READ
        return None

    # ------------------------------------------------------------------
    # text helpers
    # ------------------------------------------------------------------
    def _synonyms_note(self, request) -> str:
        extra = sorted({w for t in request.terms for w in named_file.expand(t)}
                       - set(request.terms))
        if not extra:
            return ""
        return self._t(f" (procurei também: {', '.join(extra)})",
                       f" (also searched: {', '.join(extra)})")

    @staticmethod
    def _short_folder(path: str) -> str:
        folder = os.path.dirname(path)
        home = os.path.expanduser("~")
        icloud = os.path.join(home, _ICLOUD)
        if folder == icloud or folder.startswith(icloud + os.sep):
            return "iCloud Drive" + folder[len(icloud):]
        if folder == home or folder.startswith(home + os.sep):
            return "~" + folder[len(home):]
        return folder

    def _modified_note(self, path: str) -> str:
        try:
            from datetime import datetime
            day = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d/%m/%Y")
        except OSError:
            return ""
        return self._t(f" (alterado em {day})", f" (changed on {day})")
