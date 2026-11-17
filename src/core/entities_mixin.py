"""
Entity-brain skill handlers for `CortexOrchestrator`.

Extracted from `src/core/orchestrator.py` (s59) as part of the per-family
handler-mixin refactor. The dispatch registry routes "quando falei com X" /
"com quem andei falando" here. Thin wrappers over the deterministic people
graph in `src.core.entities` (local import); mixed in so `self._handle_*(...)`
resolves through the MRO.
"""
from __future__ import annotations

from typing import Tuple


class EntitiesMixin:
    """Deterministic people-graph handlers (last contact / recent people)."""

    def _handle_last_contact(self, name: str) -> Tuple[str, str]:
        """"Quando falei com X pela última vez?" — deterministic people graph."""
        from src.core import entities
        return (entities.last_contact_reply((name or "").strip(), lang=self._reply_lang),
                "ENTITY_LAST_CONTACT")

    def _handle_recent_people(self) -> Tuple[str, str]:
        """"Com quem andei falando?" — recent interactions, deterministic."""
        from src.core import entities
        return (entities.recent_people_reply(lang=self._reply_lang), "ENTITY_RECENT_PEOPLE")
