"""
Dream Loop — the reference creative idea (see ideas-backlog "Dream Loop"). Scaffold #5,
the last of the build-now-wire-later batch. Pairs with the nightly consolidator:
the consolidator compresses the day into structured memory; the Dream Loop does
the opposite — it pulls RANDOM fragments from long-term memory and looks for a
surprising, non-obvious connection between them, producing a short "dream" the
user can read and keep or discard in the morning.

Two layers, by design (same split as the other scaffolds):

  • Layer 1 — BUILT NOW, deterministic pipeline, fully testable with a mocked
    generator: random chunk sampling from the Palace, the dream prompt, the
    pending-proposal store, and the read / approve (keep) / reject (discard)
    operations.

  • Layer 2 — DORMANT, model-bound: the QUALITY of the dream. On the 8B the
    associations are flat; this gets genuinely interesting on Gemma 4 / a 70B.
    Nothing is blocked from running — the generator is injectable and defaults
    to Ollama — but the whole feature sits behind CORTEX_DREAM_LOOP (default
    off) and has no scheduler / orchestrator wiring yet. Mac Studio day = flip
    the flag, point at the strong model, and (optionally) schedule it nightly.

The generator is model-AGNOSTIC: inject `generate_fn(prompt) -> str` (tests do;
a nightly job would pass an Ollama-backed callable). Without one it lazily uses
ollama.generate, mirroring NightlyStudier.
"""
from __future__ import annotations

import json
import logging
import os
import random
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.core.audit import audit

log = logging.getLogger(__name__)

GenerateFn = Callable[[str], str]


def _dream_enabled() -> bool:
    """True when the Dream Loop feature is ON. DORMANT default: off. Gates any
    scheduled/live use; the pipeline itself runs whenever dream() is called."""
    return os.environ.get("CORTEX_DREAM_LOOP", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Dream:
    """One generated dream awaiting the user's verdict."""
    id: str
    created_at: str
    source_doc_ids: List[int]
    source_snippets: List[str]
    text: str
    status: str = "pending"  # pending | kept | discarded


_DREAM_FIELDS = {f.name for f in fields(Dream)}


def _dream_from_dict(d: Dict[str, Any]) -> Optional[Dream]:
    """Build a Dream from a stored dict, tolerant of schema drift: unknown
    keys are dropped, and a row missing a REQUIRED field is skipped (returns
    None) rather than crashing the whole store on a stale/foreign line."""
    try:
        return Dream(**{k: v for k, v in d.items() if k in _DREAM_FIELDS})
    except TypeError:
        return None


class DreamStore:
    """Append-only JSONL store of dreams with a mutable status field. Mirrors
    StudyQueue: survives reboot, inspectable with `cat data/dreams/dreams.jsonl`."""

    KEPT = "kept"
    DISCARDED = "discarded"
    PENDING = "pending"

    def __init__(self, path: str = "data/dreams/dreams.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def save(self, text: str, source_doc_ids: List[int],
             source_snippets: List[str]) -> Dream:
        dream = Dream(
            id=uuid.uuid4().hex[:8],
            created_at=_now_iso(),
            source_doc_ids=list(source_doc_ids),
            source_snippets=list(source_snippets),
            text=text,
            status=self.PENDING,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(dream), ensure_ascii=False) + "\n")
        return dream

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def list_by_status(self, status: str) -> List[Dream]:
        out: List[Dream] = []
        for d in self._read_all():
            if d.get("status") == status:
                dream = _dream_from_dict(d)
                if dream is not None:
                    out.append(dream)
        return out

    def pending(self) -> List[Dream]:
        return self.list_by_status(self.PENDING)

    def latest_pending(self) -> Optional[Dream]:
        pend = self.pending()
        return pend[-1] if pend else None

    def get(self, dream_id: str) -> Optional[Dream]:
        for d in self._read_all():
            if d.get("id") == dream_id:
                return _dream_from_dict(d)
        return None

    def set_status(self, dream_id: str, status: str) -> bool:
        """Rewrite the store with one dream's status changed. Returns True if a
        matching dream was found."""
        entries = self._read_all()
        found = False
        for d in entries:
            if d.get("id") == dream_id:
                d["status"] = status
                found = True
        if found:
            self._rewrite_atomic(entries)
        return found

    def approve(self, dream_id: str) -> bool:
        """Keep the dream (status -> kept)."""
        return self.set_status(dream_id, self.KEPT)

    def reject(self, dream_id: str) -> bool:
        """Discard the dream (status -> discarded)."""
        return self.set_status(dream_id, self.DISCARDED)

    def _rewrite_atomic(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)


class DreamLoop:
    """Random-fragment dreamer over the Palace, with a pending review store."""

    # How many memory fragments to weave into one dream.
    N_CHUNKS = 3
    # Upper bound on the pool we sample from (most recent docs).
    POOL_SIZE = 500
    # Truncate each fragment fed to the prompt / stored as a snippet.
    SNIPPET_CHARS = 300

    def __init__(
        self,
        palace: Any,
        generate_fn: Optional[GenerateFn] = None,
        store: Optional[DreamStore] = None,
        n_chunks: int = N_CHUNKS,
        pool_size: int = POOL_SIZE,
        rng: Optional[random.Random] = None,
        model_name: Optional[str] = None,
    ):
        self.palace = palace
        self.generate_fn = generate_fn
        self.store = store or DreamStore()
        self.n_chunks = n_chunks
        self.pool_size = pool_size
        self._rng = rng or random.Random()
        self.model_name = model_name or os.getenv("CORTEX_MODEL_CREATIVE", "cortex-criativo")

    # -- public API --------------------------------------------------------

    def dream(self, trace: Optional[str] = None) -> Optional[Dream]:
        """Sample fragments -> generate a dream -> store it as pending. Returns
        the Dream, or None if there was nothing to dream about / generation
        produced nothing."""
        trace = trace or audit.new_trace()
        chunks = self._sample_chunks()
        if not chunks:
            audit.log(trace, "dream", "no_chunks")
            return None

        prompt = self._build_prompt(chunks)
        text = self._generate(prompt).strip()
        if not text:
            audit.log(trace, "dream", "empty_generation")
            return None

        snippets = [self._snippet(c.get("texto", "")) for c in chunks]
        ids = [int(c["id"]) for c in chunks if "id" in c]
        dream = self.store.save(text, ids, snippets)
        audit.log(trace, "dream", "saved",
                  payload={"id": dream.id, "n_chunks": len(chunks)})
        return dream

    # Review surface — delegate to the store (future voice/text commands wire
    # here: "ler sonho", "guarda esse sonho", "descarta").
    def pending(self) -> List[Dream]:
        return self.store.pending()

    def latest(self) -> Optional[Dream]:
        return self.store.latest_pending()

    def approve(self, dream_id: str) -> bool:
        return self.store.approve(dream_id)

    def reject(self, dream_id: str) -> bool:
        return self.store.reject(dream_id)

    # -- Layer 1: pipeline (deterministic) ---------------------------------

    def _sample_chunks(self) -> List[Dict[str, Any]]:
        """Pull a random sample of documents from the Palace. Samples from the
        most recent pool_size docs (true uniform sampling over the whole corpus
        can be added later via SQL RANDOM())."""
        try:
            pool = self.palace.list_documents(limit=self.pool_size)
        except Exception as e:  # noqa: BLE001 — never crash on a memory read
            log.warning("DreamLoop: palace read failed (%s)", e)
            return []
        pool = [d for d in (pool or []) if (d.get("texto") or "").strip()]
        if not pool:
            return []
        k = min(self.n_chunks, len(pool))
        return self._rng.sample(pool, k)

    def _snippet(self, text: str) -> str:
        text = " ".join((text or "").split())
        return text[: self.SNIPPET_CHARS]

    def _build_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        frags = "\n".join(f"- {self._snippet(c.get('texto', ''))}" for c in chunks)
        return (
            "You are CORTEX dreaming. Below are unrelated fragments pulled at "
            "random from long-term memory. Find a SURPRISING, non-obvious "
            "connection between them and write a short, vivid 'dream' (3-6 "
            "sentences) that weaves them together — associative and creative, "
            "NOT a summary. Write in PT-BR.\n\n"
            f"Fragments:\n---\n{frags}\n---\nDream:"
        )

    # -- Layer 2: generation (model-bound; shines on Gemma 4) --------------

    def _generate(self, prompt: str) -> str:
        """Run the dream generator. Uses the injected generate_fn if given,
        else lazily calls ollama_client.generate (creative temperature, with
        keep_alive baked in — M8-A6). Fail-soft: returns '' on any error so
        the loop degrades instead of crashing."""
        if self.generate_fn is not None:
            try:
                return self.generate_fn(prompt) or ""
            except Exception as e:  # noqa: BLE001
                log.warning("DreamLoop: injected generator failed (%s)", e)
                return ""
        try:
            import ollama
            from src.core.ollama_client import KEEP_ALIVE, THINK
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.9, "num_predict": 400},
                keep_alive=KEEP_ALIVE, think=THINK,  # M8-A6
            )
            return (response.get("response") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("DreamLoop: ollama generation failed (%s)", e)
            return ""
