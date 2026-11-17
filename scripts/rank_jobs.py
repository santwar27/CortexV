#!/usr/bin/env python3
"""
Rank the stored jobs against your master resume (Job Hunter slice 2).

Usage:
  ./venv/bin/python scripts/rank_jobs.py [--resume PATH] [--top N] [--list]

  --resume PATH  ingest/replace the master resume from a local file
                 (.txt/.md/.pdf/.docx). Stored locally in ~/.cortex/jobs.db.
  --top N        how many top matches to print (default 10).
  --list         just PRINT the already-ranked top-N (with URLs) — no
                 re-ranking, no Ollama. The fast "show me the matches" view
                 (also `make jobs`).

Embeds the resume + each job (title + description) with the LOCAL Ollama model
and scores by cosine similarity. SOBERANIA r1: the resume never leaves the
machine — no cloud, no LLM judgement. Run `fetch_jobs.py` first to populate the
jobs; re-run after editing the resume to re-score.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.integrations.ats import format_deadline, is_deadline_urgent  # noqa: E402
from src.integrations.jobs import get_log  # noqa: E402
from src.integrations.job_ranker import rank_jobs  # noqa: E402
from src.tools.local_reader import LocalFileReader  # noqa: E402


def _print_top(log, top: int, *, header: str) -> int:
    """Print the stored top-N ranked jobs (with URLs). Read-only — no embedding,
    no Ollama. Returns a shell exit code."""
    rows = [r for r in log.list(limit=top, order="score") if r["score"] is not None]
    # The CLI legitimately shows everything, labelled — unlike the surfaces
    # that RECOMMEND, which filter through list_open (s82/s83).
    cutoffs = log.feed_cutoffs()
    if not rows:
        if log.count() == 0:
            print("Sem vagas na base. Roda o fetch_jobs.py (ou espera o briefing das 08h).")
        else:
            print("Nenhuma vaga rankeada ainda. Roda: rank_jobs.py --resume <cv>.")
        return 1
    print(f"=== top {top} ({header}) ===")
    for r in rows:
        score = f"{r['score']*100:.0f}%"
        dist = f" [{r['distance']:.1f} mi]" if r["distance"] is not None else ""
        loc = f" — {r['location']}" if r["location"] else ""
        deadline = format_deadline(r["closing_at"])
        flag = "⚠️  " if is_deadline_urgent(r["closing_at"]) else ""
        dl = f"  ·  {flag}{deadline}" if deadline else ""
        gone = "  ·  sumiu do feed" if log.is_gone_from_feed(r, cutoffs) else ""
        print(f"  {score:>4}  {r['title']}{dist}{loc}{dl}{gone}")
        print(f"        {r['url']}")
    return 0


def main() -> int:
    args = list(sys.argv[1:])
    top = 10
    resume_path = None
    list_only = "--list" in args
    args = [a for a in args if a != "--list"]
    for flag in ("--top", "--resume"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                print(f"{flag} precisa de um valor")
                return 1
            val = args[i + 1]
            del args[i:i + 2]
            if flag == "--top":
                try:
                    top = int(val)
                except ValueError:
                    print("--top precisa de um número")
                    return 1
            else:
                resume_path = val

    log = get_log()

    # --list: just show what's already ranked (fast, no resume/Ollama needed).
    if list_only:
        return _print_top(log, top, header="já rankeadas")

    if resume_path:
        p = Path(resume_path).expanduser()
        if not p.exists():
            print(f"currículo não encontrado: {p}")
            return 1
        text = LocalFileReader().read_path(str(p))
        if text.startswith("[SYSTEM ERROR]"):
            print(f"não consegui ler o currículo: {text}")
            return 1
        if not log.set_resume(text, name=p.name):
            print("não consegui guardar o currículo.")
            return 1
        print(f"✓ currículo '{p.name}' guardado ({len(text)} chars), local-only.")

    if log.get_resume() is None:
        print("Sem currículo guardado. Passa um com --resume <arquivo> "
              "(.txt/.md/.pdf/.docx).")
        return 1

    if log.count() == 0:
        print("Sem vagas na base. Roda o fetch_jobs.py primeiro.")
        return 1

    print("Rankeando (embeddings locais)…")
    n = rank_jobs(log)
    if n == 0:
        print("Não consegui rankear (o Ollama está rodando? "
              "embedding falhou?).")
        return 1
    print(f"✓ {n} vagas rankeadas vs currículo.")
    return _print_top(log, top, header="melhor match")


if __name__ == "__main__":
    raise SystemExit(main())
