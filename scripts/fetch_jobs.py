#!/usr/bin/env python3
"""
Fetch public job postings from an ATS board into the local Job Hunter ledger.

Usage:
  ./venv/bin/python scripts/fetch_jobs.py <ats> [token/city] \
      [--keyword K]... [--location LOC] [--distance N] [--limit N] [--raw]

  <ats>          "greenhouse" / "lever" / "ashby" / "recruitee" (each needs a
                 board token) or "findapprenticeship" (gov.uk; needs CORTEX_FAA_KEY)
  token/city     token boards: the board id (the "stripe" in
                 boards.greenhouse.io/stripe, "spotify" in jobs.lever.co/spotify,
                 "ramp" in jobs.ashbyhq.com/ramp). findapprenticeship: the home
                 city for the distance search (default "reading").
  --keyword K    keep only jobs whose title/description contains K
                 (repeatable; ALL must match), e.g. --keyword software
  --title        match --keyword in the TITLE only (cuts noise like an admin
                 role that just mentions "software" in its body)
  --location LOC findapprenticeship: home city for the geo search.
                 greenhouse: substring the location must contain.
  --distance N   findapprenticeship: search radius in miles (default 40).

Reads a PUBLIC source (nothing personal goes out), filters, dedups by URL, and
stores matches in ~/.cortex/jobs.db. findapprenticeship results are sorted
NEAREST FIRST (miles from the home city) using the API's distance field.

Examples:
  fetch_jobs.py findapprenticeship --keyword software        # near Reading
  fetch_jobs.py greenhouse stripe --keyword engineer --location london
  fetch_jobs.py lever spotify --title engineer --location london
  fetch_jobs.py ashby ramp --keyword backend
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# A standalone script doesn't inherit the app's env — load .env so secrets like
# CORTEX_FAA_KEY (Find an Apprenticeship API) are picked up.
load_dotenv(PROJECT_ROOT / ".env")

from src.integrations import ats  # noqa: E402
from src.integrations.jobs import get_log  # noqa: E402


def main() -> int:
    args = list(sys.argv[1:])
    limit = 8
    distance_miles = 40
    location = None
    keywords = []
    raw = "--raw" in args
    title_only = "--title" in args
    args = [a for a in args if a not in ("--raw", "--title")]
    # pull repeatable/valued flags out of args
    while "--keyword" in args:
        i = args.index("--keyword")
        if i + 1 >= len(args):
            print("--keyword precisa de um valor")
            return 1
        keywords.append(args[i + 1])
        del args[i:i + 2]
    for flag in ("--location", "--limit", "--distance"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                print(f"{flag} precisa de um valor")
                return 1
            val = args[i + 1]
            del args[i:i + 2]
            if flag in ("--limit", "--distance"):
                try:
                    num = int(val)
                except ValueError:
                    print(f"{flag} precisa de um número")
                    return 1
                if flag == "--limit":
                    limit = num
                else:
                    distance_miles = num
            else:
                location = val
    if not args:
        print('uso: fetch_jobs.py <ats> [board_token] '
              '[--keyword K]... [--location LOC] [--limit N] [--raw]')
        print(f'     ats disponíveis: {", ".join(sorted(ats.ADAPTERS))}')
        return 1

    source = args[0]
    query = args[1] if len(args) > 1 else ""
    if source in ats.TOKEN_SOURCES and not query:
        print(f"{source} precisa do board_token (ex: stripe)")
        return 1

    # findapprenticeship: no board token; geo comes from query or --location,
    # search_term from the first --keyword. api_key is read from CORTEX_FAA_KEY.
    raw_box = {}
    opts = {}
    if source == "findapprenticeship":
        # geo city for the server-side filter; default to the configured home city
        query = query or location or "reading"
        opts["distance_miles"] = distance_miles
        if raw:
            opts["raw_sink"] = lambda d: raw_box.update(data=d)

    label = f"{source}/{query}" if query else source
    try:
        postings = ats.fetch(source, query, **opts)
    except Exception as e:  # network / HTTP / JSON
        print(f"Falhou ao buscar {label}: {e}")
        return 1

    if raw and raw_box.get("data") is not None:
        import json
        print("=== RAW (1ª página, pra calibrar o parser) ===")
        print(json.dumps(raw_box["data"], indent=2, ensure_ascii=False)[:4000])
        print("=== /RAW ===")

    if not postings:
        hint = ""
        if source == "findapprenticeship":
            hint = " (chave CORTEX_FAA_KEY ausente/ inválida, ou 0 resultados?)"
        else:
            hint = " (board inexistente, vazio, ou ats não suportado?)"
        print(f"Nenhuma vaga em {label}{hint}.")
        return 0 if raw else 1

    fetched = len(postings)
    # findapprenticeship filters location server-side (geo), so don't re-apply
    # the client substring filter — it'd wrongly drop Croydon/Westminster/etc.
    client_loc = None if source == "findapprenticeship" else location
    filt = []
    if keywords:
        filt.append(("title=" if title_only else "keyword=") + "+".join(keywords))
    if location:
        filt.append(f"location~{location}" if client_loc else f"geo~{location}")
    postings = ats.filter_postings(postings, keywords=keywords,
                                   location=client_loc, title_only=title_only)
    filt_desc = (" [" + ", ".join(filt) + "]") if filt else ""

    if not postings:
        print(f"{label}: 0 de {fetched} vagas casaram o filtro{filt_desc}.")
        return 0

    new, total = get_log().upsert(postings)
    # nearest first (jobs without a distance fall to the end)
    postings.sort(key=lambda p: (p.distance is None, p.distance or 0.0))
    print(f"✓ {label}: {total} casaram de {fetched}{filt_desc} — "
          f"{new} novas. Base agora: {get_log().count()} vagas.")
    print(f"=== mais perto → mais longe (top {min(limit, len(postings))}) ===")
    for p in postings[:limit]:
        dist = f" [{p.distance:.1f} mi]" if p.distance is not None else ""
        loc = f" — {p.location}" if p.location else ""
        print(f"  • {p.title}{dist}{loc}")
        print(f"    {p.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
