#!/usr/bin/env python3
"""
Add (or update) a contact, with info — the path for filling the contact book
by hand. The chat/voice command only captures name+email; this CLI also takes
phone + free-text notes, and upserts (updates an existing name in place).

The contact book feeds name→email resolution (sending email "to João") AND the
entity brain ("com quem andei falando?" shows real names instead of addresses),
so it's worth populating.

Usage:
    venv/bin/python scripts/add_contact.py --name "João Silva" --email joao@x.com \
        --phone "+44 7..." --notes "colega do trabalho; cuida do projeto X"

    venv/bin/python scripts/add_contact.py --list          # show all contacts

Stored in ~/.cortex/contacts.db (override with CORTEX_CONTACTS_DB or --db).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.contacts import ContactBook  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Add or update a contact (with info).")
    p.add_argument("--name", help="contact name (the key)")
    p.add_argument("--email", help="email address")
    p.add_argument("--phone", default=None, help="phone (optional)")
    p.add_argument("--notes", default=None, help="free-text info about the person (optional)")
    p.add_argument("--db", default=None, help="contacts DB path (default ~/.cortex/contacts.db)")
    p.add_argument("--list", action="store_true", help="list all contacts and exit")
    args = p.parse_args(argv)

    book = ContactBook(db_path=args.db)

    if args.list:
        rows = book.list_all()
        if not rows:
            print("Nenhum contato ainda.")
            return 0
        print(f"{len(rows)} contato(s):")
        for c in rows:
            extra = []
            if c.phone:
                extra.append(c.phone)
            if c.notes:
                extra.append(c.notes)
            tail = f"  — {' · '.join(extra)}" if extra else ""
            print(f"  • {c.name} <{c.email}>{tail}")
        return 0

    if not args.name or not args.email:
        p.error("--name and --email are required (or use --list)")

    # Upsert: add; if the name already exists, update it in place.
    if book.add(args.name, args.email, phone=args.phone, notes=args.notes):
        print(f"✓ Contato adicionado: {args.name} <{args.email}>")
        return 0

    if book.lookup(args.name) is not None:
        ok = book.update(args.name, email=args.email, phone=args.phone, notes=args.notes)
        if ok:
            print(f"✓ Contato atualizado: {args.name} <{args.email}>")
            return 0
        print(f"✗ Não consegui atualizar {args.name}.")
        return 1

    print(f"✗ Não consegui adicionar {args.name} (DB indisponível?).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
