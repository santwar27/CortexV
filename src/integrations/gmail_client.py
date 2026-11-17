"""
Gmail client wrapper.

Stateless: every call rebuilds the API client from stored credentials.
The Cortex orchestrator calls `list_recent` for read-only access via
regex shortcut ("meus emails") or tool calling. `send_email` is gated
by an explicit user-confirmation step in the orchestrator before it
gets called — this module just performs the API call.
"""

import base64
import hashlib
import logging
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.core import turn_metrics
from src.integrations.google_oauth import load_credentials

log = logging.getLogger(__name__)


class GmailUnavailable(Exception):
    """Raised when credentials are missing or the API rejects the call."""


def _service():
    """Build a Gmail API service from stored credentials. Raises if absent."""
    turn_metrics.mark_online("gmail")  # Credential loading may refresh over Google.
    creds = load_credentials()
    if not creds:
        raise GmailUnavailable(
            "Google credentials not set up. Run scripts/google_oauth_setup.py first."
        )
    # cache_discovery=False silences a noisy warning on first build
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def list_recent(count: int = 5, query: Optional[str] = None) -> List[Dict]:
    """
    Return the most recent emails as a list of dicts with the fields we
    actually want to surface: id, from, subject, snippet, date_iso.

    `query` follows Gmail's search syntax — e.g. "from:joao@x.com",
    "newer_than:1d", "label:UNREAD". When None, returns the inbox.
    """
    service = _service()
    try:
        resp = service.users().messages().list(
            userId="me",
            maxResults=max(1, min(count, 25)),
            q=query,
        ).execute()
    except HttpError as e:
        raise GmailUnavailable(f"Gmail list failed: {e}") from e

    msg_refs = resp.get("messages", []) or []
    if not msg_refs:
        return []

    results: List[Dict] = []
    for ref in msg_refs:
        try:
            msg = service.users().messages().get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
        except HttpError as e:
            log.warning(f"Gmail get failed for {ref['id']}: {e}")
            continue

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        results.append({
            "id": msg.get("id"),
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", "(sem assunto)"),
            "snippet": msg.get("snippet", ""),
            "date_iso": _parse_date_iso(headers.get("Date", "")),
        })
    return results


def count_matching(query: str, cap: int = 500) -> int:
    """How many messages match `query`, paging through ids only — no message
    is fetched. Stops at `cap`. Raises GmailUnavailable like list_recent."""
    service = _service()
    total, token = 0, None
    try:
        while total < cap:
            resp = service.users().messages().list(
                userId="me", q=query, maxResults=min(500, cap - total),
                pageToken=token,
            ).execute()
            total += len(resp.get("messages", []) or [])
            token = resp.get("nextPageToken")
            if not token:
                break
    except HttpError as e:
        raise GmailUnavailable(f"Gmail list failed: {e}") from e
    return min(total, cap)


def message_id_for(effect_id: str) -> str:
    """Choose an RFC-5322 Message-ID from the approval's effect id."""
    return f"<cortex.{hashlib.sha256(effect_id.encode()).hexdigest()[:32]}@cortex.local>"


def find_sent(message_id: str) -> Optional[Dict]:
    """Find evidence of a send; absence is not permission to send again."""
    try:
        result = _service().users().messages().list(
            userId="me", q=f"rfc822msgid:{message_id}", maxResults=1,
        ).execute()
    except HttpError as e:
        raise GmailUnavailable(f"Gmail lookup failed: {e}") from e
    messages = result.get("messages", []) or []
    return {"id": messages[0]["id"]} if messages else None


def send_email(to: str, subject: str, body: str, message_id: Optional[str] = None) -> Dict:
    """
    Send a plain-text email through the user's Gmail account.

    Returns a dict with the new message id and the destination. Should
    only be called after the orchestrator's confirmation flow has
    explicitly authorised this send.
    """
    if not to or "@" not in to:
        raise GmailUnavailable(f"Invalid recipient address: {to!r}")

    service = _service()
    message = MIMEText(body or "", _charset="utf-8")
    message["to"] = to
    message["subject"] = subject or ""
    if message_id is not None:
        message["Message-ID"] = message_id
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    confirmed = False
    try:
        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()
    except Exception as e:
        if isinstance(e, HttpError) and (message_id is None or e.resp.status < 500):
            raise GmailUnavailable(f"Gmail send failed: {e}") from e
        # A lost reply permits one mailbox check, never a second send.
        if message_id is not None:
            try:
                sent = find_sent(message_id)
                confirmed = sent is not None
            except Exception:
                pass
        if not confirmed:
            if isinstance(e, HttpError):
                error = GmailUnavailable(f"Gmail send failed: {e}")
                error.uncertain = True
                raise error from e
            raise

    result = {
        "id": sent.get("id", ""),
        "to": to,
        "subject": subject or "",
    }
    if confirmed:
        result["confirmed"] = True
    return result


def format_email_list(emails: List[Dict], lang: str = "pt") -> str:
    """Render a list of email dicts as a user-facing string. The rows are
    data (sender — subject), language-neutral; only the empty case carries a
    phrase, localised by `lang` (PT default)."""
    if not emails:
        return "No emails found." if lang == "en" else "Nenhum email encontrado."
    lines = []
    for i, e in enumerate(emails, start=1):
        sender = e.get("from", "")
        # Strip the "<address>" tail when there's a display name
        if "<" in sender:
            sender = sender.split("<")[0].strip().strip('"') or sender
        lines.append(
            f"{i}. {sender} — {e.get('subject', '')}"
        )
    return "\n".join(lines)


def _parse_date_iso(rfc2822: str) -> str:
    if not rfc2822:
        return ""
    try:
        dt = parsedate_to_datetime(rfc2822)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return rfc2822
