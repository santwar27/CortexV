"""Whole-reply corrections in otherwise incremental text streams."""


class ReplyReplacement(str):
    """Authoritative text replacing all preceding chunks, never an append.

    Engines may emit this after live tokens. Orchestrators carry it in the
    final tuple; transports use it for storage/final text, not another token.
    """
