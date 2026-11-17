"""Bounded in-memory dialog state keyed by the current turn's origin."""
from collections import OrderedDict
import threading


_LOCK = threading.Lock()
_MAX_SCOPES = 64


class PerConversation:
    """Store a value per origin; callable defaults create fresh containers."""

    def __init__(self, name, default):
        self.name = name
        self.default = default

    def _scope(self, instance):
        """Called under the lock, including on instances built with __new__."""
        origin = instance._turn_origin()
        key = ((origin["channel"], origin.get("conversation"))
               if origin is not None else None)
        scopes = instance.__dict__.setdefault("_dialog_scopes", OrderedDict())
        if key not in scopes:
            scopes[key] = {}
        scopes.move_to_end(key)
        # Scripts keep their legacy scope; eviction can only forget permissions.
        if len(scopes) - (None in scopes) > _MAX_SCOPES:
            del scopes[next(k for k in scopes if k is not None)]
        return scopes[key]

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        with _LOCK:
            scope = self._scope(instance)
            if self.name not in scope:
                scope[self.name] = self.default() if callable(self.default) else self.default
            return scope[self.name]

    def __set__(self, instance, value):
        with _LOCK:
            self._scope(instance)[self.name] = value
