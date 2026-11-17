"""A SQLite connection that never leaves a failed write's transaction open.

The stores shared between processes (fitness, jobs) wrote with execute(...)
then commit(), and returned fail-soft on an error without a rollback: a
statement failing halfway through a batch left the transaction open on the
store's long-lived connection, and the API, the terminal and the launchd
jobs got "database is locked" (s88, Codex lote 11, reproduced — invariant 6,
the s86 FocusStore defect). With this factory any failing statement rolls
the transaction back before the error propagates, so a batch is all or
nothing and the lock is released.
"""
from __future__ import annotations

import sqlite3


class RollbackOnError(sqlite3.Connection):
    def _guard(self, call, *args):
        try:
            return call(*args)
        except BaseException:
            if self.in_transaction:
                try:
                    self.rollback()
                except sqlite3.Error:
                    pass
            raise

    def execute(self, *args):
        return self._guard(super().execute, *args)

    def executemany(self, *args):
        return self._guard(super().executemany, *args)

    def executescript(self, *args):
        return self._guard(super().executescript, *args)

    def commit(self):
        return self._guard(super().commit)
