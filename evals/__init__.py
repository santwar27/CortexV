"""Cortex behavioural eval framework.

Sister of `tests/`, but with a different contract:
  - `tests/` is unit/integration — every test MUST pass on every
    commit. Failure blocks merge.
  - `evals/` is behavioural — "when the user says X, the system
    produces Y". A failure here means behaviour drifted, which is
    sometimes intentional (prompt v2 changed answer style) and
    sometimes a regression. The runner reports drift; the human
    decides if it's a regression or an accepted evolution.

Run:
  venv/bin/python -m evals                  # all cases
  venv/bin/python -m evals --category spotify
  venv/bin/python -m evals --verbose
  venv/bin/python -m evals --json out.json

See `evals/README.md` for how to add a case.
"""
