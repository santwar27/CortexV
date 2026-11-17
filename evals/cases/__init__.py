"""Eval cases — one file per category.

The runner walks this package, importing every submodule and
concatenating their `CASES` lists. Adding a case = adding a list
entry in the relevant file (or creating a new file if the category
is brand-new).
"""
