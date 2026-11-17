"""
Parse a nutrition table ("Informação Nutricional" / "Nutrition") from OCR text
lines (Fitness Engine slice 3.5, s76).

Deterministic, on-device, accent-insensitive. Handles two real layout families:

  1. **Single-column** ("por porção" / inline): each nutrient and its value
     share a line — "Carboidratos 18 g". Matched by keyword + nearest number.

  2. **Multi-column** (2023 EN/Anvisa): the nutrient NAMES are one block and
     the VALUES another, one token per OCR line. We associate them by ORDER —
     the N-th value is the N-th nutrient — keeping "skip" slots for the
     sub-nutrients (kJ, saturates, sugars, fibre, salt) so the alignment holds.

Quirks tamed against a real UK label (s76 cont.36): bilingual EN/FR rows (the
French lines are dropped as noise), a "Reference intake" footer whose "2000
kcal" must NOT be read as the food's energy (we stop at it), a product name
("sea salted…") above the table that would otherwise trip the salt keyword (we
only start at the table header), and a second per-serving value column past the
footer (ignored). Keywords are EN+PT; bilingual labels carry EN, so FR/ES come
for free off the EN token.

Returns {name, serving_g, serving_unit, kcal, protein, carb, fat, basis} with
None where a field wasn't found. `basis` is "100g" when the table is labelled
per 100 g (so the caller skips per-serving scaling), else None.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

_NUM = r"(\d+(?:[.,]\d+)?)"

# A line is a bare VALUE if it's just a number (+ optional unit). "100g" is
# caught earlier as a column header; real macro values carry g/ml/kj/kcal or
# nothing.
_VALUE_RE = re.compile(r"^\s*" + _NUM + r"\s*(g|ml|kg|mg|kj|kcal)?\s*$")
# kcal anchored on its unit, so "140 kcal = 588 kJ" yields 140, not 588.
_KCAL_RE = re.compile(_NUM + r"\s*kcal")
# Combined energy on one line ("2361kJ/564kcal") → keep the kcal number.
_COMBINED_ENERGY_RE = re.compile(_NUM + r"\s*kj\s*/?\s*" + _NUM + r"\s*kcal")

# The nutrition table starts at one of these; anything above (product name,
# branding) is skipped so e.g. "sea salted" isn't read as the Salt row.
_START_MARKERS = (
    "informacao nutricional", "tabela nutricional", "nutricional",
    "nutrition", "typical values",
)
# Everything from a reference-intake / daily-value footer on is dropped — its
# "2000 kcal", "70g fat" are guidance, not the food's macros.
_STOP_MARKERS = (
    "reference intake", "ingestao de referencia", "valores diarios",
    "valor diario de referencia",
)


def _norm(s: str) -> str:
    """Lowercase + strip accents, for accent-insensitive matching."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _to_f(raw: str) -> float:
    return float(raw.replace(",", "."))


def _slot(n: str) -> Optional[Tuple[str, bool]]:
    """Map a nutrient NAME line to (field, store?), or None if not a nutrient.

    store=False means the slot still consumes a value position (to keep the
    multi-column alignment) but the value isn't recorded — sub-nutrients,
    kilojoules, salt. Order matters where keywords overlap (saturates before
    fat; carb before sugars)."""
    if "energ" in n:
        if "kcal" in n:
            return ("kcal", True)
        if "kj" in n:
            return ("_kj", False)
        return ("kcal", True)          # combined "NkJ/Mkcal" value, kcal pulled out
    if "satura" in n:                  # of which saturates / saturadas
        return ("_saturates", False)
    if "trans" in n:
        return ("_trans", False)
    if "fat" in n or "gordura" in n or "lipid" in n:
        return ("fat", True)
    if "carbo" in n or "glucid" in n:  # carbohydrate / carboidrato / glucides
        return ("carb", True)
    if "sugar" in n or "acucar" in n:  # of which sugars / açúcares
        return ("_sugars", False)
    if "fibr" in n:                    # fibre / fiber / fibra
        return ("_fibre", False)
    if "protein" in n:                 # protein / proteína / protéine
        return ("protein", True)
    if "salt" in n or "sodi" in n or n.strip() == "sal":
        return ("_salt", False)
    return None


def _inline_value(n: str, field: str) -> Optional[float]:
    """Value on the same line as its keyword. kcal is unit-anchored."""
    if field == "kcal":
        m = _KCAL_RE.search(n)
        if m:
            return _to_f(m.group(1))
    m = re.search(_NUM, n)
    return _to_f(m.group(1)) if m else None


def _value_number(n: str) -> Optional[float]:
    """The number carried by a bare value line (kcal kept for combined energy)."""
    m = _COMBINED_ENERGY_RE.search(n)
    if m:
        return _to_f(m.group(2))
    m = re.search(_NUM, n)
    return _to_f(m.group(1)) if m else None


def _serving(norm_lines: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """Find the serving size in grams/ml — inline forms only ("porção de 30 g",
    "each pack (375 g)", "30 g serving"). The split "per / 25g / serving" OCR
    layout isn't reconstructed; per-100g labels don't need it."""
    pats = (
        r"porc[ao]+\D*?" + _NUM + r"\s*(g|ml|gramas?)",
        r"each pack\D*?" + _NUM + r"\s*(g|ml)",
        r"serving\D{0,6}?" + _NUM + r"\s*(g|ml)",
        _NUM + r"\s*(g|ml)\s*serving",
    )
    for n in norm_lines:
        for p in pats:
            m = re.search(p, n)
            if m:
                unit_raw = m.group(2) if len(m.groups()) >= 2 else "g"
                return _to_f(m.group(1)), ("ml" if unit_raw.startswith("ml") else "g")
    return None, None


def parse_label(lines: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "name": None, "serving_g": None, "serving_unit": None,
        "kcal": None, "protein": None, "carb": None, "fat": None, "basis": None,
        "kcal_from_kj": False,
    }
    # Energy read in kJ is kept apart and never taken as kcal: "Energy" then
    # "420 kJ" was stored as 420 kcal (s88, Codex lote 11 — invariant 1). A
    # label with no kcal at all is converted exactly (1 kcal = 4.184 kJ) and
    # says so.
    kj: List[float] = []
    norm = [_norm(x) for x in lines]
    out["serving_g"], out["serving_unit"] = _serving(norm)

    # Skip the product-name region: start at the table header if we can find it.
    start = 0
    for i, n in enumerate(norm):
        if any(mk in n for mk in _START_MARKERS):
            start = i
            break

    pend_names: List[Tuple[str, bool]] = []
    pend_vals: List[Tuple[Optional[float], str]] = []
    seen_name = False

    def flush() -> None:
        names, vals = list(pend_names), list(pend_vals)
        energy = [(v, u) for v, u in vals if u in ("kj", "kcal")]
        if energy and any(f in ("kcal", "_kj") for f, _ in names):
            # Energy values carry their unit — "420 kJ" then "100 kcal" under
            # one Energy row: each goes by its unit, and only the rest align by
            # position, so an extra energy line can't shift fat onto carb.
            for v, u in energy:
                if v is None:
                    continue
                if u == "kcal" and out["kcal"] is None:
                    out["kcal"] = v
                elif u == "kj":
                    kj.append(v)
            names = [nm for nm in names if nm[0] not in ("kcal", "_kj")]
            vals = [vu for vu in vals if vu[1] not in ("kj", "kcal")]
        for (field, store), (val, unit) in zip(names, vals):
            if field in ("kcal", "_kj") and unit == "kj":
                if val is not None:
                    kj.append(val)
                continue
            if store and out[field] is None:
                out[field] = val
        pend_names.clear()
        pend_vals.clear()

    for n in norm[start:]:
        if any(mk in n for mk in _STOP_MARKERS):
            flush()
            break

        # per-100g column header (also disambiguates a stray "100g" line).
        if n in ("100g", "100 g") or "per 100" in n or "por 100" in n:
            out["basis"] = "100g"
            continue

        slot = _slot(n)
        if slot is not None:
            has_num = re.search(_NUM, n) is not None
            if has_num:                         # inline name+value (single column)
                flush()
                field, store = slot
                if field == "_kj":
                    val = _inline_value(n, field)
                    if val is not None:
                        kj.append(val)
                elif store and out[field] is None:
                    out[field] = _inline_value(n, field)
            else:                               # name-only (multi-column)
                if pend_vals:                   # a new name after values → flush group
                    flush()
                pend_names.append(slot)
            seen_name = True
            continue

        vm = _VALUE_RE.match(n) if seen_name else None
        if vm:                                  # value-only line in a column
            pend_vals.append((_value_number(n), (vm.group(2) or "").lower()))
            continue
        # "1700 kJ / 405 kcal" as a value line of its own: not a bare value, so
        # it was dropped as noise and the values below it moved up a row —
        # the fat's 3.0 read as kcal (s88, found checking Codex lote 11).
        cm = _COMBINED_ENERGY_RE.fullmatch(n.strip()) if seen_name else None
        if cm:
            pend_vals.append((_to_f(cm.group(1)), "kj"))
            pend_vals.append((_to_f(cm.group(2)), "kcal"))
            continue
        # else: noise (FR rows, descriptions, pre-table strays) — ignore
    else:
        flush()

    if out["kcal"] is None and kj:
        out["kcal"] = round(kj[0] / 4.184, 1)
        out["kcal_from_kj"] = True
    return out


def missing_macros(parsed: Dict[str, Optional[float]]) -> List[str]:
    """The macros the label did not yield — None, as opposed to a 0 read."""
    return [k for k in ("protein", "carb", "fat") if parsed.get(k) is None]


def to_per_100g(parsed: Dict[str, Optional[float]],
                per_serving: bool = True) -> Optional[Dict[str, float]]:
    """Turn parsed macros into per-100g values for add_food.

    `per_serving=True`  → values are per the serving size; scale by
    100/serving_g (needs serving_g). `per_serving=False` → values are already
    per 100 g (the 2023 Anvisa column); used as-is. Returns None if there's no
    energy value, or if a macro wasn't read (missing_macros): a protein line
    the OCR missed became protein 0 in the food table, under-counting every
    meal after it (s88, Codex lote 11). A "0 g" printed on the label is read
    as 0.0 and passes — only what was not read at all stops it."""
    if parsed.get("kcal") is None or missing_macros(parsed):
        return None

    def v(key: str) -> float:
        return float(parsed[key])

    serving = parsed.get("serving_g")
    if per_serving and serving:
        factor = 100.0 / serving
    else:
        factor = 1.0
    return {
        "kcal": round(v("kcal") * factor, 1),
        "protein": round(v("protein") * factor, 1),
        "carb": round(v("carb") * factor, 1),
        "fat": round(v("fat") * factor, 1),
    }
