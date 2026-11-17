#!/usr/bin/env python3
"""
Read a nutrition-label photo (Apple Vision OCR) and register the food.

Usage:
  ./venv/bin/python scripts/read_label.py <imagem> ["nome"] [--per100g] [--dry-run]

OCRs the "Informação Nutricional" table ON-DEVICE (no cloud), extracts the
macros, and saves the food in the Fitness Engine's local food table (per
100 g) so "comi 1 <nome>" / "comi Ng de <nome>" works afterwards. Prints the
RAW OCR text so the parser can be tuned to your real labels.

The parser auto-detects whether the table is per-100g or per-serving. Override
with --per100g if it guesses wrong (forces "already per 100 g", no scaling).
Pass --dry-run to OCR + parse + print WITHOUT touching the food table — handy
while calibrating the parser against new labels.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.ocr import ocr_image  # noqa: E402
from src.integrations.nutrition_label import (  # noqa: E402
    missing_macros, parse_label, to_per_100g,
)
from src.integrations.fitness import get_log  # noqa: E402


def main() -> int:
    args = list(sys.argv[1:])
    flag_per100 = "--per100g" in args
    flag_dry = "--dry-run" in args
    args = [a for a in args if a not in ("--per100g", "--dry-run")]
    if not args:
        print('uso: read_label.py <imagem> ["nome"] [--per100g] [--dry-run]')
        return 1
    image = args[0]
    name = args[1] if len(args) > 1 else Path(image).stem
    if not Path(image).exists():
        print(f"imagem não encontrada: {image}")
        return 1

    lines = ocr_image(image)
    print("=== OCR (texto lido) ===")
    for ln in lines:
        print("  ", ln)
    if not lines:
        print("Não li nenhum texto. Tenta uma foto mais nítida / reta / bem iluminada.")
        return 1

    parsed = parse_label(lines)
    su = parsed["serving_unit"] or "g"
    print("=== extraído ===")
    print(f"  porção: {parsed['serving_g']} {su if parsed['serving_g'] else ''}".rstrip())
    print(f"  kcal={parsed['kcal']}  proteína={parsed['protein']}  "
          f"carbo={parsed['carb']}  gordura={parsed['fat']}")
    if parsed.get("kcal_from_kj"):
        print("  (o rótulo só trazia kJ — kcal convertido: kJ ÷ 4,184)")

    # auto: if the table is labelled per-100g, don't rescale; --per100g forces it.
    is_per100 = flag_per100 or parsed.get("basis") == "100g"
    absent = missing_macros(parsed)
    if parsed["kcal"] is not None and absent:
        names = {"protein": "proteína", "carb": "carboidrato", "fat": "gordura"}
        print(f"Não li {', '.join(names[a] for a in absent)} no rótulo — não cadastrei "
              f"(gravar 0 baixaria os totais de toda refeição com esse alimento). "
              f"Tira outra foto, ou cadastra à mão com os valores do rótulo.")
        return 1
    per100 = to_per_100g(parsed, per_serving=not is_per100)
    if per100 is None:
        print("Não achei o valor energético (kcal) — não dá pra cadastrar. "
              "Confere a foto / o rótulo, ou usa --per100g.")
        return 1
    if is_per100:
        how = "já por 100g" + (" (--per100g)" if flag_per100 else " (auto)")
    elif parsed["serving_g"]:
        how = "por porção→/100g"
    else:
        how = "sem porção (assumido por 100g)"
    print(f"=== por 100g (pra base) [{how}] ===")
    print(f"  kcal={per100['kcal']}  proteína={per100['protein']}  "
          f"carbo={per100['carb']}  gordura={per100['fat']}")

    if flag_dry:
        print("=== dry-run: NÃO cadastrei (só leitura) ===")
        return 0

    unit_g = parsed["serving_g"]
    ok = get_log().add_food(name, per100["kcal"], per100["protein"],
                            per100["carb"], per100["fat"], unit_g=unit_g)
    if not ok:
        print("Não consegui cadastrar (erro no banco).")
        return 1
    srv = f" — 1 porção = {unit_g:g}{su}" if unit_g else ""
    print(f"✓ Cadastrei '{name}' na base{srv}.")
    print(f"  Agora: \"comi 1 {name}\" (1 porção) ou \"comi 50g de {name}\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
