#!/usr/bin/env python3
"""
Import running workouts from an Apple Health export into the Fitness Engine.

How to get the export (on the iPhone):
  Saúde (Health) → toque na foto do perfil (canto sup. direito) →
  "Exportar Todos os Dados de Saúde" → gera um export.zip → AirDrop/copie
  pro Mac.

Run:
  ./venv/bin/python scripts/import_apple_health.py ~/Downloads/export.zip
  (aceita o .zip direto, ou o export.xml já extraído)

Idempotent: re-running skips runs already imported (dedup by timestamp).
Only running workouts are imported (Apple Health has no per-exercise sets/reps
for strength), and everything stays local (~/.cortex/fitness.db).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.apple_health import import_export  # noqa: E402
from src.integrations.fitness import get_log  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: import_apple_health.py <caminho/export.zip ou export.xml>")
        return 1
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"arquivo não encontrado: {path}")
        return 1
    try:
        stats = import_export(path, get_log())
    except Exception as e:  # noqa: BLE001 — CLI: surface a friendly error
        print(f"erro ao importar: {e}")
        return 1
    if stats["failed"]:
        print(
            f"Corridas: {stats['imported']} importadas, "
            f"{stats['skipped']} já existiam, {stats['failed']} não consegui gravar "
            f"(de {stats['total']} encontradas no export)."
        )
        return 1
    print(
        f"Corridas: {stats['imported']} importadas, "
        f"{stats['skipped']} já existiam "
        f"(de {stats['total']} encontradas no export)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
