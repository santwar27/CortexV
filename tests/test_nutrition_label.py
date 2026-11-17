"""Nutrition-label parser tests (Fitness Engine slice 3.5, s76).

Two layout families are covered:
  - single-column "por porção" labels where each nutrient + its value share a
    line (the old PT-BR format), and
  - multi-column labels where the nutrient NAMES are one block and the VALUES
    another (the 2023 EN/Anvisa format), associated by order.
The EN/FR fixture below is the RAW Apple-Vision OCR of a real UK crisps packet
(s76 cont.36), kept verbatim as a regression case.
"""
import pytest

from src.integrations.nutrition_label import parse_label, to_per_100g


# --- single-column "por porção" (old PT-BR layout) -----------------------
_OCR = [
    "INFORMACAO NUTRICIONAL",
    "Porcao de 30 g (2 col. sopa)",
    "Valor energetico 140 kcal = 588 kJ",
    "Carboidratos 18 g",
    "Proteinas 3,5 g",
    "Gorduras totais 6 g",
    "Gorduras saturadas 1,5 g",
]


# --- multi-column EN/FR (real OCR of a UK label, two value columns) -------
_EN_OCR = [
    "sea salted beetroot,",
    "parsnip and sweet",
    "potato crisps",
    "NUTRITION",
    "Serves/Portions: 4",
    "Typical values",
    "per",
    "100g",
    "Valeurs moyennes",
    "Energy kJ/Énergie",
    "Energy kcal/Énergie",
    "Fat/Matières grasses",
    "of which saturates",
    "2214",
    "534",
    "38.1 g",
    "3.0g",
    "dont acides gras saturés",
    "Carbohydrate/Glucides",
    "of which sugars",
    "36.3 g",
    "21.1 g",
    "dont sucres",
    "Fibre/Fibres alimentaires 11.5g",
    "Protein/Protéines",
    "Salt/Sel",
    "5.6 g",
    "0.98 g",
    "Reference intake (adult)",
    "Energy 8400kJ/2000kcal Fat 70g",
    "Saturates 20g Sugars 90g Salt 6g",
    "STORAGE",
    "per",
    "25g",
    "serving",
    "554",
    "134",
    "9.5g",
    "0.8g",
    "9.1 g",
    "5.3 g",
    "2.9 g",
    "1.4g",
    "0.25g",
]


def test_parse_label_basic():
    p = parse_label(_OCR)
    assert p["serving_g"] == 30.0 and p["serving_unit"] == "g"
    assert p["kcal"] == 140.0          # 140 kcal, not 588 kJ
    assert p["carb"] == 18.0
    assert p["protein"] == 3.5         # comma decimal
    assert p["fat"] == 6.0             # totais, not saturadas (1,5)


def test_parse_label_accents_and_ml():
    p = parse_label(["Porção de 200 ml", "Valor energético 90 kcal", "Proteínas 7 g"])
    assert p["serving_g"] == 200.0 and p["serving_unit"] == "ml"
    assert p["kcal"] == 90.0 and p["protein"] == 7.0


def test_parse_label_missing_fields_are_none():
    p = parse_label(["um texto qualquer", "sem nutrientes"])
    assert p["kcal"] is None and p["carb"] is None and p["serving_g"] is None


def test_to_per_100g_scales_from_serving():
    h = to_per_100g(parse_label(_OCR), per_serving=True)
    assert h["kcal"] == 466.7          # 140 × 100/30
    assert h["carb"] == 60.0           # 18 × 100/30
    assert h["protein"] == 11.7        # 3.5 × 100/30
    assert h["fat"] == 20.0            # 6 × 100/30


def test_to_per_100g_already_per_100():
    h = to_per_100g(parse_label(_OCR), per_serving=False)
    assert h["kcal"] == 140.0 and h["carb"] == 18.0   # no scaling


def test_to_per_100g_no_energy_returns_none():
    assert to_per_100g({"kcal": None, "serving_g": 30}, per_serving=True) is None


# --- multi-column EN/FR ---------------------------------------------------

def test_parse_label_en_multicolumn():
    """Real UK label: EN keywords, names/values in separate blocks, per-100g."""
    p = parse_label(_EN_OCR)
    assert p["kcal"] == 534.0          # the kcal next to kJ, not 2214 (kJ)
    assert p["fat"] == 38.1
    assert p["carb"] == 36.3
    assert p["protein"] == 5.6


def test_parse_label_en_ignores_reference_intake():
    """The '2000 kcal' / '70g fat' reference-intake footer must be dropped."""
    p = parse_label(_EN_OCR)
    assert p["kcal"] != 2000.0
    assert p["fat"] != 70.0
    assert p["protein"] != 90.0        # not the '90g Sugars' reference line


def test_parse_label_en_does_not_grab_serving_column():
    """The second (per 25 g) value column lives past the footer → ignored."""
    p = parse_label(_EN_OCR)
    assert p["kcal"] not in (134.0, 554.0)   # 25 g serving column


def test_parse_label_en_detects_per_100g_basis():
    p = parse_label(_EN_OCR)
    assert p["basis"] == "100g"
    # with a per-100g basis the values are used as-is (no scaling)
    h = to_per_100g(p, per_serving=False)
    assert h["kcal"] == 534.0 and h["fat"] == 38.1


def test_parse_label_en_product_name_salt_not_a_nutrient():
    """'sea salted' above the table must not be read as the Salt nutrient."""
    p = parse_label(_EN_OCR)
    # salt isn't stored, but the bug would have mis-aligned everything; the
    # real macros surviving is the proof it didn't.
    assert p["kcal"] == 534.0 and p["protein"] == 5.6


# --- multi-column PT (synthetic; refine against a real Anvisa photo) ------

def test_parse_label_pt_multicolumn_names_then_values():
    """PT keywords in the split layout (names block then values block)."""
    p = parse_label([
        "INFORMACAO NUTRICIONAL",
        "Porcao de 30 g",
        "Valor energetico",
        "Carboidratos",
        "Proteinas",
        "Gorduras totais",
        "150",
        "20",
        "4",
        "5",
    ])
    assert p["serving_g"] == 30.0
    assert p["kcal"] == 150.0
    assert p["carb"] == 20.0
    assert p["protein"] == 4.0
    assert p["fat"] == 5.0


# --- multi-column PT-BR Anvisa-2023 (real OCR, interleaved name/value) -----
_PT_OCR = [
    "INFORMAÇÃO NUTRICIONAL",
    "Porções por embalagem: 3",
    "Porção: 25 g (1/2 xícara)",
    "100 g",
    "Valor energético (kcal)",
    "506",
    "Carboidratos (g)",
    "65",
    "Açúcares totais (g)",
    "2.1",
    "Açúcares adicionados (g)",
    "1",
    "Proteinas (g)",
    "6.4",
    "Gorduras totais (g)",
    "24",
    "Gorduras saturadas (g)",
    "3.6",
    "Gorduras trans (g)",
    "0,3",
    "Fibras alimentares (g)",
    "2,1",
    "Sódio (mg)",
    "575",
    "*Percentual de valores diários fornecidos pela porção",
    "25 g",
    "125",
    "16",
    "0,5",
    "0,2",
    "1,6",
    "6",
    "0,9",
    "0,1",
    "0,5",
    "144",
    "%VD*",
    "6",
    "5",
    "0",
    "3",
    "9",
    "5",
    "5",
    "2",
    "7",
]


def test_parse_label_pt_anvisa_2023_real():
    """Real Doritos label: PT keywords, 3 columns, interleaved name/value."""
    p = parse_label(_PT_OCR)
    assert p["serving_g"] == 25.0 and p["serving_unit"] == "g"
    assert p["basis"] == "100g"
    assert p["kcal"] == 506.0          # per-100g column, not 125 (per 25 g)
    assert p["carb"] == 65.0
    assert p["protein"] == 6.4
    assert p["fat"] == 24.0            # totais, not saturadas/trans


def test_parse_label_pt_anvisa_ignores_other_columns():
    """The per-serving (25 g) and %VD columns sit past the footer → dropped."""
    p = parse_label(_PT_OCR)
    assert p["kcal"] not in (125.0, 144.0)   # per-serving / %VD energy
    assert p["carb"] not in (16.0, 5.0)      # per-serving / %VD carb


def test_parse_label_en_inline_single_column():
    """EN keywords also work when name + value share a line."""
    p = parse_label([
        "Nutrition",
        "Energy 250 kcal",
        "Fat 12 g",
        "Carbohydrate 30 g",
        "Protein 9 g",
    ])
    assert p["kcal"] == 250.0 and p["fat"] == 12.0
    assert p["carb"] == 30.0 and p["protein"] == 9.0


# --- the script that feeds this parser -------------------------------------

def test_read_label_script_imports():
    """scripts/read_label.py imported an OCR function that the s83 PDF work
    deleted; nothing imported the script, so it stayed broken for two weeks
    (s85). Importing it is the whole check."""
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "read_label.py"
    spec = importlib.util.spec_from_file_location("read_label", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_ocr_image_is_fail_soft(tmp_path):
    from src.tools.ocr import ocr_image
    assert ocr_image(str(tmp_path / "no_such_photo.jpg")) == []


class TestEnergyKeepsItsUnit:
    """"Energy" then "420 kJ" with no kcal line was stored as 420 kcal — the
    value's unit was dropped (s88, Codex lote 11 — invariant 1)."""

    @pytest.mark.parametrize("lines", [
        ["Nutrition", "per 100 g", "Energy", "420 kJ", "Fat", "3.0 g",
         "Carbohydrate", "60 g", "Protein", "8 g"],
        ["Nutrition", "per 100 g", "Energy 420 kJ", "Fat 3.0 g",
         "Carbohydrate 60 g", "Protein 8 g"],
    ])
    def test_kj_only_is_converted_and_says_so(self, lines):
        p = parse_label(lines)
        assert p["kcal"] == 100.4 and p["kcal_from_kj"] is True
        assert (p["fat"], p["carb"], p["protein"]) == (3.0, 60.0, 8.0)

    def test_kj_then_kcal_under_one_row_takes_the_kcal(self):
        p = parse_label(["Nutrition", "Energy", "Fat", "Carbohydrate",
                         "420 kJ", "100 kcal", "3 g", "60 g"])
        assert (p["kcal"], p["fat"], p["carb"]) == (100.0, 3.0, 60.0)
        assert p["kcal_from_kj"] is False


class TestANutrientNotReadIsNotZero:
    """The OCR missed the protein line: protein=None became 0.0 and was
    registered, under-counting every later meal (s88, Codex lote 11)."""

    FULL = {"kcal": 405.0, "protein": 8.0, "carb": 60.0, "fat": 3.0,
            "serving_g": None, "basis": "100g"}

    def test_a_missing_macro_stops_the_conversion(self):
        from src.integrations.nutrition_label import missing_macros
        parsed = {**self.FULL, "protein": None}
        assert missing_macros(parsed) == ["protein"]
        assert to_per_100g(parsed, per_serving=False) is None

    def test_a_zero_printed_on_the_label_passes(self):
        parsed = {**self.FULL, "protein": 0.0, "carb": 0.0}
        assert to_per_100g(parsed, per_serving=False)["protein"] == 0.0

    def test_read_label_registers_nothing(self, monkeypatch, tmp_path, capsys):
        import importlib
        import sys
        rl = importlib.import_module("scripts.read_label")
        img = tmp_path / "label.jpg"
        img.write_bytes(b"x")
        monkeypatch.setattr(rl, "ocr_image", lambda p: [
            "Nutrition", "per 100 g", "Energy 405 kcal", "Fat 3 g", "Carbohydrate 60 g"])
        added = []
        monkeypatch.setattr(rl, "get_log", lambda: type("L", (), {
            "add_food": lambda self, *a, **k: added.append(a) or True})())
        monkeypatch.setattr(sys, "argv", ["read_label.py", str(img), "granola"])
        assert rl.main() == 1
        assert added == []
        assert "Não li proteína" in capsys.readouterr().out


@pytest.mark.parametrize("lines,want", [
    (["Nutrition", "per 100 g", "Energy", "1700 kJ / 405 kcal", "Fat", "3.0 g",
      "Carbohydrate", "60 g", "Protein", "8 g"], (405.0, 3.0, 60.0, 8.0)),
    (["Nutrition", "Energy", "Fat", "Carbohydrate", "Protein",
      "1700kJ/405kcal", "3 g", "60 g", "8 g"], (405.0, 3.0, 60.0, 8.0)),
])
def test_a_combined_energy_value_line_keeps_the_rows_aligned(lines, want):
    """"1700 kJ / 405 kcal" on a line of its own was dropped as noise and the
    rows below moved up: kcal read as the fat's 3.0 (s88)."""
    p = parse_label(lines)
    assert (p["kcal"], p["fat"], p["carb"], p["protein"]) == want
