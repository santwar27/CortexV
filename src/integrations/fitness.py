"""
Fitness log — local SQLite store for meals + macros (Fitness Engine v1, s76).

Slice 1 of Pedro's "Fitness & Meal Engine": log what he eats by voice/text and
get the macros, plus a running daily total. "comi 200g de frango e 2 ovos" →
look the foods up in a local table, compute kcal/protein/carb/fat, store the
meal, and report the meal + today's totals.

Why STRUCTURED SQLite and not the RAG/Palace: macros are NUMBERS you AGGREGATE
(sum the day, average the week). SQL does that exactly; an 8B summarising
retrieved chunks would fabricate the arithmetic. The vector store is reserved
for unstructured training NOTES (a later slice). See memory/ideas_backlog.md.

Design mirrors `contacts.py` / `cost_tracker.py`: a class with an injectable
path plus a module-level singleton. Errors NEVER propagate into the request
path — a missing/corrupt DB degrades to "couldn't log" rather than breaking
dispatch.

Privacy (SOBERANIA r1): the DB lives in ~/.cortex/fitness.db (env override
CORTEX_FITNESS_DB), outside the project tree. Health/meal data is among the
most personal there is — it is local forever, never vectorised into Palace,
never sent anywhere. The backup script picks up ~/.cortex; git never sees it.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import get_close_matches
from typing import Dict, List, Literal, Optional, Tuple

from src.core.sqlite_safe import RollbackOnError

log = logging.getLogger(__name__)

WriteResult = Literal["added", "duplicate", "invalid", "error"]


@dataclass(frozen=True)
class DeleteResult:
    removed: int = 0
    error: bool = False
    workout: Optional[dict] = None


# ----------------------------------------------------------------------
# Seed food table — per 100 g (approximate, common PT-BR foods). `unit_g`
# is the typical weight of ONE unit for countable foods (egg, banana,
# slice of bread), used to turn "2 ovos" into grams; NULL when a food is
# only ever weighed. Values are rounded standard references — good enough
# for day-to-day tracking, and every food is editable / extensible via
# add_food ("cadastra alimento X"). aliases help the matcher.
# (name, kcal, protein, carb, fat, unit_g, aliases)
# ----------------------------------------------------------------------
_SEED_FOODS: Tuple[Tuple[str, float, float, float, float, Optional[float], str], ...] = (
    # --- proteins --- (aliases carry common EN names so logging works in both)
    ("frango", 165, 31, 0, 3.6, None, "peito de frango,frango grelhado,chicken,chicken breast"),
    ("ovo", 143, 13, 1.1, 9.5, 50, "ovos,ovo inteiro,ovo cozido,egg,eggs"),
    ("clara de ovo", 52, 11, 0.7, 0.2, 33, "claras,clara,egg white,egg whites"),
    ("carne moída", 187, 27, 0, 9, None, "patinho,carne moida,carne,ground beef,beef"),
    ("alcatra", 168, 27, 0, 6, None, "bife,bife de alcatra,steak"),
    ("tilápia", 128, 26, 0, 2.7, None, "tilapia,peixe,fish"),
    ("salmão", 208, 20, 0, 13, None, "salmao,salmon"),
    ("atum", 116, 26, 0, 1, None, "atum em lata,tuna"),
    ("whey", 380, 80, 8, 6, 30, "whey protein,proteina,protein,suplemento"),
    ("iogurte natural", 61, 3.5, 4.7, 3.3, 170, "iogurte,yogurt,yoghurt"),
    ("iogurte grego", 97, 9, 4, 5, 100, "grego,greek yogurt"),
    ("queijo cottage", 98, 11, 3, 4, None, "cottage,cottage cheese"),
    ("queijo mussarela", 280, 28, 3, 17, 20, "mussarela,queijo,cheese,mozzarella"),
    ("leite", 61, 3.2, 4.8, 3.3, 200, "leite integral,leite desnatado,milk"),
    # --- carbs ---
    ("arroz", 130, 2.7, 28, 0.3, None, "arroz branco,arroz cozido,rice,white rice"),
    ("arroz integral", 123, 2.7, 26, 1, None, "brown rice"),
    ("feijão", 76, 4.8, 13.6, 0.5, None, "feijao,feijão cozido,beans,black beans"),
    ("batata", 87, 2, 20, 0.1, 150, "batata cozida,batata inglesa,potato"),
    ("batata doce", 86, 1.6, 20, 0.1, 150, "batata-doce,sweet potato"),
    ("macarrão", 158, 5.8, 31, 0.9, None, "macarrao,massa,espaguete,pasta,spaghetti"),
    ("pão francês", 300, 9, 58, 3, 50, "pao,pão,pãozinho,pao frances,bread"),
    ("pão de forma", 253, 9, 49, 3.5, 25, "pao de forma,fatia de pão,fatia de pao,torrada,toast,sliced bread"),
    ("aveia", 389, 17, 66, 7, None, "farelo de aveia,aveia em flocos,oats,oatmeal"),
    ("mandioca", 125, 1, 30, 0.3, None, "aipim,macaxeira,cassava"),
    ("tapioca", 240, 0, 60, 0, None, None),
    ("cuscuz", 112, 3.6, 23, 0.2, None, "cuscuz nordestino,couscous"),
    ("granola", 471, 10, 64, 20, None, None),
    # --- fruits ---
    ("banana", 89, 1.1, 23, 0.3, 120, "bananas"),
    ("maçã", 52, 0.3, 14, 0.2, 180, "maca,maçãs,apple"),
    ("laranja", 47, 0.9, 12, 0.1, 180, "laranjas,orange"),
    ("morango", 32, 0.7, 7.7, 0.3, None, "morangos,strawberry,strawberries"),
    ("abacate", 160, 2, 9, 15, 200, "avocado"),
    # --- fats / nuts ---
    ("azeite", 884, 0, 0, 100, None, "azeite de oliva,óleo,oleo,olive oil,oil"),
    ("pasta de amendoim", 588, 25, 20, 50, None, "amendoim,manteiga de amendoim,peanut butter,peanuts"),
    ("castanha", 656, 14, 12, 65, None, "castanha de caju,castanha do pará,nozes,amêndoas,nuts,cashew,almonds"),
    # --- veg ---
    ("brócolis", 34, 2.8, 7, 0.4, None, "brocolis,broccoli"),
    ("alface", 15, 1.4, 2.9, 0.2, None, "lettuce"),
    ("tomate", 18, 0.9, 3.9, 0.2, 100, "tomates,tomato"),
    ("cenoura", 41, 0.9, 10, 0.2, 80, "cenouras,carrot"),
    # --- s82 expansion: everyday items that came back "não reconheci" in real
    #     use. Values are standard per-100g reference figures (raw/cooked as
    #     noted by the name), not lab measurements — good enough for tracking
    #     trends, not for clinical precision.
    #     DELIBERATELY no composite dishes (sandwich, pizza, lasagne): the
    #     parser also reads the fillings the user lists, so "sanduíche com
    #     frango" would count the chicken twice. Composites need a recipe
    #     model, not a food row.
    ("milho", 86, 3.2, 19, 1.2, None, "milho verde,milho em conserva,sweetcorn,corn"),
    ("ervilha", 81, 5.4, 14, 0.4, None, "ervilhas,peas"),
    ("grão de bico", 164, 8.9, 27, 2.6, None, "grao de bico,chickpeas"),
    ("lentilha", 116, 9, 20, 0.4, None, "lentilhas,lentils"),
    ("quinoa", 120, 4.4, 21, 1.9, None, None),
    ("pepino", 15, 0.7, 3.6, 0.1, None, "pepinos,cucumber"),
    ("cebola", 40, 1.1, 9.3, 0.1, None, "cebolas,onion"),
    ("pimentão", 31, 1, 6, 0.3, None, "pimentao,pimenta,bell pepper,pepper"),
    ("abobrinha", 17, 1.2, 3.1, 0.3, None, "courgette,zucchini"),
    ("espinafre", 23, 2.9, 3.6, 0.4, None, "spinach"),
    ("couve", 49, 4.3, 8.8, 0.9, None, "couve manteiga,kale"),
    ("couve-flor", 25, 1.9, 5, 0.3, None, "couve flor,cauliflower"),
    ("repolho", 25, 1.3, 6, 0.1, None, "cabbage"),
    ("beterraba", 43, 1.6, 10, 0.2, None, "beetroot"),
    ("cogumelo", 22, 3.1, 3.3, 0.3, None, "cogumelos,champignon,mushroom,mushrooms"),
    ("presunto", 145, 18, 1.5, 7.5, 25, "ham"),
    ("peru", 135, 29, 0, 1.7, None, "peito de peru,turkey"),
    ("bacon", 541, 37, 1.4, 42, 20, None),
    ("linguiça", 301, 16, 1.9, 26, None, "linguica,sausage,sausages"),
    ("sardinha", 208, 25, 0, 11, None, "sardinhas,sardines"),
    ("camarão", 99, 24, 0.2, 0.3, None, "camarao,shrimp,prawns"),
    ("queijo cheddar", 402, 25, 1.3, 33, 30, "cheddar"),
    ("requeijão", 257, 11, 3, 22, None, "requeijao,cream cheese"),
    ("manteiga", 717, 0.9, 0.1, 81, 10, "butter"),
    ("pão integral", 247, 13, 41, 3.4, 40, "pao integral,wholemeal bread,brown bread"),
    ("tortilha", 306, 8, 51, 7.5, 60, "wrap,tortilla,tortillas"),
    ("polenta", 85, 2, 18, 0.4, None, None),
    ("inhame", 118, 1.5, 28, 0.2, None, "yam"),
    ("mel", 304, 0.3, 82, 0, 20, "honey"),
    ("açúcar", 387, 0, 100, 0, 5, "acucar,sugar"),
    ("chocolate", 546, 4.9, 61, 31, 25, "barra de chocolate,chocolate ao leite"),
    ("chia", 486, 17, 42, 31, 15, "semente de chia,chia seeds"),
    ("linhaça", 534, 18, 29, 42, 15, "linhaca,flaxseed"),
    ("abacaxi", 50, 0.5, 13, 0.1, 150, "pineapple"),
    ("manga", 60, 0.8, 15, 0.4, 200, "mango"),
    ("uva", 69, 0.7, 18, 0.2, 100, "uvas,grapes"),
    ("melancia", 30, 0.6, 7.6, 0.2, 200, "watermelon"),
    ("mamão", 43, 0.5, 11, 0.3, 150, "mamao,papaya"),
    ("pera", 57, 0.4, 15, 0.1, 170, "peras,pear"),
    ("kiwi", 61, 1.1, 15, 0.5, 75, "kiwis"),
    ("mirtilo", 57, 0.7, 14, 0.3, None, "mirtilos,blueberry,blueberries"),
    ("suco de laranja", 45, 0.7, 10, 0.2, 250, "suco,orange juice"),
    ("café", 2, 0.1, 0, 0, 200, "cafe,coffee,cafe preto"),
)


# Item separators. The digit lookarounds on "." and "," are load-bearing:
# a bare split on those would cut "1,5 kg de frango" into "1" and "5 kg de
# frango". s82 — the missing "." cost a real breakfast: "55 gramas de aveia.
# 100 gramas de yoghurt grego" stayed ONE chunk, so the item regex read qty=55
# and fed the whole remainder to the food lookup, which fuzzy-matched the
# LAST food named. The oats vanished and the yoghurt was logged at 55g
# instead of 100g — silently, because nothing landed in `unknown`.
_CONNECTORS_RE = re.compile(
    r"\s*(?:(?<!\d)[.,](?!\d)|;|\+|\be\b|\bmais\b|\bcom\b|\band\b|\bwith\b)\s*",
    re.IGNORECASE)
# qty + optional unit + optional "de/do/da" + food. Number may use , or . decimal.
_ITEM_RE = re.compile(
    r"^\s*(?P<qty>\d+(?:[.,]\d+)?)?\s*"
    r"(?P<unit>kg|gramas?|gr|g|ml|unidades?|unidade|un)?\s*"
    r"(?:de\s+|do\s+|da\s+|dos\s+|das\s+)?"
    r"(?P<food>.+?)\s*$",
    re.IGNORECASE,
)


# A household measure in front of the food (s88, Codex lote 11 — invariant 1):
# "2 colheres de azeite" matched azeite inside "colheres de azeite" and the 2
# became 2 g, 17.7 kcal, unmarked. A spoon is 5 g or 45 g depending on which
# and of what, so a measure is never turned into grams: the item is held back
# and the user is asked for the weight.
_MEASURE_RE = re.compile(
    r"^(?:(?:meia|meio|uma?|duas|dois|half(?:\s+a)?|an?)\s+)?"
    r"(?:colher(?:es|zinhas?)?|x[íi]caras?|copos?|fatias?|conchas?|pitadas?|"
    r"punhados?|peda[çc]os?|latas?|potes?|por[çc][õo]es|por[çc][ãa]o|tigelas?|"
    r"pratos?|cups?|tbsp|tsp|table\s*spoons?|tea\s*spoons?|spoons?|slices?|"
    r"glass(?:es)?|handfuls?|pieces?|cans?|bowls?|scoops?)\b",
    re.IGNORECASE,
)


# Typical serving in grams, applied when the user gives NO weight ("comi um
# sanduíche", "não sei o peso, faz na média"). s82 — before this, a weightless
# food silently became 100 g, which is absurd for rice and generous for butter.
# Kept as a side table rather than an 8th seed column so the food list stays
# readable and a portion can be tuned without touching macros.
#
# An applied portion is ALWAYS flagged (LoggedItem.assumed) and the reply says
# so: an assumed number must never be mistaken for a measured one.
_TYPICAL_PORTIONS = {
    # staples served by the plate, not the gram
    "arroz": 150, "arroz integral": 150, "feijão": 100, "macarrão": 180,
    "batata": 150, "batata doce": 150, "mandioca": 150, "polenta": 150,
    "inhame": 150, "cuscuz": 120, "quinoa": 120, "aveia": 40, "granola": 45,
    "tapioca": 60,
    # proteins — a typical serving, not a whole pack
    "frango": 150, "peru": 150, "alcatra": 150, "carne moída": 130,
    "tilápia": 150, "salmão": 130, "atum": 80, "sardinha": 80, "camarão": 100,
    "linguiça": 80, "queijo cottage": 100, "queijo mussarela": 30,
    "queijo cheddar": 30, "requeijão": 20, "pasta de amendoim": 20,
    "castanha": 30, "azeite": 10, "whey": 30,
    # veg / sides
    "brócolis": 80, "alface": 30, "espinafre": 60, "couve": 60, "repolho": 60,
    "couve-flor": 80, "cenoura": 80, "pepino": 50, "cebola": 40,
    "pimentão": 60, "abobrinha": 100, "beterraba": 60, "cogumelo": 60,
    "milho": 80, "ervilha": 80, "grão de bico": 120, "lentilha": 120,
    "tomate": 80,
    # fruit
    "morango": 100, "mirtilo": 80, "uva": 100,
}


# Words that mark a chunk as a CLAUSE, not a food ("nao sei o peso", "mas
# faça na media"). Reporting those as unrecognised foods is noise — the honest
# report is for things that look like food and weren't found ("um sanduiche"),
# not for the sentence the user wrapped around them. Kept tiny and literal:
# a big stoplist would start swallowing real foods.
_NON_FOOD_MARKERS = (
    "sei", "faz", "faça", "faca", "quero", "acho", "pode", "poderia",
    "obrigado", "por favor", "talvez", "mais ou menos", "please", "dunno",
    "don't know", "dont know", "make it", "guess",
)


# Meal NAMES are context, not food. s82 — "hoje eu comi no café da manhã; 55
# gramas de aveia…" logged **200 g de café**: the ";" made "no cafe da manha"
# its own chunk and the lookup happily matched the word "café" inside it. The
# system invented a food out of a label for WHEN the eating happened.
#
# Note what is NOT in here: bare "café". Coffee is a real drink someone logs
# ("200ml de café"), so only the full "café da manhã" phrase counts as a meal
# name. Same for "lanche" vs an actual snack food.
_MEAL_CONTEXT_RE = re.compile(
    r"^(?:n[oa]s?|d[oae]s?|pr[oa]|para\s+[oa]|a[oa]?|em|pel[oa]|"
    r"at|for|in(?:\s+the)?|during)?\s*"
    r"(?:cafe\s+da\s+manha|almoco|jantar|janta|lanche(?:\s+da\s+(?:tarde|noite))?|"
    r"ceia|breakfast|lunch|dinner|supper|brunch|"
    r"manha|tarde|noite|madrugada)"
    r"(?:\s+de)?(?:\s+(?:hoje|ontem|today|yesterday))?$",
    re.IGNORECASE,
)


def _looks_like_meal_context(chunk: str) -> bool:
    """True when the chunk names WHEN the meal happened, not WHAT was eaten."""
    return bool(_MEAL_CONTEXT_RE.match(_norm(chunk)))


def _looks_like_a_clause(chunk: str) -> bool:
    c = f" {chunk.lower()} "
    return any(f" {w} " in c or c.strip().startswith(w) for w in _NON_FOOD_MARKERS)


def _norm(s: str) -> str:
    """Lowercase + strip accents, for accent-insensitive food matching."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


@dataclass(frozen=True)
class Food:
    name: str
    kcal: float
    protein: float
    carb: float
    fat: float
    unit_g: Optional[float]
    # Typical serving, used when the user gives no weight ("comi um sanduíche",
    # "não sei o peso"). Distinct from unit_g, which is the weight of ONE
    # countable item ("2 ovos"). None → fall back to unit_g, then 100 g.
    portion_g: Optional[float] = None


@dataclass(frozen=True)
class LoggedItem:
    food: str
    grams: float
    kcal: float
    protein: float
    carb: float
    fat: float
    # True when the user gave no weight and we applied a typical portion.
    # The reply says so — an assumed number must never read as a measured one.
    assumed: bool = False


class FitnessLog:
    """SQLite-backed meal + macro ledger. Thread-safe, fail-soft."""

    DEFAULT_PATH = os.path.expanduser("~/.cortex/fitness.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("CORTEX_FITNESS_DB", self.DEFAULT_PATH)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._food_cache: Optional[List[Food]] = None

    # ------------------------------------------------------------------
    # connection (lazy — never fail at import time) + seed
    # ------------------------------------------------------------------

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        try:
            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                   factory=RollbackOnError)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS foods (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    name     TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    kcal     REAL NOT NULL,
                    protein  REAL NOT NULL,
                    carb     REAL NOT NULL,
                    fat      REAL NOT NULL,
                    unit_g   REAL,
                    aliases  TEXT,
                    portion_g REAL
                );
                CREATE TABLE IF NOT EXISTS meals (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT NOT NULL,
                    food     TEXT NOT NULL,
                    grams    REAL NOT NULL,
                    kcal     REAL NOT NULL,
                    protein  REAL NOT NULL,
                    carb     REAL NOT NULL,
                    fat      REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_meals_ts ON meals(ts);
                CREATE TABLE IF NOT EXISTS workouts (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts           TEXT NOT NULL,
                    kind         TEXT NOT NULL,   -- 'run' | 'lift'
                    name         TEXT NOT NULL,   -- 'corrida' | exercise (supino...)
                    distance_km  REAL,            -- run
                    duration_min REAL,            -- run
                    sets         INTEGER,         -- lift
                    reps         INTEGER,         -- lift
                    weight_kg    REAL             -- lift
                );
                CREATE INDEX IF NOT EXISTS idx_workouts_name
                    ON workouts(name COLLATE NOCASE);
                """
            )
            self._conn = conn
            self._add_missing_columns(conn)
            self._seed_foods(conn)
            return self._conn
        except Exception:
            log.exception("fitness: could not open ledger at %s", self.db_path)
            return None

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection) -> None:
        """Columns introduced after a DB was first created (idempotent).
        CREATE TABLE IF NOT EXISTS never alters an existing table, so an
        install predating a column needs the ALTER — otherwise every read of
        that column raises on the machine that has the most data to lose."""
        for table, column, decl in (("foods", "portion_g", "REAL"),
                                    ("workouts", "start_utc", "TEXT")):
            try:
                cols = {r["name"] for r in
                        conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    conn.commit()
            except Exception:  # noqa: BLE001 — a failed migration must not
                log.exception(  # take the whole engine down with it
                    "fitness: could not add %s.%s", table, column)
        # One row per run instant, enforced by the database: the check-then-
        # insert let two processes importing the same run both insert it
        # (s88, Codex lote 11, reproduced).
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_workouts_run_start "
                         "ON workouts(start_utc) WHERE start_utc IS NOT NULL")
            conn.commit()
        except Exception:  # noqa: BLE001
            log.exception("fitness: could not create the run-instant index")

    def _seed_foods(self, conn: sqlite3.Connection) -> None:
        """Sync the shipped food table into this DB.

        s82 — this used to bail out entirely when the table was non-empty
        ("only seed on first init"), which meant an existing install NEVER
        received a new food or a corrected alias. Pedro's DB still carried the
        original two-alias rows months later, so "yoghurt" resolved by luck.

        The sync is written to be safe on a DB a user has edited:
          - INSERT OR IGNORE adds foods that aren't there; existing rows keep
            their macros, because `add_food` lets the user correct those and
            re-seeding must not silently undo a correction.
          - aliases and portion_g ARE refreshed from the seed: nothing but the
            seed ever writes them (add_food doesn't touch either), so there is
            no user edit to lose, and stale aliases are exactly what broke
            lookup.
        """
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO foods "
                "(name, kcal, protein, carb, fat, unit_g, aliases) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                _SEED_FOODS,
            )
            conn.executemany(
                "UPDATE foods SET aliases = ? WHERE name = ? COLLATE NOCASE",
                [(row[6], row[0]) for row in _SEED_FOODS],
            )
            conn.executemany(
                "UPDATE foods SET portion_g = ? WHERE name = ? COLLATE NOCASE",
                [(g, name) for name, g in _TYPICAL_PORTIONS.items()],
            )
            conn.commit()
        except Exception:
            log.exception("fitness: seed failed")

    # ------------------------------------------------------------------
    # food lookup
    # ------------------------------------------------------------------

    def _all_foods(self) -> List[Tuple[Food, List[str]]]:
        """All foods with their normalised match keys (name + aliases)."""
        conn = self._connect()
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT name, kcal, protein, carb, fat, unit_g, aliases, "
            "portion_g FROM foods"
        ).fetchall()
        out: List[Tuple[Food, List[str]]] = []
        for r in rows:
            keys = [_norm(r["name"])]
            if r["aliases"]:
                keys += [_norm(a) for a in r["aliases"].split(",") if a.strip()]
            out.append((
                Food(r["name"], r["kcal"], r["protein"], r["carb"], r["fat"],
                     r["unit_g"], r["portion_g"]),
                keys,
            ))
        return out

    def lookup_food(self, name: str) -> Optional[Food]:
        """Resolve a spoken food name to a Food row. Exact (accent/case-
        insensitive) on name or any alias first, then a fuzzy fallback so
        "frnago"/"arrloz" still land. Returns None if nothing is close."""
        q = _norm(name)
        if not q:
            return None
        foods = self._all_foods()
        # exact on name or alias
        for food, keys in foods:
            if q in keys:
                return food
        # Substring stage — "peito frango" resolves to "frango". Candidates are
        # SCORED, not first-come (s82): taking the first match in table order
        # is why "yoghurt grego" resolved to *iogurte natural* on a fresh DB —
        # the plain yoghurt's "yoghurt" alias is a substring of the query and
        # its row happens to come first, so the qualified food never got a
        # look-in. Silent and 40% off on protein.
        #
        # The discriminator is the food's own NAME: "iogurte grego" shares the
        # token "grego" with the query, while "iogurte natural" shares nothing
        # (only its alias matched). So rank by how many of the food's name
        # tokens the query actually contains, then prefer the longest matching
        # key as a specificity tiebreak.
        q_tokens = set(q.split())
        scored = []
        for food, keys in foods:
            hits = [k for k in keys if q in k or k in q]
            if not hits:
                continue
            name_overlap = len(set(_norm(food.name).split()) & q_tokens)
            scored.append(((name_overlap, max(len(k) for k in hits)), food))
        if scored:
            return max(scored, key=lambda pair: pair[0])[1]
        # fuzzy
        flat: Dict[str, Food] = {}
        for food, keys in foods:
            for k in keys:
                flat[k] = food
        match = get_close_matches(q, list(flat.keys()), n=1, cutoff=0.78)
        return flat[match[0]] if match else None

    # ------------------------------------------------------------------
    # meal parsing → logging
    # ------------------------------------------------------------------

    def _resolve_grams(self, qty: Optional[float], unit: Optional[str],
                       food: Food) -> Tuple[float, bool]:
        """Turn a parsed (qty, unit) into grams for this food.

        - explicit g/grama/gr/ml → that many grams (ml treated 1:1, fine for v1)
        - kg → ×1000
        - unidade/un → count × the food's unit_g
        - NO unit: a small count (≤12) of a COUNTABLE food → count × unit_g
          ("2 ovos"); otherwise grams ("200 arroz" → 200 g)
        - NO qty at all → the food's TYPICAL PORTION, else 1 unit if countable,
          else 100 g. Returns (grams, assumed): `assumed` is True exactly when
          the user named no weight, so the reply can admit the number is a
          standard serving rather than something Pedro measured (s82 — he
          wrote "não sei o peso, mas faça na média" and got an invented
          "aproximadamente 350 gramas" back)."""
        u = (unit or "").lower()
        if qty is None:
            if food.portion_g:
                return food.portion_g, True
            return (food.unit_g if food.unit_g else 100.0), True
        if u == "kg":
            return qty * 1000.0, False
        if u.startswith("grama") or u in ("g", "gr"):
            return qty, False
        if u == "ml":
            return qty, False
        if u.startswith("unidade") or u == "un":
            return qty * (food.unit_g or 100.0), False
        # no unit
        if food.unit_g and qty <= 12:
            return qty * food.unit_g, False
        return qty, False

    def parse_meal(self, text: str) -> Tuple[List[LoggedItem], List[str]]:
        """Parse "200g de frango e 2 ovos" into (resolved_items, unknown_names).
        Resolution needs the food DB (for count→grams and macros), so this lives
        on the log. Does NOT write — log_meal does. An item given in a
        household measure counts as unknown here; parse_meal_detailed tells
        the two apart."""
        items, unknown, unmeasured = self.parse_meal_detailed(text)
        return items, unknown + unmeasured

    def parse_meal_detailed(self, text: str
                            ) -> Tuple[List[LoggedItem], List[str], List[str]]:
        """parse_meal, plus the items held back because their amount was a
        household measure ("2 colheres de azeite") — returned as written."""
        items: List[LoggedItem] = []
        unknown: List[str] = []
        unmeasured: List[str] = []
        text = (text or "").strip()
        if not text:
            return items, unknown, unmeasured
        for chunk in _CONNECTORS_RE.split(text):
            chunk = chunk.strip(" .")
            if not chunk:
                continue
            # A chunk that survives the split but resolves to nothing must be
            # REPORTED, never dropped (s82): a silent drop is how a whole food
            # disappeared from a logged meal with no signal to the user. The
            # handler surfaces `unknown`, so an unparseable chunk becomes a
            # visible "não reconheci X" instead of a quietly wrong total.
            if _looks_like_a_clause(chunk) or _looks_like_meal_context(chunk):
                continue          # sentence fragment / meal label, not a food
            m = _ITEM_RE.match(chunk)
            if not m:
                unknown.append(chunk)
                continue
            raw_food = (m.group("food") or "").strip(" .")
            if not raw_food:
                unknown.append(chunk)
                continue
            if not m.group("unit") and _MEASURE_RE.match(raw_food):
                unmeasured.append(chunk)
                continue
            food = self.lookup_food(raw_food)
            if food is None:
                unknown.append(raw_food)
                continue
            qty = m.group("qty")
            qty_f = float(qty.replace(",", ".")) if qty else None
            grams, assumed = self._resolve_grams(qty_f, m.group("unit"), food)
            factor = grams / 100.0
            items.append(LoggedItem(
                food=food.name,
                grams=round(grams, 1),
                kcal=round(food.kcal * factor, 1),
                protein=round(food.protein * factor, 1),
                carb=round(food.carb * factor, 1),
                fat=round(food.fat * factor, 1),
                assumed=assumed,
            ))
        return items, unknown, unmeasured

    def log_meal(self, items: List[LoggedItem], days_ago: int = 0) -> bool:
        """Persist already-resolved meal items. Returns False fail-soft.
        `days_ago` dates the meal back ("ontem" = 1) — decided by a fixed
        rule in the handler, never by the model."""
        if not items:
            return False
        # Microsecond precision so each log_meal call is a DISTINCT meal even
        # when two land in the same second (rapid logging / tests). All items
        # of ONE call share this exact ts → they group/undo as one meal;
        # SQLite date(ts) still extracts the date for "today" queries.
        ts = (datetime.now() - timedelta(days=max(0, int(days_ago)))).strftime(
            "%Y-%m-%d %H:%M:%S.%f")
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                conn.executemany(
                    "INSERT INTO meals (ts, food, grams, kcal, protein, carb, fat) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(ts, it.food, it.grams, it.kcal, it.protein, it.carb, it.fat)
                     for it in items],
                )
                conn.commit()
            self._food_cache = None
            return True
        except Exception:
            log.exception("fitness: log_meal failed")
            return False

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def today_totals(self, days_ago: int = 0) -> Dict[str, float]:
        """Sum kcal/protein/carb/fat + meal count for the LOCAL today (or the
        day `days_ago` before it)."""
        empty = {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0, "n": 0}
        today = (datetime.now() - timedelta(days=max(0, int(days_ago)))).strftime("%Y-%m-%d")
        try:
            conn = self._connect()
            if conn is None:
                return empty
            r = conn.execute(
                "SELECT COUNT(*) n, "
                "COALESCE(SUM(kcal),0) kcal, COALESCE(SUM(protein),0) protein, "
                "COALESCE(SUM(carb),0) carb, COALESCE(SUM(fat),0) fat "
                "FROM meals WHERE date(ts) = ?",
                (today,),
            ).fetchone()
            return {
                "kcal": round(r["kcal"], 1), "protein": round(r["protein"], 1),
                "carb": round(r["carb"], 1), "fat": round(r["fat"], 1),
                "n": r["n"],
            }
        except Exception:
            log.exception("fitness: today_totals failed")
            return empty

    def list_today(self) -> List[dict]:
        """Today's meals grouped by log time (items logged together share a
        timestamp → one meal). Each dict: {time, items:[(grams,food)],
        kcal, protein, carb, fat}. Read-only; lets the user SEE where the
        day's total came from (the 5077-kcal "where did that come from")."""
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            conn = self._connect()
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT ts, food, grams, kcal, protein, carb, fat "
                "FROM meals WHERE date(ts) = ? ORDER BY ts",
                (today,),
            ).fetchall()
        except Exception:
            log.exception("fitness: list_today failed")
            return []
        meals: List[dict] = []
        cur_ts = None
        cur: Optional[dict] = None
        for r in rows:
            if r["ts"] != cur_ts:
                cur_ts = r["ts"]
                cur = {"time": r["ts"][11:16], "items": [],
                       "kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
                meals.append(cur)
            cur["items"].append((r["grams"], r["food"]))
            for k in ("kcal", "protein", "carb", "fat"):
                cur[k] += r[k]
        for m in meals:
            for k in ("kcal", "protein", "carb", "fat"):
                m[k] = round(m[k], 1)
        return meals

    def undo_last_meal(self) -> int:
        return self.undo_last_meal_result().removed

    def undo_last_meal_result(self) -> DeleteResult:
        """Delete the most-recently-logged meal (all items sharing the latest
        timestamp), distinguishing an empty log from a failed deletion."""
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return DeleteResult(error=True)
                # The last LOGGED, by insertion order (s88, Codex lote 11): ts
                # is local wall time, so on 25/10 a 01:10 GMT meal sorted before
                # the 01:50 BST one logged an hour earlier, and the undo took
                # the wrong one (reproduced); a meal dated "ontem" would too.
                # Chosen and deleted in ONE statement: two undos at once both
                # picked the same meal and both said they removed it (s88,
                # Codex lote 11); now each removes, and reports, its own.
                cur = conn.execute(
                    "DELETE FROM meals WHERE ts = "
                    "(SELECT ts FROM meals ORDER BY id DESC LIMIT 1)")
                conn.commit()
                return DeleteResult(removed=cur.rowcount)
        except Exception:
            log.error("fitness: undo_last_meal failed")
            return DeleteResult(error=True)

    def clear_today(self) -> int:
        return self.clear_today_result().removed

    def clear_today_result(self) -> DeleteResult:
        """Delete TODAY's meals (e.g. wipe a day of test logs). Returns the
        number of items removed. Only touches today — history is untouched."""
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return DeleteResult(error=True)
                cur = conn.execute("DELETE FROM meals WHERE date(ts) = ?", (today,))
                conn.commit()
                return DeleteResult(removed=cur.rowcount)
        except Exception:
            log.error("fitness: clear_today failed")
            return DeleteResult(error=True)

    # ------------------------------------------------------------------
    # extend the food table
    # ------------------------------------------------------------------

    def add_food(self, name: str, kcal: float, protein: float, carb: float,
                 fat: float, *, unit_g: Optional[float] = None) -> bool:
        """Insert or replace a food (per 100 g). Lets Pedro teach the engine a
        food it doesn't know yet. Returns False fail-soft."""
        name = (name or "").strip()
        if not name:
            return False
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return False
                conn.execute(
                    "INSERT INTO foods (name, kcal, protein, carb, fat, unit_g) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET "
                    "kcal=excluded.kcal, protein=excluded.protein, "
                    "carb=excluded.carb, fat=excluded.fat, unit_g=excluded.unit_g",
                    (name, kcal, protein, carb, fat, unit_g),
                )
                conn.commit()
            return True
        except Exception:
            log.exception("fitness: add_food failed")
            return False

    # ------------------------------------------------------------------
    # workouts (slice 2: runs + lifts + trends). Same fail-soft contract.
    # ------------------------------------------------------------------

    def _now_ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def log_run(self, distance_km: float, duration_min: float) -> bool:
        return self.log_run_result(distance_km, duration_min) == "added"

    def log_run_result(self, distance_km: float, duration_min: float) -> WriteResult:
        """Log a run. Pace is derived on read (duration/distance)."""
        if not _finite_positive(distance_km, duration_min):
            return "invalid"
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return "error"
                conn.execute(
                    "INSERT INTO workouts (ts, kind, name, distance_km, duration_min) "
                    "VALUES (?, 'run', 'corrida', ?, ?)",
                    (self._now_ts(), distance_km, duration_min),
                )
                conn.commit()
            return "added"
        except Exception:
            log.error("fitness: log_run failed")
            return "error"

    def log_lift(self, name: str, sets: int, reps: int, weight_kg: float) -> bool:
        return self.log_lift_result(name, sets, reps, weight_kg) == "added"

    def log_lift_result(self, name: str, sets: int, reps: int,
                        weight_kg: float) -> WriteResult:
        """Log a strength set. Volume (sets×reps×weight) is derived on read."""
        name = (name or "").strip().lower()
        if not name or sets <= 0 or reps <= 0:
            return "invalid"
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return "error"
                conn.execute(
                    "INSERT INTO workouts (ts, kind, name, sets, reps, weight_kg) "
                    "VALUES (?, 'lift', ?, ?, ?, ?)",
                    (self._now_ts(), name, sets, reps, weight_kg),
                )
                conn.commit()
            return "added"
        except Exception:
            log.error("fitness: log_lift failed")
            return "error"

    def exercise_history(self, name: str, limit: int = 20) -> List[dict]:
        """Sessions of one lift (matched accent-insensitively, so "supino"
        finds "supino reto"), oldest→newest, with volume. For "evoluí no X?"."""
        q = _norm(name)
        if not q:
            return []
        try:
            conn = self._connect()
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT ts, name, sets, reps, weight_kg FROM workouts "
                "WHERE kind = 'lift' ORDER BY ts"
            ).fetchall()
        except Exception:
            log.exception("fitness: exercise_history failed")
            return []
        out: List[dict] = []
        for r in rows:
            nm = _norm(r["name"])
            if q in nm or nm in q:
                out.append({
                    "ts": r["ts"], "date": r["ts"][:10], "name": r["name"],
                    "sets": r["sets"], "reps": r["reps"], "weight_kg": r["weight_kg"],
                    "volume": round((r["sets"] or 0) * (r["reps"] or 0)
                                    * (r["weight_kg"] or 0), 1),
                })
        return out[-limit:]

    def import_run(self, distance_km: float, duration_min: float, ts: str,
                   start_utc: Optional[str] = None) -> bool:
        return self.import_run_result(distance_km, duration_min, ts, start_utc) == "added"

    def import_run_result(self, distance_km: float, duration_min: float, ts: str,
                          start_utc: Optional[str] = None) -> WriteResult:
        """Insert a HISTORICAL run at a given timestamp (Apple Health import),
        skipping if that run is already there. Idempotent: re-importing the
        same export adds nothing. Reports added/duplicate/invalid/error.

        The run's identity is its `start_utc` instant when known: by wall
        clock alone, two runs at 01:30 on 25/10 an hour apart were one (s88,
        Codex lote 11). Rows stored before the instant existed have none and
        are matched by `ts`, as they always were."""
        ts = (ts or "").strip()
        if not _finite_positive(distance_km, duration_min) or not ts:
            return "invalid"
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return "error"
                if start_utc:
                    dup = conn.execute(
                        "SELECT 1 FROM workouts WHERE kind='run' AND "
                        "(start_utc = ? OR (start_utc IS NULL AND ts = ?)) LIMIT 1",
                        (start_utc, ts),
                    ).fetchone()
                else:
                    dup = conn.execute(
                        "SELECT 1 FROM workouts WHERE kind='run' AND ts = ? LIMIT 1",
                        (ts,),
                    ).fetchone()
                if dup:
                    return "duplicate"
                cur = conn.execute(
                    "INSERT OR IGNORE INTO workouts "
                    "(ts, kind, name, distance_km, duration_min, start_utc) "
                    "VALUES (?, 'run', 'corrida', ?, ?, ?)",
                    (ts, distance_km, duration_min, start_utc),
                )
                conn.commit()
            return "added" if cur.rowcount == 1 else "duplicate"
        except Exception:
            log.error("fitness: import_run failed")
            return "error"

    def run_history(self, limit: int = 10) -> List[dict]:
        """Recent runs (newest first) with derived pace (min/km)."""
        try:
            conn = self._connect()
            if conn is None:
                return []
            rows = conn.execute(
                "SELECT ts, distance_km, duration_min FROM workouts "
                "WHERE kind = 'run' ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception:
            log.exception("fitness: run_history failed")
            return []
        out: List[dict] = []
        for r in rows:
            d = r["distance_km"] or 0.0
            t = r["duration_min"] or 0.0
            out.append({
                "ts": r["ts"], "date": r["ts"][:10],
                "distance_km": d, "duration_min": t,
                "pace_min_km": round(t / d, 2) if d else 0.0,
            })
        return out

    def undo_last_workout(self, kind: Optional[str] = None) -> Optional[dict]:
        return self.undo_last_workout_result(kind).workout

    def undo_last_workout_result(self, kind: Optional[str] = None) -> DeleteResult:
        """Delete the most-recently-logged workout — of `kind` ("run"/"lift")
        when given, reporting its row only after a successful commit."""
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return DeleteResult(error=True)
                # One statement, like undo_last_meal: the SELECT-then-DELETE
                # let two undos announce the same removal (s88, Codex lote 11).
                where, params = ("WHERE kind = ? ", (kind,)) if kind else ("", ())
                row = conn.execute(
                    "DELETE FROM workouts WHERE id = "
                    f"(SELECT id FROM workouts {where}ORDER BY id DESC LIMIT 1) "
                    "RETURNING id, kind, name, distance_km, duration_min, sets, reps, weight_kg",
                    params,
                ).fetchone()
                conn.commit()
                return DeleteResult(removed=1, workout=dict(row)) if row else DeleteResult()
        except Exception:
            log.error("fitness: undo_last_workout failed")
            return DeleteResult(error=True)

    def clear_workouts_today(self) -> int:
        return self.clear_workouts_today_result().removed

    def clear_workouts_today_result(self) -> DeleteResult:
        """Delete TODAY's workouts (e.g. wipe test entries). Returns the count."""
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with self._lock:
                conn = self._connect()
                if conn is None:
                    return DeleteResult(error=True)
                cur = conn.execute(
                    "DELETE FROM workouts WHERE date(ts) = ?", (today,))
                conn.commit()
                return DeleteResult(removed=cur.rowcount)
        except Exception:
            log.error("fitness: clear_workouts_today failed")
            return DeleteResult(error=True)


def _finite_positive(*values: float) -> bool:
    """The store's own border: an infinite or NaN distance/duration was stored
    and broke every pace read after it (s88, Codex lote 11)."""
    try:
        return all(math.isfinite(float(v)) and float(v) > 0 for v in values)
    except (TypeError, ValueError):
        return False


# Module-level singleton (lazy). Mirrors contacts.book.
_log_singleton: Optional[FitnessLog] = None


def get_log() -> FitnessLog:
    global _log_singleton
    if _log_singleton is None:
        _log_singleton = FitnessLog()
    return _log_singleton
