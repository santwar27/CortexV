"""One local-date snapshot and deterministic period for a research topic."""
from dataclasses import dataclass
from datetime import date
import re
import unicodedata


def local_today() -> date:
    """Local wall clock (not UTC); kept separate for offline fixed-clock tests."""
    return date.today()


EVENT_TIME_RULES = (
    "Write about what happened up to today as past, using only the evidence. "
    'A snippet announcing an event "planned for" a date before today is not evidence it happened. '
    'Describe an event after today as "previsto"; do not present it as already happened.\n'
)

_NUMBERS = {
    "um": 1, "uma": 1, "one": 1, "dois": 2, "duas": 2, "two": 2,
    "tres": 3, "three": 3, "quatro": 4, "four": 4, "cinco": 5, "five": 5,
    "seis": 6, "six": 6, "sete": 7, "seven": 7, "oito": 8, "eight": 8,
    "nove": 9, "nine": 9, "dez": 10, "ten": 10,
}
_RANGE_RE = re.compile(
    r"\b([12]\d{3})\s*(?:[-–—]|a|ate|to|and)\s*([12]\d{3})\b"
)
_LAST_N_RE = re.compile(r"\b(?:ultimos?\s+(\w+)\s+anos?|last\s+(\w+)\s+years?)\b")


def _resolve_years(topic: str, year: int) -> tuple[int, ...]:
    text = "".join(c for c in unicodedata.normalize("NFKD", topic.casefold())
                   if not unicodedata.combining(c))
    # Specific dates outrank vague words such as 'novidades'. Keep all explicit
    # ranges/years, including future ones: the synthesis rules handle tense.
    ranges = list(_RANGE_RE.finditer(text))
    if ranges:
        years = set()
        for match in ranges:
            start, end = sorted(map(int, match.groups()))
            years.update(range(start, end + 1))
        return tuple(sorted(years))
    explicit = re.findall(r"\b[12]\d{3}\b", text)
    if explicit:
        return tuple(sorted({int(y) for y in explicit}))

    match = _LAST_N_RE.search(text)
    if match:
        token = match.group(1) or match.group(2)
        count = int(token) if token.isdecimal() else _NUMBERS.get(token, 0)
        if 1 <= count <= year:
            # N calendar years, including the current (possibly partial) year.
            return tuple(range(year - count + 1, year + 1))
    if re.search(r"\b(?:nos ultimos anos|last years)\b", text):
        return tuple(range(year - 2, year + 1))
    if re.search(r"\b(?:este ano|this year)\b", text):
        return (year,)
    if re.search(r"\b(?:ano passado|last year)\b", text):
        return (year - 1,)
    if re.search(r"\b(?:recent|recentes?|novidades|latest|new features)\b", text):
        return (year - 1, year)
    return ()


@dataclass(frozen=True)
class StudyTime:
    today: date
    years: tuple[int, ...]

    @classmethod
    def for_topic(cls, topic: str) -> "StudyTime":
        today = local_today()
        return cls(today, _resolve_years(topic, today.year))

    def prompt_block(self) -> str:
        # Avoid strftime('%A'): its weekday depends on the process locale.
        weekday = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday")[self.today.weekday()]
        block = (
            f"TODAY: {self.today.isoformat()} ({weekday}). Your own knowledge ends around 2023; "
            "anything after that you know only from the snippets. "
            f'"Recent", "latest", "last N years" mean the years up to {self.today.year}.\n'
        )
        if self.years:
            labels = [str(y) for y in self.years]
            years = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
            block += f"The topic covers {years}.\n"
        return block

    def anchor_queries(self, queries: list[str]) -> list[str]:
        """Every dated-topic query must contain a range year, even on fallback.

        Preserve the query verbatim; append the latest range year if it has
        none (an unrelated/old year does not satisfy the guarantee).
        """
        if not self.years:
            return queries
        years = {str(y) for y in self.years}
        return [q if years.intersection(re.findall(r"\b\d{4}\b", q))
                else f"{q} {self.years[-1]}" for q in queries]
