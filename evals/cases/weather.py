"""Deterministic weather route (idea #27, s73).

What these cases pin, per the spec:
  - "weather does NOT go through the LLM": the DISPATCH cases prove the
    query resolves in the deterministic registry (mode WEATHER, matcher
    WEATHER_RE) — anything reaching the LLM would report a different mode.
  - offline behaviour: DISPATCH_REPLY runs the real handler against the
    fake orch, whose gatekeeper NEVER has a lease — the reply must be the
    honest internet offer, with zero network and zero invented numbers.
"""
from evals.core import EvalCase, EvalKind


CASES: list[EvalCase] = [
    EvalCase(
        name="weather_route_pt", kind=EvalKind.DISPATCH, category="weather",
        input="clima em Londres",
        expected_mode="WEATHER", expected_matcher="WEATHER_RE",
        description="Weather resolves in the deterministic registry — no LLM.",
        tags=["regression"],
    ),
    EvalCase(
        name="weather_route_temperatura", kind=EvalKind.DISPATCH, category="weather",
        input="qual a temperatura em São Paulo hoje",
        expected_mode="WEATHER", expected_matcher="WEATHER_RE",
    ),
    EvalCase(
        name="weather_route_en", kind=EvalKind.DISPATCH, category="weather",
        input="weather in London",
        expected_mode="WEATHER", expected_matcher="WEATHER_RE",
    ),
    EvalCase(
        name="weather_no_city_not_routed", kind=EvalKind.DISPATCH, category="weather",
        input="como tá o tempo?",
        expected_mode="OFFLINE",
        description="No city → never guess; falls through to the old path.",
    ),
    EvalCase(
        name="weather_duration_not_routed", kind=EvalKind.DISPATCH, category="weather",
        input="tempo de cozimento do arroz",
        expected_mode="OFFLINE",
        description="'tempo de X' is duration, not weather.",
    ),
    EvalCase(
        name="weather_offline_degrades_honestly", kind=EvalKind.DISPATCH_REPLY,
        category="weather",
        input="clima em Londres",
        language="pt",
        expected_contains="internet",
        expected_not_contains="°C",
        description=(
            "Fake orch has no lease, ever: the reply must be the honest "
            "offer to grant access — no temperatures invented offline."
        ),
        tags=["regression"],
    ),
    EvalCase(
        name="weather_offline_degrades_honestly_en", kind=EvalKind.DISPATCH_REPLY,
        category="weather",
        input="weather in London",
        language="en",
        expected_contains="internet",
        expected_not_contains="°C",
    ),
]
