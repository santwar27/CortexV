"""
Deterministic weather client — Open-Meteo, no key, no LLM (idea #27, s73).

"temperatura em Londres" used to cost 19.4s: Tavily search + the 8B
synthesising over ~3k tokens. Weather is a STRUCTURED lookup — an API
returns numbers, a template formats them. The LLM adds latency and a
hallucination surface and nothing else, so this path never touches it
(an eval pins that).

Provider: Open-Meteo (geocoding + forecast endpoints, both keyless JSON).
wttr.in was the listed alternative; Open-Meteo wins on structured JSON and
explicit metric units. SOBERANIA checklist: the only byte leaving the
machine is the city name the user just said — nothing from the Palace.

Network access still respects the lease — the ORCHESTRATOR checks it
before calling here (this module is a dumb client, deliberately free of
gatekeeper imports so it stays testable in isolation).
"""
from __future__ import annotations

import logging
from typing import Dict

import requests

from src.core import turn_metrics

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 8

# WMO weather interpretation codes → short PT/EN descriptions. Grouped by
# the ranges Open-Meteo documents; unknown codes fall back to a neutral
# phrase rather than crashing the turn.
_WMO: Dict[int, Dict[str, str]] = {
    0: {"pt": "céu limpo", "en": "clear sky"},
    1: {"pt": "predominantemente limpo", "en": "mainly clear"},
    2: {"pt": "parcialmente nublado", "en": "partly cloudy"},
    3: {"pt": "nublado", "en": "overcast"},
    45: {"pt": "nevoeiro", "en": "fog"},
    48: {"pt": "nevoeiro com geada", "en": "depositing rime fog"},
    51: {"pt": "garoa fraca", "en": "light drizzle"},
    53: {"pt": "garoa", "en": "drizzle"},
    55: {"pt": "garoa forte", "en": "dense drizzle"},
    61: {"pt": "chuva fraca", "en": "slight rain"},
    63: {"pt": "chuva", "en": "rain"},
    65: {"pt": "chuva forte", "en": "heavy rain"},
    66: {"pt": "chuva congelante fraca", "en": "light freezing rain"},
    67: {"pt": "chuva congelante", "en": "freezing rain"},
    71: {"pt": "neve fraca", "en": "slight snow"},
    73: {"pt": "neve", "en": "snow"},
    75: {"pt": "neve forte", "en": "heavy snow"},
    77: {"pt": "grãos de neve", "en": "snow grains"},
    80: {"pt": "pancadas de chuva fracas", "en": "slight rain showers"},
    81: {"pt": "pancadas de chuva", "en": "rain showers"},
    82: {"pt": "pancadas de chuva violentas", "en": "violent rain showers"},
    85: {"pt": "pancadas de neve fracas", "en": "slight snow showers"},
    86: {"pt": "pancadas de neve", "en": "snow showers"},
    95: {"pt": "trovoada", "en": "thunderstorm"},
    96: {"pt": "trovoada com granizo fraco", "en": "thunderstorm with slight hail"},
    99: {"pt": "trovoada com granizo forte", "en": "thunderstorm with heavy hail"},
}


class WeatherUnavailable(Exception):
    """Raised when geocoding/forecast fails or the city is unknown."""


def describe_wmo(code: int, lang: str = "pt") -> str:
    entry = _WMO.get(int(code))
    if entry is None:
        return "condições variáveis" if lang == "pt" else "mixed conditions"
    return entry.get(lang, entry["en"])


def fetch_weather(city: str, lang: str = "pt", day: int = 0) -> Dict:
    """Geocode `city`, fetch current conditions + today's range — or, for
    `day` 1 or 2 (tomorrow, the day after), that day's forecast. Returns a
    flat dict for `format_weather`. Raises WeatherUnavailable on any failure
    (network, unknown city, malformed payload) — callers degrade honestly."""
    name = (city or "").strip()
    if not name:
        raise WeatherUnavailable("Empty city name.")

    try:
        turn_metrics.mark_online("weather")  # Covers both requests, even on failure.
        geo = requests.get(
            GEOCODE_URL,
            params={"name": name, "count": 1, "language": lang, "format": "json"},
            timeout=TIMEOUT_S,
        )
        geo.raise_for_status()
        payload = geo.json()
    except (requests.RequestException, ValueError) as e:
        raise WeatherUnavailable(f"Geocoding failed: {e}") from e
    # A 200 with an unexpected shape ([{}], results as a dict) raised
    # AttributeError/KeyError past the handler, which catches only
    # WeatherUnavailable (s88, Codex lote 13 — invariant 7). No "results"
    # key is how Open-Meteo says it found no such city.
    if not isinstance(payload, dict):
        raise WeatherUnavailable("Geocoding failed: unexpected payload")
    hits = payload.get("results") or []
    if not isinstance(hits, list) or (hits and not isinstance(hits[0], dict)):
        raise WeatherUnavailable("Geocoding failed: unexpected payload")
    if not hits:
        raise WeatherUnavailable(f"Unknown city: {name!r}")

    place = hits[0]
    try:
        fc = requests.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,"
                           "relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,weather_code",
                "forecast_days": day + 1,
                "timezone": "auto",
            },
            timeout=TIMEOUT_S,
        )
        fc.raise_for_status()
        data = fc.json() or {}
        daily = data["daily"]
        rain = daily.get("precipitation_probability_max") or []
        out = {
            "city": place.get("name") or name,
            "country": place.get("country") or "",
            "day": day,
            "tmax_c": round(float(daily["temperature_2m_max"][day])),
            "tmin_c": round(float(daily["temperature_2m_min"][day])),
            "rain_prob": (round(float(rain[day]))
                          if len(rain) > day and rain[day] is not None else None),
        }
        if day:
            # Current conditions are today's; another day has only its
            # forecast, and its own weather code.
            out["code"] = int(daily["weather_code"][day])
            return out
        cur = data["current"]
        out.update({
            "temp_c": round(float(cur["temperature_2m"])),
            "feels_c": round(float(cur["apparent_temperature"])),
            "humidity": round(float(cur["relative_humidity_2m"])),
            "wind_kmh": round(float(cur["wind_speed_10m"])),
            "code": int(cur["weather_code"]),
        })
        return out
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        raise WeatherUnavailable(f"Forecast failed: {e}") from e


def format_weather(d: Dict, lang: str = "pt") -> str:
    """Pure template — metric only (Open-Meteo already returns °C/km/h)."""
    desc = describe_wmo(d["code"], lang)
    where = d["city"] + (f", {d['country']}" if d.get("country") else "")
    rain = d.get("rain_prob")
    if d.get("day"):
        when = {1: ("amanhã", "tomorrow"),
                2: ("depois de amanhã", "the day after tomorrow")}[d["day"]]
        if lang == "en":
            parts = [f"Forecast for {when[1]} in {where}: {desc}, "
                     f"{d['tmin_c']}°C to {d['tmax_c']}°C."]
            if rain is not None:
                parts.append(f"Chance of rain: {rain}%.")
        else:
            parts = [f"Previsão para {when[0]} em {where}: {desc}, "
                     f"de {d['tmin_c']}°C a {d['tmax_c']}°C."]
            if rain is not None:
                parts.append(f"Chance de chuva: {rain}%.")
        return " ".join(parts)
    if lang == "en":
        parts = [
            f"Weather in {where}: {d['temp_c']}°C ({desc}), "
            f"feels like {d['feels_c']}°C.",
            f"Today: {d['tmin_c']}°C to {d['tmax_c']}°C.",
            f"Humidity {d['humidity']}%, wind {d['wind_kmh']} km/h.",
        ]
        if rain is not None:
            parts.append(f"Chance of rain: {rain}%.")
    else:
        parts = [
            f"Tempo em {where}: {d['temp_c']}°C ({desc}), "
            f"sensação de {d['feels_c']}°C.",
            f"Hoje: {d['tmin_c']}°C a {d['tmax_c']}°C.",
            f"Umidade {d['humidity']}%, vento {d['wind_kmh']} km/h.",
        ]
        if rain is not None:
            parts.append(f"Chance de chuva: {rain}%.")
    return " ".join(parts)
