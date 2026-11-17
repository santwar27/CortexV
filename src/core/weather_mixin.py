"""
Deterministic weather handler for `CortexOrchestrator` (idea #27, s73).

Same mixin pattern as notes_mixin (s59). The contract this path pins:
  - NO LLM anywhere: API numbers → template string (eval pins mode).
  - Lease respected like any read: no lease → the SAME offer-to-search
    flow the realtime path uses (_pending_query + AWAITING_APPROVAL), so
    "sim" grants the lease and re-dispatches deterministically.
  - Offline/API failure → honest degrade, never a guess.
  - Audited like any external read (stage weather, events fetch/error).
"""
from __future__ import annotations

from typing import Optional, Tuple

from src.core.audit import audit


class WeatherMixin:
    """Weather handler. Relies on the host providing `self.gatekeeper`,
    `self.lease_hours`, and the pending-query slots (`_pending_query`,
    `_pending_engine`)."""

    def _handle_weather(self, city: str, raw_text: str = "") -> Optional[Tuple[str, str]]:
        city = (city or "").strip().strip('"\'')
        if not city:
            return None  # fall through — never guess a city

        if not self.gatekeeper.is_lease_active():
            # Mirror the realtime offer flow: stage the ORIGINAL text so an
            # approval re-dispatches through this same deterministic route
            # (the approval branch in _process_core re-checks WEATHER_RE).
            self._pending_query = raw_text or f"clima em {city}"
            self._pending_engine = getattr(self, "engine_analytical", None)
            return (
                self._t(
                    f"Essa pergunta precisa de dados atuais da internet. "
                    f"Posso liberar acesso por {self.lease_hours}h? "
                    f"(responda 'sim' ou 'não')",
                    f"This question needs current internet data. May I grant "
                    f"access for {self.lease_hours} hours? (reply 'yes' or 'no')",
                ),
                "AWAITING_APPROVAL",
            )

        from src.core.dispatch.weather import requested_day
        from src.tools import weather as weather_tool

        lang = getattr(self, "_reply_lang", "pt") or "pt"
        day = requested_day(raw_text)   # "amanhã" is not today (s88, lote 13)
        try:
            data = weather_tool.fetch_weather(city, lang=lang, day=day)
        except weather_tool.WeatherUnavailable as e:
            audit.log("–", "weather", "error",
                      payload={"city": city[:60], "error": str(e)[:120]})
            low = str(e).lower()
            if "unknown city" in low:
                return (
                    self._t(f'Não achei a cidade "{city}" no serviço de clima.',
                            f'I couldn\'t find the city "{city}" in the weather service.'),
                    "WEATHER_ERROR",
                )
            # Network/API down — degrade honestly, never guess numbers.
            return (
                self._t("Não consegui consultar o serviço de clima agora "
                        "(sem resposta da API). Tenta de novo em instantes.",
                        "I couldn't reach the weather service just now "
                        "(no API response). Try again shortly."),
                "WEATHER_ERROR",
            )

        audit.log("–", "weather", "fetch",
                  payload={"city": data["city"], "country": data["country"],
                           "day": day, "temp_c": data.get("temp_c"),
                           "code": data["code"]})
        return (weather_tool.format_weather(data, lang), "WEATHER")
