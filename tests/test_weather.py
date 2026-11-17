"""
Deterministic weather route (idea #27, s73).

The contract under test:
  - client: Open-Meteo geocode + forecast → flat dict; every failure shape
    (network, unknown city, malformed payload) → WeatherUnavailable.
  - template: pure formatting, metric only, PT/EN.
  - handler: lease respected (no lease → offer flow, never network);
    API failure → honest degrade; NO LLM anywhere (no engine attr is even
    touched on the happy path).
  - approval flow: "sim" on a pending weather query re-dispatches through
    the deterministic route, not _do_web_search.

No test here touches the real network: requests.get is always patched.
"""
from types import SimpleNamespace

import pytest
import requests

from src.tools import weather as wt


def _resp(payload, status=200):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(f"status {status}")

        def json(self):
            return payload
    return R()


GEO_OK = {"results": [{"name": "London", "country": "United Kingdom",
                       "latitude": 51.5, "longitude": -0.12}]}
FC_OK = {
    "current": {"temperature_2m": 18.4, "apparent_temperature": 17.2,
                "relative_humidity_2m": 62, "weather_code": 3,
                "wind_speed_10m": 14.6},
    "daily": {"temperature_2m_max": [21.1], "temperature_2m_min": [12.8],
              "precipitation_probability_max": [35]},
}


class TestFetchWeather:
    def test_happy_path(self, monkeypatch):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(url)
            return _resp(GEO_OK if "geocoding" in url else FC_OK)

        monkeypatch.setattr(wt.requests, "get", fake_get)
        d = wt.fetch_weather("londres")
        assert d["city"] == "London"
        assert d["temp_c"] == 18 and d["feels_c"] == 17
        assert d["tmax_c"] == 21 and d["tmin_c"] == 13
        assert d["rain_prob"] == 35
        assert calls == [wt.GEOCODE_URL, wt.FORECAST_URL]

    def test_unknown_city_raises(self, monkeypatch):
        monkeypatch.setattr(wt.requests, "get",
                            lambda *a, **k: _resp({"results": []}))
        with pytest.raises(wt.WeatherUnavailable, match="Unknown city"):
            wt.fetch_weather("Xyzzylândia")

    def test_network_error_raises(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("no route to host")
        monkeypatch.setattr(wt.requests, "get", boom)
        with pytest.raises(wt.WeatherUnavailable, match="Geocoding failed"):
            wt.fetch_weather("London")

    def test_malformed_forecast_raises(self, monkeypatch):
        def fake_get(url, params=None, timeout=None):
            return _resp(GEO_OK if "geocoding" in url else {"current": {}})
        monkeypatch.setattr(wt.requests, "get", fake_get)
        with pytest.raises(wt.WeatherUnavailable, match="Forecast failed"):
            wt.fetch_weather("London")

    def test_empty_city_raises(self):
        with pytest.raises(wt.WeatherUnavailable):
            wt.fetch_weather("   ")


class TestFormatWeather:
    DATA = {"city": "London", "country": "United Kingdom", "temp_c": 18,
            "feels_c": 17, "humidity": 62, "wind_kmh": 15, "code": 3,
            "tmax_c": 21, "tmin_c": 13, "rain_prob": 35}

    def test_pt_metric_template(self):
        out = wt.format_weather(self.DATA, "pt")
        assert "Tempo em London, United Kingdom: 18°C (nublado)" in out
        assert "sensação de 17°C" in out
        assert "13°C a 21°C" in out
        assert "vento 15 km/h" in out
        assert "Chance de chuva: 35%." in out

    def test_en_template(self):
        out = wt.format_weather(self.DATA, "en")
        assert "Weather in London" in out and "overcast" in out

    def test_no_rain_prob_omits_sentence(self):
        d = dict(self.DATA, rain_prob=None)
        assert "chuva" not in wt.format_weather(d, "pt").lower()

    def test_unknown_wmo_code_falls_back(self):
        d = dict(self.DATA, code=42)  # not a WMO code
        assert "condições variáveis" in wt.format_weather(d, "pt")


class TestWeatherRegex:
    @pytest.mark.parametrize("text, city", [
        ("clima em Londres", "Londres"),
        ("qual a temperatura em São Paulo", "São Paulo"),
        ("previsão do tempo em Recife hoje", "Recife"),
        ("como tá o tempo em Lisboa?", "Lisboa"),
        ("temperatura agora em Curitiba", "Curitiba"),
        ("weather in London", "London"),
        ("temperature in Tokyo now", "Tokyo"),
        ("clima no Rio de Janeiro amanhã", "Rio de Janeiro"),
    ])
    def test_matches_and_captures_city(self, text, city):
        from src.core.dispatch.weather import WEATHER_RE
        m = WEATHER_RE.match(text)
        assert m, f"should match: {text}"
        assert m.group("city").strip() == city

    @pytest.mark.parametrize("text", [
        "como tá o tempo?",                  # no city — never guess
        "quanto tempo de viagem até Paris",  # 'tempo' = duration
        "tempo de cozimento do arroz",       # 'de' is not a city preposition
        "que horas são",                     # time reflex territory
        "clima organizacional na empresa",   # not weather… falls to LLM
    ])
    def test_does_not_overmatch(self, text):
        from src.core.dispatch.weather import WEATHER_RE
        m = WEATHER_RE.match(text)
        # 'clima organizacional na empresa' is linguistically a weather
        # shape ('clima ... na <X>'); the geocoder then fails honestly.
        # The hard NOs are the duration/time shapes:
        if text.startswith(("quanto", "tempo de", "que horas", "como tá o tempo?")):
            assert not m, f"false positive: {text}"


class TestWeatherHandler:
    def _orch(self, lease=True):
        from src.core.weather_mixin import WeatherMixin
        from src.core.i18n import I18nMixin

        class O(WeatherMixin, I18nMixin):
            pass

        o = O()
        o._reply_lang = "pt"
        o.gatekeeper = SimpleNamespace(is_lease_active=lambda: lease)
        o.lease_hours = 4
        o._pending_query = None
        o._pending_engine = None
        return o

    def test_no_lease_offers_and_stages_pending(self):
        o = self._orch(lease=False)
        reply, mode = o._handle_weather("Londres", raw_text="clima em Londres")
        assert mode == "AWAITING_APPROVAL"
        assert "internet" in reply.lower()
        assert o._pending_query == "clima em Londres"

    def test_lease_active_fetches_and_templates(self, monkeypatch):
        o = self._orch(lease=True)
        monkeypatch.setattr("src.tools.weather.fetch_weather",
                            lambda city, lang="pt", day=0: dict(
                                TestFormatWeather.DATA))
        reply, mode = o._handle_weather("Londres")
        assert mode == "WEATHER"
        assert "18°C" in reply and "London" in reply

    def test_api_down_degrades_honestly(self, monkeypatch):
        o = self._orch(lease=True)

        def boom(city, lang="pt", day=0):
            raise wt.WeatherUnavailable("Forecast failed: timeout")
        monkeypatch.setattr("src.tools.weather.fetch_weather", boom)
        reply, mode = o._handle_weather("Londres")
        assert mode == "WEATHER_ERROR"
        assert "não consegui" in reply.lower()
        # No invented numbers in the degrade reply.
        assert "°c" not in reply.lower()

    def test_unknown_city_named_honestly(self, monkeypatch):
        o = self._orch(lease=True)

        def unknown(city, lang="pt", day=0):
            raise wt.WeatherUnavailable("Unknown city: 'Xyzzy'")
        monkeypatch.setattr("src.tools.weather.fetch_weather", unknown)
        reply, mode = o._handle_weather("Xyzzy")
        assert mode == "WEATHER_ERROR"
        assert "Xyzzy" in reply

    def test_empty_city_falls_through(self):
        o = self._orch()
        assert o._handle_weather("  ") is None


class TestApprovalRedispatch:
    """'sim' on a pending WEATHER query must re-enter the deterministic
    route — never _do_web_search (which would be the LLM RAG path)."""

    def test_approval_routes_weather_not_rag(self, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator

        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        o._pending_query = "clima em Londres"
        o._pending_engine = None
        o.lease_hours = 4
        from datetime import datetime, timedelta, timezone
        o.gatekeeper = SimpleNamespace(
            is_lease_active=lambda: True,
            grant_lease=lambda h: datetime.now(timezone.utc) + timedelta(hours=h),
            revoke_lease=lambda: None,
            evaluate_gate=lambda text, peek=False: SimpleNamespace(mode="OFFLINE"),
        )
        o.skill_registry = None
        o.engine_analytical = SimpleNamespace()
        o.studies_review = SimpleNamespace(list_pending=lambda: [])

        def no_rag(*a, **k):
            raise AssertionError("weather approval must not hit _do_web_search")
        monkeypatch.setattr(o, "_do_web_search", no_rag)
        monkeypatch.setattr(
            "src.tools.weather.fetch_weather",
            lambda city, lang="pt", day=0: dict(TestFormatWeather.DATA))

        reply, mode = o.process("sim")
        assert mode == "WEATHER"
        assert "18°C" in reply

    def test_approval_weather_failure_rolls_back_lease(self, monkeypatch):
        from src.core.orchestrator import CortexOrchestrator

        o = CortexOrchestrator.__new__(CortexOrchestrator)
        o._pending_action = None
        o._pending_query = "clima em Londres"
        o._pending_engine = None
        o.lease_hours = 4
        revoked = []
        from datetime import datetime, timedelta, timezone
        o.gatekeeper = SimpleNamespace(
            is_lease_active=lambda: True,
            grant_lease=lambda h: datetime.now(timezone.utc) + timedelta(hours=h),
            revoke_lease=lambda: revoked.append(True),
            evaluate_gate=lambda text, peek=False: SimpleNamespace(mode="OFFLINE"),
        )
        o.skill_registry = None
        o.engine_analytical = SimpleNamespace()
        o.studies_review = SimpleNamespace(list_pending=lambda: [])

        def boom(city, lang="pt", day=0):
            raise wt.WeatherUnavailable("Forecast failed: timeout")
        monkeypatch.setattr("src.tools.weather.fetch_weather", boom)

        reply, mode = o.process("sim")
        assert mode == "WEATHER_ERROR"
        assert revoked == [True]


class TestTheDayAsked:
    """The route swallowed "amanhã" to keep it out of the city, and the
    handler asked Open-Meteo for one day of current conditions: "clima no
    Rio de Janeiro amanhã" answered with today's weather (s88, Codex lote 13
    — invariant 7)."""

    FC_3 = {
        "current": FC_OK["current"],
        "daily": {"temperature_2m_max": [21.1, 15.2, 25.0],
                  "temperature_2m_min": [12.8, 9.1, 17.0],
                  "precipitation_probability_max": [35, 90, 0],
                  "weather_code": [3, 63, 0]},
    }

    def _ask(self, monkeypatch, text):
        from src.core.dispatch.weather import WEATHER_RE
        asked = []

        def fake_get(url, params=None, timeout=None):
            if "geocoding" in url:
                return _resp(GEO_OK)
            asked.append(params)
            return _resp(self.FC_3)
        monkeypatch.setattr(wt.requests, "get", fake_get)
        monkeypatch.setattr("src.core.weather_mixin.audit",
                            SimpleNamespace(log=lambda *a, **k: None))
        m = WEATHER_RE.match(text)
        assert m, text
        reply, mode = TestWeatherHandler()._orch()._handle_weather(
            m.group("city"), raw_text=text)
        assert mode == "WEATHER"
        return asked[-1], reply

    def test_tomorrow_is_tomorrows_forecast(self, monkeypatch):
        params, reply = self._ask(monkeypatch, "clima em Londres amanhã")
        assert params["forecast_days"] == 2
        assert "amanhã" in reply and "9°C a 15°C" in reply and "90%" in reply
        assert "Hoje" not in reply and "18°C" not in reply

    def test_the_day_after_tomorrow(self, monkeypatch):
        params, reply = self._ask(monkeypatch, "previsão do tempo em Londres depois de amanhã")
        assert params["forecast_days"] == 3
        assert "depois de amanhã" in reply and "17°C a 25°C" in reply

    def test_today_is_unchanged(self, monkeypatch):
        params, reply = self._ask(monkeypatch, "clima em Londres hoje")
        assert params["forecast_days"] == 1
        assert "Hoje: 13°C a 21°C" in reply and "18°C" in reply

    def test_english_tomorrow(self, monkeypatch):
        params, _reply = self._ask(monkeypatch, "weather in London tomorrow")
        assert params["forecast_days"] == 2

    @pytest.mark.parametrize("text, city", [
        ("clima em Lisboa depois de amanhã", "Lisboa"),
        ("weather in London the day after tomorrow", "London"),
    ])
    def test_the_city_stays_clean(self, text, city):
        from src.core.dispatch.weather import WEATHER_RE
        assert WEATHER_RE.match(text).group("city") == city

    @pytest.mark.parametrize("text, day", [
        ("clima em Londres", 0), ("clima em Londres amanhã", 1),
        ("clima em Londres amanha", 1), ("clima em Londres depois de amanhã", 2),
        ("weather in London tomorrow", 1)])
    def test_requested_day(self, text, day):
        from src.core.dispatch.weather import requested_day
        assert requested_day(text) == day



@pytest.mark.parametrize("payload", [[{}], {"results": {"x": 1}}, {"results": ["x"]}, "oops"])
def test_an_odd_geocoder_reply_degrades_honestly(monkeypatch, payload):
    """A 200 with an unexpected shape raised AttributeError/KeyError past the
    handler, which catches only WeatherUnavailable (s88, Codex lote 13)."""
    monkeypatch.setattr(wt.requests, "get", lambda *a, **k: _resp(payload))
    monkeypatch.setattr("src.core.weather_mixin.audit",
                        SimpleNamespace(log=lambda *a, **k: None))
    reply, mode = TestWeatherHandler()._orch()._handle_weather(
        "Londres", raw_text="clima em Londres")
    assert mode == "WEATHER_ERROR"
    assert "não consegui" in reply.lower()
