import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
from ddgs import DDGS

from src.core import turn_metrics
from src.core.cost_tracker import cost_tracker as _default_cost_tracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchMarket:
    """A ranking preference, not a guarantee of source country or currency."""

    code: str
    tavily_country: str
    ddgs_region: str
    name_pt: str
    name_en: str
    currency: str
    symbol: str
    location_pt: str  # Full locative phrase: preposition + name, owned by this table.


_SEARCH_MARKETS = {market.code: market for market in (
    SearchMarket("GB", "united kingdom", "uk-en", "Reino Unido", "the United Kingdom", "GBP", "£", "no Reino Unido"),
    SearchMarket("US", "united states", "us-en", "Estados Unidos", "the United States", "USD", "$", "nos Estados Unidos"),
    SearchMarket("BR", "brazil", "br-pt", "Brasil", "Brazil", "BRL", "R$", "no Brasil"),
    SearchMarket("PT", "portugal", "pt-pt", "Portugal", "Portugal", "EUR", "€", "em Portugal"),
    SearchMarket("IE", "ireland", "ie-en", "Irlanda", "Ireland", "EUR", "€", "na Irlanda"),
    SearchMarket("CA", "canada", "ca-en", "Canadá", "Canada", "CAD", "$", "no Canadá"),
)}


class WebSearcher:
    """
    Tool to fetch real-time information from the internet.

    Strategy:
      - Primary: Tavily (LLM-optimized source snippets)
      - Fallback: DuckDuckGo (no auth, unlimited) when Tavily token is
        missing, the request fails, or the monthly free cap is hit

    Provider is controlled by env vars:
      CORTEX_SEARCH_PROVIDER  -- "tavily" (default) or "ddg" to skip Tavily
      TAVILY_API_KEY          -- required for Tavily; if absent, falls back
      CORTEX_SEARCH_COUNTRY   -- optional supported ISO alpha-2 market code
    """

    TAVILY_URL = "https://api.tavily.com/search"
    TAVILY_TIMEOUT_S = 12

    # Per-source content cap. Tavily/DDG can return very long `content` blobs
    # (a sports page dumps a whole fixture list + navigation cruft — "Mais
    # buscados agora: 1º são paulo…"). That bloat does two bad things: it slows
    # the 8B synthesis (prompt-eval on the M1 scales with context — a Flamengo
    # "último jogo" turn took 27s, most of it reading the context), and it
    # buries the actual answer in noise the model then mis-picks from. Keep the
    # first slice of each source (where the relevant snippet sits) and drop the
    # tail. Generous enough to hold a real answer, tight enough to stay fast.
    MAX_SOURCE_CHARS = 700

    def __init__(self, cost_tracker=None):
        self.tavily_key = (os.getenv("TAVILY_API_KEY", "") or "").strip() or None
        self.provider_pref = (os.getenv("CORTEX_SEARCH_PROVIDER", "tavily") or "tavily").strip().lower()
        country = os.getenv("CORTEX_SEARCH_COUNTRY")
        self._market = (
            _SEARCH_MARKETS.get(country.strip().upper())
            if country is not None and country.isascii() else None
        )
        if country not in (None, "") and self._market is None:
            log.warning("Invalid or unsupported CORTEX_SEARCH_COUNTRY; treating as unset")
        # Injectable for tests; defaults to the module singleton. Only the
        # Tavily path is metered — DuckDuckGo (the fallback) is free.
        self.cost_tracker = cost_tracker or _default_cost_tracker

    def get_market(self) -> Optional[SearchMarket]:
        """Return the immutable construction-time search market, or None."""
        return self._market

    def search(self, query: str, max_results: int = 3) -> str:
        """Executes a web search and formats the top results into a context string."""
        prefer_tavily = self.provider_pref == "tavily" and self.tavily_key

        if prefer_tavily:
            tavily_result = self._search_tavily(query, max_results)
            if tavily_result is not None:
                return tavily_result
            # Fall through to DDG on Tavily failure (network, rate limit,
            # etc.) — loudly: in production this was silent, so there was no
            # way to know how many searches were degrading to the fallback.
            log.warning("Tavily search failed; falling back to DDG (query=%r)",
                        query[:80])

        return self._search_ddg(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> Optional[str]:
        """Returns formatted context on success, None on failure (caller falls back)."""
        try:
            turn_metrics.mark_online("web_search")  # Failure may follow transmission.
            payload = {
                "api_key": self.tavily_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            }
            if self._market is not None:
                # Tavily's country boost is available only for general searches.
                payload.update(topic="general", country=self._market.tavily_country)
            response = requests.post(
                self.TAVILY_URL,
                json=payload,
                timeout=self.TAVILY_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None

        results = data.get("results") or []

        if not results:
            return None

        # Billable call succeeded — log it (1 credit for a basic search).
        # cost_usd stays None: we count credits, not dollars (see
        # cost_tracker docstring). Fail-soft inside record().
        self.cost_tracker.record(
            "tavily", "search",
            units=1, unit_kind="credit",
            metadata={"query": query[:80], "max_results": max_results},
        )

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or "No Title"
            content = (r.get("content") or "").strip()[:self.MAX_SOURCE_CHARS]
            url = r.get("url") or ""
            parts.append(f"Source {i} ({title}):\n{content}\n[{url}]")

        return "\n\n".join(parts)

    def _search_ddg(self, query: str, max_results: int) -> str:
        """Fallback path — used when Tavily is unavailable or disabled."""
        try:
            # verify defaults to True (TLS on). It was explicitly disabled
            # until s63 — an on-path attacker could inject forged "results"
            # straight into the LLM context (M4-A4, review Fable 5). A test
            # greps this file to keep it from coming back.
            turn_metrics.mark_online("web_search")  # Includes provider setup.
            with DDGS() as ddgs:
                if self._market is None:
                    results = list(ddgs.text(query, max_results=max_results))
                else:
                    results = list(ddgs.text(query, max_results=max_results,
                                             region=self._market.ddgs_region))

            if not results:
                return "[SYSTEM NOTICE: No relevant information found on the web for this query.]"

            context = ""
            for i, r in enumerate(results, 1):
                # Include the real URL in the same bracket format Tavily uses, so
                # the synthesiser inherits a verifiable source instead of inventing
                # 'example.com' style placeholders. ddgs returns 'href'.
                url = (r.get("href") or "").strip()
                body = (r.get("body") or "No Body")[:self.MAX_SOURCE_CHARS]
                context += (
                    f"Result {i} ({r.get('title', 'No Title')}): "
                    f"{body}\n[{url}]\n"
                )
            return context

        except Exception as e:
            # The exception goes to the log, never into the result: the follow-up is
            # told to repeat a tool error verbatim, and the 8B read it out 6 times in
            # 6 ("O erro foi: EOF marker not found" — s88, Codex lote 8).
            log.warning("web search failed: %r", e)
            return "[SYSTEM ERROR: Web search retrieval failed.]"
