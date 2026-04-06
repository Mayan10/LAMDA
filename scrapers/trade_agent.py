"""
LAMDA Trade Agent — Port 8004
Estimates trade volume for each port.
Tries UN Comtrade → World Bank trade API → SerpAPI + Claude estimation.
"""

import time
from datetime import datetime
from flask import jsonify
import requests as http_requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from base_scraper import BaseScraper, SUPPORTED_NODES
from node_metadata import NODE_CATALOG


class TradeAgent(BaseScraper):
    """Trade volume scraper — Port 8004."""

    def __init__(self):
        super().__init__(name="trade-agent", port=8004, cache_ttl_seconds=3600)  # 60 min
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/latest/<node_id>", methods=["GET"])
        def latest(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._fetch_trade(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    def _fetch_trade(self, node_id: str) -> dict:
        city = self.node_to_city(node_id)
        volume = None
        source = "unavailable"

        # 1. Try UN Comtrade public API (with retry / exponential backoff)
        volume = self._try_comtrade(node_id)
        if volume is not None:
            source = "un_comtrade"

        # 2. World Bank annual trade indicator
        if volume is None:
            volume = self._try_world_bank(node_id)
            if volume is not None:
                source = "world_bank"

        if volume is None:
            volume = self._try_serpapi_gemini(node_id, city)
            if volume is not None:
                source = "serpapi_gemini"

        # 4. Final fallback — unknown is represented explicitly instead of a mock baseline.
        if volume is None:
            volume = 0.0
            self.logger.warning(f"No live trade signal available for {node_id}")

        return {
            "node_id": node_id,
            "volume_usd": round(volume, 2),
            "period": "weekly",
            "source": source,
            "timestamp": self.utc_timestamp(),
        }

    # ------------------------------------------------------------------
    def _try_comtrade(self, node_id: str) -> float | None:
        """Try UN Comtrade free API with exponential backoff."""
        metadata = NODE_CATALOG.get(node_id)
        code = metadata.comtrade_reporter_code if metadata else None
        if not code:
            return None

        url = (
            f"https://comtradeapi.un.org/public/v1/preview/C/A/HS"
            f"?reporterCode={code}&period=2025&flowCode=M"
        )

        for attempt in range(3):
            try:
                resp = http_requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # Extract total import value if available
                    datasets = data.get("data", [])
                    if datasets:
                        total = sum(
                            d.get("primaryValue", 0) for d in datasets[:10]
                        )
                        if total > 0:
                            # Convert annual to approximate weekly
                            weekly = total / 52
                            self.logger.info(
                                f"Comtrade OK for {node_id}: {weekly:.0f} USD/week"
                            )
                            return weekly
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    self.logger.warning(f"Comtrade rate limited, retry in {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    self.logger.warning(
                        f"Comtrade status {resp.status_code} for {node_id}"
                    )
                    break
            except Exception as exc:
                self.logger.warning(f"Comtrade error for {node_id}: {exc}")
                break

        return None

    # ------------------------------------------------------------------
    def _try_world_bank(self, node_id: str) -> float | None:
        metadata = NODE_CATALOG.get(node_id)
        if metadata is None:
            return None

        indicator = "NE.TRD.GNFS.CD"
        url = (
            "https://api.worldbank.org/v2/country/"
            f"{metadata.world_bank_country_code}/indicator/{indicator}"
            "?format=json&per_page=5"
        )

        try:
            response = http_requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self.logger.warning(f"World Bank error for {node_id}: {exc}")
            return None

        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return None

        for row in payload[1]:
            value = row.get("value")
            if value:
                weekly = float(value) / 52
                self.logger.info(
                    f"World Bank OK for {node_id}: {weekly:.0f} USD/week"
                )
                return weekly

        return None

    # ------------------------------------------------------------------
    def _try_serpapi_gemini(self, node_id: str, city: str) -> float | None:
        """SerpAPI search + Gemini to estimate trade volume."""
        year = datetime.utcnow().strftime("%Y")
        results = self.serpapi_search(
            f"{city} port trade volume monthly {year}"
        )
        snippets = []
        for r in results.get("organic_results", [])[:5]:
            snippet = r.get("snippet", "")
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return None

        prompt = (
            f"Based on these search results about trade at {city} port, "
            f"estimate the weekly trade volume in USD. Consider any disruptions, "
            f"growth, throughput figures, or reported cargo activity.\n"
            f"If disruptions → lower the estimate. If growth → raise it.\n"
            f'Return JSON only: {{"volume_usd": <float>, "reasoning": "<brief>"}}\n\n'
            + "\n\n".join(snippets[:5])
        )
        raw = self.llm_analyze(prompt)
        parsed = self.parse_json_from_text(raw)
        if parsed and "volume_usd" in parsed:
            vol = float(parsed["volume_usd"])
            self.logger.info(f"Gemini estimated trade for {node_id}: {vol:.0f}")
            return vol

        return None


if __name__ == "__main__":
    agent = TradeAgent()
    agent.run()
