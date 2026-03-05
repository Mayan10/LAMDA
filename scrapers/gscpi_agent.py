"""
LAMDA GSCPI Agent — Port 8001
Estimates a per-node Global Supply Chain Pressure Index proxy.
Uses SerpAPI for supply chain news + Claude to synthesize a numeric value in [0.0, 3.0].
"""

import json
from datetime import datetime
from flask import jsonify

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES


class GSCPIAgent(BaseScraper):
    """GSCPI proxy scraper — Port 8001."""

    FALLBACK_VALUE = 1.0  # moderate pressure

    def __init__(self):
        super().__init__(name="gscpi-agent", port=8001, cache_ttl_seconds=1800)  # 30 min
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/latest/<node_id>", methods=["GET"])
        def latest(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._fetch_gscpi(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    def _fetch_gscpi(self, node_id: str) -> dict:
        city = self.node_to_city(node_id)
        now = datetime.utcnow()
        month = now.strftime("%B")
        year = now.strftime("%Y")
        value = None

        snippets = []

        # 1. SerpAPI — supply chain pressure
        results1 = self.serpapi_search(
            f"{city} port supply chain pressure {month} {year}"
        )
        for r in results1.get("organic_results", [])[:5]:
            snippet = r.get("snippet", "")
            if snippet:
                snippets.append(snippet)

        # 2. SerpAPI — freight / shipping delays
        results2 = self.serpapi_search(
            f"{city} shipping delays freight rates {year}"
        )
        for r in results2.get("organic_results", [])[:3]:
            snippet = r.get("snippet", "")
            if snippet and snippet not in snippets:
                snippets.append(snippet)

        # 3. Claude synthesis → numeric value
        if snippets:
            prompt = (
                f"Based on these news snippets about {city} port/supply chain activity, "
                f"estimate a supply chain pressure index value between 0.0 (no pressure) "
                f"and 3.0 (extreme pressure). Consider congestion, delays, freight rates, "
                f"and disruptions.\n"
                f'Return only a JSON: {{"value": <float>}}\n\n'
                + "\n\n".join(snippets[:8])
            )
            raw = self.claude_analyze(prompt)
            parsed = self.parse_json_from_text(raw)
            if parsed and "value" in parsed:
                value = float(parsed["value"])
                value = max(0.0, min(3.0, value))  # clamp
                self.logger.info(f"Claude GSCPI for {node_id}: {value}")

        # Fallback
        if value is None:
            value = self.FALLBACK_VALUE
            self.logger.warning(
                f"Using fallback GSCPI {value} for {node_id}"
            )

        return {
            "node_id": node_id,
            "value": round(value, 2),
            "timestamp": self.utc_timestamp(),
        }


if __name__ == "__main__":
    agent = GSCPIAgent()
    agent.run()
