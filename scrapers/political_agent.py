"""
LAMDA Political Agent — Port 8003
Fetches political risk information relevant to trade/shipping.
Uses SerpAPI search + Claude synthesis.
"""

import json
from datetime import datetime
from flask import jsonify

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES


class PoliticalAgent(BaseScraper):
    """Political risk scraper — Port 8003."""

    def __init__(self):
        super().__init__(name="political-agent", port=8003, cache_ttl_seconds=1800)  # 30 min
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/latest/<node_id>", methods=["GET"])
        def latest(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._fetch_political(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    def _fetch_political(self, node_id: str) -> dict:
        city = self.node_to_city(node_id)
        now = datetime.utcnow()
        year = now.strftime("%Y")
        month = now.strftime("%B")

        snippets = []

        # 1. SerpAPI — sanctions / restrictions
        results1 = self.serpapi_search(
            f"{city} trade sanctions port restrictions political {year}"
        )
        for r in results1.get("organic_results", [])[:5]:
            snippet = r.get("snippet", "")
            if snippet:
                snippets.append(snippet)

        # 2. SerpAPI — protests / strikes
        results2 = self.serpapi_search(
            f"{city} protests strikes port workers {month} {year}"
        )
        for r in results2.get("organic_results", [])[:3]:
            snippet = r.get("snippet", "")
            if snippet and snippet not in snippets:
                snippets.append(snippet)

        # 3. Claude synthesis
        report = ""
        if snippets:
            prompt = (
                f"Summarize the political situation in {city} relevant to shipping "
                f"and trade in max 500 characters. Include any sanctions, restrictions, "
                f"or labor disputes.\n"
                f'Return JSON: {{"report": "<text>"}}\n\n'
                + "\n\n".join(snippets[:8])
            )
            raw = self.claude_analyze(prompt)
            parsed = self.parse_json_from_text(raw)
            if parsed:
                report = parsed.get("report", "")

        # Fallback
        if not report:
            report = (
                f"No active trade sanctions or port restrictions found for {city}. "
                f"Political environment appears stable."
            )
            self.logger.warning(f"Using fallback report for {node_id}")

        return {
            "node_id": node_id,
            "report": report[:500],
            "timestamp": self.utc_timestamp(),
        }


if __name__ == "__main__":
    agent = PoliticalAgent()
    agent.run()
