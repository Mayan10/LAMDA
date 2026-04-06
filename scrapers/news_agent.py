"""
LAMDA News Agent — Port 8002
Fetches latest supply-chain-relevant news for each port/city.
Uses SerpAPI Google News + Claude summarization.
"""

import json
from flask import jsonify

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES


class NewsAgent(BaseScraper):
    """News scraper — Port 8002."""

    def __init__(self):
        super().__init__(name="news-agent", port=8002, cache_ttl_seconds=900)  # 15 min
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/latest/<node_id>", methods=["GET"])
        def latest(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._fetch_news(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    def _fetch_news(self, node_id: str) -> dict:
        city = self.node_to_city(node_id)

        articles = []
        sources = []

        # 1. SerpAPI news search — supply chain focus
        results1 = self.serpapi_search(
            f"{city} port shipping supply chain news", search_type="news"
        )
        for item in results1.get("news_results", [])[:5]:
            articles.append(f"{item.get('title', '')}: {item.get('snippet', '')}")
            link = item.get("link", item.get("source", {}).get("name", ""))
            if link:
                sources.append(link)

        # 2. SerpAPI news search — disruptions
        results2 = self.serpapi_search(
            f"{city} logistics disruption strike protest", search_type="news"
        )
        for item in results2.get("news_results", [])[:3]:
            articles.append(f"{item.get('title', '')}: {item.get('snippet', '')}")
            link = item.get("link", item.get("source", {}).get("name", ""))
            if link and link not in sources:
                sources.append(link)

        # 3. Claude summarization
        summary = ""
        if articles:
            prompt = (
                f"Summarize the following news about {city} in max 500 characters, "
                f"focusing on supply chain impacts. If no disruptions, say so clearly.\n"
                f'Return JSON: {{"summary": "<text>", "sources": ["url1", "url2"]}}\n\n'
                + "\n\n".join(articles[:8])
            )
            raw = self.claude_analyze(prompt)
            parsed = self.parse_json_from_text(raw)
            if parsed:
                summary = parsed.get("summary", "")
                extra_sources = parsed.get("sources", [])
                for s in extra_sources:
                    if s not in sources:
                        sources.append(s)

        # Fallback
        if not summary:
            summary = (
                f"No significant supply chain disruptions reported for {city}. "
                f"Port operations appear normal."
            )
            self.logger.warning(f"Using fallback summary for {node_id}")

        return {
            "node_id": node_id,
            "summary": summary[:500],
            "sources": sources[:5],
            "timestamp": self.utc_timestamp(),
        }


if __name__ == "__main__":
    agent = NewsAgent()
    agent.run()
