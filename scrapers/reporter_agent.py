"""
LAMDA Reporter Credibility Agent — Port 8006
Scores the reliability of data sources for each node.
Semi-static service with hardcoded priors and optional dynamic adjustment
via internal HTTP calls to the other scrapers.
"""

import json
from flask import jsonify
import requests as http_requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES

# ---------------------------------------------------------------------------
# Base credibility priors
# ---------------------------------------------------------------------------
BASE_CREDIBILITY = {
    "Hong_Kong":   {"news": 0.80, "political": 0.70, "weather": 0.90},
    "Singapore":   {"news": 0.85, "political": 0.80, "weather": 0.92},
    "Shanghai":    {"news": 0.70, "political": 0.60, "weather": 0.88},
    "Tokyo":       {"news": 0.88, "political": 0.85, "weather": 0.95},
    "Los_Angeles": {"news": 0.90, "political": 0.87, "weather": 0.95},
}

# Other scraper endpoints for dynamic checks
SCRAPER_ENDPOINTS = {
    "news":      "http://localhost:8002/api/latest/{node_id}",
    "political": "http://localhost:8003/api/latest/{node_id}",
    "weather":   "http://localhost:8005/api/latest/{node_id}",
}


class ReporterAgent(BaseScraper):
    """Reporter credibility service — Port 8006."""

    def __init__(self):
        super().__init__(name="reporter-agent", port=8006, cache_ttl_seconds=3600)  # 60 min
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/credibility/<node_id>", methods=["GET"])
        def credibility(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._compute_credibility(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    def _compute_credibility(self, node_id: str) -> dict:
        base = BASE_CREDIBILITY.get(
            node_id, {"news": 0.75, "political": 0.70, "weather": 0.85}
        )
        scores = {k: v for k, v in base.items()}

        # Dynamic adjustment: probe other scrapers
        for category, url_template in SCRAPER_ENDPOINTS.items():
            url = url_template.format(node_id=node_id)
            try:
                resp = http_requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    # Check if response has substantive data (not just fallback)
                    has_data = False
                    if category == "news" and data.get("sources"):
                        has_data = len(data["sources"]) >= 2
                    elif category == "political" and data.get("report"):
                        has_data = len(data["report"]) > 50
                    elif category == "weather" and data.get("conditions"):
                        has_data = "unavailable" not in data["conditions"].lower()

                    if has_data:
                        # Multiple independent sources found → boost
                        scores[category] = min(1.0, scores[category] + 0.10)
                        self.logger.info(
                            f"Boosted {category} credibility for {node_id} to {scores[category]:.2f}"
                        )
                    else:
                        # Only 1 source or thin data → lower
                        scores[category] = max(0.0, scores[category] - 0.15)
                        self.logger.info(
                            f"Lowered {category} credibility for {node_id} to {scores[category]:.2f}"
                        )
                else:
                    # Scraper returned error → lower credibility
                    scores[category] = max(0.0, scores[category] - 0.15)
            except Exception as exc:
                # Scraper unreachable — keep base prior, log warning
                self.logger.warning(
                    f"Could not reach {category} scraper for {node_id}: {exc}"
                )

        # Round all scores
        scores = {k: round(v, 2) for k, v in scores.items()}

        return {
            "node_id": node_id,
            "scores": scores,
            "updated_at": self.utc_timestamp(),
        }


if __name__ == "__main__":
    agent = ReporterAgent()
    agent.run()
