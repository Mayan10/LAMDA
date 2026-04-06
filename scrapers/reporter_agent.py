"""
LAMDA Reporter Credibility Agent — Port 8006
Scores the reliability of data sources for each node.
Uses neutral starting scores and adjusts them dynamically based on the
quality and richness of other scraper responses.
"""

from flask import jsonify
import requests as http_requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES

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
        scores = {"news": 0.5, "political": 0.5, "weather": 0.55}

        # Dynamic adjustment: probe other scrapers
        for category, url_template in SCRAPER_ENDPOINTS.items():
            url = url_template.format(node_id=node_id)
            try:
                resp = http_requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    scores[category] = self._score_category(category, data, scores[category])
                    self.logger.info(
                        f"Adjusted {category} credibility for {node_id} to {scores[category]:.2f}"
                    )
                else:
                    scores[category] = max(0.0, scores[category] - 0.15)
            except Exception as exc:
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

    def _score_category(self, category: str, data: dict, base_score: float) -> float:
        score = base_score

        if category == "news":
            sources = data.get("sources", [])
            summary = data.get("summary", "")
            score += min(len(sources), 4) * 0.1
            if len(summary) > 120:
                score += 0.1

        elif category == "political":
            report = data.get("report", "")
            if len(report) > 80:
                score += 0.15
            if any(
                keyword in report.lower()
                for keyword in ("sanction", "restriction", "conflict", "protest", "strike")
            ):
                score += 0.1

        elif category == "weather":
            conditions = data.get("conditions", "")
            alerts = data.get("alerts", [])
            if conditions and "unavailable" not in conditions.lower():
                score += 0.15
            if any(token in conditions.lower() for token in ("°c", "winds", "precipitation")):
                score += 0.1
            if alerts:
                score += 0.1

        return round(max(0.1, min(score, 0.95)), 2)


if __name__ == "__main__":
    agent = ReporterAgent()
    agent.run()
