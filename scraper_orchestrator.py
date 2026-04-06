from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests

from intelligence_processor import ScraperData
from node_metadata import build_graph_edges, build_graph_nodes, SUPPORTED_NODES
from scrapers.gscpi_agent import GSCPIAgent
from scrapers.news_agent import NewsAgent
from scrapers.political_agent import PoliticalAgent
from scrapers.trade_agent import TradeAgent
from scrapers.weather_agent import WeatherAgent


class ScraperOrchestrator:
    def __init__(self, timeout_seconds: float = 20.0):
        self.timeout_seconds = timeout_seconds
        self.http_enabled = os.getenv("SCRAPER_HTTP_ENABLED", "false").lower() not in {
            "0",
            "false",
            "no",
        }
        self.endpoints = {
            "gscpi": os.getenv(
                "GSCPI_SCRAPER_URL",
                "http://127.0.0.1:8001/api/latest/{node_id}",
            ),
            "news": os.getenv(
                "NEWS_SCRAPER_URL",
                "http://127.0.0.1:8002/api/latest/{node_id}",
            ),
            "political": os.getenv(
                "POLITICAL_SCRAPER_URL",
                "http://127.0.0.1:8003/api/latest/{node_id}",
            ),
            "trade": os.getenv(
                "TRADE_SCRAPER_URL",
                "http://127.0.0.1:8004/api/latest/{node_id}",
            ),
            "weather": os.getenv(
                "WEATHER_SCRAPER_URL",
                "http://127.0.0.1:8005/api/latest/{node_id}",
            ),
            "reporter": os.getenv(
                "REPORTER_SCRAPER_URL",
                "http://127.0.0.1:8006/api/credibility/{node_id}",
            ),
        }
        self._local_agents: Dict[str, object] = {}

    def get_supported_nodes(self) -> List[str]:
        return SUPPORTED_NODES[:]

    def build_graph_structure(self) -> tuple[List[dict], List[dict]]:
        return build_graph_nodes(), build_graph_edges()

    def fetch_all(self) -> List[ScraperData]:
        payloads_by_node = {
            node_id: self.fetch_raw_payloads(node_id)
            for node_id in self.get_supported_nodes()
        }
        return [
            self._build_scraper_data(node_id, payloads)
            for node_id, payloads in payloads_by_node.items()
        ]

    def fetch_raw_payloads(self, node_id: str) -> Dict[str, dict]:
        payloads = {
            "gscpi": self._fetch_scraper_payload("gscpi", node_id),
            "trade": self._fetch_scraper_payload("trade", node_id),
            "news": self._fetch_scraper_payload("news", node_id),
            "political": self._fetch_scraper_payload("political", node_id),
            "weather": self._fetch_scraper_payload("weather", node_id),
        }

        reporter_payload = self._fetch_http_payload("reporter", node_id)
        if reporter_payload is None:
            reporter_payload = self._build_reporter_payload(
                node_id=node_id,
                news_payload=payloads["news"],
                political_payload=payloads["political"],
                weather_payload=payloads["weather"],
            )

        payloads["reporter"] = reporter_payload
        return payloads

    def _build_scraper_data(self, node_id: str, payloads: Dict[str, dict]) -> ScraperData:
        weather_conditions = payloads["weather"].get("conditions", "")
        alerts = payloads["weather"].get("alerts", [])
        if alerts:
            alert_summary = "; ".join(
                f"{alert.get('type', 'ALERT')}: {alert.get('description', '')}"
                for alert in alerts[:3]
            )
            weather_text = f"{weather_conditions} Alerts: {alert_summary}"
        else:
            weather_text = weather_conditions

        return ScraperData(
            node_id=node_id,
            gscpi=float(payloads["gscpi"].get("value", 0.0)),
            trade=float(payloads["trade"].get("volume_usd", 0.0)),
            news=payloads["news"].get("summary", ""),
            political=payloads["political"].get("report", ""),
            weather=weather_text,
            reporter_credibility=payloads["reporter"].get(
                "scores",
                {"news": 0.5, "political": 0.5, "weather": 0.5},
            ),
        )

    def _fetch_scraper_payload(self, scraper_name: str, node_id: str) -> dict:
        payload = self._fetch_http_payload(scraper_name, node_id)
        if payload is not None:
            return payload

        return self._fetch_local_payload(scraper_name, node_id)

    def _fetch_http_payload(self, scraper_name: str, node_id: str) -> Optional[dict]:
        if not self.http_enabled:
            return None

        url = self.endpoints[scraper_name].format(node_id=node_id)
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            return None

        try:
            return response.json()
        except ValueError:
            return None

    def _fetch_local_payload(self, scraper_name: str, node_id: str) -> dict:
        if scraper_name == "gscpi":
            return self._get_local_agent("gscpi")._fetch_gscpi(node_id)
        if scraper_name == "trade":
            return self._get_local_agent("trade")._fetch_trade(node_id)
        if scraper_name == "news":
            return self._get_local_agent("news")._fetch_news(node_id)
        if scraper_name == "political":
            return self._get_local_agent("political")._fetch_political(node_id)
        if scraper_name == "weather":
            return self._get_local_agent("weather")._fetch_weather(node_id)
        raise ValueError(f"Unsupported scraper: {scraper_name}")

    def _get_local_agent(self, scraper_name: str):
        if scraper_name in self._local_agents:
            return self._local_agents[scraper_name]

        constructors = {
            "gscpi": GSCPIAgent,
            "news": NewsAgent,
            "political": PoliticalAgent,
            "trade": TradeAgent,
            "weather": WeatherAgent,
        }
        agent = constructors[scraper_name]()
        self._local_agents[scraper_name] = agent
        return agent

    def _build_reporter_payload(
        self,
        node_id: str,
        news_payload: dict,
        political_payload: dict,
        weather_payload: dict,
    ) -> dict:
        summary = news_payload.get("summary", "")
        news_sources = news_payload.get("sources", [])
        political_report = political_payload.get("report", "")
        weather_conditions = weather_payload.get("conditions", "")
        weather_alerts = weather_payload.get("alerts", [])

        scores = {
            "news": self._score_news_credibility(summary, news_sources),
            "political": self._score_political_credibility(political_report),
            "weather": self._score_weather_credibility(
                weather_conditions,
                weather_alerts,
            ),
        }

        return {
            "node_id": node_id,
            "scores": scores,
            "updated_at": weather_payload.get("timestamp")
            or political_payload.get("timestamp")
            or news_payload.get("timestamp"),
        }

    @staticmethod
    def _score_news_credibility(summary: str, sources: List[str]) -> float:
        if not summary:
            return 0.4
        score = 0.45
        score += min(len(sources), 4) * 0.1
        if len(summary) >= 120:
            score += 0.1
        if "fallback" in summary.lower():
            score -= 0.15
        return round(max(0.1, min(score, 0.95)), 2)

    @staticmethod
    def _score_political_credibility(report: str) -> float:
        if not report:
            return 0.4
        score = 0.5
        if len(report) >= 100:
            score += 0.15
        if any(
            keyword in report.lower()
            for keyword in ("sanction", "restriction", "strike", "protest", "conflict")
        ):
            score += 0.1
        return round(max(0.1, min(score, 0.9)), 2)

    @staticmethod
    def _score_weather_credibility(conditions: str, alerts: List[dict]) -> float:
        if not conditions:
            return 0.45
        score = 0.55
        if "unavailable" not in conditions.lower():
            score += 0.15
        if any(token in conditions.lower() for token in ("°c", "winds", "precipitation")):
            score += 0.1
        if alerts:
            score += 0.1
        return round(max(0.1, min(score, 0.95)), 2)
