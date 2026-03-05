"""
LAMDA Base Scraper — Shared infrastructure for all scraper agents.

Provides: SerpAPI client, Anthropic Claude wrapper, Flask app factory,
in-memory TTL cache, node-to-city mapping, common endpoints, colorlog logging.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from flask import Flask, jsonify
from flask_cors import CORS
import requests as http_requests
import colorlog

# ---------------------------------------------------------------------------
# Node reference
# ---------------------------------------------------------------------------
SUPPORTED_NODES = ["Hong_Kong", "Singapore", "Shanghai", "Tokyo", "Los_Angeles"]

NODE_CITY_MAP = {
    "Hong_Kong": "Hong Kong",
    "Singapore": "Singapore",
    "Shanghai": "Shanghai",
    "Tokyo": "Tokyo",
    "Los_Angeles": "Los Angeles",
}


class BaseScraper:
    """Base class for all LAMDA scraper agents."""

    def __init__(self, name: str, port: int, cache_ttl_seconds: int = 1800):
        self.name = name
        self.port = port
        self.cache_ttl_seconds = cache_ttl_seconds

        # In-memory cache: {node_id: {"data": ..., "expires_at": datetime}}
        self._cache: Dict[str, Dict[str, Any]] = {}

        # SerpAPI
        self.serpapi_key = os.environ.get("SERPAPI_API_KEY", "")
        self._serpapi_call_count = 0

        # Anthropic
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._anthropic_client = None
        if self.anthropic_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            except Exception:
                pass

        # Logging
        self.logger = self._setup_logging()

        # Flask
        self.app = Flask(self.name)
        CORS(self.app)
        self._register_common_routes()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _setup_logging(self) -> logging.Logger:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------------
    # Common Flask routes
    # ------------------------------------------------------------------
    def _register_common_routes(self):
        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "scraper": self.name, "port": self.port})

        @self.app.route("/api/nodes", methods=["GET"])
        def nodes():
            return jsonify({"nodes": SUPPORTED_NODES})

        @self.app.route("/api/cache/clear", methods=["GET"])
        def cache_clear():
            self._cache.clear()
            self.logger.info("Cache cleared")
            return jsonify({"cleared": True})

    # ------------------------------------------------------------------
    # SerpAPI
    # ------------------------------------------------------------------
    def serpapi_search(self, query: str, search_type: str = "search") -> dict:
        """Perform a SerpAPI search. search_type: 'search' | 'news'."""
        if not self.serpapi_key:
            self.logger.warning("SERPAPI_API_KEY not set — returning empty results")
            return {}

        try:
            from serpapi import GoogleSearch

            params: Dict[str, Any] = {
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5,
            }
            if search_type == "news":
                params["tbm"] = "nws"

            self._serpapi_call_count += 1
            self.logger.info(
                f"SerpAPI call #{self._serpapi_call_count}: {query[:80]}"
            )

            search = GoogleSearch(params)
            results = search.get_dict()
            return results

        except Exception as exc:
            self.logger.error(f"SerpAPI error: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Claude (Anthropic)
    # ------------------------------------------------------------------
    def claude_analyze(self, prompt: str, system: str = None) -> str:
        """Call Claude claude-sonnet-4-20250514 and return the text response."""
        if not self._anthropic_client:
            self.logger.warning("Anthropic client not available — returning empty")
            return ""

        try:
            kwargs: Dict[str, Any] = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = self._anthropic_client.messages.create(**kwargs)
            text = response.content[0].text
            return text

        except Exception as exc:
            self.logger.error(f"Claude API error: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def get_cached(self, node_id: str) -> Optional[dict]:
        entry = self._cache.get(node_id)
        if entry and datetime.utcnow() < entry["expires_at"]:
            self.logger.info(f"Cache HIT for {node_id}")
            return entry["data"]
        return None

    def set_cached(self, node_id: str, data: dict):
        self._cache[node_id] = {
            "data": data,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.cache_ttl_seconds),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def node_to_city(self, node_id: str) -> str:
        return NODE_CITY_MAP.get(node_id, node_id.replace("_", " "))

    @staticmethod
    def utc_timestamp() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def parse_json_from_text(self, text: str) -> dict:
        """Strip markdown fences and parse JSON."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.error(f"JSON parse failed: {cleaned[:200]}")
            return {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        self.logger.info(f"Starting {self.name} on port {self.port}")
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)
