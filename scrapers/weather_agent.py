"""
LAMDA Weather Agent — Port 8005
Fetches current weather and maritime alerts for each port.
Primary: Open-Meteo (free, no key). Secondary: SerpAPI for alerts.
Optional: OpenWeatherMap if OPENWEATHERMAP_API_KEY is set.
"""

import os
import json
from flask import jsonify
import requests as http_requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from base_scraper import BaseScraper, SUPPORTED_NODES

# ---------------------------------------------------------------------------
# Port coordinates
# ---------------------------------------------------------------------------
PORT_COORDS = {
    "Hong_Kong":   {"lat": 22.3193, "lon": 114.1694},
    "Singapore":   {"lat": 1.3521,  "lon": 103.8198},
    "Shanghai":    {"lat": 31.2304, "lon": 121.4737},
    "Tokyo":       {"lat": 35.6762, "lon": 139.6503},
    "Los_Angeles": {"lat": 33.7288, "lon": -118.2620},
}

# ---------------------------------------------------------------------------
# WMO weather-code → human-readable description
# ---------------------------------------------------------------------------
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherAgent(BaseScraper):
    """Weather scraper — Port 8005."""

    def __init__(self):
        super().__init__(name="weather-agent", port=8005, cache_ttl_seconds=600)  # 10 min
        self.owm_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")
        self._register_routes()

    def _register_routes(self):
        @self.app.route("/api/latest/<node_id>", methods=["GET"])
        def latest(node_id):
            if node_id not in SUPPORTED_NODES:
                return jsonify({"error": f"Unsupported node: {node_id}"}), 404

            cached = self.get_cached(node_id)
            if cached:
                return jsonify(cached)

            data = self._fetch_weather(node_id)
            self.set_cached(node_id, data)
            return jsonify(data)

    # ------------------------------------------------------------------
    # Core fetch logic
    # ------------------------------------------------------------------
    def _fetch_weather(self, node_id: str) -> dict:
        coords = PORT_COORDS.get(node_id, {"lat": 0, "lon": 0})
        city = self.node_to_city(node_id)

        conditions = ""
        alerts = []

        # 1. Open-Meteo (primary — free, no key)
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={coords['lat']}&longitude={coords['lon']}"
                f"&current=temperature_2m,windspeed_10m,weathercode,precipitation"
            )
            resp = http_requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                temp = current.get("temperature_2m", "N/A")
                wind = current.get("windspeed_10m", "N/A")
                code = current.get("weathercode", 0)
                precip = current.get("precipitation", 0)
                description = WMO_CODES.get(code, "Unknown")

                conditions = (
                    f"{description}, {temp}°C, winds {wind} km/h. "
                    f"Precipitation: {precip} mm. Port operations "
                    f"{'may be affected' if code >= 61 else 'normal'}."
                )
                self.logger.info(f"Open-Meteo OK for {node_id}")
        except Exception as exc:
            self.logger.warning(f"Open-Meteo failed for {node_id}: {exc}")

        # 2. Optional OWM for richer data + alerts
        if self.owm_key:
            try:
                url = (
                    f"https://api.openweathermap.org/data/2.5/weather"
                    f"?lat={coords['lat']}&lon={coords['lon']}"
                    f"&appid={self.owm_key}&units=metric"
                )
                resp = http_requests.get(url, timeout=10)
                if resp.status_code == 200:
                    owm = resp.json()
                    if not conditions:
                        desc = owm.get("weather", [{}])[0].get("description", "")
                        temp = owm["main"]["temp"]
                        wind = owm["wind"]["speed"]
                        conditions = f"{desc.capitalize()}, {temp}°C, winds {wind} m/s."
                    self.logger.info(f"OWM OK for {node_id}")
            except Exception as exc:
                self.logger.warning(f"OWM failed for {node_id}: {exc}")

        # 3. SerpAPI for weather alerts / marine advisories
        try:
            from datetime import datetime
            current_date = datetime.utcnow().strftime("%Y-%m-%d")
            results = self.serpapi_search(
                f"{city} port weather alert typhoon storm fog {current_date}"
            )
            snippets = []
            for r in results.get("organic_results", [])[:3]:
                snippet = r.get("snippet", "")
                title = r.get("title", "")
                if snippet:
                    snippets.append(f"{title}: {snippet}")

            # Detect alerts from snippets
            alert_keywords = {
                "typhoon": ("TYPHOON_WARNING", "severe"),
                "hurricane": ("HURRICANE_WARNING", "severe"),
                "storm": ("STORM_WARNING", "high"),
                "fog": ("FOG_ADVISORY", "moderate"),
                "tsunami": ("TSUNAMI_WARNING", "severe"),
                "flood": ("FLOOD_WARNING", "high"),
                "cyclone": ("CYCLONE_WARNING", "severe"),
            }
            for snippet_text in snippets:
                lower = snippet_text.lower()
                for kw, (alert_type, severity) in alert_keywords.items():
                    if kw in lower:
                        alerts.append({
                            "type": alert_type,
                            "severity": severity,
                            "description": snippet_text[:200],
                            "expires": (
                                datetime.utcnow()
                                .replace(hour=23, minute=59, second=59)
                                .isoformat() + "Z"
                            ),
                        })
                        break
        except Exception as exc:
            self.logger.warning(f"SerpAPI weather alerts error for {node_id}: {exc}")

        # Fallback conditions
        if not conditions:
            conditions = f"Weather data temporarily unavailable for {city}. Port operations assumed normal."
            self.logger.warning(f"Using fallback conditions for {node_id}")

        return {
            "node_id": node_id,
            "conditions": conditions[:300],
            "alerts": alerts,
            "timestamp": self.utc_timestamp(),
        }


if __name__ == "__main__":
    agent = WeatherAgent()
    agent.run()
