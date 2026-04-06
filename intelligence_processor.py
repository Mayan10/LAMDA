import json
import os
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None
    genai_types = None


@dataclass
class ScraperData:
    node_id: str
    gscpi: float
    trade: float
    news: str
    political: str
    weather: str
    reporter_credibility: Dict[str, float]


@dataclass
class RiskVector:
    node_id: str
    gscpi_risk: float
    news_risk: float
    political_risk: float
    trade_risk: float
    weather_risk: float
    reporter_confidence: float
    timestamp: str


class IntelligenceProcessor:
    
    def __init__(self, api_key: str = None):
        self.provider = os.getenv("INTELLIGENCE_LLM_PROVIDER", "").strip().lower()
        self.gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.model = None

        if not self.provider:
            if self.gemini_api_key and genai is not None:
                self.provider = "gemini"
            else:
                self.provider = "heuristic"

        if self.provider == "gemini" and self.gemini_api_key and genai is not None:
            self.client = genai.Client(api_key=self.gemini_api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            self.provider = "heuristic"
        
        self.gscpi_min = 0.0
        self.gscpi_max = 3.0

    def normalize_gscpi(self, gscpi_value: float) -> float:
        normalized = (gscpi_value - self.gscpi_min) / (self.gscpi_max - self.gscpi_min)
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_trade(
        self,
        trade_volume: float,
        trade_min: float,
        trade_max: float,
    ) -> float:
        if trade_max <= trade_min:
            return 0.5

        normalized = (trade_volume - trade_min) / (trade_max - trade_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return 1.0 - normalized
    
    def analyze_text_with_llm(self, nodes_data: List[ScraperData]) -> Dict[str, Dict[str, float]]:
        if self.client is None or self.provider == "heuristic":
            return self.analyze_text_heuristically(nodes_data)

        batch_context = self._build_batch_prompt(nodes_data)
        
        try:
            if self.provider == "gemini":
                response_text = self._analyze_with_gemini(batch_context)
            else:
                return self.analyze_text_heuristically(nodes_data)

            risk_scores = self._parse_llm_response(response_text)
            
            return risk_scores or self.analyze_text_heuristically(nodes_data)
            
        except Exception as e:
            print(f"Error calling {self.provider} API: {e}")
            return self.analyze_text_heuristically(nodes_data)

    def _analyze_with_gemini(self, batch_context: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=batch_context,
            config=genai_types.GenerateContentConfig(
                system_instruction=self._get_system_prompt(),
                temperature=0.3,
                max_output_tokens=4000,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text or ""

    def analyze_text_heuristically(
        self,
        nodes_data: List[ScraperData],
    ) -> Dict[str, Dict[str, float]]:
        return {
            node.node_id: {
                "news_risk": self._score_text(
                    node.news,
                    high_risk_keywords={
                        "delay": 0.08,
                        "disruption": 0.16,
                        "closure": 0.25,
                        "congestion": 0.12,
                        "strike": 0.22,
                        "shortage": 0.15,
                        "backlog": 0.12,
                        "slowdown": 0.08,
                    },
                    low_risk_keywords={
                        "normal": -0.12,
                        "stable": -0.1,
                        "record throughput": -0.1,
                        "on schedule": -0.08,
                        "smoothly": -0.08,
                    },
                ),
                "political_risk": self._score_text(
                    node.political,
                    high_risk_keywords={
                        "sanction": 0.22,
                        "restriction": 0.18,
                        "conflict": 0.22,
                        "protest": 0.15,
                        "strike": 0.14,
                        "tariff": 0.12,
                        "export control": 0.14,
                        "instability": 0.18,
                    },
                    low_risk_keywords={
                        "stable": -0.12,
                        "facilitation": -0.08,
                        "agreement": -0.08,
                        "strong regulatory": -0.06,
                    },
                ),
                "weather_risk": self._score_text(
                    node.weather,
                    high_risk_keywords={
                        "typhoon": 0.3,
                        "hurricane": 0.3,
                        "storm": 0.18,
                        "fog": 0.12,
                        "flood": 0.18,
                        "thunderstorm": 0.18,
                        "heavy rain": 0.14,
                        "warning": 0.16,
                        "affected": 0.12,
                    },
                    low_risk_keywords={
                        "clear": -0.12,
                        "normal": -0.1,
                        "no weather disruptions": -0.14,
                        "partly cloudy": -0.04,
                    },
                ),
            }
            for node in nodes_data
        }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert supply chain risk analyst. Your task is to analyze news, political, and weather information for various cities/ports and output risk scores.

For each location, analyze:
1. NEWS: Strikes, disruptions, port closures, infrastructure issues
2. POLITICAL: Sanctions, conflicts, regulatory changes, instability
3. WEATHER: Storms, typhoons, extreme conditions affecting logistics

Output ONLY a JSON object with this exact structure:
{
  "node_id_1": {
    "news_risk": 0.0-1.0,
    "political_risk": 0.0-1.0,
    "weather_risk": 0.0-1.0
  },
  "node_id_2": { ... }
}

Risk scale:
- 0.0-0.3: Low risk (normal operations)
- 0.3-0.6: Moderate risk (minor delays possible)
- 0.6-0.8: High risk (significant disruptions likely)
- 0.8-1.0: Severe risk (avoid if possible)

Be concise and objective. Output ONLY the JSON, no explanation."""
    
    def _build_batch_prompt(self, nodes_data: List[ScraperData]) -> str:
        prompt = "Analyze the following supply chain locations:\n\n"
        
        for node in nodes_data:
            prompt += f"{node.node_id}\n"
            prompt += f"NEWS: {node.news[:500]}...\n"
            prompt += f"POLITICAL: {node.political[:500]}...\n"
            prompt += f"WEATHER: {node.weather[:300]}...\n\n"
        
        prompt += "\nProvide risk scores for all locations in JSON format."
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Dict[str, float]]:
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            risk_scores = json.loads(response.strip())
            return risk_scores
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response: {e}")
            print(f"Response: {response}")
            return {}

    def _score_text(
        self,
        text: str,
        high_risk_keywords: Dict[str, float],
        low_risk_keywords: Dict[str, float],
    ) -> float:
        if not text:
            return 0.5

        score = 0.5
        lowered = text.lower()

        for keyword, weight in high_risk_keywords.items():
            if keyword in lowered:
                score += weight

        for keyword, weight in low_risk_keywords.items():
            if keyword in lowered:
                score += weight

        if len(lowered) > 250:
            score += 0.03

        return float(np.clip(score, 0.0, 1.0))
    
    def apply_reporter_weights(self, 
                               base_risk: float, 
                               credibility: float) -> float:
        adjusted_risk = base_risk * credibility + 0.5 * (1 - credibility)
        return np.clip(adjusted_risk, 0.0, 1.0)
    
    def process_batch(self, nodes_data: List[ScraperData]) -> List[RiskVector]:
        print(f"Processing {len(nodes_data)} nodes")
        
        print("Step 1: Normalizing GSCPI and Trade data")
        trade_values = [max(node.trade, 0.0) for node in nodes_data]
        trade_min = min(trade_values) if trade_values else 0.0
        trade_max = max(trade_values) if trade_values else 1.0

        gscpi_risks = {
            node.node_id: self.normalize_gscpi(node.gscpi)
            for node in nodes_data
        }
        trade_risks = {
            node.node_id: self.normalize_trade(node.trade, trade_min, trade_max)
            for node in nodes_data
        }
        print(f"Step 2: Analyzing text data with {self.provider} intelligence")
        llm_risks = self.analyze_text_with_llm(nodes_data)
        
        print("Step 3: Applying Reporter credibility weights")
        risk_vectors = []
        timestamp = datetime.utcnow().isoformat()
        
        for node in nodes_data:
            gscpi_risk = gscpi_risks[node.node_id]
            trade_risk = trade_risks[node.node_id]
            
            llm_risk = llm_risks.get(node.node_id, {
                "news_risk": 0.5,
                "political_risk": 0.5,
                "weather_risk": 0.5
            })
            reporter_confidence = np.mean(list(node.reporter_credibility.values()))
            
            news_risk = self.apply_reporter_weights(
                llm_risk["news_risk"],
                node.reporter_credibility.get("news", 0.5)
            )
            political_risk = self.apply_reporter_weights(
                llm_risk["political_risk"],
                node.reporter_credibility.get("political", 0.5)
            )
            weather_risk = self.apply_reporter_weights(
                llm_risk["weather_risk"],
                node.reporter_credibility.get("weather", 0.5)
            )
            
            risk_vector = RiskVector(
                node_id=node.node_id,
                gscpi_risk=gscpi_risk,
                news_risk=news_risk,
                political_risk=political_risk,
                trade_risk=trade_risk,
                weather_risk=weather_risk,
                reporter_confidence=reporter_confidence,
                timestamp=timestamp
            )
            
            risk_vectors.append(risk_vector)
        
        print(f"Processed {len(risk_vectors)} risk vectors")
        return risk_vectors
    
    def export_to_json(self, risk_vectors: List[RiskVector], filepath: str):
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_nodes": len(risk_vectors),
            "risk_vectors": [
                {
                    "node_id": rv.node_id,
                    "gscpi_risk": float(rv.gscpi_risk),
                    "news_risk": float(rv.news_risk),
                    "political_risk": float(rv.political_risk),
                    "trade_risk": float(rv.trade_risk),
                    "weather_risk": float(rv.weather_risk),
                    "reporter_confidence": float(rv.reporter_confidence)
                }
                for rv in risk_vectors
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported risk vectors to {filepath}")

if __name__ == "__main__":
    print("Run api_server.py to execute the live intelligence pipeline.")
