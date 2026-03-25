from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging
from typing import Dict, List, Any
from intelligence_processor import IntelligenceProcessor, ScraperData
from graph_risk_engine import GraphRiskEngine
from route_optimizer import RouteOptimizer

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

intelligence_processor = None
graph_engine = None
route_optimizer = None

CONFIG = {
    "UPDATE_INTERVAL_MINUTES": 30,
    "GRAPH_STATE_FILE": "graph_state.json",
    "RISK_VECTORS_FILE": "risk_vectors_output.json",
    "DB_PATH": "supply_chain_graph.db",
    "GRAPH_NODES_FILE": "graph_nodes.json",
    "GRAPH_EDGES_FILE": "graph_edges.json"
}

def fetch_scraper_data() -> List[ScraperData]:
    """
    Fetch data from all 6 scrapers.
    
    PRODUCTION INTEGRATION:
    Replace the mock data below with actual API calls to your scrapers.
    See SCRAPER_INTEGRATION.md for detailed instructions.
    
    Expected scraper endpoints:
    - GSCPI: GET http://gscpi-scraper:8001/api/latest/{node_id}
    - News: GET http://news-scraper:8002/api/latest/{node_id}
    - Political: GET http://political-scraper:8003/api/latest/{node_id}
    - Trade: GET http://trade-scraper:8004/api/latest/{node_id}
    - Weather: GET http://weather-scraper:8005/api/latest/{node_id}
    - Reporter: GET http://reporter-service:8006/api/credibility/{node_id}
    
    Returns:
        List of ScraperData objects
    """
    logger.info("Fetching data from scrapers")
    
    # TODO: PRODUCTION - Replace with actual scraper API calls
    # Example implementation:
    # import requests
    # node_ids = list(graph_engine.nodes.keys())
    # nodes_data = []
    # for node_id in node_ids:
    #     gscpi = requests.get(f"http://gscpi-scraper:8001/api/latest/{node_id}", timeout=10).json()
    #     news = requests.get(f"http://news-scraper:8002/api/latest/{node_id}", timeout=10).json()
    #     # ... fetch from all 6 scrapers
    #     nodes_data.append(ScraperData(
    #         node_id=node_id,
    #         gscpi=gscpi["value"],
    #         trade=trade["volume_usd"],
    #         news=news["summary"],
    #         political=political["report"],
    #         weather=weather["conditions"],
    #         reporter_credibility=reporter["scores"]
    #     ))
    # return nodes_data
    
    # Mock data for demonstration (REMOVE IN PRODUCTION)
    mock_data = [
        ScraperData(
            node_id="Hong_Kong",
            gscpi=1.45,
            trade=8500000,
            news="Port operations normal. Minor delays due to customs processing.",
            political="Stable governance. New trade facilitation measures announced.",
            weather="Clear skies. No weather disruptions expected this week.",
            reporter_credibility={"news": 0.85, "political": 0.75, "weather": 0.95}
        ),
        ScraperData(
            node_id="Singapore",
            gscpi=0.85,
            trade=12000000,
            news="Record throughput at Singapore port. Expansion project on schedule.",
            political="Strong regulatory framework. Free trade agreements active.",
            weather="Typical tropical conditions. No major weather events forecasted.",
            reporter_credibility={"news": 0.90, "political": 0.85, "weather": 0.90}
        ),
        
    ]
    
    logger.info(f"Fetched data for {len(mock_data)} nodes")
    return mock_data


def update_graph_pipeline():

    try:
        logger.info("="*60)
        logger.info("Starting scheduled graph update pipeline")
        logger.info("="*60)
        
        scraper_data = fetch_scraper_data()
        
        logger.info("Step 1: Processing through Intelligence Processor...")
        risk_vectors = intelligence_processor.process_batch(scraper_data)
        intelligence_processor.export_to_json(
            risk_vectors, 
            CONFIG["RISK_VECTORS_FILE"]
        )
        
        logger.info("Step 2: Updating Graph Risk Engine...")
        graph_engine.update_risk_vectors(CONFIG["RISK_VECTORS_FILE"])
        updated_risks = graph_engine.propagate_risks()
        graph_engine.store_snapshot()
        graph_engine.export_graph_state(CONFIG["GRAPH_STATE_FILE"])
        
        logger.info("Step 3: Reloading Route Optimizer...")
        global route_optimizer
        route_optimizer = RouteOptimizer(CONFIG["GRAPH_STATE_FILE"])
        
        logger.info("✓ Graph update pipeline completed successfully")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in graph update pipeline: {e}", exc_info=True)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Supply Chain Risk Prediction System API",
        "status": "ok",
        "docs": "See README.md for full API documentation.",
        "endpoints": {
            "health": "GET /api/health",
            "analyze_route": "POST /api/analyze_route",
            "node_status": "GET /api/node_status/<node_id>",
            "graph_snapshot": "GET /api/graph_snapshot",
            "historical_trends": "GET /api/historical_trends/<node_id>?limit=10",
            "available_nodes": "GET /api/available_nodes",
            "update_graph": "POST /api/update_graph"
        },
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return "", 204


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "intelligence_processor": intelligence_processor is not None,
            "graph_engine": graph_engine is not None,
            "route_optimizer": route_optimizer is not None
        }
    })


@app.route('/api/analyze_route', methods=['POST'])
def analyze_route():

    try:
        data = request.get_json()
        
        if not data or 'source' not in data or 'destination' not in data:
            return jsonify({
                "error": "Missing required fields: source, destination"
            }), 400
        
        source = data['source']
        destination = data['destination']
        num_routes = data.get('num_routes', 3)
        
        logger.info(f"Route analysis requested: {source} → {destination}")
        
        routes = route_optimizer.find_k_best_routes(
            source, 
            destination, 
            k=num_routes
        )
        
        if not routes:
            return jsonify({
                "error": f"No routes found between {source} and {destination}"
            }), 404
        
        response = {
            "source": source,
            "destination": destination,
            "num_routes": len(routes),
            "routes": [
                {
                    "rank": i + 1,
                    "path": route.path,
                    "total_distance_km": round(route.total_distance_km, 2),
                    "avg_risk": round(route.avg_risk, 3),
                    "estimated_time_hours": round(route.estimated_time_hours, 2),
                    "estimated_time_days": round(route.estimated_time_hours / 24, 1),
                    "route_score": round(route.route_score, 2),
                    "analysis": route_optimizer.analyze_route(route)
                }
                for i, route in enumerate(routes)
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Found {len(routes)} routes")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/node_status/<node_id>', methods=['GET'])
def get_node_status(node_id: str):
    try:
        if node_id not in graph_engine.nodes:
            return jsonify({
                "error": f"Node '{node_id}' not found"
            }), 404
        
        node = graph_engine.nodes[node_id]
        risk_vector = node.risk_vector
        overall_risk = graph_engine.get_node_risk_score(node_id)
        
        if overall_risk < 0.3:
            risk_level = "LOW"
        elif overall_risk < 0.6:
            risk_level = "MODERATE"
        elif overall_risk < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "SEVERE"
        
        response = {
            "node_id": node_id,
            "latitude": node.latitude,
            "longitude": node.longitude,
            "current_risk": {
                "gscpi_risk": round(float(risk_vector[0]), 3),
                "news_risk": round(float(risk_vector[1]), 3),
                "political_risk": round(float(risk_vector[2]), 3),
                "trade_risk": round(float(risk_vector[3]), 3),
                "weather_risk": round(float(risk_vector[4]), 3),
                "reporter_confidence": round(float(risk_vector[5]), 3),
                "overall_risk": round(overall_risk, 3)
            },
            "risk_level": risk_level,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_node_status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/graph_snapshot', methods=['GET'])
def get_graph_snapshot():

    try:
        with open(CONFIG["GRAPH_STATE_FILE"], 'r') as f:
            graph_state = json.load(f)
        
        risks = [node["overall_risk"] for node in graph_state["nodes"]]
        avg_risk = sum(risks) / len(risks)
        high_risk_nodes = sum(1 for r in risks if r > 0.7)
        low_risk_nodes = sum(1 for r in risks if r < 0.3)
        
        response = {
            "timestamp": graph_state["timestamp"],
            "num_nodes": len(graph_state["nodes"]),
            "num_edges": len(graph_state["edges"]),
            "nodes": graph_state["nodes"],
            "edges": graph_state["edges"],
            "statistics": {
                "avg_risk": round(avg_risk, 3),
                "high_risk_nodes": high_risk_nodes,
                "moderate_risk_nodes": len(risks) - high_risk_nodes - low_risk_nodes,
                "low_risk_nodes": low_risk_nodes
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_graph_snapshot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/historical_trends/<node_id>', methods=['GET'])
def get_historical_trends(node_id: str):
    try:
        limit = request.args.get('limit', default=10, type=int)
        
        history = graph_engine.get_historical_trends(node_id, limit)
        
        if not history:
            return jsonify({
                "error": f"No historical data found for node '{node_id}'"
            }), 404
        
        response = {
            "node_id": node_id,
            "num_records": len(history),
            "history": history
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_historical_trends: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/update_graph', methods=['POST'])
def trigger_graph_update():
    try:
        update_graph_pipeline()
        
        return jsonify({
            "status": "success",
            "message": "Graph update completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in trigger_graph_update: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/available_nodes', methods=['GET'])
def get_available_nodes():
    try:
        nodes_info = [
            {
                "node_id": node_id,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "overall_risk": round(graph_engine.get_node_risk_score(node_id), 3)
            }
            for node_id, node in graph_engine.nodes.items()
        ]
        
        return jsonify({
            "num_nodes": len(nodes_info),
            "nodes": nodes_info
        })
        
    except Exception as e:
        logger.error(f"Error in get_available_nodes: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def initialize_system():
    global intelligence_processor, graph_engine, route_optimizer
    
    logger.info("="*60)
    logger.info("Initializing Supply Chain Risk Prediction System")
    logger.info("="*60)
    
    try:
        logger.info("1. Initializing Intelligence Processor...")
        intelligence_processor = IntelligenceProcessor()
        logger.info("2. Initializing Graph Risk Engine...")
        graph_engine = GraphRiskEngine(CONFIG["DB_PATH"])
        
        if os.path.exists(CONFIG["GRAPH_NODES_FILE"]) and \
           os.path.exists(CONFIG["GRAPH_EDGES_FILE"]):
            graph_engine.load_graph_structure(
                CONFIG["GRAPH_NODES_FILE"],
                CONFIG["GRAPH_EDGES_FILE"]
            )
        else:
            logger.warning("Graph structure files not found. Please provide:")
            logger.warning(f"  - {CONFIG['GRAPH_NODES_FILE']}")
            logger.warning(f"  - {CONFIG['GRAPH_EDGES_FILE']}")
        
        if not os.path.exists(CONFIG["GRAPH_STATE_FILE"]):
            logger.info("3. Running initial graph update...")
            update_graph_pipeline()
        else:
            logger.info("3. Loading existing graph state...")
        
        logger.info("4. Initializing Route Optimizer...")
        route_optimizer = RouteOptimizer(CONFIG["GRAPH_STATE_FILE"])
        
        logger.info("="*60)
        logger.info("✓ System initialization complete")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}", exc_info=True)
        return False


def setup_scheduler():
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        func=update_graph_pipeline,
        trigger="interval",
        minutes=CONFIG["UPDATE_INTERVAL_MINUTES"],
        id="graph_update",
        name="Update supply chain graph",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info(f"Scheduler started (updates every {CONFIG['UPDATE_INTERVAL_MINUTES']} min)")
    
    return scheduler

if __name__ == '__main__':

    if not initialize_system():
        logger.error("Failed to initialize system. Exiting.")
        exit(1)
    
    scheduler = setup_scheduler()
    
    logger.info("\n" + "="*60)
    logger.info("Starting Flask API Server")
    logger.info("="*60)
    logger.info("API Endpoints:")
    logger.info("  POST   /api/analyze_route")
    logger.info("  GET    /api/node_status/<node_id>")
    logger.info("  GET    /api/graph_snapshot")
    logger.info("  GET    /api/historical_trends/<node_id>")
    logger.info("  GET    /api/available_nodes")
    logger.info("  POST   /api/update_graph")
    logger.info("  GET    /api/health")
    logger.info("="*60 + "\n")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False
        )
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Server shutdown complete")