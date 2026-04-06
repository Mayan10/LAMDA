from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import json
import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional local convenience dependency
    def load_dotenv():
        return False

# Load environment variables from .env file
load_dotenv()

import logging
from typing import List
from intelligence_processor import IntelligenceProcessor, ScraperData
from graph_risk_engine import GraphRiskEngine
from route_optimizer import RouteOptimizer
from scraper_orchestrator import ScraperOrchestrator

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
scraper_orchestrator = ScraperOrchestrator()

CONFIG = {
    "UPDATE_INTERVAL_MINUTES": int(os.getenv("UPDATE_INTERVAL_MINUTES", "30")),
    "GRAPH_STATE_FILE": os.getenv("GRAPH_STATE_FILE", "graph_state.json"),
    "RISK_VECTORS_FILE": os.getenv("RISK_VECTORS_FILE", "risk_vectors_output.json"),
    "DB_PATH": os.getenv("DB_PATH", "supply_chain_graph.db"),
    "GRAPH_NODES_FILE": os.getenv("GRAPH_NODES_FILE", "graph_nodes.json"),
    "GRAPH_EDGES_FILE": os.getenv("GRAPH_EDGES_FILE", "graph_edges.json"),
    "PORT": int(os.getenv("PORT", "5001")),
}


def persist_graph_structure(nodes: List[dict], edges: List[dict]):
    with open(CONFIG["GRAPH_NODES_FILE"], "w") as node_file:
        json.dump({"nodes": nodes}, node_file, indent=2)

    with open(CONFIG["GRAPH_EDGES_FILE"], "w") as edge_file:
        json.dump({"edges": edges}, edge_file, indent=2)

def fetch_scraper_data() -> List[ScraperData]:
    logger.info("Fetching data from scrapers")
    live_data = scraper_orchestrator.fetch_all()
    logger.info(f"Fetched data for {len(live_data)} nodes")
    return live_data


def update_graph_pipeline():

    try:
        logger.info("="*60)
        logger.info("Starting scheduled graph update pipeline")
        logger.info("="*60)
        
        scraper_data = fetch_scraper_data()
        if not scraper_data:
            raise RuntimeError("No scraper data returned from orchestrator")

        node_trade_map = {item.node_id: item.trade for item in scraper_data}
        graph_engine.update_edge_trade_volumes(node_trade_map)
        
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
            "route_optimizer": route_optimizer is not None,
            "anthropic_enabled": bool(getattr(intelligence_processor, "client", None)),
            "scraper_http_enabled": scraper_orchestrator.http_enabled,
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

        logger.info("3. Building graph structure from supported nodes...")
        nodes, edges = scraper_orchestrator.build_graph_structure()
        persist_graph_structure(nodes, edges)
        graph_engine.load_graph_structure_from_data(nodes, edges)

        logger.info("4. Running initial graph update...")
        update_graph_pipeline()
        
        logger.info("5. Initializing Route Optimizer...")
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
            port=CONFIG["PORT"],
            debug=False
        )
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Server shutdown complete")
