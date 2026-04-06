# Supply Chain Risk Prediction System

A production-ready supply chain risk prediction system using AI-powered intelligence processing, graph neural networks for risk propagation, and multi-objective route optimization.

## System Overview

This system provides real-time supply chain risk assessment and optimal route recommendations for logistics companies. It processes data from 6 different sources, uses Claude AI for semantic analysis, propagates risks through a graph neural network, and finds optimal routes using A* pathfinding.

### Key Features

**AI-Powered Intelligence Processing** - Claude API analyzes unstructured text data when available, with local heuristic fallback when it is not  
**Graph Neural Network** - Spatial risk propagation considering geographical proximity and trade relationships  
**Temporal Memory** - Tracks risk history for trend analysis (last 10 updates)  
**Multi-Objective Optimization** - Balances risk, distance, time, and trade volume  
**REST API** - Easy integration with web/mobile frontends  
**Integrated Scraper Runtime** - The backend can use the scrapers directly in-process or call separately hosted scraper services  
**Automated Updates** - Background worker updates graph every 30 minutes  
**Production-Ready** - Database persistence, error handling, logging  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCRAPER LAYER (6 scrapers)               │
│  gscpi  │  news  │ political │ reporter │ trade │ weather  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         SCRIPT 1: Intelligence Processor                    │
│  • Normalizes numerical data (GSCPI, Trade)                 │
│  • Claude API analyzes text (News, Political, Weather)      │
│  • Applies Reporter credibility weights                     │
│  • Outputs: 6D risk vectors per node                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         SCRIPT 2: Graph Risk Engine                         │
│  • Graph Attention Network (GAT) for risk propagation       │
│  • Considers neighbor influence & trade relationships       │
│  • Temporal memory (stores last 10 snapshots)               │
│  • Outputs: Updated risk scores for all nodes               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         SCRIPT 3: Route Optimizer                           │
│  • A* pathfinding with multi-objective cost function        │
│  • Returns top-K alternative routes                         │
│  • Detailed risk analysis per route                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         SCRIPT 4: Flask API Server                          │
│  • REST API endpoints for route analysis                    │
│  • Background worker (updates every 30 min)                 │
│  • Database for persistence                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Software
- Python 3.8+
- pip (Python package manager)
- SQLite3 (comes with Python)

### Optional API Keys
- **SERPAPI_API_KEY** - Recommended for live news/search enrichment
- **OPENWEATHERMAP_API_KEY** - Optional richer weather data
- **ANTHROPIC_API_KEY** - Optional Claude enrichment for semantic scoring

The backend will still start without `ANTHROPIC_API_KEY`; it falls back to deterministic local scoring until you add the key.

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 1GB free space

---

## Installation

### Step 1: Clone/Download the System

Clone the repo to your machine:
```bash
git clone <your-repo-url> lamda
cd lamda
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

```bash
# Copy the sample env file
cp .env.example .env

# Edit the values you have available
# SERPAPI_API_KEY=...
# OPENWEATHERMAP_API_KEY=...
# ANTHROPIC_API_KEY=...   # optional
```

### Step 4: Start the Backend

```bash
python api_server.py
```

On first start the backend now:
- builds the graph structure automatically from the supported nodes
- fetches live data from the integrated scraper layer
- generates `graph_nodes.json`, `graph_edges.json`, `graph_state.json`, `risk_vectors_output.json`, and `supply_chain_graph.db`
- starts the API on `http://localhost:5001`

If you want the API server to call separately running scraper services instead of using the scrapers directly in-process, set:

```bash
SCRAPER_HTTP_ENABLED=true
```

---

## Usage

### Running the Full API Server

```bash
python api_server.py
```

The server will:
1. Initialize all components
2. Build graph files automatically if they do not exist yet
3. Run the initial graph update
4. Start the background worker (updates every 30 min)
5. Start Flask API on **http://localhost:5001**

To use a different port:
```bash
PORT=8080 python api_server.py
```

### Linux 24x7 Deployment

1. Install system packages:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```
2. Clone the repo onto the laptop:
```bash
sudo mkdir -p /opt/lamda
sudo chown "$USER":"$USER" /opt/lamda
git clone <your-repo-url> /opt/lamda
cd /opt/lamda
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
3. Edit `/opt/lamda/.env` and set the keys you have. For the single-service setup, keep `SCRAPER_HTTP_ENABLED=false`.
4. Copy the example systemd unit:
```bash
sudo cp deploy/lamda-api.service.example /etc/systemd/system/lamda-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now lamda-api
```
5. Monitor the service:
```bash
sudo systemctl status lamda-api
journalctl -u lamda-api -f
```

If you want the app reachable from other machines, open port `5001` in the firewall or place Nginx in front of it as a reverse proxy.

---

## API Documentation

### Base URL
```
http://localhost:5001/api
```
(Use the same host/port if you set `PORT` when starting the server.)

### Endpoints

#### 1. Analyze Route (POST)
Find optimal routes between two locations.

**Endpoint:** `POST /api/analyze_route`

**Request Body:**
```json
{
  "source": "Hong_Kong",
  "destination": "Los_Angeles",
  "num_routes": 3
}
```

**Response:**
```json
{
  "source": "Hong_Kong",
  "destination": "Los_Angeles",
  "num_routes": 3,
  "routes": [
    {
      "rank": 1,
      "path": ["Hong_Kong", "Tokyo", "Los_Angeles"],
      "total_distance_km": 11019.45,
      "avg_risk": 0.43,
      "estimated_time_hours": 297.82,
      "estimated_time_days": 12.4,
      "route_score": 5234.21,
      "analysis": {
        "route_summary": {...},
        "node_risks": [...],
        "high_risk_segments": [...],
        "bottlenecks": [...]
      }
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5001/api/analyze_route \
  -H "Content-Type: application/json" \
  -d '{"source": "Hong_Kong", "destination": "Los_Angeles", "num_routes": 3}'
```

---

#### 2. Get Node Status (GET)
Get current risk status of a specific node.

**Endpoint:** `GET /api/node_status/<node_id>`

**Response:**
```json
{
  "node_id": "Hong_Kong",
  "latitude": 22.3193,
  "longitude": 114.1694,
  "current_risk": {
    "gscpi_risk": 0.45,
    "news_risk": 0.67,
    "political_risk": 0.54,
    "trade_risk": 0.23,
    "weather_risk": 0.78,
    "reporter_confidence": 0.85,
    "overall_risk": 0.53
  },
  "risk_level": "MODERATE",
  "last_updated": "2024-01-01T12:00:00"
}
```

**cURL Example:**
```bash
curl http://localhost:5001/api/node_status/Hong_Kong
```

---

#### 3. Get Graph Snapshot (GET)
Get current state of entire graph.

**Endpoint:** `GET /api/graph_snapshot`

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "num_nodes": 100,
  "num_edges": 250,
  "nodes": [...],
  "edges": [...],
  "statistics": {
    "avg_risk": 0.45,
    "high_risk_nodes": 12,
    "moderate_risk_nodes": 43,
    "low_risk_nodes": 45
  }
}
```

---

#### 4. Get Historical Trends (GET)
Get historical risk trends for a node.

**Endpoint:** `GET /api/historical_trends/<node_id>?limit=10`

**Response:**
```json
{
  "node_id": "Hong_Kong",
  "num_records": 10,
  "history": [
    {
      "timestamp": "2024-01-01T12:00:00",
      "gscpi_risk": 0.45,
      "news_risk": 0.67,
      "political_risk": 0.54,
      "trade_risk": 0.23,
      "weather_risk": 0.78,
      "overall_risk": 0.53
    }
  ]
}
```

---

#### 5. Get Available Nodes (GET)
Get list of all nodes in the graph.

**Endpoint:** `GET /api/available_nodes`

**Response:**
```json
{
  "num_nodes": 100,
  "nodes": [
    {
      "node_id": "Hong_Kong",
      "latitude": 22.3193,
      "longitude": 114.1694,
      "overall_risk": 0.53
    }
  ]
}
```

---

#### 6. Trigger Manual Update (POST)
Manually trigger graph update (admin endpoint).

**Endpoint:** `POST /api/update_graph`

**Response:**
```json
{
  "status": "success",
  "message": "Graph update completed",
  "timestamp": "2024-01-01T12:00:00"
}
```

---

#### 7. Health Check (GET)
Check system health.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "services": {
    "intelligence_processor": true,
    "graph_engine": true,
    "route_optimizer": true
  }
}
```

**cURL Example:**
```bash
curl http://localhost:5001/api/health
```

---

## How to Test the API

With the server running (`python api_server.py`), use these commands in a new terminal. Replace `localhost:5001` with your host if different (e.g. after setting `PORT=8080`).

**1. Root / API info (no 404 in browser):**
```bash
curl http://localhost:5001/
```

**2. Health check:**
```bash
curl http://localhost:5001/api/health
```

**3. List available nodes:**
```bash
curl http://localhost:5001/api/available_nodes
```

**4. Node status (use a node_id from step 3, e.g. Hong_Kong):**
```bash
curl http://localhost:5001/api/node_status/Hong_Kong
```

**5. Graph snapshot:**
```bash
curl http://localhost:5001/api/graph_snapshot
```

**6. Analyze route (POST):**
```bash
curl -X POST http://localhost:5001/api/analyze_route \
  -H "Content-Type: application/json" \
  -d '{"source": "Hong_Kong", "destination": "Los_Angeles", "num_routes": 3}'
```

**7. Historical trends for a node:**
```bash
curl "http://localhost:5001/api/historical_trends/Hong_Kong?limit=5"
```

**8. Trigger manual graph update (admin):**
```bash
curl -X POST http://localhost:5001/api/update_graph
```

---

## Configuration

Edit configuration in `api_server.py`:

```python
CONFIG = {
    "UPDATE_INTERVAL_MINUTES": 30,  # How often to update graph
    "GRAPH_STATE_FILE": "graph_state.json",
    "RISK_VECTORS_FILE": "risk_vectors_output.json",
    "DB_PATH": "supply_chain_graph.db",
    "GRAPH_NODES_FILE": "graph_nodes.json",
    "GRAPH_EDGES_FILE": "graph_edges.json"
}
```

---

## Integrating Your Scrapers

Replace the `fetch_scraper_data()` function in `api_server.py`:

```python
def fetch_scraper_data() -> List[ScraperData]:
    """Fetch data from your actual scrapers."""
    
    # Call your scraper APIs
    gscpi_response = requests.get("http://your-gscpi-scraper/api/latest")
    news_response = requests.get("http://your-news-scraper/api/latest")
    # ... etc
    
    # Parse and return ScraperData objects
    nodes_data = []
    for node_id in your_node_list:
        nodes_data.append(ScraperData(
            node_id=node_id,
            gscpi=gscpi_response.json()[node_id],
            trade=trade_response.json()[node_id],
            news=news_response.json()[node_id],
            political=political_response.json()[node_id],
            weather=weather_response.json()[node_id],
            reporter_credibility=reporter_response.json()[node_id]
        ))
    
    return nodes_data
```

---

## Database Schema

SQLite database (`supply_chain_graph.db`) contains:

### Table: nodes
```sql
node_id TEXT PRIMARY KEY,
latitude REAL,
longitude REAL,
current_risk REAL,
last_updated TEXT
```

### Table: edges
```sql
edge_id INTEGER PRIMARY KEY,
source TEXT,
target TEXT,
distance_km REAL,
trade_volume REAL
```

### Table: risk_history
```sql
id INTEGER PRIMARY KEY,
node_id TEXT,
timestamp TEXT,
gscpi_risk REAL,
news_risk REAL,
political_risk REAL,
trade_risk REAL,
weather_risk REAL,
reporter_confidence REAL,
overall_risk REAL
```

---

## Frontend Integration Example

```javascript
// React example
const analyzeRoute = async (source, destination) => {
  const response = await fetch('http://localhost:5001/api/analyze_route', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source: source,
      destination: destination,
      num_routes: 3
    })
  });
  
  const data = await response.json();
  
  // Display routes
  data.routes.forEach(route => {
    console.log(`Route ${route.rank}: ${route.path.join(' → ')}`);
    console.log(`Risk: ${route.avg_risk}, Distance: ${route.total_distance_km} km`);
  });
};
```

---

## Deployment

### Option 1: Single Server (Simple)

```bash
# Install as systemd service
sudo nano /etc/systemd/system/supply-chain-api.service
```

```ini
[Unit]
Description=Supply Chain Risk Prediction API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/supply-chain-risk-system
Environment="ANTHROPIC_API_KEY=your-key"
ExecStart=/home/ubuntu/supply-chain-risk-system/venv/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable supply-chain-api
sudo systemctl start supply-chain-api
```

### Option 2: Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .
COPY *.json .

ENV ANTHROPIC_API_KEY=""

CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t supply-chain-api .
docker run -d -p 5001:5001 -e ANTHROPIC_API_KEY=your-key supply-chain-api
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

See `DEPLOYMENT.md` for detailed cloud deployment instructions.

---

## Performance & Scalability

- **100 nodes**: ~2 seconds per update
- **1000 nodes**: ~20 seconds per update
- **10,000 nodes**: Consider distributed architecture

For large-scale deployment:
1. Use Redis for caching
2. Deploy multiple API instances with load balancer
3. Use Neo4j instead of SQLite
4. Move GNN inference to GPU servers

---

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"
**Solution:** Set the environment variable before running:
```bash
export ANTHROPIC_API_KEY='your-key'
```

### Issue: "Module 'torch_geometric' not found"
**Solution:** Install PyTorch Geometric:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue: "Database is locked"
**Solution:** Only one process should write to SQLite at a time. Use PostgreSQL for production.

### Issue: "Claude API rate limit exceeded"
**Solution:** The system batches all nodes in one API call. If you have >100 nodes, split into multiple batches.

---

## License

MIT License - Free for commercial use

---

## Support & Contact

For enterprise deployment support, contact your development team.

---

## Roadmap

- [ ] GPU acceleration for GNN
- [ ] Real-time alerts system
- [ ] Dashboard UI
- [ ] Mobile app integration
- [ ] Multi-modal transport (air, rail, sea)
- [ ] Carbon footprint optimization

---
