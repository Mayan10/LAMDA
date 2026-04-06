import json
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt


@dataclass
class Route:
    path: List[str]
    total_distance_km: float
    total_risk: float
    avg_risk: float
    estimated_time_hours: float
    route_score: float


class RouteOptimizer:
    def __init__(self, graph_state_file: str):
        with open(graph_state_file, 'r') as f:
            self.graph_state = json.load(f)
        self.nodes = {
            node["node_id"]: {
                "latitude": node["latitude"],
                "longitude": node["longitude"],
                "overall_risk": node["overall_risk"]
            }
            for node in self.graph_state["nodes"]
        }
        self.adjacency = {node_id: [] for node_id in self.nodes.keys()}
        for edge in self.graph_state["edges"]:
            self.adjacency[edge["source"]].append({
                "target": edge["target"],
                "distance_km": edge["distance_km"],
                "trade_volume": edge.get("trade_volume", 1000000)
            })
            self.adjacency[edge["target"]].append({
                "target": edge["source"],
                "distance_km": edge["distance_km"],
                "trade_volume": edge.get("trade_volume", 1000000)
            })
        
        print(f"Route Optimizer initialized with {len(self.nodes)} nodes")
    
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        r = 6371
        
        return c * r
    
    def calculate_heuristic(self, current: str, goal: str) -> float:
        curr_node = self.nodes[current]
        goal_node = self.nodes[goal]
        
        distance = self.haversine_distance(
            curr_node["latitude"], curr_node["longitude"],
            goal_node["latitude"], goal_node["longitude"]
        )
        
        avg_risk = (curr_node["overall_risk"] + goal_node["overall_risk"]) / 2
        
        return distance * (1 + avg_risk)
    
    def calculate_edge_cost(self, 
                           source: str, 
                           target: str, 
                           edge_info: Dict,
                           weights: Dict[str, float]) -> float:
        distance = edge_info["distance_km"]
        trade_volume = edge_info["trade_volume"]
        
        source_risk = self.nodes[source]["overall_risk"]
        target_risk = self.nodes[target]["overall_risk"]
        avg_risk = (source_risk + target_risk) / 2
        
        distance_cost = distance * weights["distance"]
        risk_cost = avg_risk * distance * weights["risk"]
        
        trade_factor = 1.0 / (1.0 + trade_volume / 1_000_000)
        trade_cost = distance * trade_factor * weights["trade"]
        
        total_cost = distance_cost + risk_cost + trade_cost
        
        return total_cost
    
    def find_optimal_route(self,
                          source: str,
                          destination: str,
                          weights: Optional[Dict[str, float]] = None) -> Optional[Route]:
        if weights is None:
            weights = {
                "risk": 1.0,
                "distance": 0.3,
                "trade": 0.2
            }
        
        if source not in self.nodes or destination not in self.nodes:
            print(f"Invalid source or destination")
            return None
        
        open_set = []
        heapq.heappush(open_set, (0, source))
        
        came_from = {}
        g_score = {node_id: float('inf') for node_id in self.nodes}
        g_score[source] = 0
        
        f_score = {node_id: float('inf') for node_id in self.nodes}
        f_score[source] = self.calculate_heuristic(source, destination)
        
        distance_to = {source: 0.0}
        risk_to = {source: 0.0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == destination:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(source)
                path.reverse()
                
                total_distance = distance_to[destination]
                total_risk = risk_to[destination]
                avg_risk = total_risk / len(path)
                
                estimated_time = total_distance / 37.0
                
                route_score = g_score[destination]
                
                route = Route(
                    path=path,
                    total_distance_km=total_distance,
                    total_risk=total_risk,
                    avg_risk=avg_risk,
                    estimated_time_hours=estimated_time,
                    route_score=route_score
                )
                
                return route
            
            for neighbor_info in self.adjacency[current]:
                neighbor = neighbor_info["target"]
                
                edge_cost = self.calculate_edge_cost(
                    current, neighbor, neighbor_info, weights
                )
                tentative_g_score = g_score[current] + edge_cost
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_heuristic(
                        neighbor, destination
                    )
                    
                    distance_to[neighbor] = distance_to[current] + neighbor_info["distance_km"]
                    node_risk = self.nodes[neighbor]["overall_risk"]
                    risk_to[neighbor] = risk_to[current] + node_risk
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def find_k_best_routes(self,
                          source: str,
                          destination: str,
                          k: int = 3,
                          diversity_penalty: float = 0.3) -> List[Route]:
        routes = []
        weight_configs = [
            {"risk": 1.0, "distance": 0.3, "trade": 0.2},
            {"risk": 0.5, "distance": 0.8, "trade": 0.2},
            {"risk": 0.7, "distance": 0.4, "trade": 0.5},
            {"risk": 1.2, "distance": 0.2, "trade": 0.1},
            {"risk": 0.3, "distance": 1.0, "trade": 0.3},
        ]

        candidate_paths = self._enumerate_simple_paths(source, destination)
        unique_routes: Dict[tuple, Route] = {}

        for weights in weight_configs:
            for path in candidate_paths:
                route = self._build_route_from_path(path, weights)
                if route is None:
                    continue

                key = tuple(route.path)
                existing = unique_routes.get(key)
                if existing is None or route.route_score < existing.route_score:
                    unique_routes[key] = route

        for route in sorted(unique_routes.values(), key=lambda item: item.route_score):
            is_diverse = True
            for existing_route in routes:
                overlap = len(set(route.path) & set(existing_route.path))
                if overlap / len(route.path) > 0.8:
                    is_diverse = False
                    break

            if is_diverse:
                routes.append(route)

            if len(routes) >= k:
                break

        return routes[:k]

    def _enumerate_simple_paths(
        self,
        source: str,
        destination: str,
        max_depth: Optional[int] = None,
    ) -> List[List[str]]:
        if source not in self.nodes or destination not in self.nodes:
            return []

        if max_depth is None:
            max_depth = len(self.nodes) - 1

        paths: List[List[str]] = []

        def dfs(current: str, path: List[str], visited: set):
            if len(path) - 1 > max_depth:
                return

            if current == destination:
                paths.append(path[:])
                return

            for neighbor_info in sorted(
                self.adjacency[current],
                key=lambda item: item["distance_km"],
            ):
                neighbor = neighbor_info["target"]
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

        dfs(source, [source], {source})
        return paths

    def _build_route_from_path(
        self,
        path: List[str],
        weights: Dict[str, float],
    ) -> Optional[Route]:
        if len(path) < 2:
            return None

        total_distance = 0.0
        total_cost = 0.0
        total_risk = self.nodes[path[0]]["overall_risk"]

        for source, target in zip(path, path[1:]):
            edge_info = next(
                (edge for edge in self.adjacency[source] if edge["target"] == target),
                None,
            )
            if edge_info is None:
                return None

            total_distance += edge_info["distance_km"]
            total_cost += self.calculate_edge_cost(source, target, edge_info, weights)
            total_risk += self.nodes[target]["overall_risk"]

        avg_risk = total_risk / len(path)
        estimated_time = total_distance / 37.0

        return Route(
            path=path,
            total_distance_km=total_distance,
            total_risk=total_risk,
            avg_risk=avg_risk,
            estimated_time_hours=estimated_time,
            route_score=total_cost,
        )
    
    def analyze_route(self, route: Route) -> Dict:
        analysis = {
            "route_summary": {
                "path": " → ".join(route.path),
                "total_distance_km": round(route.total_distance_km, 2),
                "avg_risk": round(route.avg_risk, 3),
                "estimated_time_days": round(route.estimated_time_hours / 24, 1),
                "route_score": round(route.route_score, 2)
            },
            "node_risks": [],
            "high_risk_segments": [],
            "bottlenecks": []
        }
        
        for node_id in route.path:
            node = self.nodes[node_id]
            analysis["node_risks"].append({
                "node": node_id,
                "risk": round(node["overall_risk"], 3)
            })
            
            if node["overall_risk"] > 0.7:
                analysis["high_risk_segments"].append({
                    "node": node_id,
                    "risk": round(node["overall_risk"], 3),
                    "warning": "High risk - consider alternative"
                })
        
        for i, node_id in enumerate(route.path[1:-1], 1):
            num_alternatives = len([
                n for n in self.adjacency[node_id]
                if n["target"] in route.path[i-1:i+2]
            ])
            
            if num_alternatives <= 2:
                analysis["bottlenecks"].append({
                    "node": node_id,
                    "alternatives": num_alternatives,
                    "warning": "Limited alternative routes"
                })
        
        return analysis

if __name__ == "__main__":
    print("Run api_server.py and call /api/analyze_route to use live route optimization.")
