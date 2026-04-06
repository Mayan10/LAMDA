from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List


@dataclass(frozen=True)
class NodeMetadata:
    node_id: str
    city: str
    latitude: float
    longitude: float
    comtrade_reporter_code: str
    world_bank_country_code: str


NODE_CATALOG: Dict[str, NodeMetadata] = {
    "Hong_Kong": NodeMetadata(
        node_id="Hong_Kong",
        city="Hong Kong",
        latitude=22.3193,
        longitude=114.1694,
        comtrade_reporter_code="344",
        world_bank_country_code="HKG",
    ),
    "Singapore": NodeMetadata(
        node_id="Singapore",
        city="Singapore",
        latitude=1.3521,
        longitude=103.8198,
        comtrade_reporter_code="702",
        world_bank_country_code="SGP",
    ),
    "Shanghai": NodeMetadata(
        node_id="Shanghai",
        city="Shanghai",
        latitude=31.2304,
        longitude=121.4737,
        comtrade_reporter_code="156",
        world_bank_country_code="CHN",
    ),
    "Tokyo": NodeMetadata(
        node_id="Tokyo",
        city="Tokyo",
        latitude=35.6762,
        longitude=139.6503,
        comtrade_reporter_code="392",
        world_bank_country_code="JPN",
    ),
    "Los_Angeles": NodeMetadata(
        node_id="Los_Angeles",
        city="Los Angeles",
        latitude=33.7288,
        longitude=-118.2620,
        comtrade_reporter_code="842",
        world_bank_country_code="USA",
    ),
}

SUPPORTED_NODES: List[str] = list(NODE_CATALOG.keys())
NODE_CITY_MAP = {node_id: metadata.city for node_id, metadata in NODE_CATALOG.items()}


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def build_graph_nodes() -> List[dict]:
    return [
        {
            "node_id": metadata.node_id,
            "latitude": metadata.latitude,
            "longitude": metadata.longitude,
        }
        for metadata in NODE_CATALOG.values()
    ]


def build_graph_edges(max_neighbors: int = 2) -> List[dict]:
    node_ids = SUPPORTED_NODES[:]
    pair_distances = {}

    for index, source in enumerate(node_ids):
        source_meta = NODE_CATALOG[source]
        for target in node_ids[index + 1 :]:
            target_meta = NODE_CATALOG[target]
            pair_distances[(source, target)] = haversine_distance_km(
                source_meta.latitude,
                source_meta.longitude,
                target_meta.latitude,
                target_meta.longitude,
            )

    edges = set()
    connected = {node_ids[0]}

    while len(connected) < len(node_ids):
        candidate_pair = None
        candidate_distance = float("inf")

        for (source, target), distance in pair_distances.items():
            if (source in connected) ^ (target in connected):
                if distance < candidate_distance:
                    candidate_pair = (source, target)
                    candidate_distance = distance

        if candidate_pair is None:
            break

        edges.add(tuple(sorted(candidate_pair)))
        connected.update(candidate_pair)

    for source in node_ids:
        nearest = sorted(
            [
                (
                    target,
                    pair_distances.get((source, target))
                    or pair_distances.get((target, source)),
                )
                for target in node_ids
                if target != source
            ],
            key=lambda item: item[1],
        )
        for target, _ in nearest[:max_neighbors]:
            edges.add(tuple(sorted((source, target))))

    return [
        {
            "source": source,
            "target": target,
            "distance_km": round(
                pair_distances.get((source, target))
                or pair_distances[(target, source)],
                2,
            ),
            "trade_volume": 0.0,
        }
        for source, target in sorted(edges)
    ]
