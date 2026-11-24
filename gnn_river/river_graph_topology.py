"""
================================================================================
RAINWISE — Gujarat River-Network Graph Topology (Pillar 3: GNN)
================================================================================
Defines the directed hydrological graph of Gujarat's major river corridors:
  1. Tapi River Basin (Ukai Dam -> Surat City -> Dumas Estuary)
  2. Vishwamitri / Dhadhar Basin (Pavagadh -> Ajwa -> Vadodara -> Padra)
  3. Sabarmati Basin (Dharoi Dam -> Gandhinagar -> Ahmedabad -> Vasna Barrage)
  4. Narmada Basin (Sardar Sarovar Dam -> Kevadiya -> Bharuch -> Dahej)

Solves the Fundamental Hydrological Gap:
  Single-city isolated models assume flood risk is purely a function of local rainfall.
  In reality, water flows DOWNSTREAM along river networks:
  When Ukai Dam releases 900,000 cusecs, Surat floods even with 0 mm local rain!
================================================================================
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. Node Metadata: Dams, Reservoirs, Gauges, Cities, Estuaries
# ─────────────────────────────────────────────────────────────────────────────
GUJARAT_RIVER_NODES = [
    # ── TAPI BASIN (0 - 4) ──────────────────────────────────────────────────
    {
        "id": 0, "name": "Ukai Dam", "basin": "Tapi", "type": "Dam",
        "lat": 21.25, "lon": 73.58, "elev_m": 105.0, "danger_level": 105.0,
        "warning_level": 102.0, "capacity_cusecs": 1200000.0, "is_city": False
    },
    {
        "id": 1, "name": "Mandvi Gauge", "basin": "Tapi", "type": "Gauge",
        "lat": 21.26, "lon": 73.30, "elev_m": 45.0, "danger_level": 48.0,
        "warning_level": 44.0, "capacity_cusecs": 850000.0, "is_city": False
    },
    {
        "id": 2, "name": "Kamrej Weir", "basin": "Tapi", "type": "Gauge",
        "lat": 21.27, "lon": 72.96, "elev_m": 24.0, "danger_level": 26.0,
        "warning_level": 23.0, "capacity_cusecs": 600000.0, "is_city": False
    },
    {
        "id": 3, "name": "Surat", "basin": "Tapi", "type": "City",
        "lat": 21.17, "lon": 72.83, "elev_m": 13.0, "danger_level": 10.0,
        "warning_level": 8.5, "capacity_cusecs": 350000.0, "is_city": True
    },
    {
        "id": 4, "name": "Dumas Estuary", "basin": "Tapi", "type": "Estuary",
        "lat": 21.09, "lon": 72.71, "elev_m": 5.0, "danger_level": 6.5,
        "warning_level": 5.5, "capacity_cusecs": 900000.0, "is_city": False
    },

    # ── VISHWAMITRI / DHADHAR BASIN (5 - 10) ────────────────────────────────
    {
        "id": 5, "name": "Pavagadh Ridge", "basin": "Vishwamitri", "type": "Catchment",
        "lat": 22.46, "lon": 73.53, "elev_m": 450.0, "danger_level": 400.0,
        "warning_level": 350.0, "capacity_cusecs": 200000.0, "is_city": False
    },
    {
        "id": 6, "name": "Ajwa Reservoir", "basin": "Vishwamitri", "type": "Reservoir",
        "lat": 22.36, "lon": 73.40, "elev_m": 65.0, "danger_level": 65.5,
        "warning_level": 64.0, "capacity_cusecs": 150000.0, "is_city": False
    },
    {
        "id": 7, "name": "Pratappura Reservoir", "basin": "Vishwamitri", "type": "Reservoir",
        "lat": 22.34, "lon": 73.32, "elev_m": 48.0, "danger_level": 50.0,
        "warning_level": 47.0, "capacity_cusecs": 80000.0, "is_city": False
    },
    {
        "id": 8, "name": "Vadodara", "basin": "Vishwamitri", "type": "City",
        "lat": 22.30, "lon": 73.18, "elev_m": 39.0, "danger_level": 26.0, # Vishwamitri 26ft danger mark
        "warning_level": 22.0, "capacity_cusecs": 25000.0, "is_city": True
    },
    {
        "id": 9, "name": "Padra Gauge", "basin": "Vishwamitri", "type": "Gauge",
        "lat": 22.24, "lon": 73.08, "elev_m": 28.0, "danger_level": 30.0,
        "warning_level": 26.0, "capacity_cusecs": 60000.0, "is_city": True
    },
    {
        "id": 10, "name": "Dhadhar Estuary", "basin": "Vishwamitri", "type": "Estuary",
        "lat": 21.90, "lon": 72.60, "elev_m": 6.0, "danger_level": 7.0,
        "warning_level": 5.8, "capacity_cusecs": 120000.0, "is_city": False
    },

    # ── SABARMATI BASIN (11 - 16) ───────────────────────────────────────────
    {
        "id": 11, "name": "Dharoi Dam", "basin": "Sabarmati", "type": "Dam",
        "lat": 24.00, "lon": 72.85, "elev_m": 180.0, "danger_level": 189.0,
        "warning_level": 186.0, "capacity_cusecs": 650000.0, "is_city": False
    },
    {
        "id": 12, "name": "Himmatnagar Gauge", "basin": "Sabarmati", "type": "Gauge",
        "lat": 23.60, "lon": 72.96, "elev_m": 125.0, "danger_level": 130.0,
        "warning_level": 127.0, "capacity_cusecs": 400000.0, "is_city": False
    },
    {
        "id": 13, "name": "Gandhinagar", "basin": "Sabarmati", "type": "City",
        "lat": 23.22, "lon": 72.68, "elev_m": 81.0, "danger_level": 85.0,
        "warning_level": 82.0, "capacity_cusecs": 300000.0, "is_city": True
    },
    {
        "id": 14, "name": "Ahmedabad", "basin": "Sabarmati", "type": "City",
        "lat": 23.02, "lon": 72.57, "elev_m": 51.0, "danger_level": 48.0,
        "warning_level": 45.0, "capacity_cusecs": 220000.0, "is_city": True
    },
    {
        "id": 15, "name": "Vasna Barrage", "basin": "Sabarmati", "type": "Dam",
        "lat": 22.98, "lon": 72.55, "elev_m": 42.0, "danger_level": 43.5,
        "warning_level": 41.5, "capacity_cusecs": 250000.0, "is_city": False
    },
    {
        "id": 16, "name": "Dholka Estuary", "basin": "Sabarmati", "type": "Estuary",
        "lat": 22.72, "lon": 72.44, "elev_m": 15.0, "danger_level": 16.0,
        "warning_level": 14.5, "capacity_cusecs": 350000.0, "is_city": False
    },

    # ── NARMADA BASIN (17 - 21) ─────────────────────────────────────────────
    {
        "id": 17, "name": "Sardar Sarovar Dam", "basin": "Narmada", "type": "Dam",
        "lat": 21.83, "lon": 73.75, "elev_m": 163.0, "danger_level": 138.68,
        "warning_level": 135.0, "capacity_cusecs": 3000000.0, "is_city": False
    },
    {
        "id": 18, "name": "Garudeshwar Weir", "basin": "Narmada", "type": "Gauge",
        "lat": 21.85, "lon": 73.65, "elev_m": 48.0, "danger_level": 52.0,
        "warning_level": 49.0, "capacity_cusecs": 2500000.0, "is_city": False
    },
    {
        "id": 19, "name": "Rajpipla", "basin": "Narmada", "type": "City",
        "lat": 21.87, "lon": 73.50, "elev_m": 38.0, "danger_level": 41.0,
        "warning_level": 39.0, "capacity_cusecs": 2200000.0, "is_city": True
    },
    {
        "id": 20, "name": "Bharuch", "basin": "Narmada", "type": "City",
        "lat": 21.70, "lon": 72.97, "elev_m": 18.0, "danger_level": 24.0, # Golden Bridge danger 24ft
        "warning_level": 22.0, "capacity_cusecs": 600000.0, "is_city": True
    },
    {
        "id": 21, "name": "Dahej Estuary", "basin": "Narmada", "type": "Estuary",
        "lat": 21.71, "lon": 72.58, "elev_m": 5.0, "danger_level": 6.0,
        "warning_level": 5.2, "capacity_cusecs": 3000000.0, "is_city": False
    }
]

NUM_NODES = len(GUJARAT_RIVER_NODES)
NODE_NAME_TO_ID = {n["name"]: n["id"] for n in GUJARAT_RIVER_NODES}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Directed River Edges: (Source -> Target, Distance km, Transit Lag hours)
# ─────────────────────────────────────────────────────────────────────────────
GUJARAT_RIVER_EDGES = [
    # Tapi River Flow
    {"source": 0, "target": 1, "dist_km": 32.0, "lag_hrs": 2.5, "river": "Tapi"}, # Ukai -> Mandvi
    {"source": 1, "target": 2, "dist_km": 36.0, "lag_hrs": 3.0, "river": "Tapi"}, # Mandvi -> Kamrej
    {"source": 2, "target": 3, "dist_km": 20.0, "lag_hrs": 1.8, "river": "Tapi"}, # Kamrej -> Surat
    {"source": 3, "target": 4, "dist_km": 16.0, "lag_hrs": 1.2, "river": "Tapi"}, # Surat -> Dumas

    # Vishwamitri River Flow
    {"source": 5, "target": 6, "dist_km": 18.0, "lag_hrs": 1.5, "river": "Vishwamitri"}, # Pavagadh -> Ajwa
    {"source": 6, "target": 7, "dist_km": 10.0, "lag_hrs": 1.0, "river": "Vishwamitri"}, # Ajwa -> Pratappura
    {"source": 7, "target": 8, "dist_km": 16.0, "lag_hrs": 2.0, "river": "Vishwamitri"}, # Pratappura -> Vadodara
    {"source": 8, "target": 9, "dist_km": 14.0, "lag_hrs": 1.8, "river": "Vishwamitri"}, # Vadodara -> Padra
    {"source": 9, "target": 10, "dist_km": 42.0, "lag_hrs": 4.5, "river": "Vishwamitri"}, # Padra -> Dhadhar

    # Sabarmati River Flow
    {"source": 11, "target": 12, "dist_km": 48.0, "lag_hrs": 4.0, "river": "Sabarmati"}, # Dharoi -> Himmatnagar
    {"source": 12, "target": 13, "dist_km": 45.0, "lag_hrs": 3.5, "river": "Sabarmati"}, # Himmatnagar -> Gandhinagar
    {"source": 13, "target": 14, "dist_km": 24.0, "lag_hrs": 2.0, "river": "Sabarmati"}, # Gandhinagar -> Ahmedabad
    {"source": 14, "target": 15, "dist_km": 8.0,  "lag_hrs": 0.8, "river": "Sabarmati"}, # Ahmedabad -> Vasna Barrage
    {"source": 15, "target": 16, "dist_km": 32.0, "lag_hrs": 2.8, "river": "Sabarmati"}, # Vasna -> Dholka

    # Narmada River Flow
    {"source": 17, "target": 18, "dist_km": 12.0, "lag_hrs": 1.0, "river": "Narmada"}, # Sardar Sarovar -> Garudeshwar
    {"source": 18, "target": 19, "dist_km": 18.0, "lag_hrs": 1.5, "river": "Narmada"}, # Garudeshwar -> Rajpipla
    {"source": 19, "target": 20, "dist_km": 55.0, "lag_hrs": 4.5, "river": "Narmada"}, # Rajpipla -> Bharuch
    {"source": 20, "target": 21, "dist_km": 40.0, "lag_hrs": 3.0, "river": "Narmada"}  # Bharuch -> Dahej
]


def build_graph_tensors() -> Dict[str, torch.Tensor]:
    """
    Constructs PyTorch adjacency and normalized graph convolution matrices:
      A_hat = D_inv_sqrt * (A + I) * D_inv_sqrt
    Also builds directed downstream message passing index.
    """
    N = NUM_NODES
    adj = np.zeros((N, N), dtype=np.float32)
    lag_matrix = np.zeros((N, N), dtype=np.float32)
    dist_matrix = np.zeros((N, N), dtype=np.float32)

    edge_sources = []
    edge_targets = []
    edge_attrs = []

    for edge in GUJARAT_RIVER_EDGES:
        u, v = edge["source"], edge["target"]
        d = edge["dist_km"]
        lag = edge["lag_hrs"]

        # Directed edge: u -> v (downstream flow)
        adj[u, v] = 1.0
        lag_matrix[u, v] = lag
        dist_matrix[u, v] = d

        edge_sources.append(u)
        edge_targets.append(v)
        edge_attrs.append([d, lag, 1.0 / (d + 1e-3)])

    # Self-loops for identity persistence: A_tilde = A + I
    adj_tilde = adj + np.eye(N, dtype=np.float32)

    # Kipf & Welling Degree Normalization: D^(-1/2) * A_tilde * D^(-1/2)
    degree = np.sum(adj_tilde, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)

    norm_adj = D_inv_sqrt @ adj_tilde @ D_inv_sqrt

    # Upstream Inflow Matrix: Transpose of adj so node v collects from upstream u
    upstream_adj = adj.T

    return {
        "num_nodes": N,
        "adj": torch.tensor(adj, dtype=torch.float32),
        "norm_adj": torch.tensor(norm_adj, dtype=torch.float32),
        "upstream_adj": torch.tensor(upstream_adj, dtype=torch.float32),
        "lag_matrix": torch.tensor(lag_matrix, dtype=torch.float32),
        "dist_matrix": torch.tensor(dist_matrix, dtype=torch.float32),
        "edge_index": torch.tensor([edge_sources, edge_targets], dtype=torch.long),
        "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32)
    }


def get_node_by_name(name: str) -> Dict[str, Any]:
    """Finds node metadata by city or station name."""
    clean_name = name.lower().strip()
    for n in GUJARAT_RIVER_NODES:
        if n["name"].lower().strip() == clean_name:
            return n
    # Fallback partial matching
    for n in GUJARAT_RIVER_NODES:
        if clean_name in n["name"].lower() or n["name"].lower() in clean_name:
            return n
    return GUJARAT_RIVER_NODES[3] # default to Surat


def get_upstream_path(target_node_id: int) -> List[int]:
    """Returns the ordered upstream lineage leading to target_node_id."""
    path = [target_node_id]
    curr = target_node_id
    while True:
        upstream_nodes = [e["source"] for e in GUJARAT_RIVER_EDGES if e["target"] == curr]
        if not upstream_nodes:
            break
        curr = upstream_nodes[0]
        path.insert(0, curr)
    return path
