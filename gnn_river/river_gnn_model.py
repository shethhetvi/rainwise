"""
================================================================================
RAINWISE — River-Network Graph Neural Network Architecture (Pillar 3: GNN)
================================================================================
Implements:
  1. Directed Hydrological Graph Convolution Layer (Downstream Message Passing)
  2. Spatio-Temporal River GNN (ST-GCN + GRU Temporal Routing)
  3. Downstream Wave Routing & Early-Warning Lead-Time Predictor
  4. RiverGNNEngine for live civic emergency simulation
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from gnn_river.river_graph_topology import (
    GUJARAT_RIVER_NODES,
    GUJARAT_RIVER_EDGES,
    NUM_NODES,
    NODE_NAME_TO_ID,
    build_graph_tensors,
    get_node_by_name,
    get_upstream_path
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Directed Hydrological Graph Convolution Layer
# ─────────────────────────────────────────────────────────────────────────────
class DirectedRiverConv(nn.Module):
    """
    Directed Graph Convolution capturing downstream fluid routing.
    Aggregates upstream flood waves with channel distance attenuation:
      H' = ReLU( A_upstream @ H @ W_up + H @ W_self + Bias )
    """
    def __init__(self, in_features: int, out_features: int):
        super(DirectedRiverConv, self).__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_upstream = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, upstream_adj: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Num_Nodes, In_Features]
        upstream_adj: [Num_Nodes, Num_Nodes] (where adj[v, u] = 1 if u flows into v)
        """
        # Self-representation
        h_self = self.w_self(x)

        # Message passing from upstream neighbors
        # [Num_Nodes, Num_Nodes] @ [Batch, Num_Nodes, In_Features]
        if x.dim() == 3:
            # Batched multiplication: upstream_adj broadcasted over batch
            h_upstream_in = torch.matmul(upstream_adj.unsqueeze(0), x)
        else:
            h_upstream_in = torch.matmul(upstream_adj, x)

        h_up = self.w_upstream(h_upstream_in)

        # Combine, activate, and normalize
        out = F.silu(h_self + h_up + self.bias)
        out = self.layer_norm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatio-Temporal River GNN Model
# ─────────────────────────────────────────────────────────────────────────────
class RiverNetworkGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for River Basin Flood Routing.
    Takes 22 Gujarat river nodes with 6 hydrological features:
      [local_rain_mm, inflow_cusecs, elevation_m, rain_3day_mm, river_dist_m, dam_discharge_flag]
    Outputs for each node:
      - surge_cusecs: downstream discharge volume
      - flood_risk_prob: inundation risk probability [0.0, 1.0]
      - lead_time_hours: estimated peak arrival time from upstream release
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 64, num_nodes: int = NUM_NODES):
        super(RiverNetworkGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Input feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        # Directed River Graph Convolutions (4 hops downstream message passing across river corridor)
        self.gconv1 = DirectedRiverConv(hidden_dim, hidden_dim)
        self.gconv2 = DirectedRiverConv(hidden_dim, hidden_dim)
        self.gconv3 = DirectedRiverConv(hidden_dim, hidden_dim)
        self.gconv4 = DirectedRiverConv(hidden_dim, hidden_dim)

        # Temporal hydrodynamic wave persistence (GRU cell)
        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Multi-Head Prediction Heads (per node)
        self.head_surge = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.ReLU() # non-negative discharge
        )

        self.head_risk = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0 to 1 flood probability
        )

        self.head_lead_time = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.ReLU() # hours until peak
        )

    def forward(
        self,
        x: torch.Tensor,
        upstream_adj: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Num_Nodes, 6] or [Num_Nodes, 6]
        Returns: (surge_cusecs, flood_risk, lead_time_hours, h_new)
        """
        is_single = (x.dim() == 2)
        if is_single:
            x = x.unsqueeze(0) # [1, Num_Nodes, In_Features]

        B, N, F_in = x.shape

        # 1. Feature encoding
        h = self.encoder(x) # [B, N, hidden_dim]

        # 2. Directed downstream message passing across 4 hops
        h1 = self.gconv1(h, upstream_adj)
        h2 = self.gconv2(h1, upstream_adj) + h1
        h3 = self.gconv3(h2, upstream_adj) + h2
        h4 = self.gconv4(h3, upstream_adj) + h3

        # Residual connection
        h_spatial = h + h4

        # 3. Temporal routing update
        h_flat = h_spatial.view(B * N, self.hidden_dim)
        if h_prev is None:
            h_prev_flat = torch.zeros_like(h_flat)
        else:
            h_prev_flat = h_prev.view(B * N, self.hidden_dim)

        h_next_flat = self.temporal_gru(h_flat, h_prev_flat)
        h_out = h_next_flat.view(B, N, self.hidden_dim)

        # 4. Predictions
        surge = self.head_surge(h_out)        # [B, N, 1]
        risk = self.head_risk(h_out)          # [B, N, 1]
        lead_time = self.head_lead_time(h_out) # [B, N, 1]

        if is_single:
            surge = surge.squeeze(0)
            risk = risk.squeeze(0)
            lead_time = lead_time.squeeze(0)
            h_out = h_out.squeeze(0)

        return surge, risk, lead_time, h_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. High-Level River GNN Orchestration Engine
# ─────────────────────────────────────────────────────────────────────────────
class RiverGNNEngine:
    """
    High-level operational interface for evaluating Gujarat river basins,
    simulating dam releases, and comparing GNN vs. Single-City isolated models.
    """
    def __init__(
        self,
        model_path: Optional[str] = "gnn_river/models/river_gnn.pth",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.graph_data = build_graph_tensors()
        self.upstream_adj = self.graph_data["upstream_adj"].to(self.device)
        self.lag_matrix = self.graph_data["lag_matrix"]

        self.model = RiverNetworkGNN(in_features=6, hidden_dim=64).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def build_node_features(
        self,
        local_rainfall: Dict[str, float] = None,
        dam_releases: Dict[str, float] = None,
        soil_saturation: float = 0.4
    ) -> torch.Tensor:
        """
        Builds [NUM_NODES, 6] tensor:
          [rain_mm, inflow_cusecs/1e5, elev/100, rain_3day/100, river_dist/1000, is_dam_active]
        """
        local_rainfall = local_rainfall or {}
        dam_releases = dam_releases or {}

        features = np.zeros((NUM_NODES, 6), dtype=np.float32)

        for i, node in enumerate(GUJARAT_RIVER_NODES):
            name = node["name"]
            basin = node["basin"]
            node_type = node["type"]

            # 1. Local precipitation
            r = local_rainfall.get(name, local_rainfall.get(basin, 0.0))

            # 2. Inflow / Dam discharge
            if node_type in ["Dam", "Reservoir"]:
                release = dam_releases.get(name, 5000.0) # baseline normal release
                is_active_dam = 1.0 if release > 50000.0 else 0.0
            else:
                release = 1000.0
                is_active_dam = 0.0

            # 3. Elevation & river proximity
            elev = node["elev_m"]
            dist_to_main = 100.0 if node_type != "City" else 450.0

            # 4. Antecedent moisture
            r3 = r * 1.6

            features[i] = [
                r / 100.0,                    # scaled rain
                release / 100000.0,           # scaled cusecs (per 1 lakh)
                elev / 300.0,                 # scaled elevation
                r3 / 150.0,                   # scaled 3-day rain
                dist_to_main / 2000.0,        # scaled river distance
                is_active_dam                 # dam gate indicator
            ]

        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def simulate_basin_event(
        self,
        target_city: str = "Surat",
        dam_name: str = "Ukai Dam",
        dam_release_cusecs: float = 500000.0,
        local_rain_mm: float = 0.0
    ) -> Dict[str, Any]:
        """
        Simulates an upstream dam release event (e.g. Ukai Dam release)
        and compares:
          - GNN Prediction (captures downstream flood propagation)
          - Single-City Isolated Model (misses upstream dam release, predicts 0%)
        """
        node = get_node_by_name(target_city)
        node_id = node["id"]
        basin = node["basin"]

        # Build feature tensor with upstream dam release and specified local rain
        node_rain = {target_city: local_rain_mm}
        dam_dict = {dam_name: dam_release_cusecs}
        if "Ajwa" in dam_name:
            dam_dict["Pratappura Reservoir"] = dam_release_cusecs * 0.95

        x_tensor = self.build_node_features(local_rainfall=node_rain, dam_releases=dam_dict)

        with torch.no_grad():
            surge, risk, lead_time, _ = self.model(x_tensor, self.upstream_adj)
            surge_vals = surge.squeeze(-1).cpu().numpy() * 100000.0
            risk_vals = risk.squeeze(-1).cpu().numpy()
            lead_vals = lead_time.squeeze(-1).cpu().numpy()

        city_surge = float(surge_vals[node_id])
        city_risk = float(risk_vals[node_id])
        city_lead_time = float(lead_vals[node_id])

        # Baseline Isolated Single-City Model: evaluates ONLY local_rain_mm
        # If local_rain == 0, isolated model outputs near 0% risk!
        isolated_risk = float(1.0 / (1.0 + np.exp(-(0.04 * local_rain_mm - 3.5))))

        # Upstream transit path calculation
        path = get_upstream_path(node_id)
        path_names = [GUJARAT_RIVER_NODES[i]["name"] for i in path]
        total_dist_km = sum(
            e["dist_km"] for e in GUJARAT_RIVER_EDGES
            if e["source"] in path and e["target"] in path
        )
        total_transit_hrs = sum(
            e["lag_hrs"] for e in GUJARAT_RIVER_EDGES
            if e["source"] in path and e["target"] in path
        )

        return {
            "target_city": target_city,
            "basin": basin,
            "upstream_dam": dam_name,
            "dam_release_cusecs": dam_release_cusecs,
            "local_rain_mm": local_rain_mm,
            "gnn_flood_risk": city_risk,
            "gnn_surge_cusecs": city_surge,
            "gnn_lead_time_hrs": round(city_lead_time if city_lead_time > 0.5 else total_transit_hrs, 1),
            "isolated_model_risk": isolated_risk,
            "upstream_path": path_names,
            "total_dist_km": total_dist_km,
            "total_transit_hrs": total_transit_hrs,
            "all_nodes": [
                {
                    "name": GUJARAT_RIVER_NODES[i]["name"],
                    "basin": GUJARAT_RIVER_NODES[i]["basin"],
                    "type": GUJARAT_RIVER_NODES[i]["type"],
                    "risk": float(risk_vals[i]),
                    "surge_cusecs": float(surge_vals[i])
                }
                for i in range(NUM_NODES)
            ]
        }
