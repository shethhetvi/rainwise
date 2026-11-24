"""
================================================================================
RAINWISE — River-Network GNN Training Pipeline (Pillar 3: GNN)
================================================================================
Trains the Spatio-Temporal Graph Neural Network on Gujarat River Basins:
  - Propagates fluid mass downstream from reservoirs/dams to coastal estuaries
  - Predicts discharge surge (cusecs), inundation probability, and lead time
  - Incorporates hydrological transit attenuation across 22 network nodes
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Use temporary matplotlib config
os.environ["MPLCONFIGDIR"] = "/tmp"

from gnn_river.river_graph_topology import (
    GUJARAT_RIVER_NODES,
    GUJARAT_RIVER_EDGES,
    NUM_NODES,
    build_graph_tensors,
    get_upstream_path
)
from gnn_river.river_gnn_model import RiverNetworkGNN

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cpu")


def generate_synthetic_basin_dataset(num_samples: int = 2000):
    """
    Generates synthetic hydrological scenarios across Gujarat's 4 river basins:
      1. Tapi Basin (Ukai Dam -> Mandvi -> Kamrej -> Surat -> Dumas)
      2. Vishwamitri Basin (Pavagadh -> Ajwa -> Pratappura -> Vadodara -> Padra -> Dhadhar)
      3. Sabarmati Basin (Dharoi Dam -> Himmatnagar -> Gandhinagar -> Ahmedabad -> Vasna -> Gulf)
      4. Narmada Basin (Sardar Sarovar -> Kevadiya -> Garudeshwar -> Bharuch -> Ankleshwar -> Dahej)

    Includes diverse hydrological regimes:
      - Dry / Normal Baseflow (low rain, low release)
      - Upstream Dam Emergency Discharge with zero local rain (Surat 2006 test)
      - Localized Urban Cloudburst with closed dam gates (Ahmedabad / Vadodara flash floods)
      - Combined Catastrophic Basin Overload (monsoon peak + dam release)
    """
    X_list = []
    Y_surge_list = []
    Y_risk_list = []
    Y_lead_list = []

    for s in range(num_samples):
        # Scenario type
        scenario = random.choice([
            "normal",
            "dam_release_only",
            "local_storm_only",
            "basin_wide_monsoon",
            "extreme_compound"
        ])

        x_features = np.zeros((NUM_NODES, 6), dtype=np.float32)
        y_surge = np.zeros((NUM_NODES, 1), dtype=np.float32)
        y_risk = np.zeros((NUM_NODES, 1), dtype=np.float32)
        y_lead = np.zeros((NUM_NODES, 1), dtype=np.float32)

        # Base atmospheric conditions
        base_rain = np.random.uniform(0.0, 15.0) if scenario == "normal" else 0.0

        for i, node in enumerate(GUJARAT_RIVER_NODES):
            b_name = node["basin"]
            n_type = node["type"]
            elev = node["elev_m"]
            cap = node["capacity_cusecs"]

            # 1. Local precipitation
            if scenario == "normal":
                r = max(0.0, base_rain + np.random.normal(0, 3.0))
            elif scenario == "dam_release_only":
                # Rain is near zero everywhere
                r = np.random.uniform(0.0, 5.0)
            elif scenario == "local_storm_only":
                # High rain only at cities, dams are quiet
                r = np.random.uniform(80.0, 220.0) if node["is_city"] else np.random.uniform(5.0, 25.0)
            elif scenario == "basin_wide_monsoon":
                r = np.random.uniform(60.0, 160.0)
            else: # extreme_compound
                r = np.random.uniform(120.0, 300.0)

            # 2. Dam / reservoir release
            release = 2000.0
            is_dam = 0.0
            if n_type in ["Dam", "Reservoir"]:
                if scenario == "normal":
                    release = np.random.uniform(2000.0, 15000.0)
                elif scenario == "dam_release_only":
                    # Severe upstream emergency release
                    release = np.random.uniform(350000.0, 950000.0)
                    is_dam = 1.0
                elif scenario == "local_storm_only":
                    release = np.random.uniform(5000.0, 30000.0)
                elif scenario == "basin_wide_monsoon":
                    release = np.random.uniform(150000.0, 500000.0)
                    is_dam = 1.0 if release > 100000.0 else 0.0
                else: # extreme_compound
                    release = np.random.uniform(500000.0, 1100000.0)
                    is_dam = 1.0

            r3 = r * np.random.uniform(1.4, 2.5)
            dist_to_river = 80.0 if not node["is_city"] else 350.0

            x_features[i] = [
                r / 100.0,
                release / 100000.0,
                elev / 300.0,
                r3 / 150.0,
                dist_to_river / 2000.0,
                is_dam
            ]

        # 3. Simulate downstream hydrodynamic wave propagation
        # Traverse downstream edges and route flow
        flow_routed = {}
        transit_times = {}

        for n in GUJARAT_RIVER_NODES:
            nid = n["id"]
            if n["type"] in ["Dam", "Reservoir"]:
                flow_routed[nid] = x_features[nid, 1] * 100000.0
                transit_times[nid] = 0.0
            else:
                flow_routed[nid] = 500.0 # baseflow
                transit_times[nid] = 0.0

        for edge in GUJARAT_RIVER_EDGES:
            u = edge["source"]
            v = edge["target"]
            # Downstream wave attenuation factor based on distance
            attenuation = edge.get("attenuation", max(0.82, 1.0 - edge["dist_km"] * 0.0025))
            lag = edge["lag_hrs"]

            # Wave arriving at target = attenuated inflow + local runoff contribution
            v_node = GUJARAT_RIVER_NODES[v]
            local_runoff_contrib = (x_features[v, 0] * 100.0) * 150.0 # cusecs from rain

            inflow = flow_routed.get(u, 1000.0) * attenuation
            flow_routed[v] = flow_routed.get(v, 0.0) + inflow + local_runoff_contrib
            transit_times[v] = transit_times.get(u, 0.0) + lag

        for i, node in enumerate(GUJARAT_RIVER_NODES):
            q = flow_routed.get(i, 1000.0)
            cap = node["capacity_cusecs"]
            danger = node["danger_level"]

            # Surge target (scaled by 1e5)
            y_surge[i, 0] = q / 100000.0

            # Inundation probability based on capacity ratio
            cap_ratio = q / cap
            prob = 1.0 / (1.0 + np.exp(-6.0 * (cap_ratio - 0.85)))
            y_risk[i, 0] = float(np.clip(prob, 0.0, 1.0))

            # Lead time hours
            y_lead[i, 0] = float(transit_times.get(i, 0.0))

        X_list.append(x_features)
        Y_surge_list.append(y_surge)
        Y_risk_list.append(y_risk)
        Y_lead_list.append(y_lead)

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(Y_surge_list), dtype=torch.float32),
        torch.tensor(np.array(Y_risk_list), dtype=torch.float32),
        torch.tensor(np.array(Y_lead_list), dtype=torch.float32)
    )


def train_river_gnn():
    print("=" * 70)
    print("🌊 RAINWISE — Training River-Network Graph Neural Network (Pillar 3)")
    print("=" * 70)

    os.makedirs("gnn_river/models", exist_ok=True)
    os.makedirs("outputs/gnn", exist_ok=True)

    # 1. Build Network Tensors
    graph_data = build_graph_tensors()
    upstream_adj = graph_data["upstream_adj"].to(DEVICE)
    print(f"✅ Loaded River Graph: {NUM_NODES} nodes, {len(GUJARAT_RIVER_EDGES)} directed hydrological edges.")

    # 2. Generate Synthetic River Basin Training Dataset
    print("⚡ Synthesizing 2,000 multi-basin hydrodynamic scenarios...")
    X, Y_surge, Y_risk, Y_lead = generate_synthetic_basin_dataset(num_samples=2000)

    # 80/20 train/val split
    n_train = 1600
    X_train, X_val = X[:n_train].to(DEVICE), X[n_train:].to(DEVICE)
    Y_surge_tr, Y_surge_val = Y_surge[:n_train].to(DEVICE), Y_surge[n_train:].to(DEVICE)
    Y_risk_tr, Y_risk_val = Y_risk[:n_train].to(DEVICE), Y_risk[n_train:].to(DEVICE)
    Y_lead_tr, Y_lead_val = Y_lead[:n_train].to(DEVICE), Y_lead[n_train:].to(DEVICE)

    # 3. Model & Optimizer
    model = RiverNetworkGNN(in_features=6, hidden_dim=64, num_nodes=NUM_NODES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()

    batch_size = 32
    num_batches = n_train // batch_size
    epochs = 50

    train_losses = []
    val_losses = []

    print(f"🚀 Training RiverNetworkGNN for {epochs} epochs on CPU...")

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_train)
        epoch_loss = 0.0

        for b in range(num_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            xb = X_train[b_idx]
            yb_surge = Y_surge_tr[b_idx]
            yb_risk = Y_risk_tr[b_idx]
            yb_lead = Y_lead_tr[b_idx]

            optimizer.zero_grad()

            pred_surge, pred_risk, pred_lead, _ = model(xb, upstream_adj)

            # Multi-task loss
            l_surge = loss_mse(pred_surge, yb_surge)
            l_risk = loss_bce(pred_risk, yb_risk)
            l_lead = loss_mse(pred_lead, yb_lead)

            total_loss = l_surge + 2.0 * l_risk + 0.3 * l_lead
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        epoch_loss /= num_batches
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            v_surge, v_risk, v_lead, _ = model(X_val, upstream_adj)
            v_loss = (loss_mse(v_surge, Y_surge_val) +
                      2.0 * loss_bce(v_risk, Y_risk_val) +
                      0.3 * loss_mse(v_lead, Y_lead_val)).item()
            val_losses.append(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {v_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 4. Save Model Weights
    save_path = "gnn_river/models/river_gnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model weights saved successfully to: {save_path}")

    # 5. Plot Training Loss Curves
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(range(1, epochs + 1), train_losses, label="Train Multi-Task Loss", color="#0284c7", lw=2)
    ax.plot(range(1, epochs + 1), val_losses, label="Val Multi-Task Loss", color="#f97316", lw=2, linestyle="--")
    ax.set_title("River-Network GNN (Pillar 3) Convergence Curves", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Hydrodynamic Multi-Task Loss", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    curve_path = "outputs/gnn/river_gnn_training_curves.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"📈 Convergence curves saved to: {curve_path}")

    return model


if __name__ == "__main__":
    train_river_gnn()
