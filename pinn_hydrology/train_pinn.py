"""
================================================================================
RAINWISE — Side-by-Side PINN vs. Standard DNN Hydrological Training Pipeline
================================================================================
This script:
  1. Synthesizes representative Gujarat hydrological catchment data with physical
     mass balance budgets (Precipitation, Infiltration, Evapotranspiration, Storage).
  2. Enforces strict Train / Test split first (zero data leakage) and fits StandardScaler.
  3. Trains two models side-by-side:
       - Model A: Standard Unconstrained DNN (Trained purely on empirical loss)
       - Model B: Physics-Informed Neural Network (PINN with HydrologicalWaterBalanceLoss)
  4. Records training loss curves, validation loss, and physical violation rates.
  5. Saves model weights to pinn_hydrology/models/ and visualization to outputs/pinn/.
================================================================================
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pinn_hydrology.pinn_water_balance import (
    compute_hydrological_budget,
    HydrologicalWaterBalanceLoss,
    StandardFloodModel,
    PINNFloodModel
)

# CONFIGURATION
MODEL_SAVE_DIR = "pinn_hydrology/models"
OUTPUT_DIR = "outputs/pinn"
EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.002
DEVICE = torch.device("cpu")


def generate_hydrological_dataset(n_samples: int = 2400):
    """
    Generates realistic, physically consistent catchment hydrology for Gujarat:
    Surat (estuary, high rain), Ahmedabad/Vadodara (alluvial river basins),
    Saputara (high elevation hills), Rajkot (semi-arid Saurashtra plateau).
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Precipitation P (mm)
    # Mixture of dry days (0mm), moderate monsoon showers (15-60mm), and heavy deluges (80-250mm)
    p_dry = np.zeros(n_samples // 4)
    p_mod = np.random.exponential(scale=35.0, size=n_samples // 2)
    p_extreme = np.random.uniform(70.0, 240.0, size=n_samples // 4)
    precip = np.concatenate([p_dry, p_mod, p_extreme])
    precip = np.clip(precip, 0.0, 260.0)
    np.random.shuffle(precip)

    # 2. Antecedent Soil Moisture Saturation (0.05 to 0.95)
    soil_moisture = np.random.beta(a=2.0, b=2.5, size=n_samples)
    soil_moisture = np.clip(soil_moisture, 0.05, 0.95)

    # 3. Evapotranspiration ET (mm/day)
    et = np.random.uniform(2.5, 6.5, size=n_samples)

    # 4. Elevation (meters above sea level)
    elev_coastal = np.random.uniform(5.0, 30.0, size=n_samples // 4)
    elev_plains = np.random.uniform(30.0, 90.0, size=n_samples // 2)
    elev_high = np.random.uniform(100.0, 950.0, size=n_samples // 4)
    elevation = np.concatenate([elev_coastal, elev_plains, elev_high])
    np.random.shuffle(elevation)

    # 5. Distance to Active River Channel (meters)
    river_dist = np.random.exponential(scale=1200.0, size=n_samples) + 120.0
    river_dist = np.clip(river_dist, 100.0, 12000.0)

    # 6. Latitude & Longitude (Gujarat Bounds)
    lat = np.random.uniform(20.2, 24.5, size=n_samples)
    lon = np.random.uniform(69.0, 74.0, size=n_samples)

    # 7. Physical Mass Balance Calculation:
    # W_avail = max(0, P - I - ET)
    w_avail, i_act, _ = compute_hydrological_budget(precip, soil_moisture, et, s_max=40.0)

    # Runoff Generation:
    # When W_avail > 0, runoff coefficient C depends on soil saturation, slope, and river proximity
    river_proximity_factor = 600.0 / (river_dist + 400.0)
    slope_retention = elevation / 1200.0
    c_runoff = 0.30 + 0.45 * soil_moisture + 0.25 * river_proximity_factor - 0.15 * slope_retention
    c_runoff = np.clip(c_runoff, 0.10, 0.95)

    # Actual Surface Runoff Q_true <= W_avail (Physical Mass Conservation)
    # Add minor natural empirical variance, but strictly clip to [0, W_avail]
    noise = np.random.normal(0, 0.02, size=n_samples)
    c_noisy = np.clip(c_runoff + noise, 0.05, 0.98)
    q_true = c_noisy * w_avail
    q_true = np.clip(q_true, 0.0, w_avail)

    # Flood Inundation Probability & Binary Class
    # Severe inundation happens when runoff exceeds drainage capacity (> 25mm)
    flood_prob_true = 1.0 / (1.0 + np.exp(-(0.18 * q_true + 1.2 * river_proximity_factor - 1.8)))
    flood_prob_true = np.clip(flood_prob_true, 0.0, 1.0)
    flood_label = (flood_prob_true >= 0.50).astype(np.float32)

    X = np.column_stack([precip, soil_moisture, et, elevation, river_dist, lat, lon])
    return X, q_true, flood_label, w_avail


def train_models():
    """Trains Standard DNN and Physics-Informed PINN side-by-side."""
    print("=" * 70)
    print("🌊 RAINWISE — Side-by-Side PINN vs. Standard DNN Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate Hydrological Data
    X, q_true, y_flood, w_avail = generate_hydrological_dataset(n_samples=2400)
    print(f"Total Hydrological Instances: {len(X)}")

    # 2. Strict Train / Test Split FIRST (No Data Leakage)
    (
        X_train, X_test,
        q_train, q_test,
        y_train, y_test,
        w_train, w_test
    ) = train_test_split(
        X, q_true, y_flood, w_avail,
        test_size=0.20,
        random_state=42,
        stratify=y_flood
    )

    # 3. Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "pinn_scaler.pkl"))
    print("✅ Fitted StandardScaler saved to pinn_hydrology/models/pinn_scaler.pkl")

    # PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(q_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(w_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_train, dtype=torch.float32)  # Raw features for physics checks
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(q_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(w_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Instantiate Models
    standard_model = StandardFloodModel(input_dim=7).to(DEVICE)
    pinn_model = PINNFloodModel(input_dim=7, enforce_physics=True).to(DEVICE)

    # Standard model loss (Unconstrained MSE + BCE)
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()

    # PINN loss with Hydrological Mass Balance Penalty (lambda_physics = 4.0)
    pinn_loss_fn = HydrologicalWaterBalanceLoss(
        lambda_physics=4.0,
        lambda_bounds=2.5,
        lambda_task=1.0
    )

    opt_std = optim.Adam(standard_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    opt_pinn = optim.Adam(pinn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Metrics history
    history = {
        "epoch": [],
        "std_train_loss": [], "std_val_loss": [], "std_violation_rate": [],
        "pinn_train_loss": [], "pinn_val_loss": [], "pinn_violation_rate": []
    }

    print("\n🚀 Beginning Side-by-Side Training...")
    print(f"{'Epoch':<6} | {'Std Loss':<10} {'Std Viol%':<10} | {'PINN Loss':<10} {'PINN Viol%':<10}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        standard_model.train()
        pinn_model.train()

        std_running_loss = 0.0
        pinn_running_loss = 0.0

        for b_x_scaled, b_q, b_y, b_w, b_raw in train_loader:
            b_x_scaled = b_x_scaled.to(DEVICE)
            b_q = b_q.to(DEVICE)
            b_y = b_y.to(DEVICE)
            b_w = b_w.to(DEVICE)

            # --- A. Train Standard Unconstrained Model ---
            opt_std.zero_grad()
            std_q_pred, std_p_pred = standard_model(b_x_scaled)
            loss_std = mse_loss_fn(std_q_pred, b_q) + 0.5 * bce_loss_fn(std_p_pred, b_y)
            loss_std.backward()
            opt_std.step()
            std_running_loss += loss_std.item() * b_x_scaled.size(0)

            # --- B. Train Physics-Informed PINN Model ---
            opt_pinn.zero_grad()
            pinn_q_pred, pinn_p_pred = pinn_model(b_x_scaled, w_avail=b_w)
            loss_dict = pinn_loss_fn(
                q_pred=pinn_q_pred,
                q_true=b_q,
                w_avail=b_w,
                p_pred=pinn_p_pred,
                p_true=b_y
            )
            loss_pinn = loss_dict["total_loss"]
            loss_pinn.backward()
            opt_pinn.step()
            pinn_running_loss += loss_pinn.item() * b_x_scaled.size(0)

        # Epoch Evaluation on Test Set
        standard_model.eval()
        pinn_model.eval()

        std_test_loss = 0.0
        pinn_test_loss = 0.0
        std_violations = 0
        pinn_violations = 0
        total_test_samples = 0

        with torch.no_grad():
            for b_x_scaled, b_q, b_y, b_w, b_raw in test_loader:
                b_x_scaled = b_x_scaled.to(DEVICE)
                b_q = b_q.to(DEVICE)
                b_y = b_y.to(DEVICE)
                b_w = b_w.to(DEVICE)
                n_b = b_x_scaled.size(0)
                total_test_samples += n_b

                # Evaluate Standard Model
                s_q, s_p = standard_model(b_x_scaled)
                s_l = mse_loss_fn(s_q, b_q) + 0.5 * bce_loss_fn(s_p, b_y)
                std_test_loss += s_l.item() * n_b
                
                # Check Standard Physical Violations (Q > W_avail + eps)
                s_v = (s_q > (b_w + 1e-3)) | (s_q < -1e-3)
                std_violations += s_v.sum().item()

                # Evaluate PINN
                p_q, p_p = pinn_model(b_x_scaled, w_avail=b_w)
                p_l_dict = pinn_loss_fn(q_pred=p_q, q_true=b_q, w_avail=b_w, p_pred=p_p, p_true=b_y)
                pinn_test_loss += p_l_dict["task_loss"].item() * n_b
                
                # Check PINN Physical Violations
                p_v = (p_q > (b_w + 1e-3)) | (p_q < -1e-3)
                pinn_violations += p_v.sum().item()

        epoch_std_train = std_running_loss / len(train_dataset)
        epoch_pinn_train = pinn_running_loss / len(train_dataset)
        epoch_std_val = std_test_loss / total_test_samples
        epoch_pinn_val = pinn_test_loss / total_test_samples
        std_viol_pct = (std_violations / total_test_samples) * 100.0
        pinn_viol_pct = (pinn_violations / total_test_samples) * 100.0

        history["epoch"].append(epoch)
        history["std_train_loss"].append(epoch_std_train)
        history["std_val_loss"].append(epoch_std_val)
        history["std_violation_rate"].append(std_viol_pct)
        history["pinn_train_loss"].append(epoch_pinn_train)
        history["pinn_val_loss"].append(epoch_pinn_val)
        history["pinn_violation_rate"].append(pinn_viol_pct)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"{epoch:<6} | {epoch_std_val:<10.4f} {std_viol_pct:<9.1f}% | {epoch_pinn_val:<10.4f} {pinn_viol_pct:<9.1f}%")

    # 5. Save Trained Models
    std_path = os.path.join(MODEL_SAVE_DIR, "standard_flood_model.pth")
    pinn_path = os.path.join(MODEL_SAVE_DIR, "pinn_flood_model.pth")
    torch.save(standard_model.state_dict(), std_path)
    torch.save(pinn_model.state_dict(), pinn_path)
    print(f"\n✅ Standard Model weights saved to: {std_path}")
    print(f"✅ PINN Model weights saved to: {pinn_path}")

    # 6. Plot & Save Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0b1120")

    # Loss Curves
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")
    ax1.plot(history["epoch"], history["std_val_loss"], label="Standard DNN (Val MSE)", color="#f87171", lw=2.2, linestyle="--")
    ax1.plot(history["epoch"], history["pinn_val_loss"], label="PINN Model (Val MSE)", color="#38bdf8", lw=2.5)
    ax1.set_title("Validation Task Loss Across Epochs", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("Mean Squared Error (Runoff)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    # Violation Rate Curves
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    ax2.plot(history["epoch"], history["std_violation_rate"], label="Standard DNN Violation Rate (%)", color="#ef4444", lw=2.5)
    ax2.plot(history["epoch"], history["pinn_violation_rate"], label="PINN Violation Rate (%)", color="#10b981", lw=3.0)
    ax2.set_title("Physical Mass Balance Violation Rate (%)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Conservation Violations (%)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "pinn_training_curves.png")
    plt.savefig(curve_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"📊 Training curves saved to: {curve_path}")

    # Save training history to CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUTPUT_DIR, "pinn_training_history.csv"), index=False)
    print("📈 Training log saved to outputs/pinn/pinn_training_history.csv")


if __name__ == "__main__":
    train_models()
