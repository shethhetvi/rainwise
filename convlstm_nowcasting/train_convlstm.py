"""
================================================================================
RAINWISE — ConvLSTM Radar Spatio-Temporal Training Pipeline (Pillar 4: ConvLSTM)
================================================================================
Trains the 2D ConvLSTM Spatio-Temporal Nowcaster on Gujarat Radar Storm sequences:
  - Arabian Sea southwest monsoon cloud fronts advancing into South Gujarat
  - Severe cyclonic storms (Cyclone Biparjoy rotational vortex)
  - Convective cloudburst cells rapidly intensifying over urban catchments
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib.pyplot as plt

from convlstm_nowcasting.convlstm_model import RadarPrecipitationNowcaster, ConvLSTMEngine

# Set reproducibility seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cpu")


def generate_training_radar_dataset(num_samples: int = 500):
    """
    Synthesizes diverse spatio-temporal radar sequences covering Gujarat:
    Returns:
      X_seq: [num_samples, 4, 1, 32, 32] (past 4 frames normalized [0, 1])
      Y_seq: [num_samples, 4, 1, 32, 32] (future 4 frames normalized [0, 1])
    """
    engine = ConvLSTMEngine(model_path=None, device=DEVICE)
    scenarios = ["monsoon_front", "biparjoy_cyclone", "cloudburst_cell"]

    X_list = []
    Y_list = []

    for i in range(num_samples):
        sc = random.choice(scenarios)
        past_dbz, future_dbz = engine.generate_synthetic_storm_sequence(scenario_type=sc)

        # Normalize dBZ to [0.0, 1.0] (70.0 dBZ max)
        past_norm = np.clip(past_dbz / 70.0, 0.0, 1.0)
        future_norm = np.clip(future_dbz / 70.0, 0.0, 1.0)

        # Add random sensor noise
        past_norm = np.clip(past_norm + np.random.normal(0, 0.02, past_norm.shape), 0.0, 1.0)

        X_list.append(np.expand_dims(past_norm, axis=1))    # [4, 1, 32, 32]
        Y_list.append(np.expand_dims(future_norm, axis=1))  # [4, 1, 32, 32]

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(Y_list), dtype=torch.float32)
    )


def train_convlstm():
    print("=" * 70)
    print("🛰️ RAINWISE — Training ConvLSTM Radar Spatio-Temporal Nowcaster (Pillar 4)")
    print("=" * 70)

    os.makedirs("convlstm_nowcasting/models", exist_ok=True)
    os.makedirs("outputs/convlstm", exist_ok=True)

    # 1. Synthesize Radar Sequences
    print("⚡ Generating 500 spatio-temporal storm radar sequences (32x32 grids)...")
    X, Y = generate_training_radar_dataset(num_samples=500)

    # 80/20 train/val split
    n_train = 400
    X_train, X_val = X[:n_train].to(DEVICE), X[n_train:].to(DEVICE)
    Y_train, Y_val = Y[:n_train].to(DEVICE), Y[n_train:].to(DEVICE)

    # 2. Model, Loss & Optimizer
    model = RadarPrecipitationNowcaster(
        in_channels=1,
        hidden_channels=[32, 32],
        out_frames=4,
        img_size=32
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()

    batch_size = 16
    num_batches = n_train // batch_size
    epochs = 25

    train_losses = []
    val_losses = []

    print(f"🚀 Training ConvLSTM Nowcaster for {epochs} epochs on CPU...")

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_train)
        epoch_loss = 0.0

        for b in range(num_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            xb = X_train[b_idx]
            yb = Y_train[b_idx]

            optimizer.zero_grad()
            pred_y = model(xb)

            # Combined MSE + L1 loss (sharpens spatial precipitation gradients)
            loss = loss_mse(pred_y, yb) + 0.5 * loss_l1(pred_y, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= num_batches
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            v_loss = (loss_mse(val_preds, Y_val) + 0.5 * loss_l1(val_preds, Y_val)).item()
            val_losses.append(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.5f} | Val Loss: {v_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 3. Save Model Weights
    save_path = "convlstm_nowcasting/models/convlstm_nowcaster.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model weights saved to: {save_path}")

    # 4. Plot Convergence Curves
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(range(1, epochs + 1), train_losses, label="Train Loss (MSE + 0.5 L1)", color="#0284c7", lw=2)
    ax.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="#f97316", lw=2, linestyle="--")
    ax.set_title("ConvLSTM Spatio-Temporal Nowcaster Convergence Curves", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Spatio-Temporal Reconstruction Loss", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    curve_path = "outputs/convlstm/convlstm_training_curves.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"📈 Convergence plot saved to: {curve_path}")

    return model


if __name__ == "__main__":
    train_convlstm()
