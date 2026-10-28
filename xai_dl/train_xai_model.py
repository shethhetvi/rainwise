"""
================================================================================
RAINWISE — FloodDNN Training Pipeline with Integrated Captum Validation
================================================================================
This script:
  1. Loads Gujarat hydrological data (rain, elevation, river distance, lat, lon).
  2. Prepares train/test splits and fits a StandardScaler.
  3. Trains FloodDNN using PyTorch (BCE Loss, Adam, Dropout, BatchNorm).
  4. Tracks training loss, validation loss, and test accuracy across epochs.
  5. Computes post-training Captum Integrated Gradients on test cases.
  6. Saves newly trained model artifacts to xai_dl/models/.
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from xai_dl.explainable_flood_model import FloodDNN, ExplainableFloodModel

# CONFIGURATION
MODEL_SAVE_DIR = "xai_dl/models"
OUTPUT_DIR = "outputs/xai"
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.002
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_and_prep_data(data_path: str = "data/processed/model_dataset_clean.csv"):
    """Loads and standardizes hydrological features for Gujarat flood classification."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    
    # 5 standard features matching FloodDNN architecture:
    # rain (using precipitation_mm or rain_3day), elevation_m, distance_to_river_m, lat, lon
    rain_col = "precipitation_mm" if "precipitation_mm" in df.columns else "rain3_mm"
    
    features = [rain_col, "elevation_m", "distance_to_river_m", "lat", "lon"]
    target = "flood"
    
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features].values
    y = df_clean[target].values
    
    # Synthesize realistic environmental augmentations if dataset is small
    if len(X) < 1000:
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, size=(len(X) * 5, X.shape[1]))
        X_repeated = np.repeat(X, 5, axis=0) + noise
        y_repeated = np.repeat(y, 5, axis=0)
        # Ensure non-negative rainfall and physical bounds
        X_repeated[:, 0] = np.clip(X_repeated[:, 0], 0, 500)
        X_repeated[:, 2] = np.clip(X_repeated[:, 2], 10, 500000)
        X = np.vstack([X, X_repeated])
        y = np.concatenate([y, y_repeated])
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, scaler, X_test_scaled, y_test


def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("🌊 RAINWISE — TRAINING FLOOD-DNN WITH CAPTUM XAI VERIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    
    # 1. Load Data
    train_loader, test_loader, scaler, X_test_scaled, y_test = load_and_prep_data()
    print(f"✅ Data loaded and standardized (Input Dim: 5)")
    
    # 2. Build Model
    model = FloodDNN(input_dim=5).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 3. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss = float("inf")
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, "flood_dnn_xai.pth")
    scaler_save_path = os.path.join(MODEL_SAVE_DIR, "scaler_xai.pkl")
    
    print("\n🚀 Commencing neural optimization...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
                
        avg_train = running_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_targets, bin_preds)
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            try:
                auc = roc_auc_score(all_targets, all_preds)
                auc_str = f"ROC-AUC: {auc:.4f}"
            except Exception:
                auc_str = ""
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc*100:.2f}% | {auc_str}")
            
        # Checkpoint Best Model
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), model_save_path)
            joblib.dump(scaler, scaler_save_path)
            
    print(f"\n✅ Training Complete. Best weights saved to: {model_save_path}")
    
    # 4. Save Loss and Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')
    
    ax1.plot(history["train_loss"], label="Train Loss", color="#38bdf8", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#f43f5e", linewidth=2)
    ax1.set_title("FloodDNN Loss (Binary Cross Entropy)", color="#f8fafc", fontweight="bold")
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    
    ax2.plot([a * 100 for a in history["val_acc"]], label="Validation Accuracy (%)", color="#10b981", linewidth=2)
    ax2.set_title("Validation Accuracy Convergence", color="#f8fafc", fontweight="bold")
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    
    curve_path = os.path.join(OUTPUT_DIR, "xai_training_curves.png")
    plt.tight_layout()
    plt.savefig(curve_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"📈 Saved training curves to: {curve_path}")
    
    # 5. Immediate Post-Training Captum XAI Verification
    print("\n🔍 Running Post-Training Captum Integrated Gradients on Test Sample...")
    xai_explainer = ExplainableFloodModel(
        model_path=model_save_path,
        scaler_path=scaler_save_path
    )
    test_sample = [145.0, 48.0, 500.0, 23.02, 72.57]  # Ahmedabad scenario
    res = xai_explainer.explain_city(test_sample, city_name="Ahmedabad Post-Training Test")
    print(res["audit_summary"])
    print(f"Completeness Axiom Convergence Delta: {res['convergence_delta']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    train()
