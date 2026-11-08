"""
================================================================================
RAINWISE — Physics-Informed Neural Network (PINN) for Hydrological Water Balance
================================================================================
Pillar 2: Conservation of Mass & Hydrological Domain Constraints
Mathematical Principles:
  1. Catchment Continuity Equation: dS/dt = P - Q - ET - I
  2. Conservation of Mass Inequality: Q <= max(0, P - I - ET)
  3. Non-negativity Constraint: Q >= 0

Standard Unconstrained DNNs frequently violate the First Law of Thermodynamics 
on edge cases by hallucinating "phantom water" (Q > P - I - ET) when precipitation 
is low or zero. 

The PINN incorporates HydrologicalWaterBalanceLoss during training and physical
domain gating to achieve 0.0% physical violations.
================================================================================
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hydrological Water Balance & Abstraction Calculation
# ─────────────────────────────────────────────────────────────────────────────
def compute_hydrological_budget(
    precip: Union[torch.Tensor, np.ndarray, float],
    soil_moisture: Union[torch.Tensor, np.ndarray, float],
    et: Union[torch.Tensor, np.ndarray, float],
    s_max: float = 40.0
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Computes catchment hydrological abstractions based on the SCS Curve Number / Horton
    infiltration principle:
      - Infiltration capacity: I_cap = S_max * (1.0 - soil_moisture)
      - Actual Infiltration: I = min(P, I_cap)
      - Maximum Permissible Surface Runoff: W_avail = max(0, P - I - ET)

    Returns:
      (w_avail, actual_infiltration, et)
    """
    is_tensor = isinstance(precip, torch.Tensor)
    
    if is_tensor:
        # Moisture saturation [0.0, 1.0]
        sm = torch.clamp(soil_moisture, 0.0, 1.0)
        p = torch.clamp(precip, min=0.0)
        e = torch.clamp(et, min=0.0)
        
        i_cap = s_max * (1.0 - sm)
        i_act = torch.minimum(p, i_cap)
        w_avail = torch.clamp(p - i_act - e, min=0.0)
        return w_avail, i_act, e
    else:
        sm = np.clip(np.asarray(soil_moisture, dtype=np.float32), 0.0, 1.0)
        p = np.clip(np.asarray(precip, dtype=np.float32), 0.0, None)
        e = np.clip(np.asarray(et, dtype=np.float32), 0.0, None)
        
        i_cap = s_max * (1.0 - sm)
        i_act = np.minimum(p, i_cap)
        w_avail = np.maximum(0.0, p - i_act - e)
        return w_avail, i_act, e


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch HydrologicalWaterBalanceLoss
# ─────────────────────────────────────────────────────────────────────────────
class HydrologicalWaterBalanceLoss(nn.Module):
    """
    Physics-Informed Loss Function penalizing non-physical predictions.
    
    Total Loss:
      L_total = lambda_task * L_task + lambda_physics * L_physics + lambda_bounds * L_bounds
      
    where:
      L_task    = MSE(Q_pred, Q_true) + BCE(P_pred, P_true)
      L_physics = Mean( [ReLU( Q_pred - W_avail )]^2 )     # Mass conservation violation
      L_bounds  = Mean( [ReLU( -Q_pred )]^2 )              # Non-negativity violation
    """
    def __init__(
        self,
        lambda_physics: float = 3.5,
        lambda_bounds: float = 2.0,
        lambda_task: float = 1.0,
        eps: float = 1e-4
    ):
        super(HydrologicalWaterBalanceLoss, self).__init__()
        self.lambda_physics = float(lambda_physics)
        self.lambda_bounds = float(lambda_bounds)
        self.lambda_task = float(lambda_task)
        self.eps = float(eps)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        q_pred: torch.Tensor,
        q_true: torch.Tensor,
        w_avail: torch.Tensor,
        p_pred: Optional[torch.Tensor] = None,
        p_true: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the combined supervised and physical loss.
        """
        # 1. Supervised Task Loss
        l_runoff = self.mse(q_pred, q_true)
        if p_pred is not None and p_true is not None:
            l_prob = self.bce(p_pred, p_true)
            l_task = l_runoff + 0.5 * l_prob
        else:
            l_task = l_runoff

        # 2. Physics Mass Conservation Loss: ReLU(Q_pred - W_avail)^2
        # Penalizes any runoff exceeding available precipitation after abstractions
        excess_runoff = F.relu(q_pred - w_avail)
        l_physics = torch.mean(torch.square(excess_runoff))

        # 3. Non-negativity Loss: ReLU(-Q_pred)^2
        negative_runoff = F.relu(-q_pred)
        l_bounds = torch.mean(torch.square(negative_runoff))

        # 4. Total Loss
        total_loss = (
            self.lambda_task * l_task +
            self.lambda_physics * l_physics +
            self.lambda_bounds * l_bounds
        )

        # 5. Physical Violation Rate (%)
        with torch.no_grad():
            violations = (q_pred > (w_avail + self.eps)) | (q_pred < -self.eps)
            violation_rate = (violations.float().mean()) * 100.0

        return {
            "total_loss": total_loss,
            "task_loss": l_task,
            "physics_loss": l_physics,
            "bounds_loss": l_bounds,
            "violation_rate": violation_rate
        }

    def check_violations(
        self,
        q_pred: torch.Tensor,
        w_avail: torch.Tensor,
        threshold: float = 1e-3
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluates predictions against the mass balance envelope.
        Returns:
          (violation_mask, violation_rate_percentage)
        """
        with torch.no_grad():
            mask = (q_pred > (w_avail + threshold)) | (q_pred < -threshold)
            rate = float(mask.float().mean().item() * 100.0)
            return mask, rate


# ─────────────────────────────────────────────────────────────────────────────
# 3. Standard Unconstrained Deep Neural Network (Baseline)
# ─────────────────────────────────────────────────────────────────────────────
class StandardFloodModel(nn.Module):
    """
    Standard Unconstrained Deep Neural Network for flood runoff & probability.
    Has NO knowledge of mass conservation laws. Outputs can freely exceed precipitation.
    Architecture: 7 -> 64 -> 64 -> 32 -> Dual Heads [Runoff (mm), Inundation Prob]
    """
    def __init__(self, input_dim: int = 7):
        super(StandardFloodModel, self).__init__()
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.15),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.SiLU()
        )
        
        # Dual Heads
        self.runoff_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.ReLU() # Standard unconstrained non-negative head
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        runoff_mm = self.runoff_head(features)
        flood_prob = self.risk_head(features)
        return runoff_mm, flood_prob


# ─────────────────────────────────────────────────────────────────────────────
# 4. Physics-Informed Neural Network (PINN Flood Model)
# ─────────────────────────────────────────────────────────────────────────────
class PINNFloodModel(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Hydrological Water Balance.
    
    Guarantees physical consistency through:
      1. HydrologicalWaterBalanceLoss during gradient updates.
      2. Physics-Informed Runoff Projection:
         Q = W_avail * Sigmoid(latent_runoff_logit)
         which strictly bounds runoff within [0.0, W_avail], eliminating any possibility
         of physical violation regardless of out-of-distribution input anomalies.
    """
    def __init__(self, input_dim: int = 7, enforce_physics: bool = True):
        super(PINNFloodModel, self).__init__()
        self.input_dim = input_dim
        self.enforce_physics = enforce_physics
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(64, 32),
            nn.SiLU()
        )
        
        # Latent runoff activation head (scaled between 0 and 1 via sigmoid)
        self.runoff_latent = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        
        # Inundation risk probability head
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        w_avail: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        If w_avail is provided and enforce_physics is True:
          runoff_mm = w_avail * Sigmoid(latent)
        Otherwise:
          runoff_mm = F.relu(latent)
        """
        features = self.encoder(x)
        latent = self.runoff_latent(features)
        flood_prob = self.risk_head(features)
        
        if self.enforce_physics and w_avail is not None:
            # Physical Conservation Gate: Runoff fraction in [0.0, 1.0] times W_avail
            runoff_fraction = torch.sigmoid(latent)
            runoff_mm = runoff_fraction * w_avail
        else:
            runoff_mm = F.softplus(latent)
            
        return runoff_mm, flood_prob


# ─────────────────────────────────────────────────────────────────────────────
# 5. High-Level Predictor & Auditor Class
# ─────────────────────────────────────────────────────────────────────────────
class PINNHydrologyEngine:
    """
    High-level orchestrator for loading, evaluating, and auditing both
    the Standard DNN and the PINN Flood Model on Gujarat civic hydrological scenarios.
    """
    FEATURE_NAMES = [
        "Precipitation (mm)",
        "Soil Moisture Saturation (0-1)",
        "Evapotranspiration (mm)",
        "Elevation (m)",
        "Distance to River (m)",
        "Latitude (°N)",
        "Longitude (°E)"
    ]

    def __init__(
        self,
        pinn_path: Optional[str] = "pinn_hydrology/models/pinn_flood_model.pth",
        standard_path: Optional[str] = "pinn_hydrology/models/standard_flood_model.pth",
        scaler_path: Optional[str] = "pinn_hydrology/models/pinn_scaler.pkl",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.loss_fn = HydrologicalWaterBalanceLoss()
        
        self.standard_model = StandardFloodModel(input_dim=7).to(self.device)
        self.pinn_model = PINNFloodModel(input_dim=7, enforce_physics=True).to(self.device)
        self.scaler = None
        
        if scaler_path and os.path.exists(scaler_path):
            import joblib
            self.scaler = joblib.load(scaler_path)
            
        if standard_path and os.path.exists(standard_path):
            self.standard_model.load_state_dict(torch.load(standard_path, map_location=self.device))
            self.standard_model.eval()
            
        if pinn_path and os.path.exists(pinn_path):
            self.pinn_model.load_state_dict(torch.load(pinn_path, map_location=self.device))
            self.pinn_model.eval()

    def predict_scenario(
        self,
        precip_mm: float,
        soil_moisture: float,
        et_mm: float,
        elevation_m: float,
        river_dist_m: float,
        lat: float,
        lon: float
    ) -> Dict[str, Union[float, bool]]:
        """
        Runs side-by-side inference through both Standard DNN and PINN,
        computing physical consistency, available water budget, and mass violations.
        """
        w_avail, i_act, _ = compute_hydrological_budget(precip_mm, soil_moisture, et_mm)
        w_avail_val = float(w_avail)
        i_act_val = float(i_act)
        
        raw_feat = np.array([[precip_mm, soil_moisture, et_mm, elevation_m, river_dist_m, lat, lon]], dtype=np.float32)
        if self.scaler is not None:
            feat_scaled = self.scaler.transform(raw_feat)
        else:
            feat_scaled = raw_feat
            
        x_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(self.device)
        w_tensor = torch.tensor([[w_avail_val]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            std_q, std_prob = self.standard_model(x_tensor)
            pinn_q, pinn_prob = self.pinn_model(x_tensor, w_avail=w_tensor)
            
            std_q_val = float(std_q.cpu().item())
            std_prob_val = float(std_prob.cpu().item())
            pinn_q_val = float(pinn_q.cpu().item())
            pinn_prob_val = float(pinn_prob.cpu().item())
            
        # Conservation check
        std_violates = (std_q_val > (w_avail_val + 1e-3)) or (std_q_val < -1e-3)
        pinn_violates = (pinn_q_val > (w_avail_val + 1e-3)) or (pinn_q_val < -1e-3)
        
        std_excess = max(0.0, std_q_val - w_avail_val)
        pinn_excess = max(0.0, pinn_q_val - w_avail_val)
        
        return {
            "precip_mm": precip_mm,
            "soil_moisture": soil_moisture,
            "et_mm": et_mm,
            "infiltration_mm": i_act_val,
            "w_avail_mm": w_avail_val,
            "std_runoff_mm": std_q_val,
            "std_flood_prob": std_prob_val,
            "std_violates": std_violates,
            "std_excess_mm": std_excess,
            "pinn_runoff_mm": pinn_q_val,
            "pinn_flood_prob": pinn_prob_val,
            "pinn_violates": pinn_violates,
            "pinn_excess_mm": pinn_excess
        }
