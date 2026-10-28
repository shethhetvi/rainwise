"""
================================================================================
RAINWISE — Explainable AI (XAI) for Deep Learning via Captum & Integrated Gradients
================================================================================
Architecture: Axiomatic Feature Attribution for FloodDNN
Theoretical Foundation: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
Implementation: PyTorch + Captum (captum.attr.IntegratedGradients)

Features analyzed:
  1. rain_mm             — Daily / 3-day accumulated precipitation (mm)
  2. elevation_m         — Topographic height above sea level (meters)
  3. distance_to_river_m — Proximity to nearest active river channel (meters)
  4. lat                 — Latitude coordinate
  5. lon                 — Longitude coordinate
================================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional

# Attempt import of Captum; provide internal fallback if necessary
try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compatible FloodDNN Architecture (Matching DLmodels/flood_dnn.pth)
# ─────────────────────────────────────────────────────────────────────────────
class FloodDNN(nn.Module):
    """
    Trained Funnel Deep Neural Network for flood risk probability prediction.
    Architecture: 5 -> 64 -> 32 -> 16 -> 1 with BatchNorm and Sigmoid.
    """
    def __init__(self, input_dim: int = 5):
        super(FloodDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()  # Continuous flood probability [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Explainable Flood Model with Captum Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────
class ExplainableFloodModel:
    """
    Wrapper integrating PyTorch Captum's Integrated Gradients with RAINWISE FloodDNN.
    Solves the Black-Box dilemma by computing mathematically sound, axiomatic feature
    attributions satisfying:
      (1) Completeness Axiom: Sum of attributions == F(x) - F(baseline)
      (2) Implementation Invariance
      (3) Linearity & Sensitivity
    """

    DEFAULT_FEATURE_NAMES = [
        "Precipitation (mm)",
        "Elevation (m)",
        "River Proximity (m)",
        "Latitude",
        "Longitude"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        
        # Default to freshly trained xai_dl/models if available, otherwise DLmodels/
        if model_path is None:
            if os.path.exists("xai_dl/models/flood_dnn_xai.pth"):
                model_path = "xai_dl/models/flood_dnn_xai.pth"
            else:
                model_path = "DLmodels/flood_dnn.pth"
                
        if scaler_path is None:
            if os.path.exists("xai_dl/models/scaler_xai.pkl"):
                scaler_path = "xai_dl/models/scaler_xai.pkl"
            else:
                scaler_path = "DLmodels/scaler.pkl"
                
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Load Model
        self.model = FloodDNN(input_dim=5).to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Loaded trained FloodDNN from: {model_path}")
        else:
            print(f"⚠️ Checkpoint not found at {model_path}. Using initialized weights for demonstration.")
            self.model.eval()

        # Load Scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded StandardScaler from: {scaler_path}")
        else:
            self.scaler = None
            print(f"⚠️ Scaler not found at {scaler_path}.")

        # Initialize Captum Integrated Gradients
        if CAPTUM_AVAILABLE:
            self.ig = IntegratedGradients(self.model)
            # Layer Integrated Gradients for hidden layer neuron attribution
            self.layer_ig = LayerIntegratedGradients(self.model, self.model.network[0])
        else:
            self.ig = None
            self.layer_ig = None
            print("⚠️ Captum not found in environment. Using native Gauss-Legendre Quadrature fallback.")

    def _native_integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Pure PyTorch fallback implementation of Integrated Gradients using
        Gauss-Legendre 50-step quadrature integration:
          IG_i(x) = (x_i - x'_i) * ∫_0^1 [∂F(x' + α(x - x')) / ∂x_i] dα
        """
        inputs = inputs.clone().detach()
        baseline = baseline.clone().detach()
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        
        # Scale inputs along straight path from baseline to input
        diff = inputs - baseline
        interpolated = baseline.unsqueeze(0) + alphas.unsqueeze(1).unsqueeze(2) * diff.unsqueeze(0)
        # interpolated shape: (steps+1, batch, features)
        
        grads = []
        for i in range(steps + 1):
            step_input = interpolated[i].clone().requires_grad_(True)
            output = self.model(step_input)
            grad = torch.autograd.grad(outputs=output, inputs=step_input,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=False, retain_graph=False)[0]
            grads.append(grad.unsqueeze(0))
            
        grads = torch.cat(grads, dim=0) # (steps+1, batch, features)
        # Trapezoidal rule approximation of path integral
        avg_grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grad = avg_grads.mean(dim=0)
        
        attributions = diff * avg_grad
        return attributions

    def explain_city(
        self,
        raw_features: List[float],
        city_name: str = "Gujarat Urban Center",
        feature_names: Optional[List[str]] = None,
        n_steps: int = 50
    ) -> Dict:
        """
        Computes axiomatic feature attributions for a single geographic location.
        
        Parameters:
          raw_features: [rain_mm, elevation_m, distance_to_river_m, lat, lon]
          city_name: Name of municipal jurisdiction
          feature_names: Human-readable names of features
          n_steps: Number of integration steps for Riemannian approximation
          
        Returns:
          Dictionary with prediction, attributions, percentage importances,
          and a formal municipal explanation narrative.
        """
        names = feature_names or self.DEFAULT_FEATURE_NAMES
        raw_arr = np.array([raw_features], dtype=np.float32)
        
        # Normalize with fitted scaler
        if self.scaler is not None:
            scaled_arr = self.scaler.transform(raw_arr)
            # Baseline: normalized zero vector (or population mean)
            baseline_arr = np.zeros_like(scaled_arr)
        else:
            scaled_arr = raw_arr
            baseline_arr = np.zeros_like(scaled_arr)

        input_tensor = torch.tensor(scaled_arr, dtype=torch.float32, device=self.device)
        baseline_tensor = torch.tensor(baseline_arr, dtype=torch.float32, device=self.device)

        # 1. Forward pass prediction
        self.model.eval()
        with torch.no_grad():
            prob = self.model(input_tensor).item()
            baseline_prob = self.model(baseline_tensor).item()

        # 2. Compute Integrated Gradients
        if CAPTUM_AVAILABLE and self.ig is not None:
            input_tensor.requires_grad = True
            attributions, delta = self.ig.attribute(
                input_tensor,
                baselines=baseline_tensor,
                n_steps=n_steps,
                return_convergence_delta=True
            )
            attr_np = attributions.detach().cpu().numpy()[0]
            conv_delta = delta.detach().cpu().item() if isinstance(delta, torch.Tensor) else float(delta)
        else:
            attributions = self._native_integrated_gradients(input_tensor, baseline_tensor, steps=n_steps)
            attr_np = attributions.detach().cpu().numpy()[0]
            # Completeness verification: sum(attributions) vs (prob - baseline_prob)
            conv_delta = abs(float(attr_np.sum()) - (prob - baseline_prob))

        # 3. Calculate Relative Percentage Importance (Normalized absolute attribution)
        abs_attrs = np.abs(attr_np)
        total_abs = abs_attrs.sum()
        if total_abs > 0:
            pct_contributions = (abs_attrs / total_abs) * 100.0
        else:
            pct_contributions = np.zeros_like(abs_attrs)

        # 4. Generate Municipal Explanation Narrative
        # Sort features by positive contribution (factors that push risk UP)
        risk_drivers = []
        protective_factors = []
        
        for name, val, raw_val, pct in zip(names, attr_np, raw_features, pct_contributions):
            entry = {
                "feature": name,
                "raw_value": raw_val,
                "attribution": float(val),
                "importance_pct": float(pct),
                "direction": "Risk Escalator (Pushes UP)" if val > 0 else "Protective Buffer (Pulls DOWN)"
            }
            if val > 0:
                risk_drivers.append(entry)
            else:
                protective_factors.append(entry)

        risk_drivers.sort(key=lambda x: x["attribution"], reverse=True)
        protective_factors.sort(key=lambda x: x["attribution"])

        # Determine Severity Level
        if prob >= 0.70:
            severity = "CRITICAL (RED ALERT)"
        elif prob >= 0.40:
            severity = "MODERATE (ORANGE WATCH)"
        else:
            severity = "LOW (GREEN / SAFE)"

        # Synthesize Human-Readable Administrative Summary
        top_driver = risk_drivers[0] if risk_drivers else None
        second_driver = risk_drivers[1] if len(risk_drivers) > 1 else None

        summary_text = (
            f"MUNICIPAL AUDIT REPORT for {city_name}:\n"
            f"• Flood Risk Probability: {prob * 100:.1f}% [{severity}]\n"
            f"• Baseline Neutral Risk : {baseline_prob * 100:.1f}%\n"
            f"• Primary Risk Driver   : {top_driver['feature']} ({top_driver['raw_value']:.1f}) "
            f"contributing +{top_driver['importance_pct']:.1f}% of total risk divergence.\n"
        )
        if second_driver:
            summary_text += (
                f"• Secondary Risk Driver : {second_driver['feature']} ({second_driver['raw_value']:.1f}) "
                f"contributing +{second_driver['importance_pct']:.1f}%.\n"
            )
        if protective_factors:
            best_buffer = protective_factors[0]
            summary_text += (
                f"• Top Protective Buffer : {best_buffer['feature']} ({best_buffer['raw_value']:.1f}) "
                f"mitigating risk by -{abs(best_buffer['attribution']):.4f}.\n"
            )

        return {
            "city": city_name,
            "probability": prob,
            "severity": severity,
            "baseline_probability": baseline_prob,
            "convergence_delta": conv_delta,
            "features": names,
            "raw_features": raw_features,
            "attributions": [float(a) for a in attr_np],
            "percentage_importance": [float(p) for p in pct_contributions],
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "audit_summary": summary_text
        }


if __name__ == "__main__":
    # Smoke Test
    explainer = ExplainableFloodModel()
    
    # Test sample: High-risk scenario for Ahmedabad
    # rain=150mm, elevation=53m, river_dist=400m, lat=23.02, lon=72.57
    sample_features = [150.0, 53.0, 400.0, 23.02, 72.57]
    result = explainer.explain_city(sample_features, city_name="Ahmedabad")
    
    print("\n" + "="*60)
    print("CAPTUM INTEGRATED GRADIENTS TEST EXECUTION:")
    print("="*60)
    print(result["audit_summary"])
    print(f"Convergence Delta (Axiom Check): {result['convergence_delta']:.6f}")
