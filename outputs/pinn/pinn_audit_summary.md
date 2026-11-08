# RAINWISE — Physics-Informed Neural Network (PINN) Benchmark Report
**Pillar 2: Conservation of Mass and Hydrological Abstraction Verification**

## Executive Summary
In hydrological disaster intelligence, unconstrained deep learning models treat input features purely as statistical correlations. Under out-of-distribution extremes or localized sensor anomalies, Standard DNNs generate **"phantom water"**, predicting catastrophic flood runoff when soil is bone-dry or precipitation is zero. 

The **RAINWISE Physics-Informed Neural Network (PINN)** embeds the Catchment Continuity Equation ($dS/dt = P - Q - ET - I$) and SCS Curve Number infiltration abstractions directly into loss gradients and neural activations.

| Metric | Standard Unconstrained DNN | Physics-Informed PINN | Civic Impact |
| :--- | :---: | :---: | :--- |
| **Overall Physical Violation Rate** | **29.2%** | **0.0%** | **Zero False Hallucinations** |
| **Max Physical Excess (Phantom Water)** | **0.90 mm** | **0.00 mm** | **100% Mass Conserved** |
| **Infiltration Barrier Violations** | **33.0%** | **0.0%** | Prevents false pre-monsoon sirens |
| **Zero Rain Hallucination Violations** | **42.0%** | **0.0%** | Immunizes VMC from sensor dropouts |
| **Test Set Runoff MSE** | **45.64** | **3.08** | Superior Predictive Accuracy |

## Benchmark Artifacts
- Diagnostic Comparison: `outputs/pinn/physical_violation_benchmark.png`
- Training Curves: `outputs/pinn/pinn_training_curves.png`
- Edge-Case Battery Dataset: `outputs/pinn/pinn_benchmark_results.csv`
- Model Weights: `pinn_hydrology/models/pinn_flood_model.pth`
