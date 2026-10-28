"""
RAINWISE Explainable AI (XAI) Package
Provides PyTorch Captum Integrated Gradients for Deep Neural Networks.
"""
from xai_dl.explainable_flood_model import ExplainableFloodModel, FloodDNN
from xai_dl.visualize_attributions import plot_attribution_bars, plot_waterfall, plot_multi_city_comparison

__all__ = [
    "ExplainableFloodModel",
    "FloodDNN",
    "plot_attribution_bars",
    "plot_waterfall",
    "plot_multi_city_comparison"
]
