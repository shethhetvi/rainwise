"""
================================================================================
RAINWISE — XAI Visualization Engine for Captum Integrated Gradients
================================================================================
Generates publication-quality charts for municipal flood risk attributions:
  1. Horizontal Attribution Divergence Bar (Positive/Negative color coding)
  2. Waterfall Attribution Chart (Baseline -> Step-by-Step Drivers -> Output)
  3. Multi-City Attribution Comparison Matrix
================================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_attribution_bars(
    result: Dict,
    output_path: str = "outputs/xai/attribution_bars.png",
    figsize: tuple = (10, 5)
) -> str:
    """
    Renders a clean horizontal bar chart showing signed feature attributions.
    Red bars push flood risk UP (danger). Green bars pull flood risk DOWN (protective buffer).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    features = result["features"]
    attributions = np.array(result["attributions"])
    percentages = result["percentage_importance"]
    city = result.get("city", "City")
    prob = result.get("probability", 0.0) * 100

    # Sort by attribution value
    sort_idx = np.argsort(attributions)
    sorted_features = [features[i] for i in sort_idx]
    sorted_attrs = attributions[sort_idx]
    sorted_pcts = [percentages[i] for i in sort_idx]

    # Colors: Red for risk escalators (>0), Teal/Green for protective buffers (<0)
    colors = ['#ef4444' if val > 0 else '#10b981' for val in sorted_attrs]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_attrs, color=colors, height=0.55, edgecolor='#334155', linewidth=1)

    # Add zero reference line
    ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.2, alpha=0.8)

    # Add percentage text labels on bars
    for i, (bar, val, pct) in enumerate(zip(bars, sorted_attrs, sorted_pcts)):
        offset = 0.015 if val >= 0 else -0.015
        align = 'left' if val >= 0 else 'right'
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f} ({pct:.1f}%)",
            va='center',
            ha=align,
            color='#f8fafc',
            fontweight='bold',
            fontsize=9
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, color='#f8fafc', fontsize=11, fontweight='semibold')
    ax.tick_params(axis='x', colors='#94a3b8')

    # Titles and labels
    ax.set_title(
        f"Captum Integrated Gradients — {city} Flood Risk Attribution ({prob:.1f}%)",
        color='#38bdf8',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("Attribution Value (Axiomatic Shift from Neutral Baseline)", color='#94a3b8', fontsize=10, labelpad=10)

    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#334155')

    # Legend annotations
    ax.text(
        0.02, 0.05,
        "■ Red: Risk Escalator (Pushes Probability UP)\n■ Green: Protective Buffer (Mitigates Risk)",
        transform=ax.transAxes,
        color='#cbd5e1',
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#334155", alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return output_path


def plot_waterfall(
    result: Dict,
    output_path: str = "outputs/xai/waterfall_attribution.png",
    figsize: tuple = (11, 6)
) -> str:
    """
    Renders an explainable Waterfall Chart showing how the baseline neutral probability
    steps up and down with each feature's contribution to arrive at the final flood probability.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    city = result.get("city", "Target City")
    baseline = result.get("baseline_probability", 0.05)
    final_prob = result.get("probability", 0.5)
    features = result["features"]
    attributions = result["attributions"]

    labels = ["Baseline"] + list(features) + ["Final Risk"]
    changes = [baseline] + list(attributions) + [0.0]

    # Calculate step positions
    running_total = baseline
    bottoms = [0.0]
    bar_heights = [baseline]
    bar_colors = ['#64748b']

    for attr in attributions:
        if attr >= 0:
            bottoms.append(running_total)
            bar_heights.append(attr)
            bar_colors.append('#ef4444')
        else:
            bottoms.append(running_total + attr)
            bar_heights.append(abs(attr))
            bar_colors.append('#10b981')
        running_total += attr

    # Final bar
    bottoms.append(0.0)
    bar_heights.append(final_prob)
    bar_colors.append('#38bdf8')

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, bar_heights, bottom=bottoms, color=bar_colors, width=0.6, edgecolor='#334155')

    # Value text labels above/below bars
    for i, (b, h, bot) in enumerate(zip(bars, bar_heights, bottoms)):
        if i == 0 or i == len(labels) - 1:
            val = h
            text = f"{val*100:.1f}%"
        else:
            val = attributions[i - 1]
            text = f"{val:+.3f}"
        
        y_loc = bot + h / 2.0
        ax.text(b.get_x() + b.get_width() / 2, y_loc, text, ha='center', va='center',
                color='#f8fafc', fontweight='bold', fontsize=8.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha='right', color='#f8fafc', fontsize=9.5, fontweight='semibold')
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.set_ylabel("Flood Probability (0.0 to 1.0)", color='#94a3b8', fontsize=10)

    ax.set_title(
        f"Axiomatic Risk Decomposition (Waterfall) — {city}\n"
        f"Baseline ({baseline*100:.1f}%) → Final Evaluated Risk ({final_prob*100:.1f}%)",
        color='#38bdf8',
        fontsize=12,
        fontweight='bold',
        pad=15
    )

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return output_path


def plot_multi_city_comparison(
    city_results: List[Dict],
    output_path: str = "outputs/xai/multi_city_attribution_matrix.png",
    figsize: tuple = (12, 6)
) -> str:
    """
    Renders a comparative attribution heatmap across key Gujarat municipal hubs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cities = [r["city"] for r in city_results]
    feature_names = city_results[0]["features"]
    
    matrix = np.array([r["attributions"] for r in city_results])

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    # Diverging colormap: Red for high positive, Blue/Green for negative
    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', vmin=-0.3, vmax=0.8)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.ax.tick_params(colors='#94a3b8')
    cbar.set_label("Captum Attribution (+ Risk / - Buffer)", color='#94a3b8', fontsize=10)

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(cities)))
    ax.set_xticklabels(feature_names, rotation=25, ha='right', color='#f8fafc', fontsize=10, fontweight='semibold')
    ax.set_yticklabels(cities, color='#f8fafc', fontsize=11, fontweight='semibold')

    # Annotate numbers in heatmap
    for i in range(len(cities)):
        for j in range(len(feature_names)):
            val = matrix[i, j]
            text_color = "black" if -0.1 < val < 0.3 else "white"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    ax.set_title("Regional Comparative XAI Matrix — Gujarat Municipal Disaster Attribution",
                 color='#38bdf8', fontsize=13, fontweight='bold', pad=15)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return output_path
