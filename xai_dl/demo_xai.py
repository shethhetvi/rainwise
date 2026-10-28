"""
================================================================================
RAINWISE — XAI Demo & Comprehensive Municipal Case Study Suite
================================================================================
Runs Captum Integrated Gradients across 5 Gujarat cities under monsoon conditions:
  1. Surat     — Coastal estuary, critical river proximity (13m elevation)
  2. Ahmedabad — Sabarmati basin, high urban density (53m elevation)
  3. Vadodara  — Vishwamitri river flood plain (39m elevation)
  4. Saputara  — Dang hill station (1000m elevation) -> Demonstrates Elevation Buffer
  5. Rajkot    — Elevated semi-arid plateau (132m elevation) -> Safe Baseline

Generates explainable audit reports and visual charts in outputs/xai/.
================================================================================
"""

import os
import sys

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xai_dl.explainable_flood_model import ExplainableFloodModel
from xai_dl.visualize_attributions import (
    plot_attribution_bars,
    plot_waterfall,
    plot_multi_city_comparison
)


def run_xai_demonstration():
    print("=" * 75)
    print("🌊 RAINWISE AI — EXPLAINABLE DEEP LEARNING (CAPTUM INTEGRATED GRADIENTS)")
    print("=" * 75)

    explainer = ExplainableFloodModel(
        model_path="DLmodels/flood_dnn.pth",
        scaler_path="DLmodels/scaler.pkl"
    )

    # 5 Gujarat Municipal Case Studies
    # Features format: [rain_mm, elevation_m, distance_to_river_m, lat, lon]
    scenarios = [
        {
            "city": "Surat (Tapi River Estuary)",
            "features": [180.0, 13.0, 250.0, 21.17, 72.83],
            "note": "Low elevation + heavy rainfall + extreme river proximity"
        },
        {
            "city": "Ahmedabad (Sabarmati Basin)",
            "features": [140.0, 53.0, 600.0, 23.02, 72.57],
            "note": "Substantial rainfall + dense urban flood plain"
        },
        {
            "city": "Vadodara (Vishwamitri Plain)",
            "features": [125.0, 39.0, 450.0, 22.30, 73.18],
            "note": "Flash-flood prone river channel"
        },
        {
            "city": "Saputara (Dang Hill Station)",
            "features": [190.0, 1000.0, 3500.0, 20.58, 73.75],
            "note": "Extreme rainfall BUT 1000m mountain elevation acts as protective buffer"
        },
        {
            "city": "Rajkot (Saurashtra Plateau)",
            "features": [35.0, 132.0, 2800.0, 22.30, 70.80],
            "note": "Moderate rainfall + elevated plateau (Safe scenario)"
        }
    ]

    results = []
    output_dir = "outputs/xai"
    os.makedirs(output_dir, exist_ok=True)

    for sc in scenarios:
        city = sc["city"]
        raw = sc["features"]
        print(f"\n📍 Analyzing: {city} ...")
        res = explainer.explain_city(raw, city_name=city)
        results.append(res)
        
        print(f"   • Flood Probability : {res['probability']*100:.1f}% [{res['severity']}]")
        print(f"   • Convergence Delta : {res['convergence_delta']:.6f} (Completeness Axiom)")
        print(f"   • Primary Escalator : {res['risk_drivers'][0]['feature']} (+{res['risk_drivers'][0]['importance_pct']:.1f}%)")
        if res['protective_factors']:
            print(f"   • Primary Buffer    : {res['protective_factors'][0]['feature']} (-{abs(res['protective_factors'][0]['attribution']):.4f})")

    # Generate Individual Visualizations for Ahmedabad & Surat
    print("\n📊 Generating Publication Visualizations...")
    
    # 1. Horizontal Attribution Bars for Ahmedabad
    ahmedabad_res = results[1]
    bar_path = os.path.join(output_dir, "ahmedabad_attribution_bars.png")
    plot_attribution_bars(ahmedabad_res, output_path=bar_path)
    print(f"   💾 Saved: {bar_path}")

    # 2. Waterfall Chart for Surat
    surat_res = results[0]
    waterfall_path = os.path.join(output_dir, "surat_waterfall_attribution.png")
    plot_waterfall(surat_res, output_path=waterfall_path)
    print(f"   💾 Saved: {waterfall_path}")

    # 3. Waterfall Chart for Saputara (Demonstrating Mountain Buffer)
    saputara_res = results[3]
    saputara_waterfall = os.path.join(output_dir, "saputara_waterfall_elevation_buffer.png")
    plot_waterfall(saputara_res, output_path=saputara_waterfall)
    print(f"   💾 Saved: {saputara_waterfall}")

    # 4. Multi-City Comparison Heatmap
    matrix_path = os.path.join(output_dir, "gujarat_multi_city_attribution_matrix.png")
    plot_multi_city_comparison(results, output_path=matrix_path)
    print(f"   💾 Saved: {matrix_path}")

    # Save Complete Municipal Audit Log
    audit_log_path = os.path.join(output_dir, "municipal_audit_reports.txt")
    with open(audit_log_path, "w") as f:
        f.write("========================================================================\n")
        f.write("RAINWISE — MUNICIPAL EXPLAINABLE AI AUDIT LOG (CAPTUM INTEGRATED GRADIENTS)\n")
        f.write("========================================================================\n\n")
        for res in results:
            f.write(res["audit_summary"])
            f.write(f"Completeness Convergence Delta: {res['convergence_delta']:.6f}\n")
            f.write("-" * 72 + "\n")
    print(f"   💾 Saved: {audit_log_path}")

    print("\n" + "=" * 75)
    print("✅ CAPTUM INTEGRATED GRADIENTS SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_xai_demonstration()
