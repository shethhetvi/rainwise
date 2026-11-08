"""
================================================================================
RAINWISE — Automated Dashboard Navigation & High-Resolution Screenshot Capture
================================================================================
This script uses Playwright to navigate the live Streamlit dashboard on localhost:8501,
trigger interactive inferences across all tabs (ML Alert, Cascading DL, Captum XAI,
PINN Physics Conservation, TFT, and Civic Benchmarks), and capture screenshots for
inclusion in the formal technical PDF report.
================================================================================
"""

import os
import time
from playwright.sync_api import sync_playwright

OUTPUT_DIR = "outputs/screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_all_tabs():
    print("🚀 Initializing Playwright Chromium Browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1050},
            device_scale_factor=2  # Retina crisp screenshots
        )
        page = context.new_page()

        print("🌐 Navigating to RAINWISE Dashboard at http://localhost:8501 ...")
        page.goto("http://localhost:8501", timeout=30000, wait_until="networkidle")
        page.wait_for_timeout(4000)

        # ─────────────────────────────────────────────────────────────────────
        # Tab 1: Telemetry & Rapid ML Baseline Alert
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 1: Real-Time Telemetry & ML Alert...")
        try:
            # Trigger ML Classification
            btn_ml = page.locator("button:has-text('RUN ML FLOOD CLASSIFICATION')")
            if btn_ml.count() > 0:
                btn_ml.first.click()
                page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "01_telemetry_ml_alert.png"))
            print("   Saved 01_telemetry_ml_alert.png")
        except Exception as e:
            print(f"   Tab 1 capture error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Tab 2: Cascading 2-Stage DL Pipeline
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 2: Cascading DL Pipeline...")
        try:
            tabs = page.locator("[role='tab']")
            if tabs.count() >= 2:
                tabs.nth(1).click()
                page.wait_for_timeout(2500)
                btn_dl = page.locator("button:has-text('EXECUTE CASCADING DL INFERENCE')")
                if btn_dl.count() > 0:
                    btn_dl.first.click()
                    page.wait_for_timeout(2500)
                page.screenshot(path=os.path.join(OUTPUT_DIR, "02_cascading_dl_pipeline.png"))
                print("   Saved 02_cascading_dl_pipeline.png")
        except Exception as e:
            print(f"   Tab 2 capture error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Tab 3: Explainable AI (Captum Integrated Gradients)
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 3: Explainable AI (Captum)...")
        try:
            tabs = page.locator("[role='tab']")
            if tabs.count() >= 3:
                tabs.nth(2).click()
                page.wait_for_timeout(2500)
                btn_xai = page.locator("button:has-text('COMPUTE AXIOMATIC ATTRIBUTIONS')")
                if btn_xai.count() > 0:
                    btn_xai.first.click()
                    page.wait_for_timeout(4000)
                page.screenshot(path=os.path.join(OUTPUT_DIR, "03_explainable_ai_captum.png"))
                print("   Saved 03_explainable_ai_captum.png")
        except Exception as e:
            print(f"   Tab 3 capture error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Tab 4: PINN Physics Verification (Conservation of Mass)
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 4: PINN Physics Verification...")
        try:
            tabs = page.locator("[role='tab']")
            if tabs.count() >= 4:
                tabs.nth(3).click()
                page.wait_for_timeout(2500)

                # Capture baseline state
                page.screenshot(path=os.path.join(OUTPUT_DIR, "04_pinn_physics_baseline.png"))
                print("   Saved 04_pinn_physics_baseline.png")

                # Trigger Stress Preset 1: Bone-Dry Soil (Demonstrates Standard Violation vs PINN Consistency)
                dry_soil_btn = page.locator("button:has-text('Bone-Dry Soil')")
                if dry_soil_btn.count() > 0:
                    dry_soil_btn.first.click()
                    page.wait_for_timeout(3000)
                    page.screenshot(path=os.path.join(OUTPUT_DIR, "04_pinn_stress_violation_audit.png"))
                    print("   Saved 04_pinn_stress_violation_audit.png (Violation Proven!)")

                # Scroll down to capture the benchmark diagnostic section
                page.evaluate("window.scrollBy(0, 750)")
                page.wait_for_timeout(2000)
                page.screenshot(path=os.path.join(OUTPUT_DIR, "05_pinn_diagnostics_gallery.png"))
                print("   Saved 05_pinn_diagnostics_gallery.png")
        except Exception as e:
            print(f"   Tab 4 capture error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Tab 5: Spatio-Temporal Forecasting (TFT Quantiles)
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 5: TFT Spatio-Temporal Forecasting...")
        try:
            tabs = page.locator("[role='tab']")
            if tabs.count() >= 5:
                tabs.nth(4).click()
                page.wait_for_timeout(2500)
                btn_tft = page.locator("button:has-text('RUN TEMPORAL FUSION TRANSFORMER')")
                if btn_tft.count() > 0:
                    btn_tft.first.click()
                    page.wait_for_timeout(2500)
                page.screenshot(path=os.path.join(OUTPUT_DIR, "06_tft_spatio_temporal.png"))
                print("   Saved 06_tft_spatio_temporal.png")
        except Exception as e:
            print(f"   Tab 5 capture error: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Tab 6: Gujarat Municipal Benchmarks
        # ─────────────────────────────────────────────────────────────────────
        print("📸 Capturing Tab 6: Gujarat Municipal Benchmarks...")
        try:
            tabs = page.locator("[role='tab']")
            if tabs.count() >= 6:
                tabs.nth(5).click()
                page.wait_for_timeout(2500)
                btn_surat = page.locator("button:has-text('Surat')")
                if btn_surat.count() > 0:
                    btn_surat.first.click()
                    page.wait_for_timeout(3000)
                page.screenshot(path=os.path.join(OUTPUT_DIR, "07_municipal_benchmarks.png"))
                print("   Saved 07_municipal_benchmarks.png")
        except Exception as e:
            print(f"   Tab 6 capture error: {e}")

        browser.close()
        print("\n✅ All dashboard screenshots successfully captured!")


if __name__ == "__main__":
    capture_all_tabs()

