# RAINWISE Pillar 4: ConvLSTM Radar Spatio-Temporal Nowcaster Audit Report

## 1. Executive Summary
Traditional hydrological models ingest rainfall as a single scalar measurement at an isolated gauge station ($R \in \mathbb{R}$). However, storm systems are **dynamic 2D convective cloud and precipitation fields** moving across space and time:

$$\frac{\partial Z}{\partial t} + \mathbf{u} \cdot \nabla Z = S(Z)$$

where $\mathbf{u} = (u, v)$ is the atmospheric advection velocity field and $S(Z)$ represents convective storm growth and decay.

The **RAINWISE ConvLSTM Nowcaster (Pillar 4)** replaces scalar station assumptions with multi-step spatio-temporal radar forecasting over a $32 \times 32$ grid across Gujarat. It ingests 4 historical Doppler radar frames ($t-3\text{h}, t-2\text{h}, t-1\text{h}, t_0$) and predicts radar reflectivity and rainfall rates for **$+1\text{h}, +2\text{h}, +3\text{h}, +6\text{h}$**.

---

## 2. Quantitative Verification vs. Classical Persistence Baseline (+3h Horizon)

| Meteorological Scenario | ConvLSTM MAE (dBZ) | Persistence MAE (dBZ) | ConvLSTM CSI ($\ge$ 30 dBZ) | Persistence CSI ($\ge$ 30 dBZ) | Critical Gain |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Monsoon Front Inland Advance** | **2.58 dBZ** | 5.01 dBZ | **0.63** | 0.02 | +3050.0% CSI |
| **Cyclone Biparjoy Vortex** | **5.08 dBZ** | 12.78 dBZ | **0.748** | 0.195 | +283.6% CSI |
| **Urban Convective Cloudburst** | **2.33 dBZ** | 8.22 dBZ | **0.931** | 0.335 | +177.9% CSI |

---

## 3. Key Findings & Operational Municipal Value
1. **Critical Success Index (CSI) Superiority**: ConvLSTM achieves an average $+3\text{h}$ Critical Success Index of **0.770** compared to **0.183** for the Eulerian persistence baseline (a **+319.8% relative performance leap**).
2. **MAE Error Reduction**: Error in storm core intensity is reduced by **61.6%** (3.33 dBZ vs 8.67 dBZ).
3. **Tracking Storm Motion vs. False Alarm Smearing**: Eulerian persistence blindly keeps storm clouds frozen in place, leading to severe false alarms where the storm used to be, and missed alarms where the storm has moved. ConvLSTM captures the advective vector field $\mathbf{u}$, predicting true future storm landfall.
4. **Marshall-Palmer Hydrodynamic Conversion**: Predicted reflectivity factors ($Z$) are converted to mm/hr rain intensity using $Z = 200 \cdot R^{1.6}$, feeding directly into municipal drainage simulations.
