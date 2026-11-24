"""
================================================================================
RAINWISE — ConvLSTM Radar Spatio-Temporal Nowcasting (Pillar 4: ConvLSTM)
================================================================================
Implements:
  1. ConvLSTMCell with 2D spatial convolution gates
  2. Multi-layer ConvLSTM Spatio-Temporal Encoder-Forecaster
  3. RadarPrecipitationNowcaster (T_in=4 past frames -> T_out=4 future frames)
  4. Marshall-Palmer Hydrodynamic Reflectivity-to-Rainfall (Z-R) Inversion
  5. ConvLSTMEngine for live synthetic storm tracking and civic alert nowcasts
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConvLSTM Recurrent Cell
# ─────────────────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """
    2D Convolutional Long Short-Term Memory (ConvLSTM) Cell.
    Replaces matrix multiplications with 2D spatial convolutions:
      i_t = sigma(W_xi * X_t + W_hi * H_{t-1} + b_i)
      f_t = sigma(W_xf * X_t + W_hf * H_{t-1} + b_f)
      g_t = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)
      C_t = f_t * C_{t-1} + i_t * g_t
      o_t = sigma(W_xo * X_t + W_ho * H_{t-1} + b_o)
      H_t = o_t * tanh(C_t)
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Combined convolution for all 4 gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C_in, H, W]
        state: (h, c) each [B, C_hidden, H, W]
        Returns: (h_next, c_next)
        """
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Layer ConvLSTM Encoder-Forecaster Network
# ─────────────────────────────────────────────────────────────────────────────
class ConvLSTM(nn.Module):
    """
    Multi-Layer ConvLSTM processing temporal sequences of 2D spatial frames.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: List[int] = [32, 32],
        kernel_size: int = 3
    ):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_channels)
        self.cell_list = nn.ModuleList()

        for l in range(self.num_layers):
            cur_in = in_channels if l == 0 else hidden_channels[l - 1]
            cur_hidden = hidden_channels[l]
            self.cell_list.append(ConvLSTMCell(cur_in, cur_hidden, kernel_size))

    def forward(
        self,
        x_seq: torch.Tensor,
        init_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x_seq: [B, T, C_in, H, W]
        Returns:
          layer_outputs: [B, T, C_hidden[-1], H, W]
          last_states: list of (h, c) for each layer
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        states = []
        for l in range(self.num_layers):
            if init_states is None:
                h = torch.zeros(B, self.cell_list[l].hidden_channels, H, W, device=device)
                c = torch.zeros(B, self.cell_list[l].hidden_channels, H, W, device=device)
                states.append((h, c))
            else:
                states.append(init_states[l])

        current_input = x_seq
        for l in range(self.num_layers):
            cell = self.cell_list[l]
            h, c = states[l]
            layer_outputs = []

            for t in range(T):
                xt = current_input[:, t, :, :, :]
                h, c = cell(xt, (h, c))
                layer_outputs.append(h)

            states[l] = (h, c)
            current_input = torch.stack(layer_outputs, dim=1) # [B, T, C_l, H, W]

        return current_input, states


# ─────────────────────────────────────────────────────────────────────────────
# 3. Radar Precipitation Nowcaster (Pillar 4)
# ─────────────────────────────────────────────────────────────────────────────
class RadarPrecipitationNowcaster(nn.Module):
    """
    Spatio-Temporal Radar Nowcaster for convective storm tracking across Gujarat.
    Inputs: Past T_in=4 frames of radar reflectivity (t-3h, t-2h, t-1h, t0).
    Outputs: Future T_out=4 frames (+1h, +2h, +3h, +6h).
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: List[int] = [32, 32],
        out_frames: int = 4,
        img_size: int = 32
    ):
        super(RadarPrecipitationNowcaster, self).__init__()
        self.out_frames = out_frames
        self.img_size = img_size

        # Spatial Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1)
        )

        # Recurrent Spatio-Temporal Core
        self.convlstm = ConvLSTM(in_channels=in_channels, hidden_channels=hidden_channels)

        # Forecaster / Decoder Cell
        self.forecaster_cell = ConvLSTMCell(
            in_channels=hidden_channels[-1],
            hidden_channels=hidden_channels[-1],
            kernel_size=3
        )

        # Output Projection Head (predicts normalized reflectivity in [0, 1])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # bounds dBZ to [0.0, 1.0]
        )

    def forward(self, past_radar: torch.Tensor) -> torch.Tensor:
        """
        past_radar: [B, T_in=4, 1, H=32, W=32]
        Returns:
          future_radar: [B, T_out=4, 1, H=32, W=32]
        """
        B, T_in, C, H, W = past_radar.shape

        # Encode temporal sequence through ConvLSTM
        encoded_seq, last_states = self.convlstm(past_radar)

        # Use last hidden state of top layer as starting point for forecasting
        h_fore, c_fore = last_states[-1]
        x_fore = h_fore

        future_frames = []
        for step in range(self.out_frames):
            h_fore, c_fore = self.forecaster_cell(x_fore, (h_fore, c_fore))
            pred_frame = self.head(h_fore) # [B, 1, H, W]
            future_frames.append(pred_frame)
            # Autoregressive feedback: use predicted frame as input to next step
            # Project back to forecaster input dimension via h_fore
            x_fore = h_fore

        return torch.stack(future_frames, dim=1) # [B, T_out, 1, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Marshall-Palmer Hydrodynamic Radar Reflectivity Inversion (Z-R Law)
# ─────────────────────────────────────────────────────────────────────────────
def dbz_to_rainfall_rate(dbz: np.ndarray) -> np.ndarray:
    """
    Converts radar reflectivity factor Z (dBZ) to rain rate R (mm/hr)
    using the standard meteorological Marshall-Palmer formula:
      Z = 200 * R^(1.6)
      R = (10^(Z / 10) / 200)^(1 / 1.6)
    """
    # Clip negative values
    dbz_clean = np.clip(dbz, 0.0, 75.0)
    # Z in mm^6 / m^3
    z_linear = 10.0 ** (dbz_clean / 10.0)
    # R in mm/hr
    r = np.power(z_linear / 200.0, 1.0 / 1.6)
    # Below 15 dBZ is typically virga / non-precipitating clouds
    r[dbz_clean < 15.0] = 0.0
    return r


def rainfall_rate_to_dbz(r: np.ndarray) -> np.ndarray:
    """Converts rain rate R (mm/hr) to radar reflectivity Z (dBZ)."""
    r_pos = np.maximum(r, 0.01)
    z_linear = 200.0 * np.power(r_pos, 1.6)
    dbz = 10.0 * np.log10(z_linear)
    dbz[r < 0.1] = 0.0
    return np.clip(dbz, 0.0, 75.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. High-Level ConvLSTM Nowcasting Engine
# ─────────────────────────────────────────────────────────────────────────────
class ConvLSTMEngine:
    """
    Operational inference engine for Spatio-Temporal Radar Storm Nowcasting across Gujarat.
    """
    def __init__(
        self,
        model_path: Optional[str] = "convlstm_nowcasting/models/convlstm_nowcaster.pth",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.model = RadarPrecipitationNowcaster(
            in_channels=1,
            hidden_channels=[32, 32],
            out_frames=4,
            img_size=32
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

        # Gujarat spatial grid bounds (32 x 32)
        # Lat: 20.0N to 24.5N, Lon: 68.5E to 74.5E
        self.lat_min, self.lat_max = 20.0, 24.5
        self.lon_min, self.lon_max = 68.5, 74.5

        # City coordinates mapped to 32x32 grid indices
        self.city_grid_locs = {
            "Surat": (int((21.17 - 20.0) / (24.5 - 20.0) * 31), int((72.83 - 68.5) / (74.5 - 68.5) * 31)),
            "Vadodara": (int((22.30 - 20.0) / (24.5 - 20.0) * 31), int((73.18 - 68.5) / (74.5 - 68.5) * 31)),
            "Ahmedabad": (int((23.02 - 20.0) / (24.5 - 20.0) * 31), int((72.58 - 68.5) / (74.5 - 68.5) * 31)),
            "Rajkot": (int((22.30 - 20.0) / (24.5 - 20.0) * 31), int((70.80 - 68.5) / (74.5 - 68.5) * 31)),
            "Bhavnagar": (int((21.76 - 20.0) / (24.5 - 20.0) * 31), int((72.15 - 68.5) / (74.5 - 68.5) * 31)),
            "Bharuch": (int((21.70 - 20.0) / (24.5 - 20.0) * 31), int((72.97 - 68.5) / (74.5 - 68.5) * 31))
        }

    def generate_synthetic_storm_sequence(
        self,
        scenario_type: str = "monsoon_front"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic 8-frame sequence (4 past + 4 future) on 32x32 grid:
          - 'monsoon_front': SW monsoon cloudband moving NE from Arabian Sea into South Gujarat
          - 'biparjoy_cyclone': Intense rotating cyclonic vortex moving inland
          - 'cloudburst_cell': Localized convective cloudburst expanding rapidly over Vadodara/Surat
        Returns: (past_8frames [8, 32, 32] in dBZ, ground_truth)
        """
        T = 8
        H, W = 32, 32
        frames = np.zeros((T, H, W), dtype=np.float32)

        # Coordinate mesh
        yy, xx = np.mgrid[0:H, 0:W]

        if scenario_type == "monsoon_front":
            # SW Monsoon front starts in SW (bottom-left) and moves NE at ~2.5 grid cells / frame
            speed_y = 1.4
            speed_x = 2.0
            start_y, start_x = 4.0, 4.0

            for t in range(T):
                cy = start_y + t * speed_y
                cx = start_x + t * speed_x
                # Elongated elliptical cloud band
                dx = (xx - cx)
                dy = (yy - cy)
                dist_sq = (dx * np.cos(np.pi/4) + dy * np.sin(np.pi/4))**2 / 18.0 + \
                          (-dx * np.sin(np.pi/4) + dy * np.cos(np.pi/4))**2 / 6.0
                intensity = np.exp(-dist_sq) * 58.0 # peak 58 dBZ (heavy monsoon)
                # Background moisture
                noise = np.random.uniform(0.0, 8.0, size=(H, W))
                frames[t] = np.clip(intensity + noise, 0.0, 70.0)

        elif scenario_type == "biparjoy_cyclone":
            # Spiral vortex moving from Kutch / Arabian Sea inland
            start_y, start_x = 10.0, 6.0
            speed_y = 1.2
            speed_x = 1.6

            for t in range(T):
                cy = start_y + t * speed_y
                cx = start_x + t * speed_x
                r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                theta = np.arctan2(yy - cy, xx - cx)

                # Eye wall + spiral rain bands
                eyewall = np.exp(-((r - 5.0) / 2.5)**2) * 62.0
                spiral = np.sin(3.0 * theta - r * 0.8) * np.exp(-r / 10.0) * 22.0
                total = eyewall + spiral
                total[r < 2.0] = 12.0 # calm eye
                frames[t] = np.clip(total + np.random.uniform(0.0, 6.0, size=(H, W)), 0.0, 70.0)

        else: # cloudburst_cell
            # Convective cloudburst rapidly intensifying over Central/South Gujarat
            cy, cx = 15.0, 20.0 # near Vadodara / Surat
            for t in range(T):
                # Intensifies from 30 dBZ to 65 dBZ (extreme cloudburst)
                peak_dbz = 32.0 + t * 5.2
                radius = 3.5 + t * 0.8
                dist_sq = ((xx - cx)**2 + (yy - cy)**2) / (radius**2)
                intensity = np.exp(-dist_sq) * peak_dbz
                frames[t] = np.clip(intensity + np.random.uniform(0.0, 5.0, size=(H, W)), 0.0, 70.0)

        past_frames = frames[:4] # [4, 32, 32]
        future_frames = frames[4:] # [4, 32, 32]

        return past_frames, future_frames

    def predict_nowcast(
        self,
        past_frames_dbz: np.ndarray
    ) -> Dict[str, Any]:
        """
        Takes [4, 32, 32] past radar reflectivity frames in dBZ.
        Returns predicted future 4 frames (+1h, +2h, +3h, +6h) in dBZ and mm/hr rain rate,
        plus baseline Eulerian persistence prediction.
        """
        # Normalize dBZ to [0.0, 1.0] (dividing by 70.0 dBZ)
        norm_past = np.clip(past_frames_dbz / 70.0, 0.0, 1.0)
        tensor_in = torch.tensor(norm_past, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)

        with torch.no_grad():
            pred_tensor = self.model(tensor_in) # [1, 4, 1, 32, 32]
            pred_norm = pred_tensor.squeeze(0).squeeze(1).cpu().numpy()

        pred_dbz = pred_norm * 70.0 # denormalize to dBZ

        # Convert to rainfall rates (mm/hr)
        past_rain_rates = [dbz_to_rainfall_rate(past_frames_dbz[t]) for t in range(4)]
        pred_rain_rates = [dbz_to_rainfall_rate(pred_dbz[t]) for t in range(4)]

        # Eulerian Persistence baseline: assumes t0 frame remains static for all future steps
        t0_frame = past_frames_dbz[-1]
        persistence_dbz = np.stack([t0_frame for _ in range(4)], axis=0)
        persistence_rain_rates = [dbz_to_rainfall_rate(persistence_dbz[t]) for t in range(4)]

        # Extract city-specific rain nowcast
        city_forecasts = {}
        lead_time_labels = ["+1 Hour", "+2 Hours", "+3 Hours", "+6 Hours"]

        for city, (gy, gx) in self.city_grid_locs.items():
            city_forecasts[city] = {
                "current_dbz": float(past_frames_dbz[-1, gy, gx]),
                "current_rain_mmhr": float(past_rain_rates[-1][gy, gx]),
                "forecast_dbz": [float(pred_dbz[t, gy, gx]) for t in range(4)],
                "forecast_rain_mmhr": [float(pred_rain_rates[t][gy, gx]) for t in range(4)],
                "persistence_rain_mmhr": [float(persistence_rain_rates[t][gy, gx]) for t in range(4)],
                "lead_labels": lead_time_labels
            }

        return {
            "past_dbz": past_frames_dbz,
            "pred_dbz": pred_dbz,
            "persistence_dbz": persistence_dbz,
            "past_rain_rates": past_rain_rates,
            "pred_rain_rates": pred_rain_rates,
            "persistence_rain_rates": persistence_rain_rates,
            "city_forecasts": city_forecasts,
            "lead_time_labels": lead_time_labels
        }
