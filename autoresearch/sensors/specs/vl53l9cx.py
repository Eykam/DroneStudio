"""VL53L9CX dToF LiDAR module - datasheet-anchored spec.

Sources (ST DS14879 Rev 7, verified against extracted text 2026-09-06):
- 2.3K-zone dToF: up to 54x42 zones with binning options (DS14879 features).
- FoV 55 deg horizontal x 42 deg vertical (71 deg diagonal), software-reducible
  (Table 3, FoV angles).
- Range: precision mode <5 cm to 8.8 m; ambient mode 45 cm to 8.5 m (Sec 2,
  ranging modes).
- Frame rate up to 100 Hz (features / Table 1 sample rate).
- Profile examples (Table 9): gaming 54x42@100fps/4ms/5m/420mW(5klx);
  room mapping 54x42@30fps/6ms/8m; AR 54x42@20fps/16ms/3m(100klx);
  autofocus 24x20@15fps/5ms/8.8m; wake 12x10@1fps/2ms/8.5m.
- Junction temp -30..70C; ranging error/noise rises with temperature (2.9.1).
- Dual 940nm VCSEL flood illumination, SPAD array, on-chip histogram
  processing with cover-glass crosstalk compensation.
- Package body 12.83 x 6.10 x 4.64 mm (Fig 23 p36; authoritative - supersedes
  stale 12.1 x 5.1 x 4.5 ST product-page metadata).

Noise model parameters below marked [model] are ST FlightSense family
behavior (VL53L5CX-class public characterization), not VL53L9CX datasheet
numbers - DS14879 gives profile max ranges under ambient conditions, not
sigma curves. Labeled so they can be re-anchored when characterization data
lands.
"""
VL53L9CX_SPEC = {
    "part": "VL53L9CX",
    "kind": "dToF multizone LiDAR",
    # --- geometry (datasheet) ---
    "fov_h_deg": 55.0,
    "fov_v_deg": 42.0,
    "zones_native": (42, 54),              # rows x cols
    "zone_grids": [(42, 54), (21, 27), (20, 24), (10, 12)],  # binning options
    # --- ranging (datasheet) ---
    "range_min_m": {"precision": 0.05, "ambient": 0.45},
    "range_max_m": {"precision": 8.8, "ambient": 8.5},
    "rate_hz_max": 100.0,
    "vcsel_nm": 940,
    "vcsel_count": 2,
    # --- thermal (datasheet) ---
    "temp_range_c": (-30.0, 70.0),
    # --- modes (datasheet Table 9) ---
    "modes": {
        "gaming":       {"grid": (42, 54), "mode": "precision", "fps": 100, "exposure_ms": 4,  "max_m": 5.0, "mw": 420, "ambient": "5klx"},
        "room_mapping": {"grid": (42, 54), "mode": "ambient",   "fps": 30,  "exposure_ms": 6,  "max_m": 8.0, "mw": 200, "ambient": "5klx"},
        "ar_glasses":   {"grid": (42, 54), "mode": "ambient",   "fps": 20,  "exposure_ms": 16, "max_m": 3.0, "mw": 600, "ambient": "100klx"},
        "autofocus":    {"grid": (20, 24), "mode": "precision", "fps": 15,  "exposure_ms": 5,  "max_m": 8.8, "mw": 80,  "ambient": "indoor"},
        "wake":         {"grid": (10, 12), "mode": "ambient",   "fps": 1,   "exposure_ms": 2,  "max_m": 8.5, "mw": 12.5, "ambient": "dark"},
    },
    # --- noise model [model: ST FlightSense family behavior, not DS14879] ---
    "sigma_base_mm": 3.0,          # [model] close-range floor
    "sigma_range2": 0.0006,        # [model] sigma += k * r^2 (mm, r in m)
    "dropout_start_frac": 0.8,     # [model] dropout onset at 80% of mode max
    "dropout_slope": 5.0,          # [model] p -> 1 as range -> max
    "ambient_sigma_scale": 2.0,    # [model] sigma x this at high ambient (100klx)
    "temp_sigma_scale_per_30c": 0.25,  # [model] +25% sigma per 30C above 25C
    "min_return_snr": 0.05,        # [model] below -> status no-return
}
