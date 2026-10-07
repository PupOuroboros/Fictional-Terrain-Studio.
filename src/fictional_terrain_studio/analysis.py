from __future__ import annotations

import numpy as np


def slope_percent(elevation_m: np.ndarray, extent_km: float) -> np.ndarray:
    n_y, n_x = elevation_m.shape
    dx = extent_km * 1000.0 / (n_x - 1)
    dy = extent_km * 1000.0 / (n_y - 1)
    gy, gx = np.gradient(elevation_m.astype(np.float64), dy, dx)
    return (np.hypot(gx, gy) * 100.0).astype(np.float32)


def terrain_report(elevation_m: np.ndarray, extent_km: float, target_grade_percent: float) -> dict:
    slopes = slope_percent(elevation_m, extent_km)
    finite = np.isfinite(elevation_m)
    vals = elevation_m[finite]
    s = slopes[np.isfinite(slopes)]

    def frac(limit: float) -> float:
        return float(np.mean(s <= limit)) if s.size else 0.0

    return {
        "resolution": [int(elevation_m.shape[1]), int(elevation_m.shape[0])],
        "extent_km": float(extent_km),
        "elevation_min_m": float(np.min(vals)),
        "elevation_max_m": float(np.max(vals)),
        "total_relief_m": float(np.max(vals) - np.min(vals)),
        "slope_mean_percent": float(np.mean(s)),
        "slope_p95_percent": float(np.percentile(s, 95)),
        "fraction_under_5pct": frac(5.0),
        "fraction_under_10pct": frac(10.0),
        "fraction_under_target_grade": frac(float(target_grade_percent)),
        "target_grade_percent": float(target_grade_percent),
    }
