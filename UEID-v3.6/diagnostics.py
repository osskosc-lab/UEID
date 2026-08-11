import numpy as np


def peak_location(g, y):
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return np.nan
    return float(g[np.nanargmax(y)])


def seed_gc_stability(gc_values):
    gc_values = np.asarray(gc_values, dtype=float)
    return {"mean": float(np.nanmean(gc_values)), "sd": float(np.nanstd(gc_values, ddof=1))}
