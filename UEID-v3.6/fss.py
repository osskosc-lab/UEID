import numpy as np


def fit_fss(N_values, chi_max_values):
    N_values = np.asarray(N_values, dtype=float)
    chi_max_values = np.asarray(chi_max_values, dtype=float)
    good = (N_values > 0) & (chi_max_values > 0) & np.isfinite(chi_max_values)
    x = np.log(N_values[good])
    y = np.log(chi_max_values[good])
    if len(x) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}
