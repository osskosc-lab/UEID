import numpy as np


def graph_laplacian(W):
    W = np.asarray(W, dtype=float)
    return np.diag(np.sum(W, axis=1)) - W


def estimate_spectral_dimension(W, lower_quantile=0.05, upper_quantile=0.30):
    L = graph_laplacian(W)
    eigvals = np.linalg.eigvalsh(0.5 * (L + L.T))
    eigvals = np.sort(eigvals[eigvals > 1e-10])
    n = len(eigvals)
    if n < 10:
        return np.nan
    lo = max(1, int(lower_quantile * n))
    hi = max(lo + 4, int(upper_quantile * n))
    hi = min(hi, n)
    lam = eigvals[lo:hi]
    counts = np.arange(lo + 1, hi + 1, dtype=float)
    good = np.isfinite(lam) & (lam > 0)
    if good.sum() < 4:
        return np.nan
    slope, _ = np.polyfit(np.log(lam[good]), np.log(counts[good]), 1)
    return float(2.0 * slope)
