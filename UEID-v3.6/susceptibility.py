import numpy as np


def equilibrium_susceptibility(R, N):
    R = np.asarray(R, dtype=float)
    return N * np.cov(R, rowvar=False, ddof=1)


def ness_correction(R, A, dt=1.0, max_lag=4):
    R = np.asarray(R, dtype=float)
    A = np.asarray(A, dtype=float)
    T, K = R.shape
    Rc = R - np.nanmean(R, axis=0)
    Ac = A - np.nanmean(A, axis=0)
    delta = np.zeros((K, K), dtype=float)
    max_lag = min(max_lag, T - 1)
    for lag in range(max_lag + 1):
        r = Rc[lag:]
        a = Ac[: T - lag]
        delta += (r.T @ a) / max(1, len(r)) * dt
    return delta


def generalized_susceptibility(R, A, N, dt=1.0, max_lag=4):
    chi_eq = equilibrium_susceptibility(R, N)
    delta = ness_correction(R, A, dt=dt, max_lag=max_lag)
    return {"chi_total": chi_eq + delta, "chi_eq": chi_eq, "chi_ness": delta}


def susceptibility_strength(chi):
    chi = np.asarray(chi, dtype=float)
    eigvals = np.linalg.eigvalsh(0.5 * (chi + chi.T))
    return float(np.max(eigvals))
