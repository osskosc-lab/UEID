import numpy as np


def node_order(theta):
    theta = np.asarray(theta)
    return np.abs(np.mean(np.exp(1j * theta), axis=-1))


def edge_order(W):
    W = np.asarray(W, dtype=float)
    eigvals = np.linalg.eigvalsh(0.5 * (W + W.T))
    denom = np.sum(np.abs(eigvals)) + 1e-12
    return float(np.max(np.abs(eigvals)) / denom)


def geometry_order(curvature):
    curvature = np.asarray(curvature, dtype=float)
    if curvature.size == 0:
        return np.nan
    mu = np.mean(curvature)
    sigma = np.std(curvature)
    return float(np.abs(mu) / (np.abs(mu) + sigma + 1e-12))


def build_multiplex_order(theta_series, W_series, curvature_series):
    rows = []
    for theta, W, curvature in zip(theta_series, W_series, curvature_series):
        rows.append([float(node_order(theta)), edge_order(W), geometry_order(curvature)])
    return np.asarray(rows, dtype=float)
