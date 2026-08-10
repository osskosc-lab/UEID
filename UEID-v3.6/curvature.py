import numpy as np


def af3_curvature(W, threshold=0.5):
    """Triangle-augmented Forman-Ricci curvature AF3.

    AF3(e) = 4 - deg(u) - deg(v) + 3 * (# triangles containing e).
    The weighted graph is thresholded to a simple undirected graph first.
    """
    W = np.asarray(W, dtype=float)
    B = (W > threshold).astype(np.uint8)
    np.fill_diagonal(B, 0)
    B = np.maximum(B, B.T)
    deg = B.sum(axis=1).astype(int)
    neighbors = [set(np.flatnonzero(B[i])) for i in range(B.shape[0])]
    vals = []
    for u in range(B.shape[0]):
        for v in neighbors[u]:
            if u < v:
                tri = len(neighbors[u].intersection(neighbors[v]))
                vals.append(4 - deg[u] - deg[v] + 3 * tri)
    return np.asarray(vals, dtype=float)


def mean_curvature(curvature):
    curvature = np.asarray(curvature, dtype=float)
    return float(np.mean(curvature)) if curvature.size else np.nan
