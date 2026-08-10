import numpy as np
from curvature import af3_curvature, mean_curvature
from order_parameters import build_multiplex_order
from spectral_dimension import estimate_spectral_dimension
from susceptibility import generalized_susceptibility, susceptibility_strength


def _torus_edges(N, rng, shortcut_degree=2):
    m = int(round(np.sqrt(N)))
    if m * m != N:
        raise ValueError("Pilot sizes must be perfect squares for the torus base graph.")
    edges = set()
    def idx(i, j):
        return (i % m) * m + (j % m)
    for i in range(m):
        for j in range(m):
            u = idx(i, j)
            for di, dj in ((1, 0), (0, 1)):
                v = idx(i + di, j + dj)
                edges.add((min(u, v), max(u, v)))
    for u in range(N):
        added = 0
        tries = 0
        while added < shortcut_degree and tries < 50 * shortcut_degree:
            v = int(rng.integers(0, N))
            tries += 1
            if v == u:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges:
                edges.add(e)
                added += 1
    arr = np.asarray(sorted(edges), dtype=int)
    return arr[:, 0], arr[:, 1]


def _dense_W(N, u, v, w):
    W = np.zeros((N, N), dtype=float)
    W[u, v] = w
    W[v, u] = w
    return W


def _edge_weight_shuffle_W(N, u, v, w, rng):
    ws = np.array(w, copy=True)
    rng.shuffle(ws)
    return _dense_W(N, u, v, ws)


def simulate_one(N, g, seed, cfg):
    rng = np.random.default_rng(seed + 100_003 * N + int(round(g * 10_000)))
    u, v = _torus_edges(N, rng, cfg.shortcut_degree)
    E = len(u)
    theta = rng.uniform(-np.pi, np.pi, size=N)
    omega = rng.normal(0.0, cfg.omega_sd, size=N)
    w = rng.uniform(0.65, 0.95, size=E)

    theta_samples, W_samples, curv_samples, A_samples = [], [], [], []
    W_null_samples, curv_null_samples = [], []
    entropy_flux = []

    for step in range(cfg.steps):
        t = step * cfg.dt
        delta = theta[v] - theta[u]
        phase_flow = w * np.sin(delta)
        acc = np.zeros(N, dtype=float)
        degw = np.zeros(N, dtype=float)
        np.add.at(acc, u, phase_flow)
        np.add.at(acc, v, -phase_flow)
        np.add.at(degw, u, w)
        np.add.at(degw, v, w)
        coupling = acc / np.maximum(degw, 1e-8)

        h = cfg.drive_amp * np.sin(cfg.drive_omega * t)
        noise = np.sqrt(2 * cfg.noise_D * cfg.dt) * rng.standard_normal(N)
        theta += cfg.dt * (omega + g * coupling + h * np.sin(-theta)) + noise
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        delta_new = theta[v] - theta[u]
        target = 1.0 / (1.0 + np.exp(-cfg.plasticity_beta * (np.cos(delta_new) - cfg.plasticity_threshold)))
        dw = cfg.plasticity_eta * (target - w)
        w += cfg.dt * dw
        w = np.clip(w, 0.02, 1.0)
        entropy_flux.append(float(np.mean(dw * dw) / (cfg.noise_D + 1e-12)))

        if step >= cfg.burn_in and (step - cfg.burn_in) % cfg.sample_every == 0:
            W = _dense_W(N, u, v, w)
            curv = af3_curvature(W, threshold=cfg.binary_weight_threshold)
            Wn = _edge_weight_shuffle_W(N, u, v, w, rng)
            curvn = af3_curvature(Wn, threshold=cfg.binary_weight_threshold)
            theta_samples.append(theta.copy())
            W_samples.append(W)
            curv_samples.append(curv)
            W_null_samples.append(Wn)
            curv_null_samples.append(curvn)
            mc = mean_curvature(curv)
            A_samples.append([h, h * float(np.mean(w)), h * (0.0 if not np.isfinite(mc) else mc)])

    theta_samples = np.asarray(theta_samples)
    A = np.asarray(A_samples, dtype=float)
    R = build_multiplex_order(theta_samples, W_samples, curv_samples)
    R_null = build_multiplex_order(theta_samples, W_null_samples, curv_null_samples)

    sample_dt = cfg.dt * cfg.sample_every
    chi = generalized_susceptibility(R, A, N, dt=sample_dt, max_lag=cfg.ness_max_lag)
    chi_null = generalized_susceptibility(R_null, A, N, dt=sample_dt, max_lag=cfg.ness_max_lag)

    d_info = np.asarray([estimate_spectral_dimension(W, cfg.spectral_q_lo, cfg.spectral_q_hi) for W in W_samples], dtype=float)
    d_info_null = np.asarray([estimate_spectral_dimension(W, cfg.spectral_q_lo, cfg.spectral_q_hi) for W in W_null_samples], dtype=float)
    chi_eq = chi["chi_eq"]
    chi_total = chi["chi_total"]
    chi_ness = chi["chi_ness"]

    return {
        "N": N, "control_parameter": g, "seed": seed,
        "R_node_mean": float(np.nanmean(R[:, 0])),
        "R_edge_mean": float(np.nanmean(R[:, 1])),
        "R_geo_mean": float(np.nanmean(R[:, 2])),
        "chi_node": float(chi_eq[0, 0]),
        "chi_edge": float(chi_eq[1, 1]),
        "chi_geo": float(chi_eq[2, 2]),
        "chi_node_edge": float(chi_eq[0, 1]),
        "chi_node_geo": float(chi_eq[0, 2]),
        "chi_edge_geo": float(chi_eq[1, 2]),
        "principal_chi_eq": susceptibility_strength(chi_eq),
        "principal_chi": susceptibility_strength(chi_total),
        "principal_chi_null": susceptibility_strength(chi_null["chi_total"]),
        "chi_ness_fro": float(np.linalg.norm(chi_ness, ord="fro")),
        "chi_eq_fro": float(np.linalg.norm(chi_eq, ord="fro")),
        "ness_ratio": float(np.linalg.norm(chi_ness, ord="fro") / (np.linalg.norm(chi_eq, ord="fro") + 1e-12)),
        "d_info": float(np.nanmean(d_info)),
        "d_info_sd": float(np.nanstd(d_info)),
        "d_info_null": float(np.nanmean(d_info_null)),
        "mean_curvature": float(np.nanmean([mean_curvature(c) for c in curv_samples])),
        "entropy_production_proxy": float(np.mean(entropy_flux[cfg.burn_in:])),
        "n_samples": int(len(R)),
    }
