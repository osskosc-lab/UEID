from __future__ import annotations

import io
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / "market_ueid_v01"
OUT.mkdir(parents=True, exist_ok=True)

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_Daily_CSV.zip"
START = pd.Timestamp("1990-01-01")
TRAIN_END = pd.Timestamp("2004-12-31")
VAL_END = pd.Timestamp("2014-12-31")
WINDOW = 126
STEP = 5
CHI_HISTORY = 26
CRISIS_H = 21
CRISIS_DD = -0.08
ALERT_BUDGET = 0.05
RNG = np.random.default_rng(20260810)


def fetch_ff30() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 Market-UEID research pilot"}
    r = requests.get(FF_URL, headers=headers, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        text = zf.read(name).decode("latin1")
    lines = text.splitlines()
    header_i = None
    for i, line in enumerate(lines[:-1]):
        if line.lstrip().startswith(",") and re.match(r"^\s*\d{8}\s*,", lines[i + 1]):
            header_i = i
            break
    if header_i is None:
        raise RuntimeError("Could not locate daily industry data table in Kenneth French CSV")
    header = [x.strip() for x in lines[header_i].split(",")][1:]
    rows = []
    dates = []
    for line in lines[header_i + 1 :]:
        cells = [x.strip() for x in line.split(",")]
        if not cells or not re.fullmatch(r"\d{8}", cells[0]):
            if rows:
                break
            continue
        vals = []
        for x in cells[1 : 1 + len(header)]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
        dates.append(pd.to_datetime(cells[0], format="%Y%m%d"))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=header, dtype=float)
    df = df.replace([-99.99, -999.0, -999.99], np.nan) / 100.0
    df = df.loc[df.index >= START].sort_index()
    keep = [c for c in df.columns if df[c].notna().mean() >= 0.95]
    return df[keep]


def fetch_market() -> tuple[pd.Series, pd.Series, str]:
    try:
        raw = yf.download(["^GSPC", "^VIX"], start="1989-01-01", progress=False, auto_adjust=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
            spx = close["^GSPC"].dropna().rename("spx")
            vix = close["^VIX"].dropna().rename("vix")
        else:
            raise RuntimeError("unexpected yfinance response shape")
        spx.index = pd.to_datetime(spx.index).tz_localize(None)
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        if len(spx) < 1000:
            raise RuntimeError("insufficient S&P500 history")
        return spx, vix, "Yahoo Finance ^GSPC/^VIX"
    except Exception as e:
        print("Yahoo download failed; using Stooq S&P500 fallback and no VIX:", repr(e))
        end = datetime.utcnow().strftime("%Y%m%d")
        url = f"https://stooq.com/q/d/l/?s=%5Espx&i=d&d1=19890101&d2={end}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df or "Close" not in df:
            raise RuntimeError("Stooq fallback did not return S&P500 data")
        spx = pd.Series(df["Close"].astype(float).values, index=pd.to_datetime(df["Date"]), name="spx").sort_index()
        vix = pd.Series(index=spx.index, dtype=float, name="vix")
        return spx, vix, "Stooq ^SPX fallback; VIX unavailable"


def graph_from_corr(corr: np.ndarray, q: float = 0.70) -> nx.Graph:
    n = corr.shape[0]
    vals = np.abs(corr[np.triu_indices(n, 1)])
    thr = float(np.nanquantile(vals, q))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(float(corr[i, j]))
            if np.isfinite(w) and w >= thr:
                g.add_edge(i, j, weight=max(w, 1e-8))
    return g


def weighted_adjacency(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    a = np.zeros((n, n), dtype=float)
    for u, v, d in g.edges(data=True):
        w = float(d.get("weight", 1.0))
        a[u, v] = a[v, u] = w
    return a


def edge_concentration(g: nx.Graph) -> float:
    a = weighted_adjacency(g)
    eig = np.linalg.eigvalsh(a)
    den = np.sum(np.abs(eig)) + 1e-12
    return float(np.max(eig) / den)


def af3_geometry_score(g: nx.Graph) -> float:
    vals = []
    deg = dict(g.degree())
    for u, v in g.edges():
        triangles = len(set(g.neighbors(u)).intersection(g.neighbors(v)))
        f = 4.0 - deg[u] - deg[v]
        vals.append(f + 3.0 * triangles)
    if not vals:
        return np.nan
    arr = np.asarray(vals, dtype=float)
    m, s = float(arr.mean()), float(arr.std(ddof=0))
    return abs(m) / (abs(m) + s + 1e-12)


def spectral_dimension(g: nx.Graph) -> tuple[float, float, int]:
    a = weighted_adjacency(g)
    d = np.diag(a.sum(axis=1))
    lap = d - a
    eig = np.linalg.eigvalsh(lap)
    pos = np.sort(eig[eig > 1e-8])
    comps = nx.number_connected_components(g)
    if len(pos) < 6:
        return np.nan, np.nan, comps
    m = min(max(6, len(pos) // 3), len(pos))
    x = np.log(pos[:m])
    y = np.log(np.arange(1, m + 1, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(2.0 * slope), float(r2), comps


def rewired_null(g: nx.Graph, seed: int) -> nx.Graph:
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(g.edges())
    m = h.number_of_edges()
    if m >= 6:
        try:
            nx.double_edge_swap(h, nswap=max(1, min(5 * m, 500)), max_tries=max(100, 50 * m), seed=seed)
        except Exception:
            pass
    weights = [float(d.get("weight", 1.0)) for _, _, d in g.edges(data=True)]
    rr = np.random.default_rng(seed)
    rr.shuffle(weights)
    for (u, v), w in zip(h.edges(), weights):
        h[u][v]["weight"] = w
    return h


def build_network_features(ind: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = ind.index
    for k, i in enumerate(range(WINDOW - 1, len(ind), STEP)):
        w = ind.iloc[i - WINDOW + 1 : i + 1]
        corr = w.corr().values
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        upper = corr[np.triu_indices_from(corr, 1)]
        r_node = float(np.nanmean(upper))
        g = graph_from_corr(corr)
        r_edge = edge_concentration(g)
        r_geo = af3_geometry_score(g)
        d_info, d_info_r2, comps = spectral_dimension(g)
        ng = rewired_null(g, 20260810 + k)
        r_edge_n = edge_concentration(ng)
        r_geo_n = af3_geometry_score(ng)
        d_info_n, d_info_r2_n, comps_n = spectral_dimension(ng)
        rows.append({
            "date": idx[i], "r_node": r_node, "r_edge": r_edge, "r_geo": r_geo,
            "d_info": d_info, "d_info_fit_r2": d_info_r2, "components": comps,
            "r_edge_null": r_edge_n, "r_geo_null": r_geo_n, "d_info_null": d_info_n,
            "d_info_null_fit_r2": d_info_r2_n, "components_null": comps_n,
        })
    f = pd.DataFrame(rows).set_index("date")
    f["principal_chi"] = rolling_principal_chi(f[["r_node", "r_edge", "r_geo"]], ind.shape[1])
    f["principal_chi_null"] = rolling_principal_chi(f[["r_node", "r_edge_null", "r_geo_null"]], ind.shape[1])
    return f


def rolling_principal_chi(x: pd.DataFrame, n_nodes: int) -> pd.Series:
    vals = np.full(len(x), np.nan)
    arr = x.to_numpy(dtype=float)
    for i in range(CHI_HISTORY - 1, len(x)):
        z = arr[i - CHI_HISTORY + 1 : i + 1]
        if not np.isfinite(z).all():
            continue
        cov = np.cov(z, rowvar=False, ddof=1) * n_nodes
        vals[i] = float(np.linalg.eigvalsh(cov).max())
    return pd.Series(vals, index=x.index)


def forward_crisis_labels(close: pd.Series) -> pd.Series:
    a = close.to_numpy(dtype=float)
    y = np.full(len(a), np.nan)
    for i in range(len(a) - CRISIS_H):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        future = a[i + 1 : i + CRISIS_H + 1]
        if np.isfinite(future).sum() < CRISIS_H // 2:
            continue
        dd = np.nanmin(future) / a[i] - 1.0
        y[i] = float(dd <= CRISIS_DD)
    return pd.Series(y, index=close.index, name="crisis")


def prepare_dataset(net: pd.DataFrame, spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
    mret = spx.pct_change()
    rv20 = mret.rolling(20).std() * math.sqrt(252.0)
    labels = forward_crisis_labels(spx)
    d = net.copy()
    d["spx"] = spx.reindex(d.index, method="ffill")
    d["rv20"] = rv20.reindex(d.index, method="ffill")
    d["vix"] = vix.reindex(d.index, method="ffill")
    d["crisis"] = labels.reindex(d.index, method="ffill")
    d = d.loc[d.index >= START]
    d = d[d["crisis"].notna()]
    return d


def model_feature_map(d: pd.DataFrame) -> dict[str, list[str]]:
    base_v = ["rv20"] + (["vix"] if d["vix"].notna().mean() >= 0.20 else [])
    return {
        "vol_only": base_v,
        "node_only": base_v + ["r_node"],
        "ueid_lite": base_v + ["r_node", "r_edge", "r_geo", "d_info", "principal_chi"],
        "ueid_null": base_v + ["r_node", "r_edge_null", "r_geo_null", "d_info_null", "principal_chi_null"],
    }


def fit_predict(d: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = d.index <= TRAIN_END
    val = (d.index > TRAIN_END) & (d.index <= VAL_END)
    test = d.index > VAL_END
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")),
    ])
    pipe.fit(d.loc[train, cols], d.loc[train, "crisis"].astype(int))
    p_val = pd.Series(pipe.predict_proba(d.loc[val, cols])[:, 1], index=d.index[val])
    p_test = pd.Series(pipe.predict_proba(d.loc[test, cols])[:, 1], index=d.index[test])
    p_all = pd.Series(pipe.predict_proba(d[cols])[:, 1], index=d.index)
    return p_val, p_test, p_all


def metrics(y: pd.Series, p: pd.Series) -> dict:
    y = y.loc[p.index].astype(int)
    auc = roc_auc_score(y, p) if y.nunique() == 2 else np.nan
    ap = average_precision_score(y, p) if y.nunique() == 2 else np.nan
    brier = brier_score_loss(y, p)
    thr = float(np.nanquantile(p, 1.0 - ALERT_BUDGET))
    alert = p >= thr
    tp = int(((y == 1) & alert).sum())
    fp = int(((y == 0) & alert).sum())
    fn = int(((y == 1) & (~alert)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    years = max((p.index.max() - p.index.min()).days / 365.25, 1e-9)
    return {
        "auc": float(auc), "pr_auc": float(ap), "brier": float(brier),
        "recall_at_5pct": float(recall), "precision_at_5pct": float(precision),
        "false_alerts_per_year": float(fp / years), "threshold": thr,
        "n": int(len(y)), "positives": int(y.sum()), "event_rate": float(y.mean()),
    }


def block_bootstrap_auc_delta(y: pd.Series, p1: pd.Series, p0: pd.Series, B: int = 300, block: int = 12) -> tuple[float, float, float]:
    common = y.index.intersection(p1.index).intersection(p0.index)
    yy = y.loc[common].to_numpy(dtype=int)
    a = p1.loc[common].to_numpy(dtype=float)
    b = p0.loc[common].to_numpy(dtype=float)
    n = len(common)
    deltas = []
    for _ in range(B):
        inds = []
        while len(inds) < n:
            s = int(RNG.integers(0, max(1, n - block + 1)))
            inds.extend(range(s, min(s + block, n)))
        inds = np.asarray(inds[:n])
        if np.unique(yy[inds]).size < 2:
            continue
        deltas.append(roc_auc_score(yy[inds], a[inds]) - roc_auc_score(yy[inds], b[inds]))
    arr = np.asarray(deltas)
    return float(np.nanmean(arr)), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))


def era_deltas(d: pd.DataFrame, preds: dict[str, pd.Series]) -> dict:
    eras = {
        "2015-2019": ("2015-01-01", "2019-12-31"),
        "2020-2022": ("2020-01-01", "2022-12-31"),
        "2023-latest": ("2023-01-01", "2099-12-31"),
    }
    out = {}
    for name, (a, b) in eras.items():
        idx = d.loc[a:b].index.intersection(preds["ueid_lite"].index).intersection(preds["node_only"].index)
        y = d.loc[idx, "crisis"].astype(int)
        if len(idx) >= 20 and y.nunique() == 2:
            au = roc_auc_score(y, preds["ueid_lite"].loc[idx])
            an = roc_auc_score(y, preds["node_only"].loc[idx])
            out[name] = {"ueid_auc": float(au), "node_auc": float(an), "delta": float(au - an), "n": int(len(idx)), "positives": int(y.sum())}
        else:
            out[name] = {"ueid_auc": None, "node_auc": None, "delta": None, "n": int(len(idx)), "positives": int(y.sum()) if len(idx) else 0}
    return out


def standardized_diff(x1: pd.Series, x0: pd.Series) -> float:
    a = x1.dropna().to_numpy(dtype=float)
    b = x0.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / max(len(a)+len(b)-2, 1))
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def save_plot(d: pd.DataFrame, p: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(p.index, p.values, linewidth=1.0, label="UEID-lite predicted crisis risk")
    crisis = d.loc[p.index, "crisis"].astype(int)
    ax.fill_between(p.index, 0, 1, where=crisis.values.astype(bool), alpha=0.12, transform=ax.get_xaxis_transform(), label="forward 21d drawdown <= -8%")
    ax.set_ylabel("predicted probability")
    ax.set_title("Market-UEID v0.1 out-of-sample test risk")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUT / "market_ueid_v01_test_risk.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_pdf(summary: dict, metrics_table: pd.DataFrame, plot_path: Path) -> Path:
    path = OUT / "Market_UEID_v01_Result_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Market-UEID v0.1 Falsification Pilot", styles["Title"]), Spacer(1, 5*mm)]
    story.append(Paragraph(f"Run date: {summary['run_utc']}<br/>Decision: <b>{summary['decision']}</b><br/>Market data: {summary['market_source']}<br/>Network: Kenneth French 30 Industry Portfolios Daily", styles["BodyText"]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Test metrics (2015-latest)", styles["Heading2"]))
    disp = metrics_table[["model","auc","pr_auc","brier","recall_at_5pct","precision_at_5pct","false_alerts_per_year"]].copy()
    for c in disp.columns[1:]: disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    data = [disp.columns.tolist()] + disp.values.tolist()
    tbl = Table(data, repeatRows=1, colWidths=[33*mm, 20*mm, 20*mm, 20*mm, 23*mm, 24*mm, 27*mm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [tbl, Spacer(1, 4*mm)]
    story.append(Paragraph("Preregistered gates", styles["Heading2"]))
    gate_data = [["Gate","Pass","Observed"]] + [[k, str(v["pass"]), str(v["observed"])] for k,v in summary["gates"].items()]
    gt = Table(gate_data, repeatRows=1, colWidths=[38*mm, 20*mm, 115*mm])
    gt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [gt, Spacer(1, 4*mm)]
    story.append(Paragraph("Interpretation", styles["Heading2"]))
    story.append(Paragraph(summary["interpretation"], styles["BodyText"]))
    story += [Spacer(1, 4*mm), Image(str(plot_path), width=180*mm, height=74*mm)]
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Caveats: This is a historical market falsification pilot, not a trading strategy or proof of a physical market phase transition. NESS was intentionally excluded after the synthetic UEID v3.6 pilot failed its NESS gate. Industry portfolios reduce, but do not eliminate, data-design bias.", styles["BodyText"]))
    doc.build(story)
    return path


def main():
    ind = fetch_ff30()
    spx, vix, market_source = fetch_market()
    common_end = min(ind.index.max(), spx.index.max())
    ind = ind.loc[ind.index <= common_end]
    spx = spx.loc[spx.index <= common_end]
    vix = vix.loc[vix.index <= common_end]

    net = build_network_features(ind)
    d = prepare_dataset(net, spx, vix)
    d.to_csv(OUT / "market_ueid_features.csv", index_label="date")

    fmap = model_feature_map(d)
    preds_val, preds_test, preds_all = {}, {}, {}
    rows = []
    for name, cols in fmap.items():
        pv, pt, pa = fit_predict(d, cols)
        preds_val[name], preds_test[name], preds_all[name] = pv, pt, pa
        mval = metrics(d["crisis"], pv)
        mtest = metrics(d["crisis"], pt)
        row = {"model": name, **mtest}
        row.update({f"val_{k}": v for k, v in mval.items()})
        rows.append(row)
    mt = pd.DataFrame(rows)
    mt.to_csv(OUT / "market_ueid_model_metrics.csv", index=False)

    test_y = d.loc[preds_test["ueid_lite"].index, "crisis"].astype(int)
    mu = mt.set_index("model")
    auc_u = float(mu.loc["ueid_lite", "auc"])
    auc_n = float(mu.loc["node_only", "auc"])
    auc_null = float(mu.loc["ueid_null", "auc"])
    rec_u = float(mu.loc["ueid_lite", "recall_at_5pct"])
    rec_n = float(mu.loc["node_only", "recall_at_5pct"])
    d_effect = standardized_diff(d.loc[d["crisis"] == 1, "d_info"], d.loc[d["crisis"] == 0, "d_info"])
    eras = era_deltas(d, preds_test)
    positive_eras = sum(1 for e in eras.values() if e["delta"] is not None and e["delta"] > 0)
    boot_mean, boot_lo, boot_hi = block_bootstrap_auc_delta(test_y, preds_test["ueid_lite"], preds_test["node_only"])

    g0 = ind.shape[1] >= 25 and ind.index.min() <= pd.Timestamp("1991-01-01") and len(d) >= 1000
    gates = {
        "G0_data": {"pass": bool(g0), "observed": f"n_industries={ind.shape[1]}, first={ind.index.min().date()}, feature_rows={len(d)}"},
        "G1_incremental_auc": {"pass": bool(auc_u - auc_n >= 0.02), "observed": f"delta_auc={auc_u-auc_n:.4f}; block-bootstrap mean={boot_mean:.4f}, 95% CI [{boot_lo:.4f},{boot_hi:.4f}]"},
        "G2_null_exclusion": {"pass": bool(auc_u - auc_null >= 0.02), "observed": f"delta_auc_vs_null={auc_u-auc_null:.4f}"},
        "G3_dimension_signal": {"pass": bool(np.isfinite(d_effect) and d_effect <= -0.20), "observed": f"standardized d_info difference={d_effect:.4f}"},
        "G4_alert_recall": {"pass": bool(rec_u - rec_n >= 0.10), "observed": f"recall delta at 5% budget={rec_u-rec_n:.4f}"},
        "G5_era_stability": {"pass": bool(positive_eras >= 2), "observed": f"positive UEID-vs-node AUC eras={positive_eras}/3; details={eras}"},
    }
    supported = gates["G0_data"]["pass"] and gates["G1_incremental_auc"]["pass"] and gates["G2_null_exclusion"]["pass"] and gates["G4_alert_recall"]["pass"]
    decision = "SUPPORTED IN PILOT" if supported else "NOT SUPPORTED IN PILOT"
    if supported:
        interp = "The preregistered incremental-prediction, null-exclusion, and fixed-alert-budget gates all passed. UEID-lite therefore survives this pilot as an incremental market-structure signal. This does not establish a physical phase transition or trading profitability."
    else:
        failed = ", ".join(k for k,v in gates.items() if not v["pass"])
        interp = f"The full preregistered survival rule was not met. Failed gates: {failed}. The correct conclusion is to retain only any individually surviving diagnostics and revise/ablate before a larger confirmatory experiment."

    summary = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        "decision": decision,
        "market_source": market_source,
        "data": {"industry_count": int(ind.shape[1]), "industry_start": str(ind.index.min().date()), "industry_end": str(ind.index.max().date()), "spx_start": str(spx.index.min().date()), "spx_end": str(spx.index.max().date()), "feature_rows": int(len(d)), "test_start": str(d.loc[d.index > VAL_END].index.min().date()), "test_end": str(d.loc[d.index > VAL_END].index.max().date())},
        "gates": gates,
        "era_deltas": eras,
        "bootstrap_auc_delta_ueid_minus_node": {"mean": boot_mean, "ci95_low": boot_lo, "ci95_high": boot_hi},
        "d_info_effect": d_effect,
        "interpretation": interp,
    }
    with open(OUT / "market_ueid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    predout = pd.DataFrame({name: s for name,s in preds_test.items()})
    predout["crisis"] = d["crisis"]
    predout.to_csv(OUT / "market_ueid_test_predictions.csv", index_label="date")
    plot_path = save_plot(d, preds_test["ueid_lite"])
    pdf = make_pdf(summary, mt, plot_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PDF:", pdf)


if __name__ == "__main__":
    main()
