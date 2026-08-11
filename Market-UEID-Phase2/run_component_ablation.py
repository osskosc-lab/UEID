from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_PATH = ROOT / "Market-UEID" / "run_market_ueid.py"
OUT = ROOT / "results" / "market_ueid_v02_component_ablation"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("market_ueid_v01", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

BOOT_B = 1000
BOOT_BLOCK = 12
CI_ALPHA = 0.05 / 4.0
ALERT_BUDGET = 0.05
RNG = np.random.default_rng(20260811)


def feature_map(d: pd.DataFrame) -> dict[str, list[str]]:
    vol = ["rv20"] + (["vix"] if d["vix"].notna().mean() >= 0.20 else [])
    node = vol + ["r_node"]
    return {
        "vol_only": vol,
        "node_only": node,
        "node_plus_edge": node + ["r_edge"],
        "node_plus_af3": node + ["r_geo"],
        "node_plus_dinfo": node + ["d_info"],
        "node_plus_chi": node + ["principal_chi"],
        "node_plus_edge_null": node + ["r_edge_null"],
        "node_plus_af3_null": node + ["r_geo_null"],
        "node_plus_dinfo_null": node + ["d_info_null"],
        "node_plus_chi_null": node + ["principal_chi_null"],
        "full_ueid_reference": node + ["r_edge", "r_geo", "d_info", "principal_chi"],
    }


def calibrated_metrics(y_val: pd.Series, p_val: pd.Series, y_test: pd.Series, p_test: pd.Series) -> dict:
    yv = y_val.loc[p_val.index].astype(int)
    yt = y_test.loc[p_test.index].astype(int)
    threshold = float(np.nanquantile(p_val, 1.0 - ALERT_BUDGET))
    alert = p_test >= threshold
    tp = int(((yt == 1) & alert).sum())
    fp = int(((yt == 0) & alert).sum())
    fn = int(((yt == 1) & (~alert)).sum())
    tn = int(((yt == 0) & (~alert)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    years = max((p_test.index.max() - p_test.index.min()).days / 365.25, 1e-9)
    return {
        "auc": float(roc_auc_score(yt, p_test)) if yt.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(yt, p_test)) if yt.nunique() == 2 else np.nan,
        "brier": float(brier_score_loss(yt, p_test)),
        "validation_threshold": threshold,
        "test_alert_rate": float(alert.mean()),
        "recall_fixed_alert": float(recall),
        "precision_fixed_alert": float(precision),
        "false_alerts_per_year": float(fp / years),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": int(len(yt)), "positives": int(yt.sum()), "event_rate": float(yt.mean()),
    }


def bootstrap_auc_delta(y: pd.Series, p1: pd.Series, p0: pd.Series) -> dict:
    idx = y.index.intersection(p1.index).intersection(p0.index)
    yy = y.loc[idx].astype(int).to_numpy()
    a = p1.loc[idx].to_numpy(dtype=float)
    b = p0.loc[idx].to_numpy(dtype=float)
    n = len(idx)
    deltas = []
    for _ in range(BOOT_B):
        take = []
        while len(take) < n:
            s = int(RNG.integers(0, max(1, n - BOOT_BLOCK + 1)))
            take.extend(range(s, min(s + BOOT_BLOCK, n)))
        take = np.asarray(take[:n])
        if np.unique(yy[take]).size < 2:
            continue
        deltas.append(roc_auc_score(yy[take], a[take]) - roc_auc_score(yy[take], b[take]))
    arr = np.asarray(deltas, dtype=float)
    qlo = CI_ALPHA / 2.0
    qhi = 1.0 - CI_ALPHA / 2.0
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, qlo)),
        "ci_high": float(np.quantile(arr, qhi)),
        "ci_level": float(1.0 - CI_ALPHA),
        "replicates": int(len(arr)),
    }


def era_auc_deltas(d: pd.DataFrame, p1: pd.Series, p0: pd.Series) -> dict:
    eras = {
        "2015-2019": ("2015-01-01", "2019-12-31"),
        "2020-2022": ("2020-01-01", "2022-12-31"),
        "2023-latest": ("2023-01-01", "2099-12-31"),
    }
    out = {}
    for name, (a, b) in eras.items():
        idx = d.loc[a:b].index.intersection(p1.index).intersection(p0.index)
        y = d.loc[idx, "crisis"].astype(int)
        if len(idx) >= 20 and y.nunique() == 2:
            delta = roc_auc_score(y, p1.loc[idx]) - roc_auc_score(y, p0.loc[idx])
            out[name] = {"delta_auc": float(delta), "n": int(len(idx)), "positives": int(y.sum())}
        else:
            out[name] = {"delta_auc": None, "n": int(len(idx)), "positives": int(y.sum()) if len(idx) else 0}
    return out


def make_plot(candidate_rows: list[dict]) -> Path:
    names = [r["candidate"] for r in candidate_rows]
    vals = [r["delta_auc_vs_node"] for r in candidate_rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(names, vals)
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(0.02, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Test ROC-AUC delta vs node_only")
    ax.set_title("Market-UEID v0.2 component-wise ablation")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = OUT / "component_auc_deltas.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def make_pdf(summary: dict, metrics_df: pd.DataFrame, candidate_df: pd.DataFrame, plot_path: Path) -> Path:
    path = OUT / "Market_UEID_v02_Component_Ablation_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Market-UEID v0.2 Component Ablation", styles["Title"]), Spacer(1, 4*mm)]
    story.append(Paragraph(
        f"Run UTC: {summary['run_utc']}<br/>Decision: <b>{summary['decision']}</b><br/>"
        f"Purpose: test each surviving UEID diagnostic independently beyond the node-only baseline.",
        styles["BodyText"],
    ))
    story += [Spacer(1, 4*mm), Paragraph("Frozen test metrics", styles["Heading2"])]
    cols = ["model", "auc", "pr_auc", "brier", "recall_fixed_alert", "precision_fixed_alert", "test_alert_rate"]
    disp = metrics_df[cols].copy()
    for c in cols[1:]:
        disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    data = [disp.columns.tolist()] + disp.values.tolist()
    tbl = Table(data, repeatRows=1, colWidths=[38*mm, 18*mm, 18*mm, 18*mm, 24*mm, 25*mm, 22*mm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.6),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [tbl, Spacer(1, 4*mm), Paragraph("Candidate survival audit", styles["Heading2"])]
    ccols = ["candidate","delta_auc_vs_node","familywise_ci_low","delta_auc_vs_null","positive_eras","survives"]
    cdisp = candidate_df[ccols].copy()
    for c in ["delta_auc_vs_node","familywise_ci_low","delta_auc_vs_null"]:
        cdisp[c] = cdisp[c].map(lambda x: f"{x:.4f}")
    cdata = [cdisp.columns.tolist()] + cdisp.values.tolist()
    ct = Table(cdata, repeatRows=1, colWidths=[31*mm, 27*mm, 27*mm, 26*mm, 24*mm, 20*mm])
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.6),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [ct, Spacer(1, 4*mm), Image(str(plot_path), width=175*mm, height=98*mm), Spacer(1, 4*mm)]
    story.append(Paragraph("Interpretation", styles["Heading2"]))
    story.append(Paragraph(summary["interpretation"], styles["BodyText"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Method note: alert thresholds are calibrated only on the 2005-2014 validation period and then frozen for 2015-latest. "
        "The four candidate AUC deltas use Bonferroni familywise 98.75% moving-block bootstrap confidence intervals. "
        "This is a predictive falsification study, not a trading strategy and not evidence of a literal physical phase transition.",
        styles["BodyText"],
    ))
    doc.build(story)
    return path


def main() -> None:
    with open(HERE / "preregistration.json", "r", encoding="utf-8") as f:
        prereg = json.load(f)

    ind = base.fetch_ff30()
    spx, vix, market_source = base.fetch_market()
    common_end = min(ind.index.max(), spx.index.max())
    ind = ind.loc[ind.index <= common_end]
    spx = spx.loc[spx.index <= common_end]
    vix = vix.loc[vix.index <= common_end]

    net = base.build_network_features(ind)
    d = base.prepare_dataset(net, spx, vix)
    d.to_csv(OUT / "market_ueid_v02_features.csv", index_label="date")

    fmap = feature_map(d)
    preds_val: dict[str, pd.Series] = {}
    preds_test: dict[str, pd.Series] = {}
    metric_rows = []
    for name, cols in fmap.items():
        pv, pt, _ = base.fit_predict(d, cols)
        preds_val[name] = pv
        preds_test[name] = pt
        m = calibrated_metrics(d["crisis"], pv, d["crisis"], pt)
        metric_rows.append({"model": name, **m})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)
    mm = metrics_df.set_index("model")

    candidate_map = {
        "r_edge": ("node_plus_edge", "node_plus_edge_null"),
        "r_geo_AF3": ("node_plus_af3", "node_plus_af3_null"),
        "d_info": ("node_plus_dinfo", "node_plus_dinfo_null"),
        "principal_chi": ("node_plus_chi", "node_plus_chi_null"),
    }
    test_idx = preds_test["node_only"].index
    y_test = d.loc[test_idx, "crisis"].astype(int)
    candidate_rows = []
    detail = {}
    for candidate, (model, null_model) in candidate_map.items():
        delta_node = float(mm.loc[model, "auc"] - mm.loc["node_only", "auc"])
        delta_null = float(mm.loc[model, "auc"] - mm.loc[null_model, "auc"])
        boot = bootstrap_auc_delta(y_test, preds_test[model], preds_test["node_only"])
        eras = era_auc_deltas(d, preds_test[model], preds_test["node_only"])
        positive_eras = sum(1 for x in eras.values() if x["delta_auc"] is not None and x["delta_auc"] > 0)
        survives = bool(delta_node >= 0.02 and boot["ci_low"] > 0 and delta_null >= 0.01 and positive_eras >= 2)
        row = {
            "candidate": candidate,
            "model": model,
            "null_model": null_model,
            "delta_auc_vs_node": delta_node,
            "familywise_ci_low": boot["ci_low"],
            "familywise_ci_high": boot["ci_high"],
            "bootstrap_mean": boot["mean"],
            "delta_auc_vs_null": delta_null,
            "positive_eras": positive_eras,
            "survives": survives,
        }
        candidate_rows.append(row)
        detail[candidate] = {**row, "bootstrap": boot, "eras": eras}

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(OUT / "candidate_survival.csv", index=False)
    survivors = candidate_df.loc[candidate_df["survives"], "candidate"].tolist()
    decision = "SUPPORTED COMPONENT" if survivors else "NO INCREMENTAL UEID COMPONENT SURVIVES"
    if survivors:
        interpretation = (
            "At least one single UEID-derived diagnostic met every preregistered survival gate beyond the node-only baseline: "
            + ", ".join(survivors)
            + ". The next experiment should externally replicate only these frozen survivors, without reintroducing failed components."
        )
    else:
        interpretation = (
            "None of the four UEID-derived additions met all preregistered incremental-AUC, familywise bootstrap, matched-null, "
            "and era-stability gates. The market application should therefore retain the simpler node-only synchronization baseline "
            "and not escalate UEID complexity without a new mechanistic hypothesis."
        )

    g0 = bool(ind.shape[1] >= 25 and ind.index.min() <= pd.Timestamp("1991-01-01") and len(d) >= 1000)
    summary = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        "study_id": prereg["study_id"],
        "decision": decision,
        "survivors": survivors,
        "market_source": market_source,
        "data_gate": {"pass": g0, "industry_count": int(ind.shape[1]), "feature_rows": int(len(d)), "test_start": str(test_idx.min().date()), "test_end": str(test_idx.max().date())},
        "node_only_auc": float(mm.loc["node_only", "auc"]),
        "vol_only_auc": float(mm.loc["vol_only", "auc"]),
        "full_ueid_reference_auc": float(mm.loc["full_ueid_reference", "auc"]),
        "candidates": detail,
        "interpretation": interpretation,
    }
    with open(OUT / "decision.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({k: v for k, v in preds_test.items()})
    pred_df["crisis"] = d["crisis"]
    pred_df.to_csv(OUT / "test_predictions.csv", index_label="date")

    plot_path = make_plot(candidate_rows)
    pdf_path = make_pdf(summary, metrics_df, candidate_df, plot_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
