import argparse
import json
from pathlib import Path
import pandas as pd

from config import PILOT_CONFIG
from diagnostics import peak_location, seed_gc_stability
from fss import fit_fss
from simulate import simulate_one


def _aggregate(df):
    return df.groupby(["N", "control_parameter"], as_index=False).agg(
        principal_chi=("principal_chi", "mean"),
        principal_chi_sd=("principal_chi", "std"),
        principal_chi_eq=("principal_chi_eq", "mean"),
        principal_chi_null=("principal_chi_null", "mean"),
        chi_node=("chi_node", "mean"),
        d_info=("d_info", "mean"),
        d_info_null=("d_info_null", "mean"),
        mean_curvature=("mean_curvature", "mean"),
        ness_ratio=("ness_ratio", "mean"),
        entropy_production_proxy=("entropy_production_proxy", "mean"),
    )


def summarize(df):
    agg = _aggregate(df)
    peak_rows = []
    for N, d in agg.groupby("N"):
        peak_rows.append(d.loc[d["principal_chi"].idxmax()])
    peaks = pd.DataFrame(peak_rows)
    fss = fit_fss(peaks["N"].to_numpy(), peaks["principal_chi"].to_numpy())
    fss_scalar = fit_fss(peaks["N"].to_numpy(), [agg[agg.N.eq(N)]["chi_node"].max() for N in peaks["N"]])
    fss_null = fit_fss(peaks["N"].to_numpy(), [agg[agg.N.eq(N)]["principal_chi_null"].max() for N in peaks["N"]])
    seed_gcs = []
    for (N, seed), d in df.groupby(["N", "seed"]):
        seed_gcs.append(peak_location(d["control_parameter"], d["principal_chi"]))
    gc_stability = seed_gc_stability(seed_gcs)
    corr_kappa_d = float(df[["mean_curvature", "d_info"]].corr().iloc[0, 1])
    corr_g_d = float(df[["control_parameter", "d_info"]].corr().iloc[0, 1])
    mean_ness_ratio = float(df["ness_ratio"].mean())
    gates = {
        "chi_peak_grows_with_N": bool(fss["slope"] > 0.05),
        "loglog_fss_r2_ge_0_80": bool(fss["r2"] >= 0.80),
        "dynamic_dimension_changes": bool(max(abs(corr_kappa_d), abs(corr_g_d)) >= 0.20),
        "candidate_ness_nontrivial": bool(mean_ness_ratio >= 0.05),
        "multiplex_beats_scalar_fss": bool(fss["r2"] >= fss_scalar["r2"] + 0.02),
        "beats_shuffled_network_null": bool(fss["r2"] >= fss_null["r2"] + 0.02),
        "critical_point_seed_stable": bool(gc_stability["sd"] <= 0.15),
    }
    failed = [k for k, v in gates.items() if not v]
    return {
        "n_runs": int(len(df)), "fss_principal": fss, "fss_scalar": fss_scalar, "fss_null": fss_null,
        "gc_seed_stability": gc_stability, "corr_mean_curvature_d_info": corr_kappa_d,
        "corr_g_d_info": corr_g_d, "mean_ness_ratio": mean_ness_ratio, "gates": gates,
        "survived": [k for k, v in gates.items() if v], "failed": failed,
        "overall": "conditional_survival" if len(failed) <= 2 else "not_supported_in_pilot"
    }, agg, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = PILOT_CONFIG
    rows = []
    total = len(cfg.sizes) * len(cfg.g_values) * len(cfg.seeds)
    k = 0
    for N in cfg.sizes:
        for g in cfg.g_values:
            for seed in cfg.seeds:
                k += 1
                rows.append(simulate_one(N, float(g), int(seed), cfg))
                if k % 25 == 0 or k == total:
                    print(f"completed {k}/{total}", flush=True)
    df = pd.DataFrame(rows)
    summary, agg, peaks = summarize(df)
    df.to_csv(out / "ueid_v36_pilot_runs.csv", index=False)
    agg.to_csv(out / "ueid_v36_pilot_aggregate.csv", index=False)
    peaks.to_csv(out / "ueid_v36_pilot_peaks.csv", index=False)
    (out / "ueid_v36_pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
