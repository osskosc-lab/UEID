# UEID v3.6 Critical Scaling Engine v0.1 - Pilot Result

Date: 2026-08-10

## Decision

**NOT SUPPORTED IN PILOT**

The 375-run frozen computational pilot was reproduced successfully in GitHub Actions. The complex UEID v3.6 extension did not show a clear advantage over the node-only scalar susceptibility or the edge-weight-shuffled null.

## Primary numbers

- Principal FSS slope: 0.9079008792
- Principal FSS R2: 0.9837796130
- Node-only FSS R2: 0.9885513731
- Shuffled-null FSS R2: 0.9880510466
- Mean candidate NESS ratio: 0.0308506169 (gate: >= 0.05)
- corr(d_info, g): 0.2545919673
- corr(d_info, mean AF3 curvature): -0.4690568881
- g_c seed SD: 0.1934294247 (gate: <= 0.15)

## Gates

PASS:
- susceptibility peak grows with N
- formal log-log FSS R2 >= 0.80
- dynamic d_info changes with control/curvature

FAIL:
- candidate NESS correction is non-trivial
- multiplex FSS beats node-only scalar FSS
- multiplex beats the shuffled-network null
- critical point is seed-stable

## Critical audit note

For N=144, 196 and 256, the pilot peak occurs at the upper scan boundary g=1.50. Therefore the high FSS R2 cannot be interpreted as confirmatory evidence for a well-localized critical point. A post-hoc range-extension diagnostic (g=0.5..2.5, 252 runs) still showed one of three sizes peaking at the upper boundary and worse g_c stability (SD about 0.547).

## Consequence

Do **not** launch the frozen 7650-run confirmatory grid as-is. First add an interior-peak gate, geometry causal ablations, stronger nulls, and direct-response validation for the candidate NESS term.
