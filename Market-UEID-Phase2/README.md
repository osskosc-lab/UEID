# Market-UEID v0.2 Component Ablation

This phase follows the v0.1 pilot result where node-only outperformed the full UEID-lite bundle.

The experiment freezes four one-at-a-time additions to the node-only baseline: edge concentration, AF3 geometry, spectral information dimension, and principal susceptibility. Each candidate is compared with both node-only and its matched rewired/shuffled null where available. The primary endpoint is frozen 2015-latest ROC-AUC with a Bonferroni familywise moving-block bootstrap gate. Alert thresholds are calibrated on validation only.
