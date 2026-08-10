from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    sizes: tuple = (64, 100, 144, 196, 256)
    g_values: tuple = field(default_factory=lambda: tuple(np.round(np.linspace(0.50, 1.50, 15), 6)))
    seeds: tuple = tuple(range(5))
    steps: int = 210
    burn_in: int = 105
    sample_every: int = 15
    dt: float = 0.05
    noise_D: float = 0.18
    omega_sd: float = 0.18
    plasticity_eta: float = 0.35
    plasticity_beta: float = 5.0
    plasticity_threshold: float = 0.15
    shortcut_degree: int = 2
    drive_amp: float = 0.08
    drive_omega: float = 0.70
    binary_weight_threshold: float = 0.50
    ness_max_lag: int = 4
    spectral_q_lo: float = 0.05
    spectral_q_hi: float = 0.30


PILOT_CONFIG = ExperimentConfig()
CONFIRMATORY_SIZES = (128, 256, 512, 1024, 2048)
CONFIRMATORY_G_VALUES = tuple(np.round(np.arange(0.50, 1.5001, 0.02), 6))
CONFIRMATORY_SEEDS = tuple(range(30))
