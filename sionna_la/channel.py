"""
Time-correlated effective SINR process for downlink LA experiments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_sinr_db_trace(
    num_slots: int,
    mean_db: float = 8.0,
    rho: float = 0.95,
    innov_std_db: float = 1.2,
    sinr_min_db: float = -5.0,
    sinr_max_db: float = 25.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    AR(1) in dB:
        x_{t+1} = mean + rho (x_t - mean) + eps_t,  eps ~ N(0, innov_std^2)

    Returns shape [num_slots] float64 SINR [dB].
    """
    if not (0.0 <= abs(rho) < 1.0):
        raise ValueError("rho must satisfy |rho| < 1")
    rng = np.random.default_rng(seed)
    x = np.empty(num_slots, dtype=np.float64)
    x[0] = mean_db + innov_std_db * rng.standard_normal()
    for t in range(num_slots - 1):
        x[t + 1] = mean_db + rho * (x[t] - mean_db) + innov_std_db * rng.standard_normal()
    return np.clip(x, sinr_min_db, sinr_max_db)


def add_cqi_noise(
    sinr_true_db: np.ndarray,
    noise_std_db: float = 1.5,
    delay_slots: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Noisy / delayed CQI that the LA algorithm observes.

    delay_slots=d means feedback at t uses true SINR at t-d (plus noise).
    """
    rng = np.random.default_rng(seed)
    if delay_slots < 0:
        raise ValueError("delay_slots must be >= 0")
    if delay_slots == 0:
        delayed = sinr_true_db
    else:
        delayed = np.empty_like(sinr_true_db)
        delayed[:delay_slots] = sinr_true_db[0]
        delayed[delay_slots:] = sinr_true_db[:-delay_slots]
    noise = noise_std_db * rng.standard_normal(size=sinr_true_db.shape)
    return delayed + noise
