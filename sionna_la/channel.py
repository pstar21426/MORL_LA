# Time-correlated effective SINR process for downlink LA experiments.

import numpy as np


def generate_sinr_db_trace(
    num_slots,  # number of time slots
    mean_db=10.0,  # mean SINR [dB]
    rho=0.95,  # correlation (|rho|<1)
    innov_std_db=2.5,  # innovation noise std
    sinr_min_db=-5.0,
    sinr_max_db=30.0,
    mean_range_db=None,  # (lo, hi) -> randomize the operating point
    mean_change_prob=0.0,  # per-slot probability of re-drawing it
    seed=None,
):
    rng = np.random.default_rng(seed)
    rho = float(rho)
    innov = float(innov_std_db)
    if abs(rho) < 1.0:
        stat_std = innov / np.sqrt(1.0 - rho * rho)
    else:
        stat_std = innov

    if mean_range_db is None:
        mu = np.full(num_slots, float(mean_db))
    else:
        lo, hi = mean_range_db
        mu = np.empty(num_slots)
        current = rng.uniform(lo, hi)
        for t in range(num_slots):
            if t > 0 and rng.random() < mean_change_prob:
                current = rng.uniform(lo, hi)
            mu[t] = current

    gamma = np.empty(num_slots)
    dev = stat_std * rng.standard_normal()
    gamma[0] = mu[0] + dev
    for t in range(num_slots - 1):
        dev = rho * dev + innov * rng.standard_normal()
        gamma[t + 1] = mu[t + 1] + dev
    return np.clip(gamma, sinr_min_db, sinr_max_db)


def add_cqi_noise(
    sinr_true_db,
    noise_std_db=1.5,
    delay_slots=None,
    seed=None,
    sinr_min_db=None,
    sinr_max_db=None,
):
    # delay_slots=None -> pick fixed delay in {1,2,3,4} once per episode
    rng = np.random.default_rng(seed)

    if delay_slots is None:
        delay_slots = int(rng.integers(1, 5))  # {1,2,3,4} once

    if delay_slots == 0:
        delayed = sinr_true_db
    else:
        delayed = np.empty_like(sinr_true_db)
        delayed[:delay_slots] = sinr_true_db[0]
        delayed[delay_slots:] = sinr_true_db[:-delay_slots]

    hat = delayed + noise_std_db * rng.standard_normal(size=sinr_true_db.shape)
    lo = -np.inf if sinr_min_db is None else float(sinr_min_db)
    hi = np.inf if sinr_max_db is None else float(sinr_max_db)
    return np.clip(hat, lo, hi), delay_slots
