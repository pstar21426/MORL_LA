# Time-correlated effective SINR process for downlink LA experiments.

import numpy as np


def generate_sinr_db_trace(
    num_slots,  # number of time slots
    mean_db=8.0,  # mean SINR [dB]
    rho=0.95,  # correlation (|rho|<1)
    innov_std_db=1.2,  # innovation noise std
    sinr_min_db=-5.0,
    sinr_max_db=25.0,
    mean_range_db=None,  # (lo, hi) -> randomize the operating point
    mean_change_prob=0.0,  # per-slot probability of re-drawing it
    seed=None,
):
    # AR(1) effective SINR + noisy/delayed CQI
    rng = np.random.default_rng(seed)

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
    dev = innov_std_db * rng.standard_normal()
    gamma[0] = mu[0] + dev
    for t in range(num_slots - 1):
        dev = rho * dev + innov_std_db * rng.standard_normal()
        gamma[t + 1] = mu[t + 1] + dev
    return np.clip(gamma, sinr_min_db, sinr_max_db)


def add_cqi_noise(sinr_true_db, noise_std_db=1.5, delay_slots=None, seed=None):
    # delay_slots=None -> pick fixed delay in {1,2,3,4} once per episode
    rng = np.random.default_rng(seed)

    # once per call (= once per simulation run), not every slot
    if delay_slots is None:
        delay_slots = int(rng.integers(1, 5))  # random delay slots {1,2,3,4}

    if delay_slots == 0:
        delayed = sinr_true_db  # no delay
    else:
        delayed = np.empty_like(sinr_true_db)  # delayed SINR [dB]
        delayed[:delay_slots] = sinr_true_db[0]  # initial SINR
        delayed[delay_slots:] = sinr_true_db[:-delay_slots]  # delayed SINR

    noise = noise_std_db * rng.standard_normal(size=sinr_true_db.shape)  # noise
    return delayed + noise, delay_slots  # noisy CQI, delay_slots
