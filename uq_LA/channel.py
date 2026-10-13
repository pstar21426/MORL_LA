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
    ar_order=1,
    phi2=0.0,
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
    else:  # (lo, hi)에서 시작 평균을 뽑고, mean_change_prob마다 다시 뽑는다.
        lo, hi = mean_range_db
        mu = np.empty(num_slots)
        current = rng.uniform(lo, hi)
        for t in range(num_slots):
            if t > 0 and rng.random() < mean_change_prob:
                current = rng.uniform(lo, hi)
            mu[t] = current

    if int(ar_order) == 1:
        gamma = np.empty(num_slots)
        dev = stat_std * rng.standard_normal()
        gamma[0] = mu[0] + dev
        for t in range(num_slots - 1):
            dev = rho * dev + innov * rng.standard_normal()
            gamma[t + 1] = mu[t + 1] + dev
        return np.clip(gamma, sinr_min_db, sinr_max_db)

    # AR(2): rho is phi1. Stationary variance matches
    # innov^2 * (1-phi2) / ((1+phi2)((1-phi2)^2 - phi1^2)).
    phi1 = rho
    phi2 = float(phi2)
    scale = ar2_std_per_innov(phi1, phi2)
    var = (innov * scale) ** 2
    cov1 = phi1 / (1.0 - phi2) * var
    std = np.sqrt(var)
    corr = cov1 / var
    z0 = rng.standard_normal()
    z1 = rng.standard_normal()
    x_prev = std * z0
    x = corr * x_prev + std * np.sqrt(max(0.0, 1.0 - corr * corr)) * z1
    gamma = np.empty(num_slots)
    gamma[0] = mu[0] + x
    for t in range(num_slots - 1):
        x_next = phi1 * x + phi2 * x_prev + innov * rng.standard_normal()
        x_prev = x
        x = x_next
        gamma[t + 1] = mu[t + 1] + x
    return np.clip(gamma, sinr_min_db, sinr_max_db)


def ar2_std_per_innov(phi1, phi2):
    # Stationary std of x_t = phi1 x_{t-1} + phi2 x_{t-2} + e_t, divided by std(e).
    phi1 = float(phi1)
    phi2 = float(phi2)
    if abs(phi2) >= 1.0 or (phi1 + phi2) >= 1.0 or (phi2 - phi1) >= 1.0:
        raise ValueError(f"AR(2) not stationary: phi1={phi1}, phi2={phi2}")
    disc = (1.0 - phi2) ** 2 - phi1 ** 2
    denom = (1.0 + phi2) * disc
    if denom <= 0.0:
        raise ValueError(f"AR(2) variance undefined: phi1={phi1}, phi2={phi2}")
    return float(np.sqrt((1.0 - phi2) / denom))


def impair_decoding_sinr(
    sinr_db,
    evm_db=None,
    burst_atten_db=0.0,
    burst_mean_on=20.0,
    burst_mean_off=80.0,
    burst_seed=None,
    sinr_min_db=-5.0,
    sinr_max_db=30.0,
):
    # CQI는 이 함수를 거치기 전의 SINR로 만든다.
    # EVM: 1/g_dec = 1/g + 1/g_evm (선형). 버스트: 켜진 슬롯만 atten_db를 뺀다.
    g = np.array(sinr_db, dtype=np.float64, copy=True)
    if evm_db is not None and float(evm_db) > 0.0:
        evm = float(evm_db)
        g = -10.0 * np.log10(10.0 ** (-g / 10.0) + 10.0 ** (-evm / 10.0))
    atten = float(burst_atten_db)
    if atten > 0.0:
        rng = np.random.default_rng(burst_seed)
        on = False
        mask = np.zeros(g.shape[0], dtype=bool)
        p_leave = 1.0 / float(burst_mean_on)
        p_enter = 1.0 / float(burst_mean_off)
        for t in range(g.shape[0]):
            on = (rng.random() >= p_leave) if on else (rng.random() < p_enter)
            mask[t] = on
        g = g - np.where(mask, atten, 0.0)
    return np.clip(g, sinr_min_db, sinr_max_db)


def add_cqi_noise(
    sinr_true_db,
    noise_std_db=1.5,
    bias_db=0.0,
    delay_slots=None,
    seed=None,
    sinr_min_db=None,
    sinr_max_db=None,

):
    # delay_slots=None -> pick fixed delay in {1,2,3,4} once per episode.
    ss = np.random.SeedSequence() if seed is None else np.random.SeedSequence(int(seed))
    delay_rng, noise_rng = (np.random.default_rng(s) for s in ss.spawn(2))

    if delay_slots is None:
        delay_slots = int(delay_rng.integers(1, 5))  # {1,2,3,4} once

    if delay_slots == 0:
        delayed = sinr_true_db
    else:
        delayed = np.empty_like(sinr_true_db)
        delayed[:delay_slots] = sinr_true_db[0]
        delayed[delay_slots:] = sinr_true_db[:-delay_slots]

    hat = delayed + float(bias_db) + noise_std_db * noise_rng.standard_normal(size=sinr_true_db.shape)
    lo = -np.inf if sinr_min_db is None else float(sinr_min_db)
    hi = np.inf if sinr_max_db is None else float(sinr_max_db)
    return np.clip(hat, lo, hi), delay_slots
