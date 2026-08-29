import numpy as np
import torch
from sionna.phy.utils import db_to_lin
from sionna.phy.nr.utils import decode_mcs_index

# TS 38.214 Table 5.2.2.1-2 (CQI table 1, 64QAM)
_CQI_SE = {
    1: 0.1523,
    2: 0.2344,
    3: 0.3770,
    4: 0.6016,
    5: 0.8770,
    6: 1.1758,
    7: 1.4766,
    8: 1.9141,
    9: 2.4063,
    10: 2.7305,
    11: 3.3223,
    12: 3.9023,
    13: 4.5234,
    14: 5.1152,
    15: 5.5547,
}
CQI_MAX = 15


def _mcs_spectral_efficiency(mcs_index, mcs_table_index):
    qm, rate = decode_mcs_index(
        torch.tensor([int(mcs_index)], dtype=torch.int32),
        table_index=mcs_table_index,
        is_pusch=False,
    )
    return float(qm.item() * rate.item())


def build_cqi_to_mcs(mcs_min, mcs_max, mcs_table_index=1):
    # nearest MCS (by SE) for each CQI in 1..15
    se_mcs = {
        m: _mcs_spectral_efficiency(m, mcs_table_index) for m in range(mcs_min, mcs_max + 1)
    }
    mapping = {0: mcs_min}
    for q, se in _CQI_SE.items():
        mapping[q] = min(se_mcs, key=lambda m: abs(se_mcs[m] - se))
    return mapping


def normalize_cqi(cqi_index):
    return float(cqi_index) / float(CQI_MAX)


def report_cqi(
    phy_abs,
    sinr_eff_lin,
    num_allocated_re,
    cqi_to_mcs,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
):
    # highest CQI in {1..15} with TBLER(MCS(CQI), SINR) <= bler_target; else 0
    mcs = torch.tensor(
        [cqi_to_mcs[q] for q in range(1, CQI_MAX + 1)], dtype=torch.int32
    )
    if not torch.is_tensor(sinr_eff_lin):
        sinr_eff_lin = torch.tensor([float(sinr_eff_lin)], dtype=torch.float32)
    sinr = sinr_eff_lin.to(dtype=torch.float32).reshape(-1)[0].expand(CQI_MAX)
    n_re = num_allocated_re.to(torch.int32).reshape(-1)[0].expand(CQI_MAX)

    *_, tbler, _ = phy_abs(
        mcs,
        sinr_eff=sinr,
        num_allocated_re=n_re,
        mcs_table_index=mcs_table_index,
        mcs_category=mcs_category,
        check_mcs_index_validity=False,
    )
    tbler = tbler.detach().cpu().numpy().reshape(-1)
    ok = np.where(tbler <= float(bler_target))[0]
    if ok.size == 0:
        return 0
    return int(ok[-1] + 1)  # 1-based CQI


def _tbler_at_sinr_db(
    phy_abs,
    mcs_index,
    sinr_db,
    num_allocated_re,
    mcs_table_index=1,
    mcs_category=1,
):
    sinr_lin = db_to_lin(torch.tensor([float(sinr_db)], dtype=torch.float32))
    *_, tbler, _ = phy_abs(
        torch.tensor([int(mcs_index)], dtype=torch.int32),
        sinr_eff=sinr_lin,
        num_allocated_re=num_allocated_re,
        mcs_table_index=mcs_table_index,
        mcs_category=mcs_category,
    )
    return float(tbler.item())


def calibrate_cqi_to_sinr_db(
    phy_abs,
    cqi_to_mcs,
    num_allocated_re,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
    sinr_min_db=-15.0,
    sinr_max_db=35.0,
):
    # SNR [dB] s.t. TBLER(MCS(CQI), SNR) ~= bler_target (OLLA inner mapping)
    out = {0: float(sinr_min_db)}
    for q in range(1, CQI_MAX + 1):
        mcs = int(cqi_to_mcs[q])
        lo, hi = float(sinr_min_db), float(sinr_max_db)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if _tbler_at_sinr_db(
                phy_abs,
                mcs,
                mid,
                num_allocated_re,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
            ) > float(bler_target):
                lo = mid
            else:
                hi = mid
        out[q] = hi
    return out


def mcs_from_sinr_db(
    phy_abs,
    sinr_db,
    num_allocated_re,
    mcs_min,
    mcs_max,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
):
    # highest MCS with TBLER <= bler_target at effective SNR
    best = int(mcs_min)
    for mcs in range(int(mcs_min), int(mcs_max) + 1):
        if (
            _tbler_at_sinr_db(
                phy_abs,
                mcs,
                sinr_db,
                num_allocated_re,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
            )
            <= float(bler_target)
        ):
            best = mcs
    return best


def build_bs_cqi_trace(
    phy_abs,
    sinr_fb_db,
    num_allocated_re,
    cqi_to_mcs,
    report_period,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
):
    # UE instant CQI per slot, then BS holds period-average report until next period
    num_slots = int(len(sinr_fb_db))
    instant = np.zeros(num_slots, dtype=np.int32)
    for t in range(num_slots):
        sinr_lin = db_to_lin(
            torch.tensor([float(sinr_fb_db[t])], dtype=torch.float32)
        )
        instant[t] = report_cqi(
            phy_abs,
            sinr_lin,
            num_allocated_re,
            cqi_to_mcs,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
            bler_target=bler_target,
        )

    bs_cqi = np.zeros(num_slots, dtype=np.int32)
    period = max(1, int(report_period))
    for k in range(0, num_slots, period):
        window = instant[k : min(k + period, num_slots)]
        rep = int(np.clip(int(round(float(np.mean(window)))), 0, CQI_MAX))
        bs_cqi[k : min(k + period, num_slots)] = rep
    return bs_cqi, instant
