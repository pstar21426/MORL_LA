import numpy as np
import torch
from sionna.phy.nr.utils import calculate_tb_size, decode_mcs_index
from sionna.phy.utils import db_to_lin

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

_MCS_QM_RATE = {}


def _mcs_qm_rate_table(mcs_table_index, mcs_min, mcs_max):
    key = (int(mcs_table_index), int(mcs_min), int(mcs_max))
    if key not in _MCS_QM_RATE:
        rows = []
        for m in range(int(mcs_min), int(mcs_max) + 1):
            qm, rate = decode_mcs_index(
                torch.tensor([m], dtype=torch.int32),
                table_index=mcs_table_index,
                is_pusch=False,
            )
            rows.append((m, int(qm.item()), float(rate.item())))
        _MCS_QM_RATE[key] = rows
    return _MCS_QM_RATE[key]


def build_cqi_to_mcs(mcs_min, mcs_max, mcs_table_index=1):
    rows = _mcs_qm_rate_table(mcs_table_index, mcs_min, mcs_max)
    se_mcs = {m: qm * rate for m, qm, rate in rows}
    mapping = {0: int(mcs_min)}
    for q, se in _CQI_SE.items():
        mapping[q] = min(se_mcs, key=lambda m: abs(se_mcs[m] - se))
    return mapping


def mcs_qm_rate(mcs_index, mcs_min, mcs_max, mcs_table_index=1):
    mcs_index = int(mcs_index)
    for m, qm, rate in _mcs_qm_rate_table(mcs_table_index, mcs_min, mcs_max):
        if m == mcs_index:
            return qm, rate
    raise ValueError(f"MCS {mcs_index} not in [{mcs_min}, {mcs_max}]")


def mcs_for_ir_rate(rate_eff, qm, mcs_min, mcs_max, mcs_table_index=1):
    # same Qm: highest coderate <= R_eff, else lowest-rate MCS of that Qm
    cands = [
        (m, r)
        for m, qq, r in _mcs_qm_rate_table(mcs_table_index, mcs_min, mcs_max)
        if qq == int(qm)
    ]
    if not cands:
        return int(mcs_min)
    below = [(m, r) for m, r in cands if r <= float(rate_eff) + 1e-12]
    if below:
        return int(max(below, key=lambda x: x[1])[0])
    return int(min(cands, key=lambda x: x[1])[0])


def normalize_cqi(cqi_index):
    return float(cqi_index) / float(CQI_MAX)


def _as_int32_vec(x):
    if not torch.is_tensor(x):
        x = torch.tensor([int(x)], dtype=torch.int32)
    return x.to(dtype=torch.int32).reshape(-1)


def _as_float_vec(x):
    if not torch.is_tensor(x):
        x = torch.tensor([float(x)], dtype=torch.float32)
    return x.to(dtype=torch.float32).reshape(-1)


def _match_len(x, n):
    if x.numel() == 1 and n > 1:
        return x.expand(n)
    return x


def _tb_size_from_mcs(mcs, n_re, mcs_table_index=1, mcs_category=1):
    qm, rate = decode_mcs_index(
        mcs,
        table_index=mcs_table_index,
        is_pusch=int(mcs_category) == 0,
        check_index_validity=False,
    )
    num_coded = (qm * n_re).to(torch.int32)
    _, cb_size, num_cb, *_ = calculate_tb_size(
        qm,
        rate,
        num_coded_bits=num_coded,
        tb_scaling=1.0,
        return_cw_length=False,
    )
    tbs = (num_cb * cb_size).to(torch.int32)
    return tbs, cb_size, num_cb


def tb_layout_from_mcs(
    mcs_index,
    num_allocated_re,
    mcs_table_index=1,
    mcs_category=1,
):
    mcs = _as_int32_vec(mcs_index)
    n_re = _match_len(_as_int32_vec(num_allocated_re), int(mcs.numel()))
    return _tb_size_from_mcs(
        mcs, n_re, mcs_table_index=mcs_table_index, mcs_category=mcs_category
    )


def tbler_from_phy(
    phy_abs,
    mcs_index,
    sinr_eff_lin,
    num_allocated_re=None,
    mcs_table_index=1,
    mcs_category=1,
    cb_size=None,
    num_cb=None,
):
    mcs = _as_int32_vec(mcs_index)
    n = int(mcs.numel())
    sinr = _match_len(_as_float_vec(sinr_eff_lin), n)
    if cb_size is None or num_cb is None:
        n_re = _match_len(_as_int32_vec(num_allocated_re), n)
        tbs, cb_sz, n_cb = _tb_size_from_mcs(
            mcs, n_re, mcs_table_index=mcs_table_index, mcs_category=mcs_category
        )
    else:
        cb_sz = _match_len(_as_int32_vec(cb_size), n)
        n_cb = _match_len(_as_int32_vec(num_cb), n)
        tbs = (n_cb * cb_sz).to(torch.int32)
    bler = phy_abs.get_bler(mcs, mcs_table_index, mcs_category, cb_sz, sinr)
    one = torch.ones((), dtype=bler.dtype, device=bler.device)
    tbler = one - torch.pow(one - bler, n_cb.to(dtype=bler.dtype))
    return tbler, tbs


def report_cqi(
    phy_abs,
    sinr_eff_lin,
    num_allocated_re,
    cqi_to_mcs,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
):
    mcs = torch.tensor(
        [cqi_to_mcs[q] for q in range(1, CQI_MAX + 1)], dtype=torch.int32
    )
    tbler, _ = tbler_from_phy(
        phy_abs,
        mcs,
        sinr_eff_lin,
        num_allocated_re,
        mcs_table_index=mcs_table_index,
        mcs_category=mcs_category,
    )
    tbler = tbler.detach().cpu().numpy().reshape(-1)
    ok = np.where(tbler <= float(bler_target))[0]
    if ok.size == 0:
        return 0
    return int(ok[-1] + 1)


def calibrate_cqi_to_sinr_db(
    phy_abs,
    cqi_to_mcs,
    num_allocated_re,
    mcs_table_index=1,
    mcs_category=1,
    bler_target=0.1,
    sinr_min_db=-5.0,
    sinr_max_db=30.0,
):
    out = {0: float(sinr_min_db)}
    for q in range(1, CQI_MAX + 1):
        mcs = int(cqi_to_mcs[q])
        lo, hi = float(sinr_min_db), float(sinr_max_db)
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            sinr_lin = db_to_lin(torch.tensor([mid], dtype=torch.float32))
            tbler, _ = tbler_from_phy(
                phy_abs,
                mcs,
                sinr_lin,
                num_allocated_re,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
            )
            if float(tbler.reshape(-1)[0].item()) > float(bler_target):
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
    lo, hi = int(mcs_min), int(mcs_max)
    mcs = torch.arange(lo, hi + 1, dtype=torch.int32)
    sinr_lin = db_to_lin(torch.tensor([float(sinr_db)], dtype=torch.float32))
    tbler, _ = tbler_from_phy(
        phy_abs,
        mcs,
        sinr_lin,
        num_allocated_re,
        mcs_table_index=mcs_table_index,
        mcs_category=mcs_category,
    )
    ok = tbler.detach().cpu().numpy().reshape(-1) <= float(bler_target)
    if not np.any(ok):
        return lo
    return int(lo + int(np.where(ok)[0][-1]))
