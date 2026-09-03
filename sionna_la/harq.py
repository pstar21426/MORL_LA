# HARQ-IR: first TX uses raw SNR; retx SNR_eff = I^{-1}(mean I), same Qm, R/n, original TBS.

import numpy as np
from scipy.special import logsumexp

QM_TO_MOD = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
MOD_TO_QM = {v: k for k, v in QM_TO_MOD.items()}

_SNR_DB_GRID = np.arange(-30.0, 40.0 + 1e-9, 0.25)
_GH_NODES = 40
_MI_TABLES = {}


def _pam_constellation(order):
    a = 2.0 * np.arange(order) - (order - 1)
    return a / np.sqrt((order**2 - 1) / 3.0)


def _pam_mi(order, snr_lin):
    x = _pam_constellation(order)
    sigma = np.sqrt(1.0 / snr_lin)[:, None]
    t, w = np.polynomial.hermite.hermgauss(_GH_NODES)
    noise = np.sqrt(2.0) * sigma * t[None, :]
    weight = w / np.sqrt(np.pi)

    acc = np.zeros(snr_lin.shape)
    for xi in x:
        d = xi - x
        arg = -(d[None, None, :] ** 2 + 2.0 * d[None, None, :] * noise[:, :, None]) / (
            2.0 * sigma[:, :, None] ** 2
        )
        acc += (logsumexp(arg, axis=-1) / np.log(2.0)) @ weight
    return np.log2(order) - acc / order


def _qam_mi(qm, snr_lin):
    return 2.0 * _pam_mi(int(2 ** (qm // 2)), snr_lin)


def _mi_table(mod):
    if mod not in _MI_TABLES:
        snr_lin = 10.0 ** (_SNR_DB_GRID / 10.0)
        _MI_TABLES[mod] = _qam_mi(MOD_TO_QM[mod], snr_lin)
    return _MI_TABLES[mod]


def mod_from_qm(qm):
    return QM_TO_MOD[int(qm)]


def mi_from_snr_db(mod, snr_db):
    return float(np.interp(snr_db, _SNR_DB_GRID, _mi_table(mod)))
# SNR_DB -> MI

def snr_db_from_mi(mod, mi):
    return float(np.interp(mi, _mi_table(mod), _SNR_DB_GRID))
# MI -> SNR_DB


class HarqProcess:
    def __init__(self, max_retx=2):
        self.max_retx = int(max_retx)
        self.reset()

    def reset(self):
        self.k = 0
        self.mi_tot = 0.0
        self.mcs = None
        self.qm = None
        self.coderate = None
        self.tbs = 0
        self.cb_size = 0
        self.num_cb = 0

    def start_transmission(self, mcs_index, qm, coderate, tbs, cb_size, num_cb):
        self.mcs = int(mcs_index)
        self.qm = int(qm)
        self.coderate = float(coderate)
        self.tbs = int(tbs)
        self.cb_size = int(cb_size)
        self.num_cb = int(num_cb)
        self.mi_tot = 0.0
        self.k = 0

    @property
    def n_tx(self):
        return self.k + 1

    @property
    def rate_eff(self):
        return float(self.coderate) / self.n_tx

    def accumulate(self, snr_true_db):
        snr_true_db = float(snr_true_db)
        mod = mod_from_qm(self.qm)
        self.mi_tot += mi_from_snr_db(mod, snr_true_db)
        if self.n_tx == 1:
            return snr_true_db
        return snr_db_from_mi(mod, self.mi_tot / self.n_tx)

    def on_nack(self):
        if self.k >= self.max_retx:
            self.reset()
            return True
        self.k += 1
        return False
