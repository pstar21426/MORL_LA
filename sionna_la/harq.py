# HARQ-IR: mean MI -> SNR_eff, BLER at effective code rate

import numpy as np
from scipy.special import logsumexp

QM_TO_MOD = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
MOD_TO_QM = {v: k for k, v in QM_TO_MOD.items()}

_SNR_DB_GRID = np.arange(-30.0, 40.0 + 1e-9, 0.25)
_GH_NODES = 40
SNR_GRID_MAX_DB = float(_SNR_DB_GRID[-1])
_MI_TABLES = {}


def _pam_constellation(order):
    a = 2.0 * np.arange(order) - (order - 1)
    return a / np.sqrt((order**2 - 1) / 3.0)


def _pam_mi(order, snr_lin):
    # AWGN PAM mutual information (Gauss-Hermite)
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
    # square QAM = two independent PAM axes
    return 2.0 * _pam_mi(int(2 ** (qm // 2)), snr_lin)


def _mi_table(mod):
    # cache MI tables for each modulation -> memoization
    if mod not in _MI_TABLES:
        snr_lin = 10.0 ** (_SNR_DB_GRID / 10.0)
        _MI_TABLES[mod] = _qam_mi(MOD_TO_QM[mod], snr_lin)
    return _MI_TABLES[mod]


def mod_from_qm(qm):
    return QM_TO_MOD[int(qm)]


# X축, Y축을 입력하고, X값을 넣으면, 선형 보간을 통해 Y값을 반환
# _mi_table을 snr_lin = 10.0**(_SNR_DB_GRID / 10.0)으로 만들어서, 길이는 항상 같음
def mi_from_snr_db(mod, snr_db):
    return float(np.interp(snr_db, _SNR_DB_GRID, _mi_table(mod)))
# SNR_DB -> MI

def snr_db_from_mi(mod, mi):
    return float(np.interp(mi, _mi_table(mod), _SNR_DB_GRID))
# MI -> SNR_DB

class HarqProcess:
    # k: #retx, mi_tot: weighted sum of I(gamma), mcs/qm/R locked at initial tx

    def __init__(self, combining_rho=0.9, max_retx=3):
        self.combining_rho = float(combining_rho)
        self.max_retx = int(max_retx)
        self.reset()

    def reset(self):
        self.k = 0
        self.mi_tot = 0.0
        self.mcs = None
        self.qm = None
        self.coderate = None
        self.tbs = 0

    def start_transmission(self, mcs_index, qm, coderate):
        self.mcs = int(mcs_index)
        self.qm = int(qm)
        self.coderate = float(coderate)
        self.mi_tot = 0.0
        self.k = 0

    @property
    def n_eff(self):
        # 1 + rho + ... + rho^k  (current attempt included)
        rho = self.combining_rho
        n = self.k + 1
        if abs(rho - 1.0) < 1e-12:
            return float(n)
        return (1.0 - rho**n) / (1.0 - rho)

    @property
    def rate_eff(self):
        return float(self.coderate) / max(self.n_eff, 1e-12)

    def accumulate(self, snr_true_db):
        # I_tot += rho^k * I(gamma); SNR_eff = I^{-1}(I_tot / N_eff)
        mod = mod_from_qm(self.qm)
        self.mi_tot += (self.combining_rho**self.k) * mi_from_snr_db(mod, snr_true_db)
        i_avg = self.mi_tot / max(self.n_eff, 1e-12)
        return snr_db_from_mi(mod, i_avg)

    def on_nack(self):
        # True => drop (retx budget exhausted)
        if self.k >= self.max_retx:
            self.reset()
            return True
        self.k += 1
        return False
