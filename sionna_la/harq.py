"""
MIESM-style HARQ-IR abstraction.

Idea (replaces running the real LDPC decoder):
  1. each transmission delivers mutual information  I_m(gamma)  [bits/symbol]
  2. retransmissions accumulate MI:  I_tot += rho^k * I_m(gamma_k)
  3. map I_tot back to an *equivalent* SINR, and reuse the ordinary
     SINR -> BLER tables of Sionna PHYAbstraction

MI is precomputed once on an SINR grid per modulation, then interpolated,
so no integration happens inside the simulation loop.
"""

import numpy as np
from scipy.special import logsumexp

# modulation order Qm (bits/symbol) -> label used by the MI table dict
QM_TO_MOD = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
MOD_TO_QM = {v: k for k, v in QM_TO_MOD.items()}

# SINR grid on which MI is tabulated
_SNR_DB_GRID = np.arange(-30.0, 40.0 + 1e-9, 0.25)
_GH_NODES = 40  # Gauss-Hermite nodes for the noise expectation

# A single transmission cannot carry more than Qm bits/symbol, so once the
# accumulated MI passes that ceiling the inverse mapping has nothing left to
# resolve and pins the equivalent SINR here. Callers should treat hitting this
# value as "decoding is certain" rather than as a meaningful SINR.
SNR_GRID_MAX_DB = float(_SNR_DB_GRID[-1])

_MI_TABLES = {}


def _pam_constellation(order):
    """Unit-average-energy real PAM with `order` points."""
    a = 2.0 * np.arange(order) - (order - 1)
    return a / np.sqrt((order**2 - 1) / 3.0)


def _pam_mi(order, snr_lin):
    """
    Mutual information [bits/symbol] of a real PAM constellation over AWGN,
    evaluated by Gauss-Hermite quadrature (deterministic, no Monte Carlo).

    y = x + n,  n ~ N(0, sigma^2),  sigma^2 = 1 / snr
    """
    x = _pam_constellation(order)
    sigma = np.sqrt(1.0 / snr_lin)[:, None]  # [S, 1]

    t, w = np.polynomial.hermite.hermgauss(_GH_NODES)
    noise = np.sqrt(2.0) * sigma * t[None, :]  # [S, T]
    weight = w / np.sqrt(np.pi)  # normalized N(0,1) weights

    acc = np.zeros(snr_lin.shape, dtype=np.float64)
    for xi in x:
        d = xi - x  # [L]
        # log of exp(-((d+n)^2 - n^2) / (2 sigma^2)) for every competing point
        arg = -(d[None, None, :] ** 2 + 2.0 * d[None, None, :] * noise[:, :, None]) / (
            2.0 * sigma[:, :, None] ** 2
        )  # [S, T, L]
        acc += (logsumexp(arg, axis=-1) / np.log(2.0)) @ weight
    return np.log2(order) - acc / order


def _qam_mi(qm, snr_lin):
    """Square 2^qm-QAM splits into two independent sqrt(M)-PAM dimensions."""
    return 2.0 * _pam_mi(int(2 ** (qm // 2)), snr_lin)


def _mi_table(mod):
    """MI values on _SNR_DB_GRID, computed once per modulation and cached."""
    if mod not in _MI_TABLES:
        snr_lin = 10.0 ** (_SNR_DB_GRID / 10.0)
        _MI_TABLES[mod] = _qam_mi(MOD_TO_QM[mod], snr_lin)
    return _MI_TABLES[mod]


def mod_from_qm(qm):
    """Modulation label for a modulation order, e.g. 4 -> '16QAM'."""
    return QM_TO_MOD[int(qm)]


def mi_from_snr_db(mod, snr_db):
    """I_m(gamma): mutual information [bits/symbol] at the given SINR [dB]."""
    return float(np.interp(snr_db, _SNR_DB_GRID, _mi_table(mod)))


def snr_db_from_mi(mod, mi):
    """
    Inverse of `mi_from_snr_db`. MI is monotone in SINR, so a plain
    interpolation on the flipped axes is enough. Values at or above the
    modulation limit Qm saturate at the top of the grid.
    """
    return float(np.interp(mi, _mi_table(mod), _SNR_DB_GRID))


class HarqProcess:
    """
    Single HARQ process with MI accumulation.

    State:
      k       number of retransmissions already made for the current TB
      mi_tot  accumulated mutual information [bits/symbol]
      mcs     MCS index locked in at the initial transmission (None if idle)
      qm      modulation order of that MCS (retransmissions keep it)
    """

    def __init__(self, combining_rho=0.9, max_retx=3):
        self.combining_rho = float(combining_rho)
        self.max_retx = int(max_retx)
        self.reset()

    def reset(self):
        self.k = 0
        self.mi_tot = 0.0
        self.mcs = None
        self.qm = None

    @property
    def is_retransmission(self):
        return self.k > 0

    def start_transmission(self, mcs_index, qm):
        """Lock MCS/TBS for a new transport block."""
        self.mcs = int(mcs_index)
        self.qm = int(qm)
        self.mi_tot = 0.0
        self.k = 0

    def accumulate(self, snr_true_db):
        """
        Add this transmission's MI and return the equivalent SINR [dB] that
        the BLER tables should be evaluated at.
        """
        mod = mod_from_qm(self.qm)
        self.mi_tot += (self.combining_rho**self.k) * mi_from_snr_db(mod, snr_true_db)
        return snr_db_from_mi(mod, self.mi_tot)

    def on_nack(self):
        """
        Advance after a NACK. Returns True if the TB is dropped
        (retransmission budget exhausted).
        """
        if self.k >= self.max_retx:
            self.reset()
            return True
        self.k += 1
        return False

    def signalled_mcs(self, mcs_table_index=1):
        """
        Reserved MCS index a real gNB would signal on a retransmission:
        it carries the modulation order only, while TBS is inherited from
        the initial transmission (38.214 Tables 5.1.3.1-1/-2).

        Bookkeeping only — Sionna rejects these indices.
        """
        if mcs_table_index == 2:  # 256QAM table: 28..31 -> Qm 2,4,6,8
            return 28 + (int(self.qm) // 2 - 1)
        return 29 + (int(self.qm) // 2 - 1)  # 64QAM table: 29..31 -> Qm 2,4,6
