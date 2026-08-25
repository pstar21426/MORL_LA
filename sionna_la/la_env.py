# Gym env: 5G DL link adaptation (Sionna PHYAbstraction + HARQ-IR)
#
# obs:  [sinr_fb, d_sinr, mcs_hist x3, ack_hist x3, bler_hat, last_retx/K]
# act:  MCS (ignored on retx; initial MCS stays locked)
# ACK:  Bernoulli via PHYAbstraction(TBLER)
# r:    ACK -> Qm*coderate, NACK -> 0, drop -> -drop_penalty

from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sionna.phy.nr.utils import decode_mcs_index
from sionna.phy.utils import db_to_lin
from sionna.sys import PHYAbstraction

from channel import add_cqi_noise, generate_sinr_db_trace
from harq import SNR_GRID_MAX_DB, HarqProcess, mod_from_qm

# table index -> (min MCS, max MCS) with PHYAbstraction coverage
_MCS_RANGE = {1: (3, 28), 2: (2, 27)}
_UNSEEN = -1.0


class InitialTxHistory:
    # log initial transmissions only (slots where the agent picks MCS)

    def __init__(self, num_lags=3, bler_window=100):
        self.num_lags = num_lags
        self.bler_window = bler_window
        self.reset()

    def reset(self):
        self.mcs = deque([_UNSEEN] * self.num_lags, maxlen=self.num_lags)
        self.ack = deque([_UNSEEN] * self.num_lags, maxlen=self.num_lags)
        self._window = deque(maxlen=self.bler_window)
        self.last_tb_retx = 0

    def push_initial_tx(self, mcs_norm, ack):
        self.mcs.appendleft(float(mcs_norm))
        self.ack.appendleft(float(ack))
        self._window.append(int(ack))

    def push_tb_end(self, num_retx):
        self.last_tb_retx = int(num_retx)

    @property
    def bler(self):
        if not self._window:
            return 0.0
        return 1.0 - sum(self._window) / len(self._window)


class DownlinkLAEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_slots=400,
        mcs_table_index=1,
        mcs_category=1,
        num_allocated_re=1000,
        sinr_mean_db=8.0,
        sinr_ar_rho=0.95,
        sinr_innov_std_db=1.2,
        sinr_min_db=-5.0,
        sinr_max_db=25.0,
        sinr_mean_range_db=None,
        sinr_mean_change_prob=0.0,
        cqi_noise_std_db=1.5,
        cqi_delay_slots=None,
        mi_combining_rho=0.9,
        harq_max_retx=3,
        drop_penalty=2.0,
        obs_num_lags=3,
        obs_bler_window=100,
        phy_abs=None,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.mcs_table_index = mcs_table_index
        self.mcs_category = mcs_category
        self.num_allocated_re = num_allocated_re

        self.sinr_mean_db = sinr_mean_db
        self.sinr_ar_rho = sinr_ar_rho
        self.sinr_innov_std_db = sinr_innov_std_db
        self.sinr_min_db = sinr_min_db
        self.sinr_max_db = sinr_max_db
        self.sinr_mean_range_db = (
            None if sinr_mean_range_db is None else tuple(sinr_mean_range_db)
        )
        self.sinr_mean_change_prob = sinr_mean_change_prob
        self.cqi_noise_std_db = cqi_noise_std_db
        self.cqi_delay_slots = cqi_delay_slots
        self.drop_penalty = drop_penalty

        # Sionna BLER/ACK engine (package)
        self.phy_abs = phy_abs if phy_abs is not None else PHYAbstraction()
        self.harq = HarqProcess(combining_rho=mi_combining_rho, max_retx=harq_max_retx)
        self.hist = InitialTxHistory(num_lags=obs_num_lags, bler_window=obs_bler_window)

        self.mcs_min, self.mcs_max = _MCS_RANGE[self.mcs_table_index]
        self._mcs_span = self.mcs_max - self.mcs_min
        self.action_space = spaces.Discrete(self._mcs_span + 1)

        n = self.hist.num_lags
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -50.0] + [_UNSEEN] * (2 * n) + [0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0] + [1.0] * (2 * n) + [1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._num_re_t = torch.tensor([self.num_allocated_re], dtype=torch.int32)
        self._sinr_true_db = np.zeros(self.num_slots)
        self._sinr_fb_db = np.zeros(self.num_slots)
        self._delay_used = 0
        self._t = 0
        self._last_ack = -1

    def mcs_from_action(self, action):
        return int(action) + self.mcs_min

    def action_from_mcs(self, mcs_index):
        return int(np.clip(mcs_index, self.mcs_min, self.mcs_max)) - self.mcs_min

    def _mcs_properties(self, mcs_index):
        qm, rate = decode_mcs_index(
            torch.tensor([int(mcs_index)], dtype=torch.int32),
            table_index=self.mcs_table_index,
            is_pusch=False,
        )
        return int(qm.item()), float(rate.item())

    def _obs(self):
        idx = min(self._t, self.num_slots - 1)
        sinr_fb = float(self._sinr_fb_db[idx])
        prev = float(self._sinr_fb_db[idx - 1]) if idx > 0 else sinr_fb
        return np.array(
            [
                sinr_fb,
                sinr_fb - prev,
                *self.hist.mcs,
                *self.hist.ack,
                self.hist.bler,
                self.hist.last_tb_retx / max(self.harq.max_retx, 1),
            ],
            dtype=np.float32,
        )

    def _info(self, outcome=None):
        idx = min(self._t, self.num_slots - 1)
        info = {
            # for ILLA/OLLA, which want linear SINR and HARQ feedback
            "sinr_eff_lin": db_to_lin(
                torch.tensor([self._sinr_fb_db[idx]], dtype=torch.float32)
            ),
            "num_allocated_re": self._num_re_t,
            "harq_feedback": torch.tensor([self._last_ack], dtype=torch.int32),
            "mcs_table_index": self.mcs_table_index,
            "mcs_category": self.mcs_category,
            "mcs_min": self.mcs_min,
            "mcs_max": self.mcs_max,
            # bookkeeping
            "slot": self._t,
            "k": self.harq.k,
            "mi_tot": self.harq.mi_tot,
            "is_retransmission": self.harq.is_retransmission,
            "is_decision": not self.harq.is_retransmission,
            "last_tb_retx": self.hist.last_tb_retx,
            "bler_hat": self.hist.bler,
            "cqi_delay_slots": self._delay_used,
        }
        if outcome:
            info.update(outcome)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._sinr_true_db = generate_sinr_db_trace(
            num_slots=self.num_slots,
            mean_db=self.sinr_mean_db,
            rho=self.sinr_ar_rho,
            innov_std_db=self.sinr_innov_std_db,
            sinr_min_db=self.sinr_min_db,
            sinr_max_db=self.sinr_max_db,
            mean_range_db=self.sinr_mean_range_db,
            mean_change_prob=self.sinr_mean_change_prob,
            seed=seed,
        )
        self._sinr_fb_db, self._delay_used = add_cqi_noise(
            self._sinr_true_db,
            noise_std_db=self.cqi_noise_std_db,
            delay_slots=self.cqi_delay_slots,
            seed=None if seed is None else seed + 1,
        )

        self.harq.reset()
        self.hist.reset()
        self._t = 0
        self._last_ack = -1  # no feedback yet
        return self._obs(), self._info()

    def step(self, action):
        t = self._t
        sinr_true_db = float(self._sinr_true_db[t])
        is_initial = not self.harq.is_retransmission

        # MCS selection (locked during retransmissions)
        if is_initial:
            mcs_used = self.mcs_from_action(action)
            qm, coderate = self._mcs_properties(mcs_used)
            self.harq.start_transmission(mcs_used, qm)
        else:
            mcs_used = self.harq.mcs
            qm, coderate = self._mcs_properties(mcs_used)

        # MI accumulation (equivalent SINR seen by the BLER tables)
        sinr_eq_db = self.harq.accumulate(sinr_true_db)
        sinr_eq_lin = db_to_lin(torch.tensor([sinr_eq_db], dtype=torch.float32))

        decoded_bits, harq_fb, _, tbler, _ = self.phy_abs(
            torch.tensor([mcs_used], dtype=torch.int32),
            sinr_eff=sinr_eq_lin,
            num_allocated_re=self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        ack = int(harq_fb.item())  # 1=ACK, 0=NACK

        signalled = (
            mcs_used if is_initial else self.harq.signalled_mcs(self.mcs_table_index)
        )

        if is_initial:
            self.hist.push_initial_tx((mcs_used - self.mcs_min) / self._mcs_span, ack)

        dropped = False
        if ack == 1:
            reward = qm * coderate  # SE [bps/Hz]
            self.hist.push_tb_end(self.harq.k)
            self.harq.reset()
        else:
            reward = 0.0
            num_retx = self.harq.k
            dropped = self.harq.on_nack()
            if dropped:
                reward = -self.drop_penalty
                self.hist.push_tb_end(num_retx)

        outcome = {
            "ack": ack,
            "is_initial_tx": is_initial,
            "mcs_used": mcs_used,
            "signalled_mcs": signalled,
            "qm": qm,
            "coderate": coderate,
            "modulation": mod_from_qm(qm),
            "sinr_true_db": sinr_true_db,
            "sinr_eq_db": sinr_eq_db,
            "mi_saturated": sinr_eq_db >= SNR_GRID_MAX_DB - 1e-6,
            "tbler": float(tbler.item()),
            "decoded_bits": int(decoded_bits.item()),
            "dropped": dropped,
        }

        self._last_ack = ack
        self._t += 1
        truncated = self._t >= self.num_slots
        return self._obs(), float(reward), False, truncated, self._info(outcome)

    @classmethod
    def from_config(cls, cfg, phy_abs=None):
        # yaml keys -> constructor kwargs
        keys = (
            "num_slots",
            "mcs_table_index",
            "mcs_category",
            "num_allocated_re",
            "sinr_mean_db",
            "sinr_ar_rho",
            "sinr_innov_std_db",
            "sinr_min_db",
            "sinr_max_db",
            "sinr_mean_range_db",
            "sinr_mean_change_prob",
            "cqi_noise_std_db",
            "cqi_delay_slots",
            "mi_combining_rho",
            "harq_max_retx",
            "drop_penalty",
            "obs_num_lags",
            "obs_bler_window",
        )
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        return cls(phy_abs=phy_abs, **kwargs)
