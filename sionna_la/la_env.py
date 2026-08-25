"""
Gymnasium environment for 5G downlink link adaptation.

This is the single source of truth for the dynamics: the rule-based
baselines (ILLA/OLLA) and any RL agent both drive the same `step()`.

MDP / POMDP
-----------
observation  o_t, 10-dim by default. Every history feature is indexed by
             *initial transmissions*, not by slot, because the agent never
             acts during a retransmission:
               [0]     sinr_fb_db      noisy, delayed CQI
               [1]     d_sinr_fb_db    CQI first difference (channel trend)
               [2:5]   mcs_hist / M    MCS of the last 3 initial transmissions
               [5:8]   ack_hist        their first-attempt outcomes
               [8]     bler_hat        running first-transmission BLER
               [9]     last_tb_retx/K  retransmissions the previous TB needed
             -1 fills the two history blocks before enough data exists.
action       a_t = MCS index; ignored while k > 0 (retransmission keeps
             the MCS/TBS locked in at the initial transmission). Slots where
             the action does matter are flagged by info["is_decision"].
transition   ACK ~ Bernoulli(1 - TBLER(mcs, sinr_eq)) via PHYAbstraction,
             where sinr_eq comes from MIESM-style MI accumulation
reward       ACK  -> spectral efficiency Qm * coderate
             NACK -> 0
             drop -> -drop_penalty  (retransmission budget exhausted)

The true SINR is hidden from the agent: only the noisy/delayed CQI is in
the observation, which is the thing that makes this a POMDP.
"""

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

# MCS indices Sionna's BLER tables actually cover, per 38.214 PDSCH table.
# Above the top index the standard reserves the entry for retransmissions;
# below the bottom one PHYAbstraction has no table entry at all and returns
# tbler = inf, which the caller would see as a guaranteed NACK. Measured to
# hold for every num_allocated_re, so these are hard action-space limits
# rather than something a wider allocation could fix.
_MCS_RANGE = {1: (3, 28), 2: (2, 27)}

# fills the MCS/ACK history before that many initial transmissions have
# happened; distinguishable from every legal value of either field
_UNSEEN = -1.0


class InitialTxHistory:
    """
    Sliding-window history over initial transmissions only.

    Indexing by slot would mix in retransmission slots, where the MCS is
    inherited rather than chosen, so the MCS/ACK pairs would no longer line
    up with the agent's decisions. Here every entry is one decision and the
    first-attempt outcome it produced, which is also the sense in which the
    conventional 10% BLER target is defined.
    """

    def __init__(self, num_lags=3, bler_window=100):
        self.num_lags = int(num_lags)
        self.bler_window = int(bler_window)
        self.reset()

    def reset(self):
        self.mcs = deque([_UNSEEN] * self.num_lags, maxlen=self.num_lags)
        self.ack = deque([_UNSEEN] * self.num_lags, maxlen=self.num_lags)
        self._window = deque(maxlen=self.bler_window)
        self.last_tb_retx = 0

    def push_initial_tx(self, mcs_norm, ack):
        """Record one MCS decision and whether its first attempt decoded."""
        self.mcs.appendleft(float(mcs_norm))
        self.ack.appendleft(float(ack))
        self._window.append(int(ack))

    def push_tb_end(self, num_retx):
        """Record how many retransmissions the transport block consumed."""
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
        # NR / LA
        mcs_table_index=1,
        mcs_category=1,  # 1 = downlink (PDSCH)
        num_allocated_re=1000,
        # channel
        sinr_mean_db=8.0,
        sinr_ar_rho=0.95,
        sinr_innov_std_db=1.2,
        sinr_min_db=-5.0,
        sinr_max_db=25.0,
        sinr_mean_range_db=None,
        sinr_mean_change_prob=0.0,
        # CQI feedback
        cqi_noise_std_db=1.5,
        cqi_delay_slots=None,
        # HARQ
        mi_combining_rho=0.9,
        harq_max_retx=3,
        drop_penalty=2.0,
        # observation history (counted in initial transmissions, not slots)
        obs_num_lags=3,
        obs_bler_window=100,
        # shared Sionna object (heavy to build; reuse across envs/policies)
        phy_abs=None,
    ):
        super().__init__()

        self.num_slots = int(num_slots)
        self.mcs_table_index = int(mcs_table_index)
        self.mcs_category = int(mcs_category)
        self.num_allocated_re = int(num_allocated_re)

        self.sinr_mean_db = float(sinr_mean_db)
        self.sinr_ar_rho = float(sinr_ar_rho)
        self.sinr_innov_std_db = float(sinr_innov_std_db)
        self.sinr_min_db = float(sinr_min_db)
        self.sinr_max_db = float(sinr_max_db)
        self.sinr_mean_range_db = (
            None if sinr_mean_range_db is None else tuple(sinr_mean_range_db)
        )
        self.sinr_mean_change_prob = float(sinr_mean_change_prob)

        self.cqi_noise_std_db = float(cqi_noise_std_db)
        self.cqi_delay_slots = cqi_delay_slots

        self.drop_penalty = float(drop_penalty)

        self.phy_abs = phy_abs if phy_abs is not None else PHYAbstraction()
        self.harq = HarqProcess(
            combining_rho=mi_combining_rho, max_retx=harq_max_retx
        )
        self.hist = InitialTxHistory(
            num_lags=obs_num_lags, bler_window=obs_bler_window
        )

        self.mcs_min, self.mcs_max = _MCS_RANGE[self.mcs_table_index]
        self._mcs_span = self.mcs_max - self.mcs_min
        self.action_space = spaces.Discrete(self._mcs_span + 1)

        n_lags = self.hist.num_lags
        self.observation_space = spaces.Box(
            low=np.array(
                [-50.0, -50.0] + [_UNSEEN] * (2 * n_lags) + [0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [50.0, 50.0] + [1.0] * (2 * n_lags) + [1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # constant tensors reused every slot
        self._num_re_t = torch.tensor([self.num_allocated_re], dtype=torch.int32)

        self._sinr_true_db = np.zeros(self.num_slots)
        self._sinr_fb_db = np.zeros(self.num_slots)
        self._delay_used = 0
        self._t = 0
        self._last_ack = -1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def mcs_from_action(self, action):
        """Actions are contiguous from 0; MCS indices start at `mcs_min`."""
        return int(action) + self.mcs_min

    def action_from_mcs(self, mcs_index):
        return int(np.clip(mcs_index, self.mcs_min, self.mcs_max)) - self.mcs_min

    def _mcs_properties(self, mcs_index):
        """(modulation order Qm, coderate) for an MCS index."""
        qm, rate = decode_mcs_index(
            torch.tensor([int(mcs_index)], dtype=torch.int32),
            table_index=self.mcs_table_index,
            is_pusch=False,
        )
        return int(qm.item()), float(rate.item())

    def _obs(self):
        idx = min(self._t, self.num_slots - 1)
        sinr_fb = float(self._sinr_fb_db[idx])
        prev_fb = float(self._sinr_fb_db[idx - 1]) if idx > 0 else sinr_fb
        return np.array(
            [
                sinr_fb,
                sinr_fb - prev_fb,
                *self.hist.mcs,
                *self.hist.ack,
                self.hist.bler,
                self.hist.last_tb_retx / max(self.harq.max_retx, 1),
            ],
            dtype=np.float32,
        )

    def _info(self, outcome=None):
        """
        Pre-decision fields (what a controller needs to pick the next MCS)
        plus the outcome of the step that just finished.

        `is_decision` says whether the action passed to the *next* `step()`
        will actually be used. Retransmission slots still consume a step so
        that ILLA/OLLA keep seeing every HARQ feedback, but their action is
        discarded, so an offline dataset should filter or aggregate on this.
        """
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
        info.update(outcome or {})
        return info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # whole trace generated up front so that different policies can be
        # compared on exactly the same channel realization
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

        # --- MCS selection: locked during retransmissions ---
        if is_initial:
            mcs_used = self.mcs_from_action(action)
            qm, coderate = self._mcs_properties(mcs_used)
            self.harq.start_transmission(mcs_used, qm)
        else:
            mcs_used = self.harq.mcs
            qm, coderate = self._mcs_properties(mcs_used)

        # --- MI accumulation -> equivalent SINR seen by the BLER tables ---
        sinr_eq_db = self.harq.accumulate(sinr_true_db)
        sinr_eq_lin = db_to_lin(torch.tensor([sinr_eq_db], dtype=torch.float32))

        decoded_bits, harq_fb, _, tbler, _ = self.phy_abs(
            torch.tensor([mcs_used], dtype=torch.int32),
            sinr_eff=sinr_eq_lin,
            num_allocated_re=self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )

        ack = int(harq_fb.item())
        signalled = (
            mcs_used
            if is_initial
            else self.harq.signalled_mcs(self.mcs_table_index)
        )

        # only initial transmissions enter the history: they are the slots the
        # agent controls, and BLER targets are defined on first attempts
        if is_initial:
            self.hist.push_initial_tx(
                (mcs_used - self.mcs_min) / self._mcs_span, ack
            )

        # --- reward and HARQ state transition ---
        dropped = False
        if ack == 1:
            reward = qm * coderate  # spectral efficiency [bps/Hz]
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

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, phy_abs=None):
        """Build from the keys used in configs/downlink_la.yaml."""
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
