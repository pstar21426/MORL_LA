# Gym env: decision-step 5G DL LA (Sionna PHYAbstraction + HARQ-IR)
#
# One Gym step = one transport block (initial MCS + internal retx slots).
# Agent state: [cqi_n, past_cqi×L, past_m×L, past_b×L]  (default L=3 → 10-dim)
#   cqi_n       current reported CQI / 15
#   past_cqi    decision CQIs at n-1..n-L (most recent first); unseen = -1
#   past_m, b   MCS norm / first-tx ACK at n-1..n-L; unseen = -1
# True SINR γ is not in the agent state.
# info (next decision): cqi_index, harq_feedbacks (delay-elapsed first-ACKs), slot, ...
# info["outcome"] (TB just finished): mcs, tbler, harq_seq, decision CQI/SINR, ...
# delta_tau stays in info only (not in state)
# γ̂ = delayed + noisy SINR (UE measurement before CQI quantization)
# Reward: SE/num_tx if TB ACKs, -drop_penalty if dropped, 0 on intermediate NACKs
# HARQ-IR: BLER uses R_eff = R / N_eff and I^{-1}(mean I); MCS locked at first tx
# ACK is sampled per slot from a dedicated RNG (not CQI/OLLA TBLER lookups)
# ILLA/OLLA baselines use discrete CQI only (see policies.py)

from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sionna.phy import config as sionna_config
from sionna.phy.nr.utils import decode_mcs_index
from sionna.phy.utils import db_to_lin
from sionna.sys import PHYAbstraction

from channel import add_cqi_noise, generate_sinr_db_trace
from cqi import (
    build_cqi_to_mcs,
    mcs_for_ir_rate,
    normalize_cqi,
    report_cqi,
    tbler_from_phy,
)
from harq import SNR_GRID_MAX_DB, HarqProcess, mod_from_qm

_MCS_RANGE = {1: (3, 28), 2: (2, 27)}
_UNSEEN = -1.0


def seed_phy(seed):
    # torch / leftover Sionna RNG; ACK uses env._ack_seed (reset seed+2), not this
    if seed is None:
        return
    seed = int(seed)
    sionna_config.seed = seed
    torch.manual_seed(seed)


def _slot_uniform(ack_seed, slot):
    # U(0,1) keyed by (ack_seed, slot) so CQI/OLLA lookups cannot steal ACK draws
    ss = np.random.SeedSequence([int(ack_seed) & 0xFFFFFFFF, int(slot) & 0xFFFFFFFF])
    return float(np.random.default_rng(ss).random())


class DecisionHistory:
    # past decision CQI / MCS / first-attempt ACK (L each; aligned by decision index)

    def __init__(self, num_lags=3):
        self.num_lags = max(1, int(num_lags))
        self.reset()

    def reset(self):
        n = self.num_lags
        self.cqi = deque([_UNSEEN] * n, maxlen=n)
        self.mcs = deque([_UNSEEN] * n, maxlen=n)
        self.ack = deque([_UNSEEN] * n, maxlen=n)

    def push(self, cqi_n, mcs_norm, first_ack):
        self.cqi.appendleft(float(cqi_n))
        self.mcs.appendleft(float(mcs_norm))
        self.ack.appendleft(float(first_ack))


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
        sinr_mean_range_db=None, # range of the mean SINR
        sinr_mean_change_prob=0.0, # probability of changing the mean SINR  
        cqi_noise_std_db=1.5,
        cqi_delay_slots=None,
        cqi_bler_target=0.1,
        ack_delay_slots=0,
        mi_combining_rho=0.9, # discount ratio for MI accumulation
        harq_max_retx=3,
        drop_penalty=2.0,
        state_num_lags=3,
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
        self.cqi_bler_target = float(cqi_bler_target)
        self.ack_delay_slots = max(0, int(ack_delay_slots))
        self.drop_penalty = drop_penalty

        self.phy_abs = phy_abs if phy_abs is not None else PHYAbstraction()
        self.harq = HarqProcess(combining_rho=mi_combining_rho, max_retx=harq_max_retx)
        self.hist = DecisionHistory(num_lags=state_num_lags)

        self.mcs_min, self.mcs_max = _MCS_RANGE[self.mcs_table_index]
        self._mcs_span = self.mcs_max - self.mcs_min
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min, self.mcs_max, mcs_table_index=self.mcs_table_index
        )
        self.action_space = spaces.Discrete(self._mcs_span + 1)

        # [cqi_n, past_cqi×L, past_m×L, past_b×L]; unseen = -1, else in [0, 1]
        n = self.hist.num_lags
        state_dim = 1 + 3 * n
        self.state_space = spaces.Box(
            low=np.full(state_dim, _UNSEEN, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        # Gymnasium requires this name; same object as state_space
        self.observation_space = self.state_space

        self._num_re_t = torch.tensor([self.num_allocated_re], dtype=torch.int32)
        self._sinr_true_db = np.zeros(self.num_slots)
        self._sinr_fb_db = np.zeros(self.num_slots)
        self._delay_used = 0
        self._t = 0
        self._last_decision_slot = 0
        self._ever_decided = False
        self._pending_harq = [-1]  # fed to OLLA on next decision
        self._fb_queue = deque()  # (delivery_slot, first_ack, cqi_n, mcs_norm)
        self._ack_seed = 2

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

    def _sinr_hat_db(self, slot):
        # UE effective-SINR estimate (delayed + noisy), before CQI quantization
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_fb_db[idx])

    def _gamma_at(self, slot):
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_true_db[idx])

    def _report_cqi_at(self, slot):
        sinr_lin = db_to_lin(
            torch.tensor([self._sinr_hat_db(slot)], dtype=torch.float32)
        )
        return report_cqi(
            self.phy_abs,
            sinr_lin,
            self._num_re_t,
            self._cqi_to_mcs,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
            bler_target=self.cqi_bler_target,
        )

    def _drain_feedback(self, now_slot):
        # ACK/NACK + delayed history push (BS learns first_ack at delivery_slot)
        drained = []
        while self._fb_queue and self._fb_queue[0][0] <= now_slot:
            _, first_ack, cqi_n, mcs_norm = self._fb_queue.popleft()
            self.hist.push(cqi_n, mcs_norm, int(first_ack))
            drained.append(int(first_ack))
        self._pending_harq = drained if drained else [-1]

    def _state(self, cqi_index=None):
        # agent state at current decision slot _t
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        # [cqi_t, past_cqi(L), past_m(L), past_b(L)]
        cqi_n = normalize_cqi(cqi_index)
        # [cqi_t, past CQI x L, past MCS x L, past ACK x L]
        return np.array(
            [cqi_n, *self.hist.cqi, *self.hist.mcs, *self.hist.ack],
            dtype=np.float32,
        )

    def _info(self, outcome=None, cqi_index=None):
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        sinr_hat_db = self._sinr_hat_db(self._t)
        info = {
            # true channel (hidden from agent state vector)
            "sinr_true_db": self._gamma_at(self._t),
            # sinr_hat kept for logging; baselines use cqi_index (discrete) only
            "sinr_hat_db": sinr_hat_db,
            "cqi_index": int(cqi_index),
            "cqi_norm": normalize_cqi(cqi_index),
            "harq_feedbacks": list(self._pending_harq),
            "mcs_table_index": self.mcs_table_index,
            "mcs_category": self.mcs_category,
            "mcs_min": self.mcs_min,
            "mcs_max": self.mcs_max,
            # bookkeeping (tau not in state; logging / analysis only)
            "slot": self._t,
            "delta_tau": (
                0.0
                if not self._ever_decided
                else float(self._t - self._last_decision_slot)
            ),
            "cqi_delay_slots": self._delay_used,
            "ack_delay_slots": self.ack_delay_slots,
        }
        # TB log lives under outcome so it cannot overwrite next-decision fields
        if outcome is not None:
            info["outcome"] = outcome
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
        self._ack_seed = (
            int(seed) + 2
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        self._t = 0
        self._last_decision_slot = 0
        self._ever_decided = False
        self._pending_harq = [-1]
        self._fb_queue.clear()
        self._drain_feedback(0)
        cqi_index = self._report_cqi_at(self._t)
        return self._state(cqi_index), self._info(cqi_index=cqi_index)

    def _phy_once(self, mcs_used):
        sinr_true_db = self._gamma_at(self._t)
        sinr_eq_db = self.harq.accumulate(sinr_true_db)
        sinr_eq_lin = db_to_lin(torch.tensor([sinr_eq_db], dtype=torch.float32))
        if self.harq.k == 0:
            mcs_lookup = int(mcs_used)
        else:
            mcs_lookup = mcs_for_ir_rate(
                self.harq.rate_eff,
                self.harq.qm,
                self.mcs_min,
                self.mcs_max,
                mcs_table_index=self.mcs_table_index,
            )
        tbler_t, _ = tbler_from_phy(
            self.phy_abs,
            mcs_lookup,
            sinr_eq_lin,
            self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        tbler = float(tbler_t.reshape(-1)[0].item())
        tbs = int(self.harq.tbs)
        # PHYAbstraction: U < tbler → NACK (0), else ACK (1)
        ack = 0 if _slot_uniform(self._ack_seed, self._t) < tbler else 1
        return (
            ack,
            tbler,
            ack * tbs,
            sinr_true_db,
            sinr_eq_db,
            mcs_lookup,
            float(self.harq.rate_eff),
        )

    def step(self, action):
        if self._t >= self.num_slots:
            return self._state(), 0.0, False, True, self._info()

        decision_slot = self._t
        decision_cqi = self._report_cqi_at(self._t)
        decision_cqi_n = normalize_cqi(decision_cqi)
        decision_sinr_hat = self._sinr_hat_db(self._t)
        decision_gamma = self._gamma_at(self._t)

        mcs_used = self.mcs_from_action(action)
        qm, coderate = self._mcs_properties(mcs_used)
        self.harq.start_transmission(mcs_used, qm, coderate)
        _, tbs_t = tbler_from_phy(
            self.phy_abs,
            mcs_used,
            db_to_lin(torch.tensor([0.0], dtype=torch.float32)),
            self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        self.harq.tbs = int(tbs_t.reshape(-1)[0].item())

        reward = 0.0
        harq_seq = []
        first_ack = None
        dropped = False
        truncated_mid_tb = False
        num_tx = 0
        last_tbler = 0.0
        last_bits = 0
        last_sinr_eq = decision_gamma
        last_mcs_lookup = mcs_used
        last_rate_eff = coderate
        mi_sat = False

        # run whole TB: initial + retx until ACK/drop or slots run out
        while self._t < self.num_slots:
            ack, tbler, bits, _, sinr_eq, mcs_lookup, rate_eff = self._phy_once(
                mcs_used
            )
            harq_seq.append(ack)
            num_tx += 1
            last_tbler, last_bits, last_sinr_eq = tbler, bits, sinr_eq
            last_mcs_lookup, last_rate_eff = mcs_lookup, rate_eff
            mi_sat = sinr_eq >= SNR_GRID_MAX_DB - 1e-6  # MI grid saturation

            if first_ack is None:
                first_ack = ack

            self._t += 1

            if ack == 1:
                reward = qm * coderate / num_tx
                self.harq.reset()
                break

            dropped = self.harq.on_nack()
            if dropped:
                reward = -self.drop_penalty
                break
        else:
            # episode time limit mid-TB: not a HARQ drop (reward stays 0)
            truncated_mid_tb = True
            if self.harq.mcs is not None:
                self.harq.reset()

        mcs_norm = (mcs_used - self.mcs_min) / self._mcs_span
        first_ack_i = int(first_ack or 0)
        if harq_seq:
            self._fb_queue.append(
                (
                    self._t + self.ack_delay_slots,
                    first_ack_i,
                    decision_cqi_n,
                    mcs_norm,
                )
            )

        self._last_decision_slot = decision_slot
        self._ever_decided = True

        truncated = self._t >= self.num_slots
        if not truncated:
            self._drain_feedback(self._t)
            next_cqi = self._report_cqi_at(self._t)
        else:
            next_cqi = decision_cqi
        outcome = {
            "ack": int(first_ack or 0),  # first-attempt ACK
            "tb_success": int(reward > 0),
            "mcs_used": mcs_used,
            "mcs_lookup": last_mcs_lookup,
            "qm": qm,
            "coderate": coderate,
            "rate_eff": last_rate_eff,
            "modulation": mod_from_qm(qm),
            "sinr_true_db": decision_gamma,
            "sinr_hat_db": decision_sinr_hat,
            "cqi_index": decision_cqi,
            "sinr_eq_db": last_sinr_eq,
            "mi_saturated": mi_sat,
            "tbler": last_tbler,
            "decoded_bits": last_bits,
            "dropped": dropped,
            "truncated_mid_tb": truncated_mid_tb,
            "num_slots": num_tx,
            "num_retx": max(num_tx - 1, 0),
            "harq_feedbacks": list(harq_seq),
            "decision_slot": decision_slot,
        }
        return (
            self._state(next_cqi),
            float(reward),
            False,
            truncated,
            self._info(outcome, cqi_index=next_cqi),
        )

    @classmethod
    def from_config(cls, cfg, phy_abs=None):
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
            "ack_delay_slots",
            "cqi_bler_target",
            "mi_combining_rho",
            "harq_max_retx",
            "drop_penalty",
            "state_num_lags",
        )
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        return cls(phy_abs=phy_abs, **kwargs)
