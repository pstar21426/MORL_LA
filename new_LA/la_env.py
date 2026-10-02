# Gym env: 1 step = 1 TB (초기 전송 + HARQ 재전송). 버퍼가 비면 1슬롯 idle.
# State: [cqi_n, past_cqi×L, past_m×L, past_b×L, queue/capacity] (unseen = -1).
# 버퍼는 대기 비트. RB 수는 고정. MCS는 action임.
# 상태의 큐는 비트/용량. 보상은 비트/RE.
# ACK·HARQ 포기 때 min(큐, 코드블록 비트)를 뺌. 코드블록 비트는 TB CRC를 포함.
# 슬롯 순서: 도착을 큐에 넣은 뒤 관측하고 전송한다.
# 재전송은 gap 슬롯을 비운 뒤에 한다. gap 동안에는 비트만 도착한다.
# HARQ-IR: 초기 전송은 그 슬롯 SINR. 재전송은 SNR_eff = I^{-1}(mean I), 같은 Qm, R/n

from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sionna.phy import config as sionna_config
from sionna.phy.utils import db_to_lin
from sionna.sys import PHYAbstraction

from channel import add_cqi_noise, generate_sinr_db_trace
from cqi import (
    build_cqi_to_mcs,
    mcs_for_ir_rate,
    mcs_qm_rate,
    normalize_cqi,
    report_cqi,
    tb_layout_from_mcs,
    tbler_from_phy,
)
from harq import HarqProcess

# Sionna BLER 표에 없는 낮은 MCS는 뺀다. table 1은 3–28, table 2는 2–27.
_MCS_RANGE = {1: (3, 28), 2: (2, 27)}
_UNSEEN = -1.0

# PHY seed 설정
def seed_phy(seed):
    if seed is None:
        return
    seed = int(seed)
    sionna_config.seed = seed
    torch.manual_seed(seed)


def _slot_uniform(ack_seed, slot):
    ss = np.random.SeedSequence([int(ack_seed) & 0xFFFFFFFF, int(slot) & 0xFFFFFFFF])
    return float(np.random.default_rng(ss).random())

def _item_int(x):
    if torch.is_tensor(x):
        return int(x.reshape(-1)[0].item())
    return int(x)

# 과거 CQI, MCS, ACK 저장용 버퍼
class DecisionHistory:
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
        num_slots=1000,
        mcs_table_index=1,
        mcs_category=1,
        num_allocated_re=300,
        sinr_mean_db=10.0,
        sinr_ar_rho=0.95,
        sinr_innov_std_db=2.5,
        sinr_min_db=-5.0,
        sinr_max_db=30.0,
        sinr_mean_range_db=None,
        sinr_mean_change_prob=0.0,
        cqi_noise_std_db=1.5,
        cqi_bias_db=0.0,
        cqi_delay_slots=None,
        cqi_bler_target=0.1,
        ack_delay_slots=0,
        harq_max_retx=2,
        drop_penalty=0.0,
        harq_retx_gap_slots=3,
        state_num_lags=3,
        arrival_bits_min=400,
        arrival_bits_max=800,
        queue_capacity=15000,
        overflow_penalty=0.0,
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
        self.cqi_bias_db = float(cqi_bias_db)
        self.cqi_delay_slots = cqi_delay_slots
        self.cqi_bler_target = float(cqi_bler_target)
        self.ack_delay_slots = max(0, int(ack_delay_slots))
        self.drop_penalty = drop_penalty
        self.harq_retx_gap_slots = max(0, int(harq_retx_gap_slots))
        self.arrival_bits_min = int(arrival_bits_min)
        self.arrival_bits_max = max(self.arrival_bits_min, int(arrival_bits_max))
        self.queue_capacity = max(1, int(queue_capacity))
        self.overflow_penalty = float(overflow_penalty)

        self.phy_abs = phy_abs if phy_abs is not None else PHYAbstraction()
        self.harq = HarqProcess(max_retx=harq_max_retx)
        self.hist = DecisionHistory(num_lags=state_num_lags)

        self.mcs_min, self.mcs_max = _MCS_RANGE[self.mcs_table_index]
        self._mcs_span = self.mcs_max - self.mcs_min
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min,
            self.mcs_max,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        self.action_space = spaces.Discrete(self._mcs_span + 1)

        n = self.hist.num_lags
        state_dim = 2 + 3 * n  # CQI + lags + queue
        low = np.full(state_dim, _UNSEEN, dtype=np.float32)
        low[-1] = 0.0
        self.state_space = spaces.Box(
            low=low,
            high=np.ones(state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self.state_space

        self._num_re_t = torch.tensor([self.num_allocated_re], dtype=torch.int32)
        self._sinr_true_db = np.zeros(self.num_slots)
        self._sinr_fb_db = np.zeros(self.num_slots)
        self._delay_used = 0
        self._t = 0
        self._pending_harq = [-1]
        self._fb_queue = deque()
        self._ack_seed = 2
        self._q = 0

    def mcs_from_action(self, action):
        return int(action) + self.mcs_min

    def action_from_mcs(self, mcs_index):
        return int(np.clip(mcs_index, self.mcs_min, self.mcs_max)) - self.mcs_min

    def _sinr_hat_db(self, slot):
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_fb_db[idx])

    def _gamma_at(self, slot):
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_true_db[idx]        )

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

    # 공개 시점이 된 초전송 ACK를 히스토리에 넣음. 없으면 [-1].
    # _fb_queue: (slot, first_ack, cqi_n, mcs_norm)
    def _drain_feedback(self, now_slot):
        drained = []
        while self._fb_queue and self._fb_queue[0][0] <= now_slot:
            _, first_ack, cqi_n, mcs_norm = self._fb_queue.popleft()
            self.hist.push(cqi_n, mcs_norm, int(first_ack))
            drained.append(int(first_ack))
        self._pending_harq = drained if drained else [-1]

    def _state(self, cqi_index=None):
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        q_norm = float(self._q) / float(self.queue_capacity)
        return np.array(
            [
                normalize_cqi(cqi_index),
                *self.hist.cqi,
                *self.hist.mcs,
                *self.hist.ack,
                q_norm,
            ],
            dtype=np.float32,
        )

    def _info(self, outcome=None, cqi_index=None):
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        info = {
            "sinr_true_db": self._gamma_at(self._t),
            "sinr_hat_db": self._sinr_hat_db(self._t),
            "cqi_index": int(cqi_index),
            "cqi_norm": normalize_cqi(cqi_index),
            # delay가 끝난 초전송 ACK들. 없으면 [-1]. outcome["harq_seq"]와 다름
            "first_acks": list(self._pending_harq),
            "mcs_table_index": self.mcs_table_index,
            "mcs_category": self.mcs_category,
            "mcs_min": self.mcs_min,
            "mcs_max": self.mcs_max,
            "slot": self._t,
            "cqi_delay_slots": self._delay_used,
            "ack_delay_slots": self.ack_delay_slots,
            "queue": int(self._q),
        }
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
            bias_db=self.cqi_bias_db,
            delay_slots=self.cqi_delay_slots,
            seed=None if seed is None else seed + 1,
            sinr_min_db=self.sinr_min_db,
            sinr_max_db=self.sinr_max_db,
        )
        self.harq.reset()
        self.hist.reset()
        self._ack_seed = (
            int(seed) + 2
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        self._t = 0
        self._pending_harq = [-1]
        self._fb_queue.clear()
        self._q = 0
        self._drain_feedback(0)
        self._arrive()
        cqi_index = self._report_cqi_at(self._t)
        return self._state(cqi_index), self._info(cqi_index=cqi_index)

    def _phy_once(self):
        sinr_eq_db = self.harq.accumulate(self._gamma_at(self._t))
        # 초기 전송인 경우 초기 MCS 사용
        if self.harq.k == 0:
            mcs_lookup = self.harq.mcs
        else:
            mcs_lookup = mcs_for_ir_rate(
                self.harq.rate_eff,
                self.harq.qm,
                self.mcs_min,
                self.mcs_max,
                mcs_table_index=self.mcs_table_index,
                mcs_category=self.mcs_category,
            )
        tbler_t, _ = tbler_from_phy(
            self.phy_abs,
            mcs_lookup,
            db_to_lin(torch.tensor([sinr_eq_db], dtype=torch.float32)),
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
            cb_size=self.harq.cb_size,
            num_cb=self.harq.num_cb,
        )
        tbler = float(tbler_t.reshape(-1)[0].item())
        ack = 0 if _slot_uniform(self._ack_seed, self._t) < tbler else 1
        return ack, tbler, sinr_eq_db

    def _se(self, bits):
        return float(bits) / float(self.num_allocated_re)

    # 슬롯마다 [min, max] 비트가 균등 도착. 넘친 비트 수를 반환
    def _arrive(self):
        bits = int(self.np_random.integers(self.arrival_bits_min, self.arrival_bits_max + 1))
        space = self.queue_capacity - self._q
        if bits <= space:
            self._q += bits
            return 0
        self._q = self.queue_capacity
        return int(bits - space)

    def step(self, action):
        # 현재 slot이 num_slots보다 크거나 같으면 종료됨
        if self._t >= self.num_slots:
            cqi_index = self._report_cqi_at(self._t)
            return (
                self._state(cqi_index),
                0.0,
                False,
                True,
                self._info(self._idle_outcome(cqi_index), cqi_index=cqi_index),
            )

        if self._q <= 0:
            return self._idle_slot()

        decision_slot = self._t
        decision_cqi = self._report_cqi_at(self._t)
        decision_cqi_n = normalize_cqi(decision_cqi)
        decision_sinr_hat = self._sinr_hat_db(self._t)
        decision_gamma = self._gamma_at(self._t)

        mcs_used = self.mcs_from_action(action)
        qm, coderate = mcs_qm_rate(
            mcs_used,
            self.mcs_min,
            self.mcs_max,
            self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        tbs_t, cb_t, ncb_t = tb_layout_from_mcs(
            mcs_used,
            self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        tbs = _item_int(tbs_t)
        self.harq.start_transmission(
            mcs_used, qm, coderate, tbs, _item_int(cb_t), _item_int(ncb_t)
        )
        payload = min(self._q, tbs)

        reward = 0.0
        n_overflow = 0
        discard_bits = 0
        harq_seq = []
        first_ack = 0
        dropped = False
        truncated_mid_tb = False
        last_tbler = 0.0
        first_tbler = 0.0
        last_sinr_eq = decision_gamma

        # 이 슬롯 도착은 이미 큐에 있음. 전송 후 다음 슬롯 시작에 도착을 넣음
        while self._t < self.num_slots:
            ack, tbler, sinr_eq = self._phy_once()
            harq_seq.append(ack)
            last_tbler, last_sinr_eq = tbler, sinr_eq
            if len(harq_seq) == 1:
                first_ack = ack
                first_tbler = tbler

            finished = False
            if ack == 1:
                reward = self._se(payload)
                self._q = max(0, self._q - payload)
                self.harq.reset()
                finished = True
            else:
                dropped = self.harq.on_nack()
                if dropped:
                    reward = -self.drop_penalty * self._se(payload)
                    discard_bits = payload
                    self._q = max(0, self._q - payload)
                    finished = True

            self._t += 1
            if self._t < self.num_slots:
                n_overflow += self._arrive()
            if finished:
                break
            if self._t >= self.num_slots:
                truncated_mid_tb = True
                self.harq.reset()
                break
            # 재전송 전에 슬롯을 비움. 도착만 있고 전송은 없음.
            gap_open = True
            for _ in range(self.harq_retx_gap_slots):
                self._t += 1
                if self._t >= self.num_slots:
                    gap_open = False
                    break
                n_overflow += self._arrive()
            if not gap_open:
                truncated_mid_tb = True
                self.harq.reset()
                break
        else:
            truncated_mid_tb = True
            self.harq.reset()

        bit_reward = reward
        reward = bit_reward - self.overflow_penalty * self._se(n_overflow)

        num_tx = len(harq_seq)
        slots_used = self._t - decision_slot
        mcs_norm = (mcs_used - self.mcs_min) / self._mcs_span
        if harq_seq:
            self._fb_queue.append(
                (self._t + self.ack_delay_slots, first_ack, decision_cqi_n, mcs_norm)
            )

        truncated = self._t >= self.num_slots
        self._drain_feedback(self._t)
        next_cqi = self._report_cqi_at(self._t)

        return (
            self._state(next_cqi),
            float(reward),
            False,
            truncated,
            self._info(
                self._tb_outcome(
                    first_ack=first_ack,
                    reward=bit_reward,
                    mcs_used=mcs_used,
                    qm=qm,
                    coderate=coderate,
                    decision_gamma=decision_gamma,
                    decision_sinr_hat=decision_sinr_hat,
                    decision_cqi=decision_cqi,
                    last_sinr_eq=last_sinr_eq,
                    first_tbler=first_tbler,
                    last_tbler=last_tbler,
                    dropped=dropped,
                    truncated_mid_tb=truncated_mid_tb,
                    num_tx=num_tx,
                    slots_used=slots_used,
                    harq_seq=harq_seq,
                    decision_slot=decision_slot,
                    n_overflow=n_overflow,
                    discard_bits=discard_bits,
                    n_re=self.num_allocated_re,
                ),
                cqi_index=next_cqi,
            ),
        )

    def _idle_slot(self):
        decision_slot = self._t
        self._t += 1
        n_overflow = self._arrive() if self._t < self.num_slots else 0
        truncated = self._t >= self.num_slots
        self._drain_feedback(self._t)
        cqi_index = self._report_cqi_at(self._t)
        reward = -self.overflow_penalty * self._se(n_overflow)
        outcome = self._idle_outcome(cqi_index, n_overflow)
        outcome["reward"] = float(reward)
        outcome["decision_slot"] = int(decision_slot)
        return (
            self._state(cqi_index),
            float(reward),
            False,
            truncated,
            self._info(outcome, cqi_index=cqi_index),
        )

    def _idle_outcome(self, cqi_index, n_overflow=0):
        gamma = self._gamma_at(self._t)
        return self._tb_outcome(
            first_ack=0,
            reward=0.0,
            mcs_used=self.mcs_min,
            qm=0,
            coderate=0.0,
            decision_gamma=gamma,
            decision_sinr_hat=self._sinr_hat_db(self._t),
            decision_cqi=int(cqi_index),
            last_sinr_eq=gamma,
            first_tbler=0.0,
            last_tbler=0.0,
            dropped=False,
            truncated_mid_tb=False,
            num_tx=0,
            harq_seq=[],
            decision_slot=int(self._t),
            n_overflow=n_overflow,
            n_re=self.num_allocated_re,
            idle=True,
        )

    @staticmethod
    def _tb_outcome(
        *,
        first_ack,
        reward,
        mcs_used,
        qm,
        coderate,
        decision_gamma,
        decision_sinr_hat,
        decision_cqi,
        last_sinr_eq,
        first_tbler,
        last_tbler,
        dropped,
        truncated_mid_tb,
        num_tx,
        harq_seq,
        slots_used=None,
        decision_slot,
        n_overflow=0,
        discard_bits=0,
        n_re=1,
        idle=False,
    ):
        lost_bits = int(n_overflow) + int(discard_bits)
        delivered_se = float(reward) if reward > 0 else 0.0
        return {
            "ack": first_ack,
            "tb_success": int(reward > 0),
            "mcs_used": mcs_used,
            "qm": qm,
            "coderate": coderate,
            "sinr_true_db": decision_gamma,
            "sinr_hat_db": decision_sinr_hat,
            "cqi_index": decision_cqi,
            "sinr_eq_db": last_sinr_eq,
            "tbler": first_tbler,
            "tbler_first": first_tbler,
            "tbler_last": last_tbler,
            "dropped": dropped,
            "truncated_mid_tb": truncated_mid_tb,
            "num_slots": int(num_tx if slots_used is None else slots_used),
            "num_retx": max(num_tx - 1, 0),
            "harq_seq": list(harq_seq),
            "decision_slot": decision_slot,
            "n_overflow": int(n_overflow),
            "discard_bits": int(discard_bits),
            "lost_bits": lost_bits,
            "lost_se": float(lost_bits) / float(max(n_re, 1)),
            "delivered_se": delivered_se,
            "idle": bool(idle),
        }

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
            "cqi_bias_db",
            "cqi_delay_slots",
            "ack_delay_slots",
            "cqi_bler_target",
            "harq_max_retx",
            "drop_penalty",
            "harq_retx_gap_slots",
            "state_num_lags",
            "arrival_bits_min",
            "arrival_bits_max",
            "queue_capacity",
            "overflow_penalty",
        )
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        return cls(phy_abs=phy_abs, **kwargs)
