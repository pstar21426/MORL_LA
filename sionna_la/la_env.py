# Gym env: 1 step = 1 TB (초기 전송 + HARQ 재전송).
# State: [cqi_n, past_cqi×L, past_m×L, past_b×L] (unseen = -1). True SINR는 state에 포함되지 않음.
# HARQ-IR: 초기 전송은 SNR 사용; 재전송 시 SNR_eff = I^{-1}(mean I), 같은 Qm, R/n, 초기 전송 TBS를 사용.

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

# 맨 앞 요소만 int로 반환
def _item_int(x):
    if torch.is_tensor(x):
        return int(x.reshape(-1)[0].item())
    return int(x)

# 과거 CQI, MCS, ACK 저장용 버퍼
class DecisionHistory:
    def __init__(self, num_lags=3):
        self.num_lags = max(1, int(num_lags))
        self.reset()

    # 버퍼 초기화(현재 hat sinr, self.cqi[3], self.mcs[3], self.ack[3])
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
        cqi_delay_slots=None,
        cqi_bler_target=0.1,
        ack_delay_slots=0,
        harq_max_retx=2,
        drop_penalty=6.0,
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
        self.harq = HarqProcess(max_retx=harq_max_retx)
        self.hist = DecisionHistory(num_lags=state_num_lags)

        self.mcs_min, self.mcs_max = _MCS_RANGE[self.mcs_table_index]
        self._mcs_span = self.mcs_max - self.mcs_min
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min, self.mcs_max, mcs_table_index=self.mcs_table_index
        )
        self.action_space = spaces.Discrete(self._mcs_span + 1)

        n = self.hist.num_lags
        state_dim = 1 + 3 * n # 10차원
        self.state_space = spaces.Box(
            low=np.full(state_dim, _UNSEEN, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self.state_space

        self._num_re_t = torch.tensor([self.num_allocated_re], dtype=torch.int32)
        self._sinr_true_db = np.zeros(self.num_slots)
        self._sinr_fb_db = np.zeros(self.num_slots)
        self._delay_used = 0
        self._t = 0
        self._last_decision_slot = 0
        self._ever_decided = False
        self._pending_harq = [-1]
        self._fb_queue = deque()
        self._ack_seed = 2
    
    # --------------------------------------------------------------
    # Sionna TBLER 계산, 0~2 인덱스가 테이블에 없어서 mcs_min으로 보정
    def mcs_from_action(self, action):
        return int(action) + self.mcs_min

    def action_from_mcs(self, mcs_index):
        return int(np.clip(mcs_index, self.mcs_min, self.mcs_max)) - self.mcs_min
    # --------------------------------------------------------------
    
    def _sinr_hat_db(self, slot):
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_fb_db[idx])

    # 얘도 위랑 똑같음
    def _gamma_at(self, slot):
        idx = min(max(slot, 0), self.num_slots - 1)
        return float(self._sinr_true_db[idx])

    # 현재 t 받으면 CQI 인덱스 반환
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

    # 과거 CQI, MCS, ACK 저장용 버퍼에서 현재 slot보다 작은 것들을 모두 제거
    # _fb_queue: (slot, first_ack, cqi_n, mcs_norm)
    
    def _drain_feedback(self, now_slot):
        drained = []
        while self._fb_queue and self._fb_queue[0][0] <= now_slot:
            _, first_ack, cqi_n, mcs_norm = self._fb_queue.popleft()
            self.hist.push(cqi_n, mcs_norm, int(first_ack))
            drained.append(int(first_ack))
        self._pending_harq = drained if drained else [-1]

    # 현재 slot에서의 state 반환
    # 평상시 시뮬레이션에서는 _state()만 호출하고, 테스트 시에는 특정 cqi_index를 넣으면 해당 cqi_index의 state를 반환
    def _state(self, cqi_index=None):
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        return np.array(
            [normalize_cqi(cqi_index), *self.hist.cqi, *self.hist.mcs, *self.hist.ack],
            dtype=np.float32,
        )

    # 현재 slot에서의 info 반환
    # 평상시 시뮬레이션에서는 _info()만 호출하고, 테스트 시에는 특정 cqi_index를 넣으면 해당 cqi_index의 info를 반환
    def _info(self, outcome=None, cqi_index=None):
        if cqi_index is None:
            cqi_index = self._report_cqi_at(self._t)
        info = {
            "sinr_true_db": self._gamma_at(self._t),
            "sinr_hat_db": self._sinr_hat_db(self._t),
            "cqi_index": int(cqi_index),
            "cqi_norm": normalize_cqi(cqi_index),
            "harq_feedbacks": list(self._pending_harq),
            "mcs_table_index": self.mcs_table_index,
            "mcs_category": self.mcs_category,
            "mcs_min": self.mcs_min,
            "mcs_max": self.mcs_max,
            "slot": self._t,
            "delta_tau": (
                0.0
                if not self._ever_decided
                else float(self._t - self._last_decision_slot)
            ),
            "cqi_delay_slots": self._delay_used,
            "ack_delay_slots": self.ack_delay_slots,
        }
        # 테스트 시에는 특정 outcome을 넣으면 해당 outcome의 info를 반환
        if outcome is not None:
            info["outcome"] = outcome
        return info

    # 환경 초기화
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 테스트 시에는 특정 seed를 넣으면 해당 seed의 sinr_true_db를 생성
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
        # CQI 노이즈 추가
        self._sinr_fb_db, self._delay_used = add_cqi_noise(
            self._sinr_true_db,
            noise_std_db=self.cqi_noise_std_db,
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
        self._last_decision_slot = 0 # 마지막 결정 시점
        self._ever_decided = False
        self._pending_harq = [-1]
        self._fb_queue.clear()
        self._drain_feedback(0)
        cqi_index = self._report_cqi_at(self._t)
        return self._state(cqi_index), self._info(cqi_index=cqi_index)

    # 현재 slot에서의 PHY 시뮬레이션 한 번 실행
    # action에 대한 처리 핵심
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
        tbler = float(tbler_t.reshape(-1)[0].item()) # tbler(0에서 1사이 값) 반환값만 사용
        # 위의 tbler 값을 이용, [0,1] 사이 값을 랜덤으로 추출해서 tbler보다 작으면 0, 크면 1 반환
        ack = 0 if _slot_uniform(self._ack_seed, self._t) < tbler else 1
        return ack, tbler, ack * self.harq.tbs, sinr_eq_db

    def step(self, action):
        # 현재 slot이 num_slots보다 크거나 같으면 종료
        if self._t >= self.num_slots:
            return self._state(), 0.0, False, True, self._info()

        decision_slot = self._t
        decision_cqi = self._report_cqi_at(self._t)
        decision_cqi_n = normalize_cqi(decision_cqi)
        decision_sinr_hat = self._sinr_hat_db(self._t)
        decision_gamma = self._gamma_at(self._t)

        mcs_used = self.mcs_from_action(action)
        qm, coderate = mcs_qm_rate(
            mcs_used, self.mcs_min, self.mcs_max, self.mcs_table_index
        )
        tbs_t, cb_t, ncb_t = tb_layout_from_mcs(
            mcs_used,
            self._num_re_t,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        # 이따가 _phy_once() 메서드 호출하기 위한 정보 저장
        self.harq.start_transmission(
            mcs_used, qm, coderate, _item_int(tbs_t), _item_int(cb_t), _item_int(ncb_t)
        )

        reward = 0.0
        harq_seq = []
        first_ack = 0
        dropped = False
        truncated_mid_tb = False
        last_tbler = 0.0
        first_tbler = 0.0
        last_bits = 0
        last_sinr_eq = decision_gamma

        while self._t < self.num_slots:
            ack, tbler, bits, sinr_eq = self._phy_once()
            harq_seq.append(ack)
            last_tbler, last_bits, last_sinr_eq = tbler, bits, sinr_eq
            if len(harq_seq) == 1:
                first_ack = ack
                first_tbler = tbler
            self._t += 1

            if ack == 1:
                reward = qm * coderate / len(harq_seq) # 성공 시 보상 qm * coderate / 사용 슬롯 수
                # max throughput을 목적으로 그냥 qm * coderate을 사용하려 했는데, 그러면 에이전트가 과도한 재전송을 하게 됨
                self.harq.reset()
                break

            dropped = self.harq.on_nack()
            if dropped:
                reward = -self.drop_penalty # 실패 시 보상(상수)
                break
        else:
            truncated_mid_tb = True
            self.harq.reset()

        num_tx = len(harq_seq)
        mcs_norm = (mcs_used - self.mcs_min) / self._mcs_span
        if harq_seq:
            self._fb_queue.append(
                (self._t + self.ack_delay_slots, first_ack, decision_cqi_n, mcs_norm)
            )

        self._last_decision_slot = decision_slot
        self._ever_decided = True

        truncated = self._t >= self.num_slots
        self._drain_feedback(self._t)
        next_cqi = self._report_cqi_at(self._t)

        outcome = {
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
            "harq_max_retx",
            "drop_penalty",
            "state_num_lags",
        )
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        return cls(phy_abs=phy_abs, **kwargs)
