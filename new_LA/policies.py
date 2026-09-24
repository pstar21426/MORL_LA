# ILLA / OLLA / epsilon-greedy baselines (discrete CQI only)
# + delay-free oracles: true SINR → CQI→MCS, or Sionna InnerLoopLinkAdaptation

import numpy as np
import torch
from sionna.phy.utils import db_to_lin
from sionna.sys import InnerLoopLinkAdaptation

from cqi import (
    build_cqi_to_mcs,
    calibrate_cqi_to_sinr_db,
    mcs_from_sinr_db,
    report_cqi,
)


def _to_action(mcs, info):
    lo, hi = info["mcs_min"], info["mcs_max"]
    return int(np.clip(int(mcs), lo, hi)) - lo


def _mcs_from_cqi(cqi_index, cqi_to_mcs):
    q = int(np.clip(int(cqi_index), 0, 15))
    return int(cqi_to_mcs[q])


class IllaPolicy:
    name = "illa"

    def __init__(self, mcs_min, mcs_max, mcs_table_index=1, mcs_category=1):
        self.mcs_min = int(mcs_min)
        self.mcs_max = int(mcs_max)
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min,
            self.mcs_max,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
        )

    def reset(self):
        pass

    def __call__(self, state, info):
        mcs = _mcs_from_cqi(info["cqi_index"], self._cqi_to_mcs)
        return _to_action(mcs, info)


class OllaPolicy:
    # CQI -> SNR_base + offset -> MCS. ACK: +step_up, NACK: -step_up*(1/BLER-1)
    name = "olla"

    def __init__(
        self,
        phy_abs,
        num_allocated_re,
        mcs_min,
        mcs_max,
        mcs_table_index=1,
        mcs_category=1,
        bler_target=0.1,
        step_up_db=None,
        step_down_db=None,
        sinr_min_db=-5.0,
        sinr_max_db=30.0,
        offset_min_db=-20.0,
        offset_max_db=20.0,
    ):
        self.phy_abs = phy_abs
        self.num_allocated_re = num_allocated_re
        self.mcs_min = int(mcs_min)
        self.mcs_max = int(mcs_max)
        self.mcs_table_index = int(mcs_table_index)
        self.mcs_category = int(mcs_category)
        self.bler_target = float(bler_target)
        self.sinr_min_db = float(sinr_min_db)
        self.sinr_max_db = float(sinr_max_db)
        self.offset_min_db = float(offset_min_db)
        self.offset_max_db = float(offset_max_db)

        self.step_up_db = 0.1 if step_up_db is None else float(step_up_db)
        if step_down_db is None:
            self.step_down_db = self.step_up_db * (1.0 / self.bler_target - 1.0)
        else:
            self.step_down_db = float(step_down_db)

        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min,
            self.mcs_max,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        self._cqi_to_sinr_db = calibrate_cqi_to_sinr_db(
            self.phy_abs,
            self._cqi_to_mcs,
            self.num_allocated_re,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
            bler_target=self.bler_target,
            sinr_min_db=self.sinr_min_db,
            sinr_max_db=self.sinr_max_db,
        )
        self.reset()

    def reset(self):
        self._offset_db = 0.0

    def __call__(self, state, info):
        fb_seq = info.get("first_acks") or [-1]
        for fb in fb_seq:
            if int(fb) == -1:
                continue
            if int(fb) == 1:
                self._offset_db += self.step_up_db
            else:
                self._offset_db -= self.step_down_db
        self._offset_db = float(
            np.clip(self._offset_db, self.offset_min_db, self.offset_max_db)
        )

        cqi = int(np.clip(int(info["cqi_index"]), 0, 15))
        base_sinr_db = self._cqi_to_sinr_db[cqi]
        eff_sinr_db = float(
            np.clip(
                base_sinr_db + self._offset_db,
                self.sinr_min_db,
                self.sinr_max_db,
            )
        )
        mcs = mcs_from_sinr_db(
            self.phy_abs,
            eff_sinr_db,
            self.num_allocated_re,
            self.mcs_min,
            self.mcs_max,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
            bler_target=self.bler_target,
        )
        return _to_action(mcs, info)


class IdealIllaCqiPolicy:
    # delay 없는 true SINR → CQI 양자화 → ILLA와 같은 CQI→MCS
    name = "illa_ideal_cqi"

    def __init__(
        self,
        phy_abs,
        num_allocated_re,
        mcs_min,
        mcs_max,
        mcs_table_index=1,
        mcs_category=1,
        bler_target=0.1,
    ):
        self.phy_abs = phy_abs
        self.num_allocated_re = num_allocated_re
        self.mcs_min = int(mcs_min)
        self.mcs_max = int(mcs_max)
        self.mcs_table_index = int(mcs_table_index)
        self.mcs_category = int(mcs_category)
        self.bler_target = float(bler_target)
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min,
            self.mcs_max,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )

    def reset(self):
        pass

    def __call__(self, state, info):
        del state
        sinr_lin = db_to_lin(
            torch.tensor([float(info["sinr_true_db"])], dtype=torch.float32)
        )
        cqi = report_cqi(
            self.phy_abs,
            sinr_lin,
            self.num_allocated_re,
            self._cqi_to_mcs,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
            bler_target=self.bler_target,
        )
        mcs = _mcs_from_cqi(cqi, self._cqi_to_mcs)
        return _to_action(mcs, info)


class IdealIllaSionnaPolicy:
    # delay 없는 연속 SINR → Sionna InnerLoopLinkAdaptation
    name = "illa_ideal_sinr"

    def __init__(
        self,
        phy_abs,
        num_allocated_re,
        mcs_min,
        mcs_max,
        mcs_table_index=1,
        mcs_category=1,
        bler_target=0.1,
    ):
        if torch.is_tensor(num_allocated_re):
            self.num_allocated_re = int(num_allocated_re.reshape(-1)[0].item())
        else:
            self.num_allocated_re = int(num_allocated_re)
        self.mcs_min = int(mcs_min)
        self.mcs_max = int(mcs_max)
        self.mcs_table_index = int(mcs_table_index)
        self.mcs_category = int(mcs_category)
        self._illa = InnerLoopLinkAdaptation(
            phy_abs, bler_target=float(bler_target), fill_mcs_value=self.mcs_min
        )

    def reset(self):
        pass

    def __call__(self, state, info):
        del state
        sinr_lin = db_to_lin(
            torch.tensor([float(info["sinr_true_db"])], dtype=torch.float32)
        )
        n_re = torch.tensor([self.num_allocated_re], dtype=torch.int32)
        mcs = self._illa(
            sinr_eff=sinr_lin,
            num_allocated_re=n_re,
            mcs_table_index=self.mcs_table_index,
            mcs_category=self.mcs_category,
        )
        return _to_action(int(mcs.reshape(-1)[0].item()), info)


class EpsilonGreedyPolicy:
    def __init__(self, base, n_actions, epsilon=0.0, seed=None):
        self.base = base
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self.name = f"{getattr(base, 'name', 'policy')}_eps{self.epsilon:g}"

    def reset(self):
        self.base.reset()

    def __call__(self, state, info):
        greedy = self.base(state, info)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return greedy


def make_baseline_policy(name, env, bler_target=0.1, olla_step_up_db=None):
    if name == "illa":
        return IllaPolicy(
            mcs_min=env.mcs_min,
            mcs_max=env.mcs_max,
            mcs_table_index=env.mcs_table_index,
            mcs_category=env.mcs_category,
        )
    if name == "olla":
        kwargs = dict(
            phy_abs=env.phy_abs,
            num_allocated_re=env._num_re_t,
            mcs_min=env.mcs_min,
            mcs_max=env.mcs_max,
            mcs_table_index=env.mcs_table_index,
            mcs_category=env.mcs_category,
            bler_target=bler_target,
            sinr_min_db=env.sinr_min_db,
            sinr_max_db=env.sinr_max_db,
        )
        if olla_step_up_db is not None:
            kwargs["step_up_db"] = float(olla_step_up_db)
        return OllaPolicy(**kwargs)
    if name in ("illa_ideal_cqi", "illa_ideal_sinr"):
        kwargs = dict(
            phy_abs=env.phy_abs,
            num_allocated_re=env._num_re_t,
            mcs_min=env.mcs_min,
            mcs_max=env.mcs_max,
            mcs_table_index=env.mcs_table_index,
            mcs_category=env.mcs_category,
            bler_target=bler_target,
        )
        if name == "illa_ideal_cqi":
            return IdealIllaCqiPolicy(**kwargs)
        return IdealIllaSionnaPolicy(**kwargs)
    raise ValueError(f"unknown baseline: {name}")
