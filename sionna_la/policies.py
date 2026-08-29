# ILLA / OLLA / epsilon-greedy baselines (discrete CQI only)

import numpy as np

from cqi import build_cqi_to_mcs, calibrate_cqi_to_sinr_db, mcs_from_sinr_db


def _to_action(mcs, info):
    lo, hi = info["mcs_min"], info["mcs_max"]
    return int(np.clip(int(mcs), lo, hi)) - lo


def _mcs_from_cqi(cqi_index, cqi_to_mcs):
    q = int(np.clip(int(cqi_index), 0, 15))
    return int(cqi_to_mcs[q])


class IllaPolicy:
    """
    Inner-loop LA: reported CQI -> MCS (3GPP-style table lookup).
    Uses discrete CQI only (same info UE would report).
    """

    name = "illa"

    def __init__(self, mcs_min, mcs_max, mcs_table_index=1, bler_target=0.1):
        self.mcs_min = int(mcs_min)
        self.mcs_max = int(mcs_max)
        self.bler_target = float(bler_target)
        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min, self.mcs_max, mcs_table_index=mcs_table_index
        )

    def reset(self):
        pass

    def __call__(self, state, info):
        mcs = _mcs_from_cqi(info["cqi_index"], self._cqi_to_mcs)
        return _to_action(mcs, info)


class OllaPolicy:
    """
    Outer-loop LA (srsRAN-style): CQI -> SNR_base, add SNR offset, map to MCS.
    ACK  -> offset += step_up_db   (more aggressive)
    NACK -> offset -= step_down_db (more conservative)
    step_down_db = step_up_db * (1/BLER - 1)
    """

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
        sinr_min_db=-15.0,
        sinr_max_db=35.0,
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

        self.step_up_db = 0.001 if step_up_db is None else float(step_up_db)
        if step_down_db is None:
            self.step_down_db = self.step_up_db * (1.0 / self.bler_target - 1.0)
        else:
            self.step_down_db = float(step_down_db)

        self._cqi_to_mcs = build_cqi_to_mcs(
            self.mcs_min, self.mcs_max, mcs_table_index=self.mcs_table_index
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
        fb_seq = info.get("harq_feedbacks") or [-1]
        if fb_seq and fb_seq[0] != -1:
            if fb_seq[0] == 1:
                self._offset_db += self.step_up_db
            else:
                self._offset_db -= self.step_down_db

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
    """Build ILLA/OLLA from env MCS table settings."""
    if name == "illa":
        return IllaPolicy(
            mcs_min=env.mcs_min,
            mcs_max=env.mcs_max,
            mcs_table_index=env.mcs_table_index,
            bler_target=bler_target,
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
    raise ValueError(f"unknown baseline: {name}")
