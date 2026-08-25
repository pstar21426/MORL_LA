# ILLA / OLLA / epsilon-greedy baselines

import numpy as np
from sionna.sys import InnerLoopLinkAdaptation, OuterLoopLinkAdaptation


def _to_action(mcs, info):
    # MCS index -> Discrete action (0-based)
    lo, hi = info["mcs_min"], info["mcs_max"]
    return int(np.clip(int(mcs), lo, hi)) - lo


class IllaPolicy:
    name = "illa"

    def __init__(self, phy_abs, bler_target=0.1):
        self._illa = InnerLoopLinkAdaptation(phy_abs, bler_target=bler_target)

    def reset(self):
        pass

    def __call__(self, obs, info):
        mcs = self._illa(
            num_allocated_re=info["num_allocated_re"],
            sinr_eff=info["sinr_eff_lin"],
            mcs_table_index=info["mcs_table_index"],
            mcs_category=info["mcs_category"],
        )
        return _to_action(mcs.item(), info)


class OllaPolicy:
    name = "olla"

    def __init__(self, phy_abs, bler_target=0.1):
        self._phy_abs = phy_abs
        self._bler_target = float(bler_target)
        self.reset()

    def reset(self):
        # rebuild each episode so SINR offset does not leak
        self._olla = OuterLoopLinkAdaptation(
            self._phy_abs, num_ut=1, bler_target=self._bler_target
        )

    def __call__(self, obs, info):
        mcs = self._olla(
            num_allocated_re=info["num_allocated_re"],
            sinr_eff=info["sinr_eff_lin"],
            mcs_table_index=info["mcs_table_index"],
            mcs_category=info["mcs_category"],
            harq_feedback=info["harq_feedback"],
        )
        return _to_action(mcs.item(), info)


class EpsilonGreedyPolicy:
    def __init__(self, base, n_actions, epsilon=0.0, seed=None):
        self.base = base
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self.name = f"{getattr(base, 'name', 'policy')}_eps{self.epsilon:g}"

    def reset(self):
        self.base.reset()

    def __call__(self, obs, info):
        # always call base so OLLA internal state keeps updating
        greedy = self.base(obs, info)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return greedy
