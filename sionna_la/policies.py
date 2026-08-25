"""
Policies for DownlinkLAEnv.

All of them share the same interface, so the rule-based baselines and an RL
agent are interchangeable in the rollout loop:

    mcs = policy(obs, info)

ILLA/OLLA read the raw quantities they need (linear SINR, allocated REs,
HARQ feedback) out of `info`; the plain float `obs` vector is what an RL
agent would consume.

Sionna's LA controllers return an absolute MCS index, but the environment's
action space starts at 0 and skips the indices its BLER tables do not cover,
so the baselines convert through `info["mcs_min"]`/`["mcs_max"]`.
"""

import numpy as np
from sionna.sys import (
    InnerLoopLinkAdaptation,
    OuterLoopLinkAdaptation,
)


def _to_action(mcs, info):
    lo, hi = info["mcs_min"], info["mcs_max"]
    return int(np.clip(int(mcs), lo, hi)) - lo


class IllaPolicy:
    """Inner-loop LA: highest MCS whose BLER stays under the target."""

    name = "illa"

    def __init__(self, phy_abs, bler_target=0.1):
        self._illa = InnerLoopLinkAdaptation(phy_abs, bler_target=bler_target)

    def reset(self):
        pass  # stateless

    def __call__(self, obs, info):
        mcs = self._illa(
            num_allocated_re=info["num_allocated_re"],
            sinr_eff=info["sinr_eff_lin"],
            mcs_table_index=info["mcs_table_index"],
            mcs_category=info["mcs_category"],
        )
        return _to_action(mcs.item(), info)


class OllaPolicy:
    """Outer-loop LA: ILLA plus a HARQ-driven SINR offset."""

    name = "olla"

    def __init__(self, phy_abs, bler_target=0.1):
        self._phy_abs = phy_abs
        self._bler_target = float(bler_target)
        self._olla = None
        self.reset()

    def reset(self):
        # rebuilt rather than reset so the SINR offset never leaks across episodes
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
    """
    Wraps a base policy and picks a uniformly random MCS with probability
    `epsilon`. Used to add coverage when collecting offline datasets.
    """

    def __init__(self, base, n_actions, epsilon=0.0, seed=None):
        self.base = base
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self.name = f"{getattr(base, 'name', 'policy')}_eps{self.epsilon:g}"

    def reset(self):
        self.base.reset()

    def __call__(self, obs, info):
        # the base policy is always queried so that OLLA's loop keeps tracking
        greedy = self.base(obs, info)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return greedy
