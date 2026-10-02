# Same learner (no bootstrap, argmax E[r] - rho E[tau] over supported actions),
# different behavior data. Does local exploration around OLLA beat uniform epsilon?

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from _probe_rate_bandit import evaluate, fit
from la_env import DownlinkLAEnv, seed_phy
from policies import EpsilonGreedyPolicy, make_baseline_policy
from train_mopo import load_dataset, normalize_fit

MIN_COUNT = 10


class LocalExplore:
    # With prob p, move the base action by a uniform offset in [-k, k] \ {0}.
    def __init__(self, base, n_actions, p, k, seed):
        self.base, self.n, self.p, self.k = base, n_actions, p, k
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.base.reset()

    def __call__(self, state, info):
        a = int(self.base(state, info))
        if self.rng.random() < self.p:
            d = int(self.rng.choice([i for i in range(-self.k, self.k + 1) if i != 0]))
            a = int(np.clip(a + d, 0, self.n - 1))
        return a


def collect(env, makers, episodes, seed0):
    obs, act, rew, nxt, slots = [], [], [], [], []
    for p_i, make in enumerate(makers):
        for ep in range(episodes):
            seed = seed0 + p_i * episodes + ep
            pol = make(seed)
            pol.reset()
            seed_phy(seed)
            s, info = env.reset(seed=seed)
            done = False
            while not done:
                a = int(pol(s, info))
                ns, r, term, trunc, info = env.step(a)
                done = term or trunc
                obs.append(s); act.append(a); rew.append(r); nxt.append(ns)
                slots.append(max(int(info["outcome"]["num_slots"]), 1))
                s = ns
    return {
        "observations": np.asarray(obs, np.float32), "actions": np.asarray(act, np.int64),
        "rewards": np.asarray(rew, np.float32), "next_observations": np.asarray(nxt, np.float32),
        "slots": np.asarray(slots, np.float32),
    }


def score(env, data, label):
    mean, std = normalize_fit(data["observations"])
    cqi = np.rint(data["observations"][:, 0] * 15).astype(int)
    cnt = np.zeros((16, 26), dtype=np.int64)
    np.add.at(cnt, (cqi, data["actions"]), 1)
    sup = cnt >= MIN_COUNT
    torch.manual_seed(0)
    net = fit(data, mean, std)
    rho = 1.0
    for _ in range(2):
        r, sd, rho = evaluate(env, net, mean, std, sup, rho)
    print(f"{label:42s} N={len(data['rewards']):5d} supported cells={int(sup.sum()):3d}  "
          f"return={r:7.1f} ±{sd:5.1f}", flush=True)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    env = DownlinkLAEnv.from_config(cfg, phy_abs=PHYAbstraction())
    bler, step, n = float(cfg["bler_target"]), cfg.get("olla_step_up_db"), env.action_space.n

    def olla():
        return make_baseline_policy("olla", env, bler_target=bler, olla_step_up_db=step)

    def illa():
        return make_baseline_policy("illa", env, bler_target=bler)

    data, _ = load_dataset(ROOT / "datasets" / "offline_illa_olla.npz")
    score(env, data, "current: ILLA+OLLA, uniform eps 0.1, 10+10 ep")

    runs = [
        ("ILLA+OLLA, uniform eps 0.3, 10+10 ep",
         [lambda s: EpsilonGreedyPolicy(illa(), n, 0.3, s), lambda s: EpsilonGreedyPolicy(olla(), n, 0.3, s)], 10),
        ("OLLA only, local +-3 p=0.3, 20 ep",
         [lambda s: LocalExplore(olla(), n, 0.3, 3, s)], 20),
        ("OLLA only, local +-3 p=0.3, 60 ep",
         [lambda s: LocalExplore(olla(), n, 0.3, 3, s)], 60),
    ]
    for label, makers, eps in runs:
        score(env, collect(env, makers, eps, 0), label)


if __name__ == "__main__":
    main()
