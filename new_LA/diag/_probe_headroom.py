# Held-out B_del for baselines, simple heuristics, a genie, and the saved DDQN / MOPO.

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.phy.utils import db_to_lin
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from cqi import mcs_from_sinr_db, tb_layout_from_mcs, tbler_from_phy
from ddqn import QNet
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from train_mopo import DiscreteQ, apply_norm

SEEDS = (201, 202, 203, 204, 205)


def run(env, pol, seed):
    if hasattr(pol, "reset"):
        pol.reset()
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    ret, n, nack, slots, mcs_sum, ovf = 0.0, 0, 0, 0, 0, 0
    done = False
    while not done:
        a = int(pol(state, info))
        state, r, term, trunc, info = env.step(a)
        done = term or trunc
        out = info["outcome"]
        ret += r
        ovf += out["n_overflow"]
        if out.get("idle"):
            continue
        n += 1
        nack += 1 - int(out["ack"])
        slots += int(out["num_slots"])
        mcs_sum += int(out["mcs_used"])
    return ret, nack / max(n, 1), mcs_sum / max(n, 1), slots / max(n, 1), ovf / env.num_allocated_re


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler = float(cfg["bler_target"])
    step = cfg.get("olla_step_up_db")
    gap = env.harq_retx_gap_slots + 1

    tbs = {}
    for m in range(env.mcs_min, env.mcs_max + 1):
        t, _, _ = tb_layout_from_mcs(m, env._num_re_t)
        tbs[m] = int(t.reshape(-1)[0].item())
    mcs_all = torch.arange(env.mcs_min, env.mcs_max + 1, dtype=torch.int32)

    def olla(target):
        return make_baseline_policy("olla", env, bler_target=target, olla_step_up_db=step)

    class Shifted:
        # CQI only: calibrated SINR of the CQI minus a fixed shift, then MCS at the target.
        def __init__(self, shift, target):
            self.shift, self.target = shift, target
            self.base = make_baseline_policy("olla", env, bler_target=bler)

        def __call__(self, state, info):
            s = self.base._cqi_to_sinr_db[int(info["cqi_index"])] - self.shift
            m = mcs_from_sinr_db(phy, s, env._num_re_t, env.mcs_min, env.mcs_max, bler_target=self.target)
            return m - env.mcs_min

    class QueueCap:
        # Base policy, then the smallest MCS whose TBS already covers the queue.
        def __init__(self, base):
            self.base = base

        def reset(self):
            self.base.reset()

        def __call__(self, state, info):
            a = self.base(state, info)
            q = info["queue"]
            m = a + env.mcs_min
            for mm in range(env.mcs_min, m):
                if tbs[mm] >= q:
                    return mm - env.mcs_min
            return a

    def genie(state, info):
        # True SINR at decision; maximize TBS(1-p) / (1 + gap * p) (first-tx rate per slot).
        lin = db_to_lin(torch.tensor([info["sinr_true_db"]], dtype=torch.float32))
        p, _ = tbler_from_phy(phy, mcs_all, lin, env._num_re_t)
        p = p.numpy().reshape(-1)
        q = info["queue"]
        pay = np.minimum(np.asarray([tbs[int(m)] for m in mcs_all]), q)
        score = pay * (1 - p) / (1 + gap * p)
        return int(np.argmax(score))

    def load_mopo(name):
        ck = torch.load(ROOT / "outputs" / name, map_location="cpu", weights_only=False)
        ag = DiscreteQ(int(ck["state_dim"]), int(ck["n_actions"]), gamma=float(ck.get("gamma", 0.9)))
        ag.q.load_state_dict(ck["q"])
        ag.q.eval()
        mu, sd = ck["state_mean"], ck["state_std"]
        return lambda s, i: ag.act(apply_norm(np.asarray(s, np.float32).reshape(1, -1), mu, sd), greedy=True)

    ck = torch.load(ROOT / "outputs" / "ddqn_seed0.pt", map_location="cpu", weights_only=False)
    ddqn_q = QNet(int(ck["state_dim"]), int(ck["n_actions"]), hidden=int(ck["hidden"]))
    ddqn_q.load_state_dict(ck["q"])
    ddqn_q.eval()

    def ddqn(state, info):
        with torch.no_grad():
            return int(ddqn_q(torch.as_tensor(state, dtype=torch.float32).view(1, -1)).argmax(1).item())

    policies = [
        ("ILLA", make_baseline_policy("illa", env, bler_target=bler)),
        ("OLLA t=0.10", olla(0.10)),
        ("OLLA t=0.05", olla(0.05)),
        ("OLLA t=0.02", olla(0.02)),
        ("CQI-3dB t=0.10", Shifted(3.0, 0.10)),
        ("CQI-4dB t=0.05", Shifted(4.0, 0.05)),
        ("OLLA t=0.10 +qcap", QueueCap(olla(0.10))),
        ("genie rate/slot", genie),
        ("DDQN (saved)", ddqn),
        ("MOPO g0.90", load_mopo("mopo_slot_g090_seed0.pt")),
        ("MOPO g0.95", load_mopo("mopo_slot_g095_seed0.pt")),
    ]

    print(f"seeds {SEEDS}; return = B_del / N_RE")
    for name, pol in policies:
        t0 = time.time()
        rows = np.asarray([run(env, pol, s) for s in SEEDS])
        m = rows.mean(0)
        print(f"{name:20s} return={m[0]:7.1f} ±{rows[:, 0].std():5.1f}  BLER1={m[1]:.3f}  "
              f"MCS={m[2]:5.1f}  slots/TB={m[3]:.2f}  ovf={m[4]:6.1f}   ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
