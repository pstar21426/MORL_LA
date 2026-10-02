# MCS held on each slot for ILLA, OLLA, and greedy MOPO. Seed 201.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from train_mopo import DiscreteQ, apply_norm

SEED = 201
OUT = HERE / "outputs" / "mcs_slots_illa_olla_mopo_seed201_s50k.png"


def expand(values, num_slots, total, served):
    y = np.full(int(total), np.nan, dtype=np.float64)
    t = 0
    for v, ns, ok in zip(values, num_slots, served):
        width = max(int(ns), 1)
        end = min(t + width, total)
        if ok and end > t:
            y[t:end] = float(v)
        t = end
        if t >= total:
            break
    return y


def collect(env, policy, seed):
    if hasattr(policy, "reset"):
        policy.reset()
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    mcs, ns, served = [], [], []
    done = False
    while not done:
        action = int(policy(state, info))
        state, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        out = info["outcome"]
        mcs.append(int(out["mcs_used"]))
        ns.append(int(out["num_slots"]))
        served.append(not out.get("idle"))
    sinr = np.asarray(env._sinr_true_db[: env.num_slots], dtype=np.float64)
    return np.asarray(mcs), np.asarray(ns), np.asarray(served), sinr


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler = float(cfg["bler_target"])
    total = int(env.num_slots)
    ckpt = torch.load(
        ROOT / "outputs" / "mopo_slot_g090_seed0.pt", map_location="cpu", weights_only=False
    )
    agent = DiscreteQ(int(ckpt["state_dim"]), int(ckpt["n_actions"]), gamma=float(ckpt["gamma"]))
    agent.q.load_state_dict(ckpt["q"])
    agent.q.eval()
    mean, std = ckpt["state_mean"], ckpt["state_std"]

    def mopo(state, info):
        del info
        x = apply_norm(np.asarray(state, dtype=np.float32).reshape(1, -1), mean, std)
        return agent.act(x, greedy=True)

    illa = make_baseline_policy("illa", env, bler_target=bler)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler, olla_step_up_db=cfg.get("olla_step_up_db")
    )
    policies = (("ILLA", illa, "C0"), ("OLLA", olla, "C1"), ("MOPO", mopo, "C2"))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, sharey=True)
    slots = np.arange(total)
    sinr = None
    for ax, (name, pol, color) in zip(axes, policies):
        mcs, ns, served, sinr = collect(env, pol, SEED)
        y = expand(mcs, ns, total, served)
        ax.step(slots, y, where="post", color=color, lw=0.9)
        ax.set_ylabel("MCS")
        ax.set_ylim(2, 30)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        print(name, "mean MCS", round(float(mcs[served].mean()), 2), "TBs", int(served.sum()))
    twin = axes[0].twinx()
    twin.plot(slots, sinr, color="0.45", lw=0.7, alpha=0.8)
    twin.set_ylabel("true SINR (dB)")
    twin.set_ylim(-5, 30)
    axes[-1].set_xlabel("slot")
    fig.suptitle("seed 201, bias +3 dB", y=1.01)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
