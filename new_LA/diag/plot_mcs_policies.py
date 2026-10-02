# MCS choices of ILLA, OLLA, and greedy MOPO on the same held-out seeds.

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

SEEDS = (201, 202, 203)
OUT = HERE / "outputs" / "mcs_illa_olla_mopo_slotg090_s50k.png"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler = float(cfg["bler_target"])
    ckpt = torch.load(ROOT / "outputs" / "mopo_slot_g090_seed0.pt", map_location="cpu", weights_only=False)
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
    policies = (("ILLA", illa), ("OLLA", olla), ("MOPO", mopo))
    colors = {"ILLA": "C0", "OLLA": "C1", "MOPO": "C2"}
    rows = {name: [] for name, _ in policies}

    for seed in SEEDS:
        for name, pol in policies:
            if hasattr(pol, "reset"):
                pol.reset()
            seed_phy(seed)
            state, info = env.reset(seed=seed)
            done = False
            while not done:
                action = int(pol(state, info))
                mcs = int(env.mcs_from_action(action))
                cqi = int(info["cqi_index"])
                queue = int(info["queue"])
                next_state, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                out = info["outcome"]
                if not out.get("idle"):
                    rows[name].append((mcs, cqi, queue, int(out["mcs_used"])))
                state = next_state
            print(name, "seed", seed, "n", len(rows[name]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.arange(2.5, 29.5, 1.0)
    for name, _ in policies:
        mcs = np.asarray([r[0] for r in rows[name]])
        axes[0].hist(
            mcs, bins=bins, histtype="step", density=True, lw=1.6, label=name, color=colors[name]
        )
        print(name, "MCS mean", round(float(mcs.mean()), 2), "p10/50/90",
              [round(float(np.percentile(mcs, p)), 1) for p in (10, 50, 90)])
    axes[0].set_xlabel("MCS")
    axes[0].set_ylabel("density")
    axes[0].set_title("MCS at decision, seeds 201-203")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, _ in policies:
        cqi = np.asarray([r[1] for r in rows[name]], dtype=float)
        mcs = np.asarray([r[0] for r in rows[name]], dtype=float)
        xs, ys = [], []
        for q in range(0, 16):
            mask = cqi == q
            if not mask.any():
                continue
            xs.append(q)
            ys.append(mcs[mask].mean())
        axes[1].plot(xs, ys, marker="o", ms=4, lw=1.4, label=name, color=colors[name])
    axes[1].set_xlabel("CQI at decision")
    axes[1].set_ylabel("mean MCS")
    axes[1].set_title("MCS vs reported CQI")
    axes[1].set_xticks(range(0, 16, 2))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)


if __name__ == "__main__":
    main()
