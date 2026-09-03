# Slot-axis cumulative return: ILLA / OLLA / DQN

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sionna.sys import PHYAbstraction

from dqn import DQNAgent
from la_env import DownlinkLAEnv, seed_phy
from train_dqn import load_config


def slot_cum_return(rewards, slots_per_tb, total_slots):
    y = np.zeros(total_slots, dtype=np.float64)
    t = 0
    cum = 0.0
    for r, ns in zip(rewards, slots_per_tb):
        ns = max(int(ns), 1)
        cum += float(r)
        end = min(t + ns, total_slots)
        y[t:end] = cum
        t = end
        if t >= total_slots:
            break
    if t < total_slots:
        y[t:] = cum
    return y


def load_npz_rollout(path):
    d = np.load(path)
    return {
        "reward": d["reward"],
        "num_slots": d["num_slots"],
    }


def dqn_rollout(env, ckpt_path, seed, hidden=256):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    hidden = int(ckpt.get("hidden", hidden))

    agent = DQNAgent(state_dim, n_actions, hidden=hidden)
    agent.q.load_state_dict(ckpt["q"])
    agent.q.eval()

    seed_phy(seed)
    state, _ = env.reset(seed=seed)
    rewards, slots = [], []

    done = False
    while not done:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
        slots.append(int(info["outcome"]["num_slots"]))

    return {"reward": np.asarray(rewards), "num_slots": np.asarray(slots)}


def plot_t_return(curves, total_slots, out_path, seed):
    t = np.arange(total_slots)
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        "ILLA": dict(color="C0", lw=1.6, label="ILLA"),
        "OLLA": dict(color="C1", lw=1.6, label="OLLA"),
        "DQN": dict(color="C2", lw=1.6, label="DQN", ls="--"),
    }
    for name, y in curves.items():
        ax.plot(t, y, **styles[name])

    ymax = max(float(y.max()) for y in curves.values())
    ax.set_xlim(0, total_slots - 1)
    ax.set_ylim(0, ymax * 1.02 if ymax > 0 else 1.0)
    ax.margins(x=0, y=0)
    ax.set_xlabel("slot index")
    ax.set_ylabel("cumulative return")
    ax.set_title(f"Cumulative return vs time (seed={seed})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--illa-npz", type=Path, default=None)
    p.add_argument("--olla-npz", type=Path, default=None)
    p.add_argument("--dqn-ckpt", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir

    total_slots = int(cfg["num_slots"])
    illa_npz = args.illa_npz or out_dir / f"la_illa_seed{seed}.npz"
    olla_npz = args.olla_npz or out_dir / f"la_olla_seed{seed}.npz"
    dqn_ckpt = args.dqn_ckpt or out_dir / f"dqn_seed{int(cfg.get('seed', 0))}.pt"

    illa = load_npz_rollout(illa_npz)
    olla = load_npz_rollout(olla_npz)

    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)

    if not dqn_ckpt.is_file():
        raise FileNotFoundError(f"DQN checkpoint not found: {dqn_ckpt}")
    dqn = dqn_rollout(env, dqn_ckpt, seed)

    curves = {
        "ILLA": slot_cum_return(illa["reward"], illa["num_slots"], total_slots),
        "OLLA": slot_cum_return(olla["reward"], olla["num_slots"], total_slots),
        "DQN": slot_cum_return(dqn["reward"], dqn["num_slots"], total_slots),
    }

    out_path = out_dir / f"t_return_seed{seed}.png"
    plot_t_return(curves, total_slots, out_path, seed)
    print(f"Saved -> {out_path}")
    for name, y in curves.items():
        print(f"  {name}: final cum return = {y[-1]:.1f}")


if __name__ == "__main__":
    main()
