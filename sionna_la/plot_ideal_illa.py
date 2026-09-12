# 4-line cum reward: ILLA / OLLA (기존 npz) + delay-free CQI ILLA + Sionna InnerLoopLinkAdaptation

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from run_la_sim import (
    _flush_xlim,
    _rolling,
    _slot_cum_return,
    _slot_hold,
    rollout,
    summarize,
)

_LABELS = {
    "illa": "ILLA",
    "olla": "OLLA",
    "illa_ideal_cqi": "ILLA-CQI (ideal)",
    "illa_ideal_sinr": "ILLA-SINR (Sionna)",
}


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_npz_rollout(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot_ideal_illa(results, num_slots, out_path, mcs_window=50):
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    slots = np.arange(num_slots)
    x_max = num_slots - 1

    for i, (name, res) in enumerate(results.items()):
        label = _LABELS.get(name, name.upper())
        mcs_slot = _slot_hold(res["mcs_used"], res["num_slots"], num_slots)
        axs[0].plot(
            slots,
            _rolling(mcs_slot, mcs_window),
            color=f"C{i}",
            lw=1.6,
            label=f"{label} {mcs_window}-slot mean",
        )
    axs[0].set_ylabel("MCS")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].grid(True, alpha=0.3)
    _flush_xlim(axs[0], x_max)

    for i, (name, res) in enumerate(results.items()):
        y = _slot_cum_return(res["reward"], res["num_slots"], num_slots)
        axs[1].plot(slots, y, color=f"C{i}", label=_LABELS.get(name, name.upper()))
    axs[1].set_ylabel("cum reward")
    axs[1].set_xlabel("slot")
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3)
    _flush_xlim(axs[1], x_max)

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
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed)
    num_slots = int(cfg["num_slots"])
    bler_target = float(cfg["bler_target"])
    olla_step_up_db = cfg.get("olla_step_up_db")

    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)

    results = {}
    for name in ("illa", "olla", "illa_ideal_cqi", "illa_ideal_sinr"):
        npz_path = out_dir / f"la_{name}_seed{seed}.npz"
        if npz_path.is_file():
            results[name] = load_npz_rollout(npz_path)
            summarize(name, results[name], bler_target, num_slots)
            continue
        if name in ("illa", "olla"):
            raise FileNotFoundError(
                f"{npz_path} 없음. 먼저 python run_la_sim.py 로 ILLA/OLLA npz를 만들 것"
            )
        policy = make_baseline_policy(
            name, env, bler_target=bler_target, olla_step_up_db=olla_step_up_db
        )
        print(f"=== Rollout: {name} ===")
        seed_phy(seed)
        res = rollout(env, policy, seed=seed)
        summarize(name, res, bler_target, num_slots)
        results[name] = res
        np.savez_compressed(
            npz_path,
            **res,
            bler_target=bler_target,
            seed=seed,
            episode_num_slots=num_slots,
        )
        print(f"Saved {len(res['action'])} TB transitions -> {npz_path}")

    path = out_dir / f"la_ideal_illa_seed{seed}.png"
    plot_ideal_illa(results, num_slots, path)
    print(f"Saved plot -> {path}")
    print("Done.")


if __name__ == "__main__":
    main()
