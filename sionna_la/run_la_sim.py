"""
Rule-based rollouts on DownlinkLAEnv: ILLA vs OLLA.

The environment owns all dynamics (channel, HARQ-IR, PHYAbstraction); this
script only drives it with baseline policies, plots the result, and dumps
transitions so the same trajectories can be reused as an offline dataset.

Two datasets come out of every rollout: the raw per-slot transitions, and a
per-transport-block version built from info["is_decision"] where each sample
is one MCS decision (see `build_decision_dataset`).

Run (conda env sionna_la):
  python run_la_sim.py
  python run_la_sim.py --num-slots 200 --seed 1
  python run_la_sim.py --epsilons 0.0 0.25    # extra coverage for offline RL
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.phy import config as sionna_config
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv
from policies import EpsilonGreedyPolicy, IllaPolicy, OllaPolicy


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rollout(env, policy, seed):
    """
    One episode. Records both the RL transition tuple and the PHY-level
    quantities needed for the link-adaptation plots.
    """
    obs, info = env.reset(seed=seed)
    policy.reset()

    log = {
        "obs": [],
        "action": [],
        "reward": [],
        "next_obs": [],
        "done": [],
        "mcs_used": [],
        "signalled_mcs": [],
        "ack": [],
        "k": [],
        "mi_tot": [],
        "sinr_true_db": [],
        "sinr_fb_db": [],
        "sinr_eq_db": [],
        "tbler": [],
        "decoded_bits": [],
        "dropped": [],
        "is_decision": [],
        "mi_saturated": [],
    }

    done = False
    while not done:
        action = policy(obs, info)
        k_before = info["k"]
        is_decision = info["is_decision"]
        sinr_fb_before = float(obs[0])

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        log["obs"].append(obs)
        log["action"].append(action)
        log["reward"].append(reward)
        log["next_obs"].append(next_obs)
        log["done"].append(done)
        log["mcs_used"].append(info["mcs_used"])
        log["signalled_mcs"].append(info["signalled_mcs"])
        log["ack"].append(info["ack"])
        log["k"].append(k_before)
        log["mi_tot"].append(info["mi_tot"])
        log["sinr_true_db"].append(info["sinr_true_db"])
        log["sinr_fb_db"].append(sinr_fb_before)
        log["sinr_eq_db"].append(info["sinr_eq_db"])
        log["tbler"].append(info["tbler"])
        log["decoded_bits"].append(info["decoded_bits"])
        log["dropped"].append(info["dropped"])
        log["is_decision"].append(is_decision)
        log["mi_saturated"].append(info["mi_saturated"])

        obs = next_obs

    return {k: np.asarray(v) for k, v in log.items()}


def build_decision_dataset(res):
    """
    Collapse the per-slot log into one transition per transport block.

    Only initial transmissions carry an action the agent chose, so a per-slot
    dataset feeds an offline learner transitions whose action had no effect.
    Aggregating each transport block into a single transition makes every
    sample a real decision, at the price of a variable time step (`num_slots`,
    which a semi-MDP discount beta^num_slots would use).

    A transport block still open when the episode truncates is discarded.
    """
    decision = res["is_decision"].astype(bool)
    finished = (res["ack"] == 1) | res["dropped"].astype(bool)
    starts = np.flatnonzero(decision)

    out = {k: [] for k in ("obs", "action", "reward", "next_obs", "done",
                           "num_slots", "num_retx", "mcs", "dropped")}
    for start in starts:
        ends = np.flatnonzero(finished[start:])
        if ends.size == 0:  # truncated mid-block
            break
        end = start + ends[0]
        out["obs"].append(res["obs"][start])
        out["action"].append(res["action"][start])
        out["reward"].append(res["reward"][start : end + 1].sum())
        out["next_obs"].append(res["next_obs"][end])
        out["done"].append(res["done"][end])
        out["num_slots"].append(end - start + 1)
        out["num_retx"].append(end - start)
        out["mcs"].append(res["mcs_used"][start])
        out["dropped"].append(res["dropped"][end])

    return {k: np.asarray(v) for k, v in out.items()}


def summarize(name, res, bler_target):
    ack = res["ack"]
    initial = res["is_decision"].astype(bool)
    retx = ~initial
    tb_done = (ack == 1) | res["dropped"]  # transport blocks that finished
    # share of retransmissions whose accumulated MI already exceeded the
    # modulation ceiling, i.e. that were guaranteed to decode
    sat = res["mi_saturated"][retx].mean() if retx.any() else 0.0
    print(
        f"[{name.upper()}] "
        f"return={res['reward'].sum():.1f} | "
        f"initial-tx BLER={1.0 - ack[initial].mean():.3f} (target {bler_target}) | "
        f"per-tx BLER={1.0 - ack.mean():.3f} | "
        f"mean MCS={res['mcs_used'][initial].mean():.1f} | "
        f"retx share={retx.mean():.3f} (MI-saturated {sat:.3f}) | "
        f"drops={int(res['dropped'].sum())}/{int(tb_done.sum())} TBs | "
        f"bits/slot={res['decoded_bits'].mean():.1f}"
    )


def _rolling(x, window):
    if len(x) < window:
        return x.astype(np.float64)
    kern = np.ones(window) / window
    return np.convolve(x.astype(np.float64), kern, mode="valid")


def plot_results(results, bler_target, out_path, mcs_window=50):
    ref_name, ref = next(iter(results.items()))
    slots = np.arange(len(ref["ack"]))
    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axs[0].plot(slots, ref["sinr_true_db"], label="true SINR", color="C0", lw=0.9)
    axs[0].plot(slots, ref["sinr_fb_db"], ":", label="noisy CQI", color="C1", alpha=0.8)
    # only the reference policy, otherwise the panel is unreadable; saturated
    # points carry no SINR information so they are marked apart
    retx = ~ref["is_decision"].astype(bool)
    sat = ref["mi_saturated"].astype(bool)
    axs[0].plot(
        slots[retx & ~sat], ref["sinr_eq_db"][retx & ~sat],
        ".", ms=4, color="C3", label=f"{ref_name.upper()} retx equiv. SINR",
    )
    axs[0].plot(
        slots[retx & sat], np.full(np.sum(retx & sat), ref["sinr_true_db"].max() + 2.0),
        "x", ms=3, color="C4", alpha=0.6, label="retx MI-saturated (decode certain)",
    )
    axs[0].set_ylabel("SINR [dB]")
    axs[0].legend(loc="upper left", fontsize=7, ncol=2)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("5G DL LA (Sionna PHYAbstraction + MIESM HARQ-IR)")

    # initial transmissions only, smoothed: the raw per-slot trace saturates
    # the panel once several policies are overlaid
    for name, res in results.items():
        init = res["is_decision"].astype(bool)
        m = _rolling(res["mcs_used"][init], mcs_window)
        axs[1].plot(slots[init][: len(m)], m, label=name.upper(), alpha=0.9)
    axs[1].set_ylabel(f"initial-tx MCS\n({mcs_window}-tx rolling mean)")
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3)

    for name, res in results.items():
        axs[2].plot(slots, np.cumsum(res["reward"]), label=name.upper())
    axs[2].set_ylabel("cumulative reward")
    axs[2].legend(loc="best", fontsize=8)
    axs[2].grid(True, alpha=0.3)

    # the 10% target is defined on first attempts, so average over initial
    # transmissions rather than over all slots
    for name, res in results.items():
        initial = res["is_decision"].astype(bool)
        ack = res["ack"][initial].astype(np.float64)
        emp_bler = 1.0 - np.cumsum(ack) / np.arange(1, len(ack) + 1)
        axs[3].plot(slots[initial], emp_bler, label=f"{name.upper()} initial-tx BLER")
    axs[3].axhline(bler_target, color="k", ls="--", label="BLER target")
    axs[3].set_ylabel("empirical BLER")
    axs[3].set_xlabel("slot")
    axs[3].legend(loc="best", fontsize=8)
    axs[3].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Rule-based LA rollouts (ILLA vs OLLA)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.num_slots is not None:
        cfg["num_slots"] = args.num_slots
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epsilons is not None:
        cfg["collect_epsilons"] = args.epsilons
    if args.out_dir is not None:
        cfg["out_dir"] = str(args.out_dir)

    seed = int(cfg["seed"])
    sionna_config.seed = seed
    torch.manual_seed(seed)
    bler_target = float(cfg["bler_target"])
    epsilons = [float(e) for e in cfg.get("collect_epsilons", [0.0])]

    print("=== Building Sionna PHYAbstraction (shared) ===")
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    n_actions = env.action_space.n
    print(f"obs={env.observation_space.shape} actions={n_actions}")

    baselines = {
        "illa": IllaPolicy(phy_abs, bler_target=bler_target),
        "olla": OllaPolicy(phy_abs, bler_target=bler_target),
    }

    results = {}
    for eps in epsilons:
        for name, base in baselines.items():
            policy = (
                base
                if eps == 0.0
                else EpsilonGreedyPolicy(base, n_actions, epsilon=eps, seed=seed)
            )
            label = name if eps == 0.0 else f"{name}_eps{eps:g}"
            print(f"=== Rollout: {label} ===")
            res = rollout(env, policy, seed=seed)
            summarize(label, res, bler_target)
            results[label] = res

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("save_plot", True):
        plot_path = out_dir / f"la_baselines_seed{seed}.png"
        plot_results(results, bler_target, plot_path)
        print(f"Saved plot -> {plot_path}")

    if cfg.get("save_npz", True):
        for label, res in results.items():
            npz_path = out_dir / f"la_{label}_seed{seed}.npz"
            np.savez_compressed(npz_path, **res, bler_target=bler_target, seed=seed)
            print(f"Saved per-slot transitions -> {npz_path}")

    if cfg.get("save_decision_npz", True):
        for label, res in results.items():
            ds = build_decision_dataset(res)
            npz_path = out_dir / f"la_{label}_seed{seed}_decisions.npz"
            np.savez_compressed(npz_path, **ds, bler_target=bler_target, seed=seed)
            print(
                f"Saved {len(ds['action'])} TB transitions "
                f"(mean {ds['num_slots'].mean():.2f} slots) -> {npz_path}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
