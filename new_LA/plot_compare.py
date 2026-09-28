# Slot-axis comparison: ILLA / OLLA / greedy DDQN
# Panels: cumulative return, cumulative delivered bits/RE, running first-tx BLER, cumulative drops
# Drops = overflow bits and HARQ-discarded bits, each divided by the RE count.
# Delivered bits/RE counts only the payload of a successful TB.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sionna.sys import PHYAbstraction

from ddqn import DDQNAgent
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from run_la_sim import _slot_cum_return, _slot_hold, load_config, rollout
from train_ddqn import resolve_held_out_eval_seed


def _running_served(values, served):
    out = np.zeros(len(values), dtype=np.float64)
    acc = 0.0
    n = 0
    last = 0.0
    for i, (v, ok) in enumerate(zip(values, served)):
        if not ok:
            out[i] = last
            continue
        n += 1
        acc += float(v)
        last = acc / n
        out[i] = last
    return out


def _curves(res, total_slots):
    slots = np.asarray(res["num_slots"])
    served = slots > 0
    reward = _slot_cum_return(res["reward"], slots, total_slots)
    ack_run = _running_served(np.asarray(res["ack"], dtype=np.float64), served)
    bler = _slot_hold(1.0 - ack_run, slots, total_slots)
    if "lost_se" in res:
        drop_n = np.asarray(res["lost_se"], dtype=np.float64)
    else:
        drop_n = np.asarray(res["dropped"], dtype=np.float64) + np.asarray(
            res["n_overflow"], dtype=np.float64
        )
    drops = _slot_cum_return(drop_n, slots, total_slots)
    if "delivered_se" in res:
        delivered_n = np.asarray(res["delivered_se"], dtype=np.float64)
    else:
        delivered_n = np.zeros(len(slots), dtype=np.float64)
    delivered = _slot_cum_return(delivered_n, slots, total_slots)
    return reward, delivered, bler, drops


class _Greedy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, state, info):
        del info
        return self.agent.select_action(state, greedy=True)


def _load_agent(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent = DDQNAgent(
        int(ckpt["state_dim"]),
        int(ckpt["n_actions"]),
        hidden=int(ckpt.get("hidden", 256)),
    )
    agent.q.load_state_dict(ckpt["q"])
    agent.q.eval()
    return agent


def plot_compare(curves, total_slots, bler_target, out_path, seed):
    t = np.arange(total_slots)
    styles = {
        "ILLA": dict(color="C0", lw=1.6),
        "OLLA": dict(color="C1", lw=1.6),
        "DDQN": dict(color="C2", lw=1.6, ls="--"),
    }
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    for name, (reward, _delivered, _bler, _drops) in curves.items():
        axs[0].plot(t, reward, label=name, **styles[name])
    axs[0].set_ylabel("cumulative return")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].grid(True, alpha=0.3)

    for name, (_reward, delivered, _bler, _drops) in curves.items():
        axs[1].plot(t, delivered, label=name, **styles[name])
    axs[1].set_ylabel("delivered bits / RE")
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3)

    for name, (_reward, _delivered, bler, _drops) in curves.items():
        axs[2].plot(t, bler, label=name, **styles[name])
    axs[2].axhline(bler_target, color="k", ls=":", lw=1.0, label="target")
    axs[2].set_ylabel("first-tx BLER")
    axs[2].set_ylim(-0.02, 1.02)
    axs[2].legend(loc="best", fontsize=8)
    axs[2].grid(True, alpha=0.3)

    for name, (_reward, _delivered, _bler, drops) in curves.items():
        axs[3].plot(t, drops, label=name, **styles[name])
    axs[3].set_ylabel("lost bits / RE")
    axs[3].set_xlabel("slot")
    axs[3].legend(loc="best", fontsize=8)
    axs[3].grid(True, alpha=0.3)

    for ax in axs:
        ax.set_xlim(0, total_slots - 1)
        ax.margins(x=0)

    fig.suptitle(f"ILLA / OLLA / DDQN (seed={seed})", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--seed", type=int, default=None, help="eval seed before held-out shift")
    p.add_argument("--ddqn-ckpt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    requested = int(args.seed if args.seed is not None else (cfg.get("train") or {}).get("eval_seeds", [1])[0])
    seed, train_seed, episodes = resolve_held_out_eval_seed(cfg, requested)
    if seed != requested:
        print(
            f"eval seed {requested} overlaps training "
            f"[{train_seed}, {train_seed + episodes}); using {seed}"
        )

    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    ckpt = args.ddqn_ckpt or out_dir / f"ddqn_seed{int(cfg.get('seed', 0))}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"DDQN checkpoint not found: {ckpt}")

    total_slots = int(cfg["num_slots"])
    bler_target = float(cfg["bler_target"])
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    agent = _load_agent(ckpt)

    policies = {
        "ILLA": make_baseline_policy("illa", env, bler_target=bler_target),
        "OLLA": make_baseline_policy(
            "olla",
            env,
            bler_target=bler_target,
            olla_step_up_db=cfg.get("olla_step_up_db"),
        ),
        "DDQN": _Greedy(agent),
    }

    curves = {}
    for name, policy in policies.items():
        seed_phy(seed)
        res = rollout(env, policy, seed)
        curves[name] = _curves(res, total_slots)
        reward, delivered, bler, drops = curves[name]
        print(
            f"{name}: return={reward[-1]:.1f}  delivered={delivered[-1]:.1f}  "
            f"BLER={bler[-1]:.3f}  drops={drops[-1]:.0f}"
        )

    out_path = out_dir / f"compare_seed{seed}.png"
    plot_compare(curves, total_slots, bler_target, out_path, seed)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
