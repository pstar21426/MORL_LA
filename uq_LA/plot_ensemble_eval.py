# Paired eval and MCS-vs-CQI maps for the ensemble DQN checkpoint vs OLLA / ILLA.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from uq_ensemble import EnsembleDQNAgent

HERE = Path(__file__).resolve().parent
ACK_COLS = slice(7, 10)
LEARNER = "ENS"
COLORS = {"ILLA": "C0", "OLLA": "C1", "ENS": "C2"}


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ensemble(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent = EnsembleDQNAgent(
        int(ck["state_dim"]),
        int(ck["n_actions"]),
        n_models=int(ck["n_models"]),
        hidden=int(ck.get("hidden", 256)),
        gamma=float(ck.get("gamma", 0.7)),
        rho=float(ck["rho"]) if ck.get("rho") is not None else None,
        seed=0,
    )
    for net, state in zip(agent.nets, ck["nets"]):
        net.load_state_dict(state)
        net.eval()
    return agent


def rollout(env, choose, seed):
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    rows = []
    done = False
    ret = 0.0
    while not done:
        action = int(choose(state, info))
        cqi = int(info["cqi_index"])
        acks = np.asarray(state[ACK_COLS])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        ret += float(reward)
        idle = bool(out.get("idle"))
        rows.append(
            dict(
                slot=min(int(out["decision_slot"]), env.num_slots - 1),
                idle=idle,
                tau=1 if idle else max(int(out["num_slots"]), 1),
                mcs=int(out["mcs_used"]),
                ack=int(out["ack"]),
                reward=float(reward),
                cqi=cqi,
                nacks=int((acks == 0).sum()),
            )
        )
    res = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    res["return"] = ret
    return res


def plot_paired(seeds, ret, out):
    names = ["ILLA", "OLLA", LEARNER]
    x = np.arange(len(seeds))
    w = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    for i, name in enumerate(names):
        axs[0].bar(x + (i - 1) * w, ret[name], w, color=COLORS[name], label=name)
    axs[0].set_xticks(x, [str(s) for s in seeds])
    axs[0].set_xlabel("eval seed")
    axs[0].set_ylabel("episode return")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, axis="y", alpha=0.3)

    d = np.asarray(ret[LEARNER]) - np.asarray(ret["OLLA"])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    axs[1].bar(x, d, color=np.where(d >= 0, "C2", "C3"))
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].axhline(d.mean(), color="k", ls="--", lw=1.0, label=f"mean {d.mean():+.1f} (SE {se:.1f})")
    axs[1].set_xticks(x, [str(s) for s in seeds])
    axs[1].set_xlabel("eval seed")
    axs[1].set_ylabel("ENS - OLLA")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Paired eval, greedy ensemble mean vs ILLA / OLLA", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bler(seeds, bler, target, out):
    names = ["ILLA", "OLLA", LEARNER]
    x = np.arange(len(seeds))
    w = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    for i, name in enumerate(names):
        axs[0].bar(x + (i - 1) * w, bler[name], w, color=COLORS[name], label=name)
    axs[0].axhline(target, color="k", ls="--", lw=1.0, label=f"target {target:g}")
    axs[0].set_xticks(x, [str(s) for s in seeds])
    axs[0].set_xlabel("eval seed")
    axs[0].set_ylabel("first-tx BLER")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, axis="y", alpha=0.3)

    d = np.asarray(bler[LEARNER]) - np.asarray(bler["OLLA"])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    axs[1].bar(x, d, color=np.where(d <= 0, "C2", "C3"))
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].axhline(d.mean(), color="k", ls="--", lw=1.0, label=f"mean {d.mean():+.3f} (SE {se:.3f})")
    axs[1].set_xticks(x, [str(s) for s in seeds])
    axs[1].set_xlabel("eval seed")
    axs[1].set_ylabel("ENS - OLLA")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("First-tx BLER on the same eval seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_policy_map(pooled, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for name in ("ILLA", "OLLA", LEARNER):
        r = pooled[name]
        busy = ~r["idle"]
        cq, mcs = r["cqi"][busy], r["mcs"][busy]
        xs = np.unique(cq)
        axs[0].plot(
            xs,
            [mcs[cq == c].mean() for c in xs],
            "o-",
            color=COLORS[name],
            ms=4,
            label=name,
        )
    axs[0].set_title("mean MCS vs reported CQI")
    axs[0].set_xlabel("CQI")
    axs[0].set_ylabel("MCS")
    axs[0].legend(fontsize=8)

    for ax, name in zip(axs[1:], ("OLLA", LEARNER)):
        r = pooled[name]
        busy = ~r["idle"]
        for k, ls in zip((0, 1, 2), ("-", "--", ":")):
            sel = busy & (np.minimum(r["nacks"], 2) == k)
            cq, mcs = r["cqi"][sel], r["mcs"][sel]
            xs = [c for c in np.unique(cq) if (cq == c).sum() >= 8]
            if not xs:
                continue
            ax.plot(
                xs,
                [mcs[cq == c].mean() for c in xs],
                "o",
                ls=ls,
                color=COLORS[name],
                ms=4,
                label=f"{k}{'+' if k == 2 else ''} NACK in last 3 (n={int(sel.sum())})",
            )
        ax.set_title(f"{name}: MCS vs CQI by recent NACKs")
        ax.set_xlabel("CQI")
        ax.legend(fontsize=7)
    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 16, 2))
    fig.suptitle("Greedy MCS choice, pooled over eval seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=HERE / "outputs" / "uq_ensemble_k5_seed0.pt")
    p.add_argument("--config", type=Path, default=HERE / "configs" / "downlink_la.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[205, 206, 207])
    p.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "plots")
    args = p.parse_args()

    cfg = load_config(args.config)
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    agent = load_ensemble(args.ckpt)
    bler_target = float(cfg["bler_target"])
    illa = make_baseline_policy("illa", env, bler_target=bler_target)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=cfg.get("olla_step_up_db")
    )

    def ens_choose(state, info):
        del info
        return agent.select_action(state, greedy=True)

    policies = (("ILLA", illa), ("OLLA", olla), (LEARNER, ens_choose))
    ret = {name: [] for name, _ in policies}
    bler = {name: [] for name, _ in policies}
    pooled = {name: [] for name, _ in policies}
    for seed in args.seeds:
        print(f"seed {seed}")
        for name, pol in policies:
            if hasattr(pol, "reset"):
                pol.reset()
            res = rollout(env, pol, seed)
            ret[name].append(res["return"])
            pooled[name].append(res)
            busy = ~res["idle"]
            bler[name].append(float(1.0 - res["ack"][busy].mean()))
            print(
                f"  {name}: return={res['return']:.1f} "
                f"MCS={res['mcs'][busy].mean():.2f} "
                f"BLER={1.0 - res['ack'][busy].mean():.3f}"
            )

    merged = {}
    for name, parts in pooled.items():
        merged[name] = {k: np.concatenate([r[k] for r in parts]) for k in parts[0] if k != "return"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_paired(args.seeds, ret, args.out_dir / "ensemble_eval_paired.png")
    plot_bler(args.seeds, bler, bler_target, args.out_dir / "ensemble_eval_bler.png")
    plot_policy_map(merged, args.out_dir / "ensemble_policy_map.png")
    print(f"Saved {args.out_dir / 'ensemble_eval_paired.png'}")
    print(f"Saved {args.out_dir / 'ensemble_eval_bler.png'}")
    print(f"Saved {args.out_dir / 'ensemble_policy_map.png'}")


if __name__ == "__main__":
    main()
