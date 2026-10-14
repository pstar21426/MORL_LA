# MCS vs CQI for the two Desideratum 3.2 policies.
# Greedy uses mean mu. Behavior uses the training action rule.

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
OUT = HERE / "outputs" / "des32"
SEEDS = [205, 206, 207]
ACK_COLS = slice(7, 10)
COLORS = {"alea": "C1", "epist": "C0", "OLLA": "C3"}


def load_config():
    with (HERE / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mode(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck["nets"][0]["backbone.0.weight"]
    mu_w = ck["nets"][0]["mu.weight"]
    agent = EnsembleDQNAgent(
        int(w.shape[1]),
        int(mu_w.shape[0]),
        n_models=int(ck["n_models"]),
        hidden=int(w.shape[0]),
        gamma=float(ck.get("gamma", 0.7)),
        rho=float(ck["rho"]) if ck.get("rho") is not None else None,
        seed=0,
    )
    for net, state in zip(agent.nets, ck["nets"]):
        net.load_state_dict(state)
        net.eval()
    agent.action_mode = ck.get("action_mode", "mean")
    return agent


def rollout(env, choose, seed):
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    rows = []
    done = False
    while not done:
        action = int(choose(state, info))
        cqi = int(info["cqi_index"])
        acks = np.asarray(state[ACK_COLS])
        state, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        if out.get("idle"):
            continue
        rows.append(
            (
                cqi,
                int(out["mcs_used"]),
                int((acks == 0).sum()),
            )
        )
    return np.asarray(rows, dtype=np.int64)


def mean_curve(rows, min_n=8):
    if rows.size == 0:
        return np.array([]), np.array([])
    cqi, mcs = rows[:, 0], rows[:, 1]
    xs, ys = [], []
    for c in range(16):
        sel = mcs[cqi == c]
        if sel.size >= min_n:
            xs.append(c)
            ys.append(float(sel.mean()))
    return np.asarray(xs), np.asarray(ys)


def main():
    cfg = load_config()
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )
    agents = {mode: load_mode(OUT / f"{mode}.pt") for mode in ("alea", "epist")}

    greedy, behavior, olla_rows = {}, {}, []
    for mode, agent in agents.items():
        g_parts, b_parts = [], []
        for s in SEEDS:
            g_parts.append(
                rollout(env, lambda st, info, a=agent: a.select_action(st, greedy=True), s)
            )
            b_parts.append(
                rollout(env, lambda st, info, a=agent: a.select_action(st, greedy=False), s)
            )
        greedy[mode] = np.concatenate(g_parts)
        behavior[mode] = np.concatenate(b_parts)
        print(
            f"{mode} greedy MCS {greedy[mode][:, 1].mean():.2f} "
            f"behavior MCS {behavior[mode][:, 1].mean():.2f} "
            f"n={len(greedy[mode])}",
            flush=True,
        )
    for s in SEEDS:
        olla.reset()
        olla_rows.append(rollout(env, olla, s))
    olla_rows = np.concatenate(olla_rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    panels = (
        (axes[0], greedy, "greedy, mean mu"),
        (axes[1], behavior, "behavior policy"),
    )
    for ax, data, title in panels:
        for mode in ("epist", "alea"):
            xs, ys = mean_curve(data[mode])
            ax.plot(xs, ys, "o-", color=COLORS[mode], ms=4, label=mode)
        xs, ys = mean_curve(olla_rows)
        ax.plot(xs, ys, "o--", color=COLORS["OLLA"], ms=4, label="OLLA")
        ax.set_title(title)
        ax.set_xlabel("CQI")
        ax.set_xlim(-0.5, 15.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean MCS")
    fig.suptitle(f"MCS vs CQI, seeds {SEEDS[0]}-{SEEDS[-1]}", fontsize=12)
    fig.tight_layout()
    path = OUT / "mcs_vs_cqi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
