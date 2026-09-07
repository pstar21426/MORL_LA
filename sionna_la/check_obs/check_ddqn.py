"""Greedy replay + same-state MCS scatter vs ILLA/OLLA (after train_ddqn)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ddqn import DDQNAgent
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agent(ckpt: Path, env: DownlinkLAEnv) -> DDQNAgent:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent = DDQNAgent(
        int(data["state_dim"]),
        int(data["n_actions"]),
        hidden=int(data.get("hidden", 256)),
    )
    agent.q.load_state_dict(data["q"])
    agent.q.eval()
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "downlink_la.yaml")
    p.add_argument("--seed", type=int, default=1, help="eval seed (avoid train seed 0)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint
    if ckpt is None:
        base = Path(cfg.get("out_dir", "outputs"))
        if not base.is_absolute():
            base = _ROOT / base
        ckpt = base / f"ddqn_seed{int(cfg.get('seed', 0))}.pt"
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    agent = load_agent(ckpt, env)
    illa = make_baseline_policy("illa", env, bler_target=float(cfg["bler_target"]))
    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )

    seed_phy(args.seed)
    state, info = env.reset(seed=args.seed)
    illa.reset()
    olla.reset()

    ddqn_a, illa_a, olla_a, replay = [], [], [], []
    done = False
    while not done:
        a_ddqn = agent.select_action(state, greedy=True)
        replay.append(int(agent.select_action(state, greedy=True)))
        a_illa = illa(state, info)
        a_olla = olla(state, info)
        ddqn_a.append(int(a_ddqn))
        illa_a.append(int(a_illa))
        olla_a.append(int(a_olla))
        state, _, term, trunc, info = env.step(int(a_ddqn))
        done = term or trunc

    ddqn_a = np.asarray(ddqn_a)
    replay = np.asarray(replay)
    illa_a = np.asarray(illa_a)
    olla_a = np.asarray(olla_a)
    n_fail = int(np.sum(ddqn_a != replay))
    n = len(ddqn_a)
    lo = env.mcs_min

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.5))
    for ax, other, name in ((axs[0], illa_a, "ILLA"), (axs[1], olla_a, "OLLA")):
        ax.scatter(lo + other, lo + ddqn_a, s=10, alpha=0.4)
        ax.plot([lo, env.mcs_max], [lo, env.mcs_max], "k--", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{name} MCS (on DDQN trajectory)")
        ax.set_ylabel("DDQN MCS")
        agree = float(np.mean(other == ddqn_a))
        ax.set_title(f"DDQN vs {name}  agree={agree:.0%}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Same-state MCS — seed {args.seed}", fontsize=11)
    fig.tight_layout()
    scatter_path = out_dir / f"obs_ddqn_mcs_scatter_seed{args.seed}.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    md = [
        f"# DDQN greedy replay (eval seed={args.seed})",
        "",
        f"- TBs: {n}",
        f"- greedy replay mismatches: **{n_fail}** / {n}",
        f"- agree ILLA: {np.mean(ddqn_a == illa_a):.1%}",
        f"- agree OLLA: {np.mean(ddqn_a == olla_a):.1%}",
        f"- mean |MCS-ILLA|: {np.mean(np.abs(ddqn_a - illa_a)):.2f} actions",
        f"- scatter: `{scatter_path.name}`",
        "",
    ]
    md_path = out_dir / f"obs_table_ddqn_greedy_seed{args.seed}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"greedy mismatches: {n_fail}/{n}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {md_path}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
