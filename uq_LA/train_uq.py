# Train UncertaintyDQN and run quick uncertainty diagnostics.

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from uq_dqn import Transition, UncertaintyDQNAgent


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def held_out_eval_seeds(train_seed, episodes, requested):
    used = set(range(int(train_seed), int(train_seed) + int(episodes)))
    step = max(int(episodes), 1)
    out = []
    for s in requested:
        s = int(s)
        while s in used:
            s += step
        used.add(s)
        out.append(s)
    return out


def rollout_metrics(env, agent, seed, *, train=True, greedy=False, collect_uq=False):
    state, info = env.reset(seed=seed)
    st = dict(ret=0.0, tbs=0, n_ack=0, n_slots=0)
    uq_rows = []
    done = False
    while not done:
        cqi = int(info.get("cqi_index", 0))
        action = agent.select_action(state, greedy=greedy)
        if collect_uq and not train:
            mu, u_a, u_e = agent.uncertainty_at(state, action)
            uq_rows.append(
                dict(
                    cqi=cqi,
                    action=action,
                    mu=mu,
                    u_alea=u_a,
                    u_epist=u_e,
                    visit=agent.cqi_visits[cqi],
                )
            )
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        if train:
            slots = 1 if out.get("idle") else max(int(out["num_slots"]), 1)
            agent.push(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    terminated=bool(terminated),
                    slots=slots,
                    cqi=cqi,
                )
            )
            agent.train_step()
        if collect_uq and uq_rows and not out.get("idle"):
            uq_rows[-1]["num_retx"] = int(out.get("num_retx", 0))
            uq_rows[-1]["ack"] = int(out.get("ack", 0))
        st["ret"] += float(reward)
        if not out.get("idle"):
            st["tbs"] += 1
            st["n_ack"] += int(out["ack"])
            st["n_slots"] += int(out["num_slots"])
        state = next_state
    st["return_per_slot"] = st["ret"] / max(env.num_slots, 1)
    return st, uq_rows


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def summarize_uq(rows):
    rows = [r for r in rows if "num_retx" in r]
    if not rows:
        return {}
    u_a = np.asarray([r["u_alea"] for r in rows], dtype=np.float64)
    u_e = np.asarray([r["u_epist"] for r in rows], dtype=np.float64)
    retx = np.asarray([r.get("num_retx", 0) for r in rows], dtype=np.float64)
    visit = np.asarray([r.get("visit", 0) for r in rows], dtype=np.float64)
    risky = retx > 0
    out = {
        "n": len(rows),
        "u_alea_mean": float(u_a.mean()),
        "u_epist_mean": float(u_e.mean()),
        "alea_retx_gt0": float(u_a[risky].mean()) if risky.any() else float("nan"),
        "alea_retx_eq0": float(u_a[~risky].mean()) if (~risky).any() else float("nan"),
        "rho_alea_retx": spearman(u_a, retx),
        "rho_epist_inv_visit": spearman(u_e, 1.0 / np.sqrt(visit + 1.0)),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--double-q", action="store_true", default=None)
    p.add_argument("--no-double-q", action="store_true", help="use vanilla DQN (paper default)")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train") or {}
    uq_cfg = cfg.get("uq_dqn") or {}

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 120))
    log_every = int(train_cfg.get("log_every", 20))

    double_q = bool(uq_cfg.get("double_q", True))
    if args.no_double_q:
        double_q = False
    if args.double_q:
        double_q = True

    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    seed_phy(seed)
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    state_dim = int(np.prod(env.state_space.shape))
    n_actions = env.action_space.n

    rho_ema = float(uq_cfg.get("rho_ema", 0.1))
    agent = UncertaintyDQNAgent(
        state_dim,
        n_actions,
        hidden=int(uq_cfg.get("hidden", 256)),
        gamma=float(uq_cfg.get("gamma", 0.7)),
        lr=float(uq_cfg.get("lr", 1e-3)),
        buffer_size=int(uq_cfg.get("buffer_size", 100_000)),
        batch_size=int(uq_cfg.get("batch_size", 64)),
        target_sync=int(uq_cfg.get("target_sync", 200)),
        epsilon_start=float(uq_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(uq_cfg.get("epsilon_end", 0.01)),
        epsilon_decay_steps=int(uq_cfg.get("epsilon_decay_steps", 30_000)),
        rho=0.0,
        double_q=double_q,
        evi_reg=float(uq_cfg.get("evi_reg", 0.1)),
        evi_loss_weight=float(uq_cfg.get("evi_loss_weight", 1.0)),
    )

    tag = "ddqn" if double_q else "dqn"
    print(
        f"=== UQ-{tag.upper()} train ({episodes} ep) gamma={agent.gamma} "
        f"queue_feat={env.queue_feat} ==="
    )

    for ep in range(episodes):
        m, _ = rollout_metrics(env, agent, seed=seed + ep, train=True)
        ep_rate = m["return_per_slot"]
        agent.rho = ep_rate if ep == 0 else (1.0 - rho_ema) * agent.rho + rho_ema * ep_rate
        if (ep + 1) % log_every == 0 or ep == 0:
            print(
                f"  ep {ep + 1:4d} | return={m['ret']:.1f} | "
                f"rate={ep_rate:.4f} | rho={agent.rho:.4f} | "
                f"eps={agent.epsilon:.3f} | q_loss={agent.last_q_loss:.4f} | "
                f"evi_loss={agent.last_evi_loss:.4f}"
            )

    ckpt = out_dir / f"uq_{tag}_seed{seed}.pt"
    torch.save(
        {
            **agent.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "gamma": agent.gamma,
            "rho": agent.rho,
            "double_q": double_q,
            "episodes": episodes,
        },
        ckpt,
    )
    print(f"Saved {ckpt}")

    eval_seeds = held_out_eval_seeds(
        seed, episodes, [int(s) for s in train_cfg.get("eval_seeds", [205, 206, 207])]
    )
    olla = make_baseline_policy(
        "olla", env, bler_target=float(cfg["bler_target"]), olla_step_up_db=cfg.get("olla_step_up_db")
    )

    id_rows = []
    ood_rows = []
    returns = []
    olla_returns = []

    cfg_id = dict(cfg)
    cfg_ood = dict(cfg)
    cfg_ood["cqi_bias_db"] = 0.0

    env_id = DownlinkLAEnv.from_config(cfg_id, phy_abs=phy_abs)
    env_ood = DownlinkLAEnv.from_config(cfg_ood, phy_abs=phy_abs)

    for es in eval_seeds:
        seed_phy(es)
        m_id, uq_id = rollout_metrics(env_id, agent, seed=es, train=False, greedy=True, collect_uq=True)
        seed_phy(es)
        m_ood, uq_ood = rollout_metrics(
            env_ood, agent, seed=es, train=False, greedy=True, collect_uq=True
        )
        seed_phy(es)
        olla.reset()
        state, info = env_id.reset(seed=es)
        olla_ret = 0.0
        done = False
        while not done:
            action = olla(state, info)
            state, reward, terminated, truncated, info = env_id.step(action)
            olla_ret += float(reward)
            done = terminated or truncated
        olla_m = {"ret": olla_ret}
        returns.append(m_id["ret"])
        olla_returns.append(olla_m["ret"])
        id_rows.extend(uq_id)
        ood_rows.extend(uq_ood)
        print(
            f"seed {es}: ID return={m_id['ret']:.1f} OOD(bias=0)={m_ood['ret']:.1f} "
            f"OLLA={olla_m['ret']:.1f}"
        )

    sum_id = summarize_uq(id_rows)
    sum_ood = summarize_uq(ood_rows)
    u_ep_id = np.asarray([r["u_epist"] for r in id_rows], dtype=np.float64)
    u_ep_ood = np.asarray([r["u_epist"] for r in ood_rows], dtype=np.float64)

    report = [
        f"# UQ-{tag} smoke report (train seed={seed}, {episodes} ep)",
        "",
        f"- Mean eval return (ID): **{np.mean(returns):.1f}** vs OLLA **{np.mean(olla_returns):.1f}** "
        f"(delta {np.mean(returns) - np.mean(olla_returns):+.1f})",
        f"- Epistemic mean ID vs OOD (cqi_bias 3→0): **{u_ep_id.mean():.3f}** vs **{u_ep_ood.mean():.3f}**",
        f"- Aleatoric mean if num_retx>0 vs ==0: **{sum_id.get('alea_retx_gt0', float('nan')):.3f}** vs "
        f"**{sum_id.get('alea_retx_eq0', float('nan')):.3f}** (Spearman vs retx {sum_id.get('rho_alea_retx', float('nan')):.3f})",
        f"- Epistemic vs 1/sqrt(visit): Spearman **{sum_id.get('rho_epist_inv_visit', float('nan')):.3f}**",
        "",
        "## Interpretation",
        "",
        "- **DQN vs DDQN**: paper uses vanilla DQN; DDQN is for stable Q in this LA MDP. "
        "Uncertainty head is trained on one-step `r - rho*tau` either way.",
        "- Positive OOD epistemic gap + higher aleatoric on HARQ retries supports disentanglement.",
    ]
    md_path = out_dir / f"uq_{tag}_report_seed{seed}.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    np.savez_compressed(
        out_dir / f"uq_{tag}_diag_seed{seed}.npz",
        id_u_alea=np.asarray([r["u_alea"] for r in id_rows]),
        id_u_epist=np.asarray([r["u_epist"] for r in id_rows]),
        id_retx=np.asarray([r.get("num_retx", 0) for r in id_rows]),
        ood_u_epist=np.asarray([r["u_epist"] for r in ood_rows]),
        eval_returns=np.asarray(returns),
        olla_returns=np.asarray(olla_returns),
        **{f"sum_id_{k}": v for k, v in sum_id.items()},
    )
    print(f"Wrote {md_path}")
    print("\n".join(report[2:6]))


if __name__ == "__main__":
    main()
